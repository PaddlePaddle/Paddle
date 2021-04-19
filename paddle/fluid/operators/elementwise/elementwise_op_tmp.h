// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ddim.h"

#define MAX_DIMS 10
#define INT_BITS 32

namespace paddle {
namespace operators {

template <typename IndexT>
struct FastDivMod {
  FastDivMod() {}

  explicit FastDivMod(IndexT d) : divisor(d) {
    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      if ((1 << shift_val) >= divisor) {
        break;
      }
    }
    uint64_t one_uint64 = 1;
    uint64_t temp_div =
        ((one_uint64 << INT_BITS) * ((one_uint64 << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __forceinline__ __device__ IndexT div(IndexT n) const {
    IndexT t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __forceinline__ __device__ IndexT *divmod(IndexT n) const {
    IndexT q = div(n);
    IndexT arr[2] = {q, n - q * divisor};
    // return DivMod<IndexT>(q, n - q * divisor);
    return arr;
  }

  IndexT divisor;
  IndexT multiplier;
  IndexT shift_val;
};

template <typename T, int Size>
struct alignas(sizeof(T) * Size) CudaAlignedVector {
  T val[Size];
};

/* 1. To compensate the lackage of input_tensors dimension;
*  2. To Merge the dimensions of input_tensors while the consequtive
* equal-dimensions appears;
*  3. To Merge the dimension of input_tensors while the consequtive
* 1-value-dimensions appears;
*  4. To calculate the strides of each input_tensor. */
struct DimensionTransform {
  using vec_t = std::vector<uint64_t>;

 private:
  __inline__ void DimsExtend(int N) {
    for (auto in_dim : vec_dims) {
      std::reverse(in_dim.begin(), in_dim.end());

      if (in_dim.size() < dim_size) {
        vec_t tmp_dim(dim_size, 1);
        in_dim.resize(dim_size, 1);
        uint64_t idx_in = 0, idx_out = 0;
        do {
          if (in_dim[idx_in] == out_dims[idx_out] || in_dim[idx_in] == 1) {
            tmp_dim[idx_out++] = in_dim[idx_in++];
          } else {
            idx_out++;
          }
        } while (idx_out < dim_size);

        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      }
    }
  }

  template <typename func_t>
  __inline__ void DimsReorganise(func_t merge_func, int N) {
    auto VectorReorganise = [](vec_t *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] =
          std::accumulate(vec->begin() + l_idx, vec->begin() + m_idx, 1,
                          std::multiplies<uint64_t>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };

    uint64_t i = 0;
    while (i < dim_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal_flag = true;
      do {
        merge_func(equal_flag, vec_dims, out_dims, i, N);
        if (equal_flag) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < dim_size);

      if (cnt > 1) {
        for (auto in_dim : vec_dims) {
          VectorReorganise(&in_dim, low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        dim_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

 public:
  DimensionTransform(std::vector<const framework::Tensor *> *ins,
                     const framework::DDim dims, int N) {
    dim_size = dims.size();
    out_dims = framework::vectorize<uint64_t>(dims);
    vec_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      vec_dims[j] = framework::vectorize<uint64_t>(((*ins)[j])->dims());
    }
    DimsExtend(N);

    auto merge_sequential_dims = [](bool &equal, std::vector<vec_t> &in_dims,
                                    vec_t &out, int i, int num) -> void {
      for (int j = 1; j < num; ++j) {
        equal = (in_dims[0][i] == in_dims[j][i]);
      }
    };

    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<vec_t> &in_dims, vec_t &out,
                                        int i, int num) -> void {
      if (in_dims[0][i] == 1) {
        for (auto in_dim : in_dims) {
          if (in_dim[i] == out[i]) {
            equal = true;
          } else {
            PADDLE_ENFORCE_NE(
                in_dim[i], out[i],
                platform::errors::InvalidArgument(
                    "One input tensor %d th dimension(%d) is neither equal to"
                    "the output tensor %d th dimension(%d), nor equal to 1.\n",
                    i, in_dim[i], out[i]));
          }
        }
      }
    };

    typedef void (*func_t)(bool &, std::vector<vec_t> &, vec_t &, int, int);
    func_t merge_ptr = merge_sequential_dims;
    DimsReorganise<func_t>(merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(vec_dims[0].begin(), vec_dims[0].end(), 1,
                                  std::multiplies<uint64_t>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(vec_dims[j].begin(), vec_dims[j].end(), 1,
                                 std::multiplies<uint64_t>());
      min_val = min_val > temp ? temp : min_val;
      min_idx = min_val == temp ? j : min_idx;
    }
    std::swap(vec_dims[0], vec_dims[min_idx]);

    merge_ptr = merge_sequential_one_dims;
    DimsReorganise<func_t>(merge_ptr, N);
  }

  uint64_t dim_size;
  vec_t out_dims;
  std::vector<vec_t> vec_dims;
};

template <typename merge_t, typename uint64_t = int>
struct OffsetPreCalculator {
 private:
  template <typename vec_t>
  __inline__ void StirdeCalculator(int N, int dim_size, const vec_t &in_dims) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < dim_size; ++i) {
        strides[j][i] = in_dims[j][i] == 1 ? 0 : strides[j][i];
        strides[j][i] = (i != 0) ? std::accumulate(in_dims[j].begin(),
                                                   in_dims[j].begin() + i, 1,
                                                   std::multiplies<uint64_t>())
                                 : strides[j][i];
      }
    }
  }

 public:
  explicit OffsetPreCalculator(merge_t *merge_dims) {
    auto N = merge_dims->vec_dims.size();
    int dim_size = merge_dims->dim_size;
    divmoder.resize(dim_size);
    strides.resize(N, std::vector<uint64_t>(dim_size, 1));

    for (int i = 0; i < dim_size; ++i) {
      divmoder[i] = FastDivMod<uint64_t>(merge_dims->out_dims[i]);
    }
    auto vec_dims = merge_dims->vec_dims;
    StirdeCalculator<decltype(vec_dims)>(N, dim_size, vec_dims);
  }

  std::vector<std::vector<uint64_t>> strides;
  std::vector<FastDivMod<uint64_t>> divmoder;
};

template <typename T, typename ArgT, typename OffsetT, int nDims, int vec_size,
          typename IndexT = uint32_t>
struct StrideVectorLoad {
 private:
  __device__ inline IndexT const get_offset(int tid) {
    IndexT offset = 0;
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoders[i].divmod(tid);
      tid = fast_divmoder[0];
      offset += fast_divmoder[1] * strides[i];
    }
    return offset;
  }

 public:
  StrideVectorLoad(const OffsetT *offset_calculator, int i) {
    memcpy(static_cast<void *>(strides), offset_calculator->strides[i].data(),
           sizeof(uint32_t) * nDims);
    memcpy(static_cast<void *>(divmoders), offset_calculator->divmoder.data(),
           sizeof(FastDivMod<uint32_t>) * nDims);
  }

  __device__ inline void operator()(const T *in_data, ArgT args, int tid,
                                    int thresh) {
    if (tid < thresh) {
      int index = vec_size * tid;
      for (int i = 0; i < vec_size; ++i) {
        IndexT offset = get_offset(index + i);
        args.val[i + i] = in_data[offset];
      }
    }
  }
  IndexT strides[MAX_DIMS];
  FastDivMod<IndexT> divmoders[MAX_DIMS];
};

template <typename T, typename OffsetT, int nDims, typename IndexT = uint32_t>
struct StrideScalarLoad {
 private:
  inline __device__ IndexT const get_offset(int tid) {
    IndexT offset = 0;
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoders[i].divmod(tid);
      tid = fast_divmoder[0];
      offset += fast_divmoder[1] * strides[i];
    }
    return offset;
  }

 public:
  explicit StrideScalarLoad(const OffsetT *offset_calculator, int i) {
    memcpy(static_cast<void *>(strides), offset_calculator->strides[i].data(),
           sizeof(uint32_t) * nDims);
    memcpy(static_cast<void *>(divmoders), offset_calculator->divmoder.data(),
           sizeof(FastDivMod<uint32_t>) * nDims);
  }

  __device__ inline void operator()(const T *in_data, T args, int tid,
                                    size_t thresh) {
    if (tid < thresh) {
      auto offset = get_offset(tid);
      args = in_data[offset];
    }
  }

  IndexT strides[MAX_DIMS];
  FastDivMod<IndexT> divmoders[MAX_DIMS];
};

template <typename T, typename ArgT>
struct CommonVectorLoad {
  __device__ inline void operator()(const T *in_data, ArgT args, int tid,
                                    int thresh) {
    if (tid < thresh) {
      const ArgT *vec_data = reinterpret_cast<const ArgT *>(in_data);
      args = vec_data[tid];
    }
  }
};

template <typename T>
struct CommonScalarLoad {
  __device__ inline void operator()(const T *in_data, T args, int tid,
                                    int thresh) {
    if (tid < thresh) {
      args = in_data[tid];
    }
  }
};

template <typename T, typename OffsetT, int N, int vec_size, int nDims>
struct TensorLoader {
  using ArgT = CudaAlignedVector<T, vec_size>;

  TensorLoader() {}

  TensorLoader(std::vector<const framework::Tensor *> *ins,
               framework::Tensor *out, const OffsetT *offset_pre = nullptr) {
    using ArgT = CudaAlignedVector<T, vec_size>;
    auto numel = out->numel();
    auto remain = numel % vec_size;
    std::cout << numel << std::endl;
    std::cout << remain << std::endl;

    for (int j = 0; j < N; ++j) {
      in_data[j] = (*ins)[j]->data<T>();

      if (vec_size != 1) {
        if ((*ins)[j]->dims() == out->dims()) {
          v_loader[j] = CommonVectorLoad<T, ArgT>();
        } else {
          v_loader[j] =
              StrideVectorLoad<T, ArgT, OffsetT, nDims, vec_size>(offset_pre, j)
                  .
                  operator();
        }
      }
    }
    // } else {
    //     if ((*ins)[j]->dims() == out->dims()) {
    //       auto common_loader = CommonLoad<T, ArgT, vec_size>();
    //       v_loader[j] = common_loader.load_scalar();
    //     } else {
    //       auto stide_loader =
    //           StrideLoad<T, ArgT, OffsetT, nDims, vec_size>(offset_pre, j);
    //       v_loader[j] = stide_loader.load_scalar();
    //     }
    // }

    // if (remain) {
    //   if ((*ins)[j]->dims() == out->dims()) {
    //     auto common_loader = CommonLoad<T, T, vec_size>();
    //     s_loader[j] = common_loader.load_scalar();
    //   }
    //     } else {
    //       auto stide_loader =
    //           StrideLoad<T, T, OffsetT, nDims, vec_size>(offset_pre, j);
    //       s_loader[j] = stide_loader.load_scalar();
    //     }

    //     for (int j = 0; j < N; ++j) {
    //       tail_data[j] = (*ins)[j]->data<T>() + (numel - remain) * sizeof(T);
    //     }
    //   }
    // }
  }

  const T *in_data[N];
  const T *tail_data[N];
  std::function<void(const T *, ArgT, int, int)> v_loader[N];
  std::function<void(const T *, T, int, int)> s_loader[N];
};

}  // namespace operators
}  // namespace paddle
