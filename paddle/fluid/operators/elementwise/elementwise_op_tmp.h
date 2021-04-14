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

// template <typename T, int vec_size>
// struct alignas(sizeof(T) * vec_size) aligned_vector {
//   T scalar_array[vec_size];
// };

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

/*
* 1. To compensate the lackage of input_tensors dimension;
* 2. To Merge the dimensions of input_tensors while the consequtive
*    equal-dimensions appears;
* 3. To Merge the dimension of input_tensors while the consequtive
*    1-value-dimensions appears;
* 4. To calculate the strides of each input_tensor.
*/
template <typename DimT = int>
struct DimensionTransform {
  using vec_t = std::vector<DimT>;

 private:
  __inline__ void DimsExtend(int out_size, int N) {
    for (int j = 0; j < N; ++j) {
      std::reverse(vec_dims[j].begin(), vec_dims[j].end());
      if (vec_dims[j].size() < out_size) {
        vec_t tmp_dims(out_size, 1);
        int idx_in = 0, idx_out = 0;
        vec_dims[j].resize(out_size, 1);
        do {
          if (vec_dims[j][idx_in] == out_dims[idx_out] ||
              vec_dims[j][idx_in] == 1) {
            tmp_dims[idx_out++] = vec_dims[j][idx_in++];
          } else {
            idx_out++;
          }
        } while (idx_out < out_size);
        std::copy(tmp_dims.begin(), tmp_dims.end(), vec_dims[j].begin());
      }
    }
  }

  template <typename func_t>
  __inline__ void DimsReorganise(int *out_size, func_t merge_func, int N) {
    int i = 0;
    auto VectorReorganise = [](vec_t *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] =
          std::accumulate(vec->begin() + l_idx, vec->begin() + m_idx, 1,
                          std::multiplies<DimT>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };
    while (i < *out_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal_flag = true;
      do {
        merge_func(&equal_flag, vec_dims, out_dims, i, N);
        if (equal_flag) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < *out_size);

      if (cnt > 1) {
        for (int j = 0; j < N; ++j) {
          VectorReorganise(&(vec_dims[j]), low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        *out_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

 public:
  DimensionTransform(const std::vector<framework::Tensor *> *ins,
                     const framework::DDim *dims, int N) {
    out_size = dims->size();
    out_dims = paddle::framework::vectorize<DimT>(*dims);

    vec_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      vec_dims[j] = paddle::framework::vectorize<DimT>((*ins)[j]->dims());
    }
    DimsExtend(out_size, N);

    auto merge_sequential_dims = [](bool &equal, std::vector<vec_t> &in_arr,
                                    vec_t &out, int i, int num) -> void {
      for (int j = 1; j < num; ++j) {
        equal &= (in_arr[0][i] == in_arr[j][i]);
      }
    };

    auto merge_sequential_one_dims = [](bool &equal, std::vector<vec_t> &in_arr,
                                        vec_t &out, int i, int num) -> void {
      equal &= (in_arr[0][i] == 1);
      if (equal) {
        bool at_least = false;
        for (int j = 1; j < num; ++j) {
          if (in_arr[j][i] == out[i]) {
            equal &= true;
            at_least = true;
          } else {
            PADDLE_ENFORCE_NE(
                in_arr[j][i], out[i],
                platform::errors::InvalidArgument(
                    "%d th input tensor %d th dimension(%d) is neither equal to"
                    "the output tensor %d th dimension(%d), nor equal to 1.\n",
                    j, i, in_arr[j][i], out[i]));
          }
        }
        PADDLE_ENFORCE_NE(
            at_least, false,
            platform::errors::InvalidArgument(
                "All input tensors %d th dimension(%d) is equal to 1, none of "
                "them equals to output tensor %d th dimension(%d).\n",
                i, i, out[i]));
      }
    };

    typedef void (*func_t)(bool *, std::vector<vec_t> *, vec_t *, int, int);
    func_t merge_ptr = merge_sequential_dims;
    DimsReorganise<vec_t, func_t>(&out_size, merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(vec_dims[0].begin(), vec_dims[0].end(), 1,
                                  std::multiplies<DimT>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(vec_dims[j].begin(), vec_dims[j].end(), 1,
                                 std::multiplies<DimT>());
      min_val = min_val > temp ? temp : min_val;
      if (min_val == temp) {
        min_idx = j;
      }
    }
    std::swap(vec_dims[0], vec_dims[min_idx]);

    merge_ptr = merge_sequential_one_dims;
    DimsReorganise<vec_t, func_t>(out_size, merge_ptr, N);
  }

  int out_size;
  vec_t out_dims;
  std::vector<vec_t> vec_dims;
};

template <typename merge_t, typename IndexT = uint32_t>
struct OffsetPreCalculator {
  __inline__ void StirdeCalculator(int dim_size, int N,
                                   std::vector<std::vector<int>> *vec_dims) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < dim_size; ++i) {
        if ((*vec_dims)[j][i] == 1) {
          strides[j][i] = 0;
        } else if (i != 1) {
          strides[j][i] = std::accumulate((*vec_dims)[j].begin(),
                                          (*vec_dims)[j].begin() + i, 1,
                                          std::multiplies<IndexT>());
        }
      }
    }
  }

  explicit OffsetPreCalculator(merge_t *merge_dims) {
    auto N = merge_dims->in_strides.size();
    auto dim_size = merge_dims->dims_size;
    divmoder.resize(dim_size);
    strides.resize(N, std::vector<IndexT>(dim_size, 1));

    for (int i = 0; i < dim_size; ++i) {
      divmoder[i] = FastDivMod<IndexT>(merge_dims->out_dims[i]);
    }
    StirdeCalculator(N, dim_size, &(merge_dims->vec_dims));
  }

  std::vector<std::vector<IndexT>> strides;
  std::vector<FastDivMod<IndexT>> divmoder;
};

template <typename OffsetT, typename T, int nDims, int vec_size,
          typename IndexT = uint32_t>
struct LoadScalar {
  LoadScalar() {}

  LoadScalar(OffsetT *offset_calculator, int i) {
    memcpy(static_cast<void *>(strides), offset_calculator->strides[i].val(),
           sizeof(IndexT) * nDims);
    memcpy(static_cast<void *>(divmoders), offset_calculator->divmoder.val(),
           sizeof(FastDivMod<IndexT>) * nDims);
  }

  __forceinline__ __device__ IndexT get_offset(int tid) {
    IndexT offset = 0;

#pragma unroll
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoders[i].divmod(tid);
      tid = fast_divmoder[0];
      offset += fast_divmoder[1] * strides[i];
    }
    return offset;
  }

  template <typename ArgT>
  __forceinline__ __device__ void load_scalar(const T *in_data, ArgT args,
                                              int tid, size_t loop) {
    //  int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < loop) {
      auto offset = get_offset(tid);
      args = in_data[offset];
    }
  }

  // template <typename ArgT>
  //  __forceinline__ __device__ void load_vector(const T * in_data,
  //                           ArgT args, int tid, size_t loop) {
  //   //  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  //   //  int loop = numel / vec_size ;
  //    if (tid < loop){
  //       int index = vec_size * tid;
  //       #pragma unroll
  //       for (int i = 0; i < vec_size; ++i) {
  //         auto offset = get_offset(index + i);
  //         args.val[i + i] = in_data[offset];
  //       }
  //    }
  // }

  IndexT strides[MAX_DIMS];
  FastDivMod<IndexT> divmoders[MAX_DIMS];
};

template <typename T, int vec_size>
struct LoadVector {
  __forceinline__ __device__ void operator()(const T *in_data, T args, int tid,
                                             int loop) {
    if (tid < loop) {
      args = in_data[tid];
    }
  }

  // __forceinline__ __device__ void operator() (const T * in_data,
  //                                 VecType args, size_t numel){
  //   int tid = threadIdx.x + blockDim.x * blockIdx.x;
  //   int loop = numel / loop;
  //   for (; tid < loop; tid += blockDim.x * gridDim.x) {
  //     VecType *vec_data = reinterpre_cast<VecType *>(in_data);
  //     args = vec_data[tid];
  //   }
  // }
};

template <typename T, typename OffsetT, int N, int nDims>
struct TensorLoader {
  void operator()(const std::vector<framework::Tensor *> *ins,
                  framework::Tensor *out, OffsetT *offset_pre = nullptr) {
    for (int j = 0; j < N; ++j) {
      data[j] = (*ins)[j].data();

      if ((*ins)[j]->dims() == out->dims()) {
        ploader[j] = load_vector<T, 1>();
      } else {
        auto load_scalar = LoadScalar<OffsetT, T, nDims, 1>(*offset_pre, j);
        ploader[j] = load_scalar<T>().load_scalar();
      }
    }
  }

  T *data[N];
  std::function<void(const T *, T, int, int)> ploader[N];
};

}  // namespace operators
}  // namespace paddle
