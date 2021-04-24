/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.1

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include <nvfunctional>
#endif

#define MAX_DIMS 10
#define INT_BITS 32
#define DEBUG_OPT 0

namespace paddle {
namespace operators {

template <typename T, int Size>
struct alignas(sizeof(T) * Size) BAlignedVector {
  T val[Size];
};

template <typename T>
int BVectorizedSizeImpl(T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<BAlignedVector<T, 4>>::value;
  constexpr int vec2 = std::alignment_of<BAlignedVector<T, 2>>::value;
  if (sizeof(T) <= 16) {
    constexpr int vec8 = std::alignment_of<BAlignedVector<T, 8>>::value;
    if (address % vec8 == 0) return 8;
  }
  if (address % vec4 == 0) return 4;
  if (address % vec2 == 0) return 2;
  return 1;
}

template <typename index_t>
struct DivMod {
  index_t div, mod;

  HOSTDEVICE inline DivMod(index_t div, index_t mod) : div(div), mod(mod) {}
};

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

  __device__ inline IndexT div(IndexT n) const {
    IndexT t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ DivMod<IndexT> divmod(IndexT n) const {
    IndexT q = div(n);
    return DivMod<IndexT>(q, n - q * divisor);
  }

  IndexT divisor;
  IndexT multiplier;
  IndexT shift_val;
};

/* 1. To compensate the lackage of input_tensors dimension;
*  2. To Merge the dimensions of input_tensors while the consequtive
* equal-dimensions appears;
*  3. To Merge the dimension of input_tensors while the consequtive
* 1-value-dimensions appears;
*  4. To calculate the strides of each input_tensor. */
struct DimensionTransform {
  using vec_t = std::vector<uint64_t>;
  typedef void (*func_t)(bool &, std::vector<vec_t> &, vec_t &, int, int);

 private:
  __inline__ void DimsExtend(int N) {
    std::reverse(out_dims.begin(), out_dims.end());

    for (auto &in_dim : vec_dims) {
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
      bool equal = true;
      do {
        merge_func(equal, vec_dims, out_dims, i, N);
        if (equal) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < dim_size);

      if (cnt > 1) {
        for (auto &in_dim : vec_dims) {
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
  DimensionTransform(const std::vector<const framework::Tensor *> &ins,
                     const framework::DDim &dims, int N) {
    dim_size = dims.size();
    out_dims = framework::vectorize<uint64_t>(dims);
    vec_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      vec_dims[j] = framework::vectorize<uint64_t>(ins[j]->dims());
    }
    DimsExtend(N);

    auto merge_sequential_dims = [](bool &equal, std::vector<vec_t> &in_dims,
                                    vec_t &out, int i, int num) {
      for (int j = 1; j < num; ++j) {
        equal = (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    auto merge_sequential_one_dims = [](
        bool &equal, std::vector<vec_t> &in_dims, vec_t &out, int i, int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal = in_dims[j][i] == out[i];
        }
      }
    };

    func_t merge_ptr = merge_sequential_dims;
    DimsReorganise<func_t>(merge_ptr, N);
#if DEBUG_OPT
    std::cout << std::endl;
    for (auto &vec_dim : vec_dims) {
      std::cout << "[1. Dims]: \t";
      for (auto &x : vec_dim) {
        std::cout << x << "\t";
      }
      std::cout << std::endl;
    }
#endif
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
        strides[j][i] =
            (i != 0 && strides[j][i] != 0)
                ? std::accumulate(in_dims[j].begin(), in_dims[j].begin() + i, 1,
                                  std::multiplies<uint64_t>())
                : strides[j][i];
      }
    }
  }

 public:
  explicit OffsetPreCalculator(const merge_t &merge_dims) {
    auto N = merge_dims.vec_dims.size();
    int dim_size = merge_dims.dim_size;
    divmoder.resize(dim_size);
    strides.resize(N, std::vector<uint64_t>(dim_size, 1));

    for (int i = 0; i < dim_size; ++i) {
      divmoder[i] = FastDivMod<uint64_t>(merge_dims.out_dims[i]);
    }
    auto vec_dims = merge_dims.vec_dims;
    StirdeCalculator<decltype(vec_dims)>(N, dim_size, vec_dims);
#if DEBUG_OPT
    std::cout << "[Out]: \t";
    for (auto &x : merge_dims.out_dims) {
      std::cout << x << "\t";
    }
    std::cout << std::endl;
    for (auto &vec_dim : merge_dims.vec_dims) {
      std::cout << "[2. Dims]: \t";
      for (auto &x : vec_dim) {
        std::cout << x << "\t";
      }
      std::cout << std::endl;
    }
    for (auto &stride : strides) {
      std::cout << "[Strides]: \t";
      for (auto &x : stride) {
        std::cout << x << "\t";
      }
      std::cout << std::endl;
    }
#endif
  }

  std::vector<std::vector<uint64_t>> strides;
  std::vector<FastDivMod<uint64_t>> divmoder;
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename ArgT, typename OffsetT, int nDims, int vec_size,
          typename IndexT = uint32_t>
struct StrideVectorLoad {
 private:
  __device__ inline IndexT const get_offset(int tid) {
    IndexT offset = 0;

#pragma unroll
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoders[i].divmod(tid);
      tid = fast_divmoder.div;
      offset += fast_divmoder.mod * strides[i];
    }
    return offset;
  }

 public:
  HOSTDEVICE StrideVectorLoad(const OffsetT &offset_calculator, int i) {
    memcpy(static_cast<void *>(strides), offset_calculator.strides[i].data(),
           sizeof(uint32_t) * nDims);
    memcpy(static_cast<void *>(divmoders), offset_calculator.divmoder.data(),
           sizeof(FastDivMod<uint32_t>) * nDims);
#if DEBUG_OPT
    printf("[%s %d] nDims : %d\n", __func__, __LINE__, nDims);
    printf("[%s %d] strides.size : %d\n", __func__, __LINE__,
           offset_calculator.strides[i].size());
    for (int j = 0; j < nDims; j++) {
      printf("%d\t", strides[j]);
    }
    printf("\n");
#endif
  }

  __device__ inline void operator()(const T *__restrict__ in_data, ArgT *args,
                                    int tid) {
    int index = vec_size * tid;

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      IndexT offset = get_offset(index + i);
      (*args).val[i + i] = in_data[offset];
    }
  }

  IndexT strides[MAX_DIMS];
  FastDivMod<IndexT> divmoders[MAX_DIMS];
};

template <typename T, typename OffsetT, int nDims, int vec_size,
          typename IndexT = uint32_t>
struct StrideScalarLoad {
  explicit HOSTDEVICE StrideScalarLoad(const OffsetT &offset_calculator, int i,
                                       int data_offset)
      : data_offset(data_offset) {
    memcpy(static_cast<void *>(strides), offset_calculator.strides[i].data(),
           sizeof(uint32_t) * nDims);
    memcpy(static_cast<void *>(divmoders), offset_calculator.divmoder.data(),
           sizeof(FastDivMod<uint32_t>) * nDims);
  }

  __device__ inline IndexT const get_offset(int tid) {
    IndexT offset = 0;

#pragma unroll
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoders[i].divmod(tid);
      tid = fast_divmoder.div;
      offset += fast_divmoder.mod * strides[i];
    }
    return offset;
  }

  __device__ inline void operator()(const T *__restrict__ in_data, T *args,
                                    int tid) {
    int offset = get_offset(tid + data_offset);
    *args = in_data[offset];
  }

  int data_offset;
  IndexT strides[MAX_DIMS];
  FastDivMod<IndexT> divmoders[MAX_DIMS];
};

template <typename T, typename ArgT>
struct CommonVectorLoad {
  __device__ inline void operator()(const T *__restrict__ in_data, ArgT *args,
                                    int tid) {
    const ArgT *vec_data = reinterpret_cast<const ArgT *>(in_data);
    *args = vec_data[tid];
  }
};

template <typename T>
struct CommonScalarLoad {
  explicit CommonScalarLoad(int data_offset) : data_offset(data_offset) {}

  __device__ inline void operator()(const T *__restrict__ in_data, T *args,
                                    int tid) {
    *args = in_data[tid + data_offset];
  }
  int data_offset;
};

template <typename T, typename OffsetT, int N, int vec_size, int nDims>
struct TensorLoader {
  using ArgT = BAlignedVector<T, vec_size>;

  HOSTDEVICE TensorLoader(const std::vector<const framework::Tensor *> &ins,
                          const OffsetT &offset_pre, framework::Tensor *out,
                          int vec_len, int remain) {
    for (int j = 0; j < N; ++j) {
      data[j] = ins[j]->data<T>();

      if (ins[j]->dims() == out->dims()) {
        v_loader[j] = CommonVectorLoad<T, ArgT>();
      } else {
        v_loader[j] =
            StrideVectorLoad<T, ArgT, OffsetT, nDims, vec_size>(offset_pre, j);
      }

      if (remain) {
        if (ins[j]->dims() == out->dims()) {
          s_loader[j] = CommonScalarLoad<T>(vec_len);
        } else {
          s_loader[j] = StrideScalarLoad<T, OffsetT, nDims, vec_size>(
              offset_pre, j, vec_len);
        }
      }
    }
  }

  const T *data[N];
  nvstd::function<void(const T *, ArgT *, int)> v_loader[N];
  nvstd::function<void(const T *, T *, int)> s_loader[N];
};
#endif

}  // namespace operators
}  // namespace paddle
