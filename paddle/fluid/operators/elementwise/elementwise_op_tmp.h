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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

template <typename index_t, typename N>
struct FastDivMod {
  FastDivMod() {}

  explicit FastDivMod(index_t d) : divisor(d) {
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

  __forceinline__ __device__ index_t div(index_t n) const {
    index_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __forceinline__ __device__ index_t *divmod(index_t n) const {
    index_t q = div(n);
    index_t arr[N] = {q, n - q * divisor};
    // return DivMod<index_t>(q, n - q * divisor);
    return arr;
  }

  index_t divisor;
  index_t multiplier;
  index_t shift_val;
};

/*
* 1. To compensate the lackage of input_tensors dimension;
* 2. To Merge the dimensions of input_tensors while the consequtive
*    equal-dimensions appears;
* 3. To Merge the dimension of input_tensors while the consequtive
*    1-value-dimensions appears;
* 4. To calculate the strides of each input_tensor.
*/
template <typename T, typename vec_input_t>
struct MergeDims {
  using vec_t = std::vector<int>;

  __inline__ void DimsCompensate(int out_size) {
    for (int j = 0; j < N; ++j) {
      std::reverse(in_strides[j].begin(), in_strides[j].end());
      if (in_strides[j].size() < out_size) {
        vec_t tmp_dims(out_size, 1);
        int idx_in = 0, idx_out = 0;
        in_strides[j].resize(out_size, 1);
        do {
          if (in_strides[j][idx_in] == out_dims[idx_out] ||
              in_strides[j][idx_in] == 1) {
            tmp_dims[idx_out++] = in_strides[j][idx_in++];
          } else {
            idx_out++;
          }
        } while (idx_out < out_size);
        std::copy(tmp_dims.begin(), tmp_dims.end(), in_strides[j].begin());
      }
    }
  }

  __inline__ void DimsReorganise(int *out_size, func_t merge_func) {
    int i = 0;
    auto VectorReorganise = [](vec_t *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] = std::accumulate(
          vec->begin() + l_idx, vec->begin() + m_idx, 1, std::multiplies<T>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };
    while (i < *out_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal_flag = true;
      do {
        merge_func(&equal_flag, &in_strides, &out_dims, i, N);
        if (equal_flag) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < *out_size);
      if (cnt > 1) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
          VectorReorganise(&(in_strides[j]), low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        *out_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

  template <typename T, typename vec_t>
  __inline__ void StirdeCalculator(int out_size) {
    std::vector<vec_t> tmp_stride(N, vec_t(out_size, 1));
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < out_size; ++i) {
        if (in_strides[j][i] == 1) {
          tmp_stride[j][i] = 0;
        } else if (i != 1) {
          tmp_stride[j][i] =
              std::accumulate(in_strides[j].begin(), in_strides[j].begin() + i,
                              1, std::multiplies<T>());
        }
      }
      std::copy(tmp_stride[j].begin(), tmp_stride[j].end(),
                in_strides[j].begin());
    }
  }

  template <typename vec_input_t>
  MergeDims(vec_input_t *ins, framework::DDim *dims, int N) {
    auto out_size = *dims.size();
    PADDLE_ENFORCE_GE(
        out_size, MAX_DIMS,
        platform::errors::InvalidArgument(
            "Output tensor`s dim is %d, bigger than upper limitation %d\n",
            out_size, MAX_DIMS));
    for (auto *in_tensor : *ins) {
      PADDLE_ENFORCE_GE(
          in_tensor->dims().size(), MAX_DIMS,
          platform::errors::InvalidArgument(
              "Input tensor`s dim is %d, bigger than upper limitation %d\n",
              in_tensor->dims().size(), MAX_DIMS));
    }

    in_strides.resize(N);
    out_dims.resize(out_size);
    for (auto &x : *out_dims) {
      out_dims.emplace_back(x);
    }
    for (int j = 0; j < N; ++j) {
      in_strides[j].resize(out_size);
      for (auto &x : (*ins)[j]->dims()) {
        in_strides[j].emplace_back(x);
      }
    }
    DimsCompensate(out_size);

    auto merge_same = [](bool *equal, std::vector<vec_t> *in_arr, vec_t out,
                         int i, int num) -> void {
      for (int j = 1; j < num; ++j) {
        *equal &= (*in_arr)[0][i] == (*in_arr)[j][i];
      }
    };

    auto merge_one = [](bool *equal, std::vector<vec_t> *in_arr, vec_t *out,
                        int i, int num) -> void {
      *equal &= (*in_arr)[0][i] == 1;
      if (*equal) {
        bool at_least = false;
        for (int j = 1; j < num; ++j) {
          if ((*in_arr)[j][i] == (*out)[i]) {
            *equal &= true;
            at_least = true;
          } else if ((*in_arr)[j][i] == 1) {
            continue;
          } else {
            PADDLE_ENFORCE_NE((*in_arr[j])[i], (*out)[i]),
              platform::errors::InvalidArgument(
                  "%d th input tensor %d th dimension(%d) is neither equal to"
                  "the output tensor %d th dimension(%d), nor equal to 1.\n",
                  j, i, (*in_arr)[j][i], (*out)[i]) );
          }
        }
        PADDLE_ENFORCE_NE(
            at_least, false,
            platform::errors::InvalidArgument(
                "All input tensors %d th dimension(%d) is equal to 1, none of "
                "them equals to output tensor %d th dimension(%d).\n",
                i, i, (*out)[i]));
      }
    };

    typedef void (*func_t)(bool *, std::vector<vec_t> *, vec_t *, int, int);
    func_t merge_ptr = merge_same;
    DimsReorganise<T, vec_t, N, func_t>(&out_size, merge_ptr);

    int min_idx = 0;
    int min_val = std::accumulate(in_strides[0].begin(), in_strides[0].end(), 1,
                                  std::multiplies<T>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(in_strides[j].begin(), in_strides[j].end(), 1,
                                 std::multiplies<T>());
      min_val = min_val > temp ? temp : min_val;
      if (min_val == temp) {
        min_idx = j;
      }
    }
    std::swap(in_strides[0], in_strides[min_idx]);

    merge_ptr = merge_one;
    DimsReorganise<T, vec_t, N, func_t>(out_size, merge_ptr);

    StirdeCalculator<T, vec_t, N>(&vec_dims, out_size);
    dims_size = out_size;
  }

  int dims_size;
  vec_t out_dims;
  std::vector<vec_t> in_strides;
};

template <typename merge_t, typename index_t = uint32_t>
struct OffsetPreCalculator {
  OffsetFastCal(merge_t merge_dims) {
    auto N = merge_dims.in_strides.size();
    auto dim_size = merge_dims.dims_size;
    divmoder.resize(dim_size);
    strides.resize(N, std::vector<index_t>(dim_size));
    is_broadcast.resize(N, true);

    for (int i = 0; i < dim_size; ++i) {
      divmoder[i] = FastDivMod<index_t, N>(merge_dims.out_dims[i]);
    }
    for (int i = 0; i < N; ++i) {
      std::copy(merge_dims.in_strides[i].begin(),
                merge_dims.in_strides[i].end(), strides[i].begin());
    }

    std::vector<std::vector<index_t>> strides;
    std::vector<FastDivMod<index_t>> divmoder;
  }
};

template <typename offset_pre_t, int nDims, typename index_t = uint32_t>
struct LoadScalar {
  template <typename offset_pre_t, int nDims, typename index_t = uint32_t>
  LoadScalar(offset_pre_t offset_pre_cal, int i) {
    memcpy(static_cast<void *> strides, offset_pre_cal.strides[i].val(),
           sizeof(index_t) * nDims);
    memcpy(static_cast<void *> stridesdivmoder, offset_pre_cal.divmoder.val(),
           sizeof(FastDivMod<index_t>) * nDims);
  }

  __forceinline__ __device__ index_t get_offset(int tid) {
    index_t offset = 0;
#pragma unroll
    for (int i = 0; i < nDims; ++i) {
      auto fast_divmoder = divmoder[i].divmod(tid);
      tid = fast_divmoder[0];
      offset += fast_divmoder[1] * strides[i];
    }
    return offset;
  }

  template <typename T>
  __forceinline__ __device__ void load(T args, int idx);

  index_t stride[MAX_DIM];
  FastDivMod<index_t> divmoder[MAX_DIM];
};

template <typename T>
struct LoadVector {
  template <typename T, int vec_size>
  __forceinline__ __device__ void load(T args, int idx);

  template <typename T>
  explicit LoadVector(int vec_size) {
    switch (vec_size) {
      case 2: {
        return load<T, 2>();
        break;
      }
      case 4: {
        return load<T, 2>();
        break;
      }
      case 8: {
        return load<T, 2>();
        break;
      }
      default:
        return load<T, 1>();
    }
  }
};

template <typename T, typename vec_input_t, typename func_t, int N, int nDims,
          typename offset_pre_t>
struct TensorLoader {
  template <typename T, int N, typename vec_input_t, typename offset_pre_t>
  TensorLoader(vec_input_t *ins, framework::Tensor *out,
               offset_pre_t *offset_pre = nullptr) {
    for (int j = 0; j < N; ++j) {
      in_data[j] = (*ins)[j].data();

      if ((*ins)[j]->dims() == out->dims()) {
        // constexpr vec_size = 8;
        // auto load_vector = LoadVector<T>(vec_size);
        // Funcp[j] = load_vector<T>.load();
        // ;
      } else {
        auto load_scalar = LoadScalar<offset_pre_t, nDims>(*offset_pre, j);
        Funcp[j] = (load_scalar<T>).load();
      }
    }
  }

  T *in_data[N];
  std::function<void(vec_t)> Funcp[N];
}
