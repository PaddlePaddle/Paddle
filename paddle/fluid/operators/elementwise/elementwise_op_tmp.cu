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
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_tmp.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, int vec_size>
struct alignas(sizeof(T) * 4) aligned_vector {
  T scalar_array[4];
};

template <typename DeviceContext, typename T, typename Functor>
class ElementwiseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto *out_tensor = ctx.Output<framework::LoDTensor>("Out");
    bool is_broadcast;

    using vec_input_t = std::vector<const framework::Tensor *>;
    vec_input_t in_array;
    in_array.reserve(in_tensors.size());
    for (auto *in_tensor : in_tensors) {
      is_broadcast = in_tensor->dims() == out_tensor->dims();
      in_array.emplace_back(*in_tensor);
    }

    if (!is_broadcast && (in_tensors.size() == 1)) {
      SameDimsElemwise<DeviceContext, T, Functor, vec_input_t>(ctx, &in_array,
                                                               out_tensor);
    } else {
      BroadcastElementwise<DeviceContext, T, Functor, vec_input_t>(
          ctx, &in_array, out_tensor);
    }
  }
};

template <int N, typename vec_input_t>
void MergeDims(vec_input_t *ins, framework::DDim *out_dims) {
  PADDLE_ENFORCE_GE(
      out_dims->size(), MAX_DIMS,
      platform::errors::InvalidArgument(
          "Output tensor`s dim is %d, bigger than upper limitation %d\n",
          paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
          paddle::framework::DataTypeToString(
              framework::proto::VarType::INT32)));

  for (*in_tensor : ins) {
    PADDLE_ENFORCE_GE(
        in_tensor->dims().size(), MAX_DIMS,
        platform::errors::InvalidArgument(
            "Input tensor`s dim is %d, bigger than upper limitation %d\n",
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32)));
  }
}

template <typename vec_t, typename T, int N>
void DimsReorganise(std::vector<vec_t> *in_tenosr_data, vec_t *out_tensor_data,
                    int *out_size, int *max_idx, int low_idx, int cnt) {
  auto VectorReorganise = [](vec_t *vec, int l_idx, int m_idx) {
    (*vec)[m_idx - 1] = std::accumulate(
        vec->begin() + l_idx, vec->begin() + m_idx, 1, std::multiplies<T>());
    vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
  };

  if (cnt > 1) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      VectorReorganise(&((*in_tenosr_data)[j]), low_idx, *max_idx);
    }
    VectorReorganise(out_tensor_data, low_idx, *max_idx);
    (*out_size) -= --cnt;
    (*max_idx) -= cnt;
  } else if (cnt < 1) {
    (*max_idx)++;
  }
}

/*
* 1. To compensate the lackage of input_tensors dimension;
* 2. To Merge the dimensions of input_tensors while the consequtive
* equal-dimensions appear;
* 3. To Merge the dimension of input_tensors while the consequtive
* 1-value-dimensions appear;
* 4. To calculate the strides of each input_tensor.
*/
template <typename vec_t, typename T, int N>
void MergeDimsTemp(std::vector<vec_t> *in_arr, vec_t *out) {
  std::vector<vec_t> a = *in_arr;
  int out_size = out->size();

  for (int j = 0; j < N; ++j) {
    std::reverse(a[j].begin(), a[j].end());
    if (a[j].size() < out_size) {
      vec_t vec_temp(out_size, 1);
      int idx_in = 0, idx_out = 0;
      a[j].resize(out_size, 1);

      do {
        if (a[j][idx_in] == (*out)[idx_out] || a[j][idx_in] == 1) {
          vec_temp[idx_out++] = a[j][idx_in++];
        } else {
          idx_out++;
        }
      } while (idx_out < out_size);
      std::copy(vec_temp.begin(), vec_temp.end(), a[j].begin());
    }
  }

  int i = 0;
  while (i < out_size) {
    int cnt = 0;
    int low_idx = i;
    bool equal_flag = true;
    do {
#pragma unroll
      for (int j = 1; j < N; j++) {
        equal_flag &= a[0][i] == a[j][i];
      }
      if (equal_flag) {
        i++;
        cnt++;
      } else {
        break;
      }
    } while (i < out_size);
    DimsReorganise<vec_t, T, N>(&a, out, &out_size, &i, low_idx, cnt);
  }

  int min_idx = 0;
  T min_val =
      std::accumulate(a[0].begin(), a[0].end(), 1, std::multiplies<T>());
#pragma unroll
  for (int j = 1; j < N; ++j) {
    T temp = std::accumulate(a[j].begin(), a[j].end(), 1, std::multiplies<T>());
    min_val = min_val > temp ? temp : min_val;
    if (min_val == temp) {
      min_idx = j;
    }
  }
  std::swap(a[0], a[min_idx]);

  i = 0;
  while (i < out_size) {
    int cnt = 0;
    int low_idx = i;
    bool equal_flag = true;
    do {
      equal_flag &= a[0][i] == 1;
      if (equal_flag) {
#pragma unroll
        for (int j = 1; j < N; ++j) {
          equal_flag &= a[j][i] == (*out)[i];
        }
      }
      if (equal_flag) {
        i++;
        cnt++;
      } else {
        break;
      }
    } while (i < out_size);
    DimsReorganise<vec_t, T, N>(&a, out, &out_size, &i, low_idx, cnt);
  }
}

template <int N, typename vec_input_t>
void StrideCalculate() {
  auto out_size =
      out_dims->size() std::vector<vec_t> in_stride(N, vec_t(out_size, 1));
#pragma unroll
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < out_size; ++i) {
      if (a[j][i] == 1) {
        in_stride[j][i] = 0;
      } else if (i != 1) {
        auto temp = std::accumulate(a[j].begin(), a[j].begin() + i, 1,
                                    std::multiplies<T>());
        in_stride[j][i] = temp;
      }
    }
  }
}

template <int N, typename vec_input_t>  // N : input tensors.
static OffsetCalculator<N, vec_input_t> input_offset_calculator(
    vec_input_t *ins, framework::Tensor *out) {
  constexpr int input_num = std::max<int>(N, 1);
  std::array<int, input_num> shift_array;
  std::array<int, input_num> mul_array;

  std::reverse(out->dims().begin(), out->dims().end());
  for (*in : ins) {
    std::reverse(in->dims().begin(), in->dims().end());
  }
  MergeDims(ins, out->dims());
  StrideCalculate(ins, out->dims());
}

template <typename T, typename Functor, typename inp_calc_t, int N, int nDims>
__devide__ void CommonElementwiseKernel(vec_data *in_data_arr[], out_data) {
  T *args[N];

#pragma unroll
  for (int i = 0; i < N; ++i) {
    (loader[i])(in_data_arr[i], args[i]);
  }
  args[N - 1] = Functor(args);
}

template <typename T, typename Functor, typename inp_calc_t, int N>
void CommonElementwiseCore(vec_input_t ins, framework::Tensor *out) {
  constexpr int out_dim_size = out->dims().size();
  using vec_data = std::vector<T *>;
  vec_data in_data_arr;
  in_data_arr.reserve(N);
  for (auto *in_tensor : ins) {
    in_data_arr.emplace_back(in_tensor->data());
  }
  T *out_data = out->data();

  switch
    out_dim_size {
      case 2:
        CommonElementwiseKernel<T, N, 2, vec_data>(in_data_arr, out_data);
        break;
      case 3:
        CommonElementwiseKernel<T, N, 3, vec_data>(in_data_arr, out_data);
        break;
      case 4:
        CommonElementwiseKernel<T, N, 4, vec_data>(in_data_arr, out_data);
        break;
      case 5:
        CommonElementwiseKernel<T, N, 5, vec_data>(in_data_arr, out_data);
        break;
      default:
        CommonElementwiseKernel<T, N, 5, vec_data>(in_data_arr, out_data);
    }
}  // 这里还是用元模板编程switch到 MAX_DIMS 吧

template <typename T, typename Functor, typename vec_input_t>
void BroadcastElementwise(const framework::ExecutionContext &ctx,
                          vec_input_t *ins, framework::Tensor *out) {
  auto in_tensor_num = ins->size();
  switch (in_tensor_num) {
    case 2: {
      auto input_calc = OffsetCalculator<vec_input_t, 2>(ins, out);
      CommonElementwiseCore<T, Functor, input_calc, 2>(ins, out, in_tensor_num,
                                                       dims);
      break;
    }
    case 3: {
      auto input_calc = OffsetCalculator<vec_input_t, 3>(ins, out);
      CommonElementwiseCore<T, Functor, input_calc, 3>(ins, out, in_tensor_num,
                                                       dims);
      break;
    }
    default: {
      auto input_calc = OffsetCalculator<vec_input_t>(ins, out);
      CommonElementwiseCore<T, Functor, input_calc>(ins, out, in_tensor_num,
                                                    dims);
    }
  }
}

// template<typename func_t, typename array_t, typename inp_calc_t, typename
// out_calc_t, typename loader_t, typename storer_t>
// __global__ void BroadcastElementwiseKernel(int N, func_t f, array_t data,
//                                             inp_calc_t ic, out_calc_t oc,
//                                             loader_t l, storer_t s)
// {
//   int remaining = N - block_work_size * blockIdx.x;
//   auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t,
//   loader_t, storer_t>(data, remaining, ic, oc, l, s);
//   ElementwiseCalculator(f, policy);
// }

// template <template <int i> typename func, int end, int current = 0>
// template <typename... Args>
// static inline void with_args(Args &&... args) {
//   func<current>::apply(std::forward<Args>(args)...);
//   static_unroll<func, end, current + 1>::with_args(args...);
// }

// template <typename args_t>
// void load(args_t *args, int idx) {
//   constexpr int arity = std::tuple_size<args_t>::value;
//   detail::static_unroll<detail::vectorized_load_helper, MAX_DIMS>::with_args(
//       *this, args, idx);
// }

}  // namespace operators
}  // namespace paddle
