// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/random.h>

#include "paddle/fluid/framework/generator.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#endif

namespace paddle {
namespace operators {

inline std::vector<int64_t> GetNewDataFromShapeTensor(
    const phi::DenseTensor* new_data_tensor) {
  if (framework::TransToProtoVarType(new_data_tensor->dtype()) ==
      framework::proto::VarType::INT64) {
    auto* new_data = new_data_tensor->data<int64_t>();
    phi::DenseTensor cpu_starts_tensor;
    if (platform::is_gpu_place(new_data_tensor->place())) {
      paddle::framework::TensorCopySync(
          *new_data_tensor, platform::CPUPlace(), &cpu_starts_tensor);
      new_data = cpu_starts_tensor.data<int64_t>();
    }
    std::vector<int64_t> vec_new_data(new_data,
                                      new_data + new_data_tensor->numel());
    return vec_new_data;
  } else if (framework::TransToProtoVarType(new_data_tensor->dtype()) ==
             framework::proto::VarType::INT32) {
    auto* new_data = new_data_tensor->data<int32_t>();
    std::vector<int64_t> vec_new_data;
    phi::DenseTensor cpu_starts_tensor;
    if (platform::is_gpu_place(new_data_tensor->place())) {
      paddle::framework::TensorCopySync(
          *new_data_tensor, platform::CPUPlace(), &cpu_starts_tensor);
      new_data = cpu_starts_tensor.data<int32_t>();
    }
    for (int i = 0; i < new_data_tensor->numel(); ++i) {
      vec_new_data.push_back(static_cast<int64_t>(*(new_data + i)));
    }
    return vec_new_data;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Expected dtype of ShapeTensor must be int32, int64. But got "
        "unsupport dtype: %s.",
        new_data_tensor->dtype()));
  }
}

inline std::vector<int64_t> GetNewDataFromShapeTensorList(
    const std::vector<const phi::DenseTensor*>& list_new_shape_tensor) {
  std::vector<int64_t> vec_new_shape;
  vec_new_shape.reserve(list_new_shape_tensor.size());
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        phi::make_ddim({1}),
        platform::errors::InvalidArgument(
            "Shape of dim tensor in uniform_random_op should be [1]"
            "But received tensor's dim=%s.",
            tensor->dims()));

    if (framework::TransToProtoVarType(tensor->dtype()) ==
        framework::proto::VarType::INT32) {
      if (platform::is_gpu_place(tensor->place())) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int32_t>()));
      } else {
        vec_new_shape.push_back(static_cast<int64_t>(*tensor->data<int32_t>()));
      }
    } else if (framework::TransToProtoVarType(tensor->dtype()) ==
               framework::proto::VarType::INT64) {
      if (platform::is_gpu_place(tensor->place())) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_new_shape.push_back(*temp.data<int64_t>());
      } else {
        vec_new_shape.push_back(*tensor->data<int64_t>());
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected dtype of ShapeTensorList of %d-th must be int32, int64. "
          "But got "
          "unsupport dtype: %s.",
          i,
          paddle::framework::DataTypeToString(
              framework::TransToProtoVarType(tensor->dtype()))));
    }
  }

  return vec_new_shape;
}

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(
      T min, T max, int seed, int diag_num, int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T>
void UniformRandom(const framework::ExecutionContext& context,
                   phi::DenseTensor* tensor) {
  int64_t size = tensor->numel();
  auto& dev_cxt = context.template device_context<phi::GPUContext>();
  T* data = tensor->mutable_data<T>(dev_cxt.GetPlace());
  if (size <= 0) return;
  unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));

  T min = static_cast<T>(context.Attr<float>("min"));
  T max = static_cast<T>(context.Attr<float>("max"));
  unsigned int diag_num =
      static_cast<unsigned int>(context.Attr<int>("diag_num"));
  unsigned int diag_step =
      static_cast<unsigned int>(context.Attr<int>("diag_step"));
  T diag_val = static_cast<T>(context.Attr<float>("diag_val"));

  if (seed == 0) {
    // Use global Generator seed
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    phi::funcs::uniform_distribution<MT> dist;
    phi::funcs::uniform_real_transform<MT> trans(min, max);
    phi::funcs::distribution_and_transform<T>(dev_cxt, tensor, dist, trans);
  } else {
    // Use OP seed
    auto func =
        UniformGenerator<T>(min, max, seed, diag_num, diag_step, diag_val);
    phi::IndexKernel<T, UniformGenerator<T>>(dev_cxt, tensor, func);
  }
}
#endif
}  // namespace operators
}  // namespace paddle
