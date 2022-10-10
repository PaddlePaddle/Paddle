/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {
inline std::vector<int> get_expand_shape(
    const framework::ExecutionContext& ctx) {
  if (ctx.HasInput("Shape")) {
    auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
    auto* shape_data = shape_tensor->data<int>();
    phi::DenseTensor cpu_shape_tensor;
    if (platform::is_gpu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(
          *shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(
          *shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(
          *shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#endif
#ifdef PADDLE_WITH_MLU
    if (platform::is_mlu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(
          *shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#endif
    auto vec_shape =
        std::vector<int>(shape_data, shape_data + shape_tensor->numel());
    return vec_shape;
  }

  auto list_expand_shapes_tensor =
      ctx.MultiInput<phi::DenseTensor>("expand_shapes_tensor");
  if (list_expand_shapes_tensor.size() > 0) {
    // get tensor from
    std::vector<int> vec_epxand_shape;
    for (size_t i = 0; i < list_expand_shapes_tensor.size(); ++i) {
      auto tensor = list_expand_shapes_tensor[i];
      if (platform::is_gpu_place(tensor->place())) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#ifdef PADDLE_WITH_ASCEND_CL
      else if (platform::is_npu_place(tensor->place())) {  // NOLINT
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#endif
#ifdef PADDLE_WITH_XPU
      else if (platform::is_xpu_place(tensor->place())) {  // NOLINT
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#endif
#ifdef PADDLE_WITH_MLU
      else if (platform::is_mlu_place(tensor->place())) {  // NOLINT
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#endif
      else {  // NOLINT
        vec_epxand_shape.push_back(*tensor->data<int32_t>());
      }
    }
    return vec_epxand_shape;
  } else {
    return ctx.Attr<std::vector<int>>("shape");
  }
}
}  // namespace operators
}  // namespace paddle
