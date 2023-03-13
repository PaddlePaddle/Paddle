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
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
class GPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    phi::DenseTensor* tensor = nullptr;
    auto out_var = context.OutputVar("Out");
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        context.MultiInput<phi::DenseTensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || context.HasInput("ShapeTensor")) {
      if (context.HasInput("ShapeTensor")) {
        auto* shape_tensor = context.Input<phi::DenseTensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    if (out_var->IsType<phi::SelectedRows>()) {
      auto* selected_rows = out_var->GetMutable<phi::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = context.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(phi::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<phi::DenseTensor>()) {
      tensor = out_var->GetMutable<phi::DenseTensor>();
      if (!new_shape.empty()) tensor->Resize(phi::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be "
          "phi::DenseTensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    UniformRandom<T>(context, tensor);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(uniform_random_batch_size_like,
                        paddle::operators::GPUUniformRandomKernel<float>,
                        paddle::operators::GPUUniformRandomKernel<double>);
