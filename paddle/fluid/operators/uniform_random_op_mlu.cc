/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class MLUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    framework::Tensor *tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");

    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ShapeTensor")) {
        auto *shape_tensor = ctx.Input<framework::Tensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    if (out_var->IsType<pten::SelectedRows>()) {
      auto *selected_rows = out_var->GetMutable<pten::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(framework::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (!new_shape.empty()) tensor->Resize(framework::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }

    tensor->mutable_data<T>(ctx.GetPlace());
    int64_t size = tensor->numel();
    const float min = static_cast<T>(ctx.Attr<float>("min"));
    const float max = static_cast<T>(ctx.Attr<float>("max"));
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    // make mlu seed
    MLUCnnlRandomGeneratorDesc random_desc(/*is_mlu200=*/false, seed);
    cnnlDataType_t data_type = ToCnnlDataType(tensor->type());
    MLUCnnl::RandomUniform(ctx, size, /*data type=*/data_type,
                           random_desc.get(), min, max, GetBasePtr(tensor));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_MLU_KERNEL(uniform_random,
                       paddle::operators::MLUUniformRandomKernel<float>);
