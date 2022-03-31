/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUUniformRandomKernel : public framework::OpKernel<T> {
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

    if (out_var->IsType<phi::SelectedRows>()) {
      auto *selected_rows = out_var->GetMutable<phi::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(phi::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (!new_shape.empty()) tensor->Resize(phi::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    tensor->mutable_data<T>(ctx.GetPlace());
    int64_t size = tensor->numel();

    Tensor cpu_tensor(tensor->dtype());
    cpu_tensor.Resize(tensor->dims());
    T *data_cpu = cpu_tensor.mutable_data<T>(platform::CPUPlace());

    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    auto engine = framework::GetCPURandomEngine(seed);

    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = dist(*engine);
    }

    unsigned int diag_num =
        static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
    auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
    if (diag_num > 0) {
      PADDLE_ENFORCE_GT(
          size, (diag_num - 1) * (diag_step + 1),
          platform::errors::InvalidArgument(
              "ShapeInvalid: the diagonal's elements is equal (num-1) "
              "* (step-1) with num %d, step %d,"
              "It should be smaller than %d, but received %d",
              diag_num, diag_step, (diag_num - 1) * (diag_step + 1), size));
      for (int64_t i = 0; i < diag_num; ++i) {
        int64_t pos = i * diag_step + i;
        data_cpu[pos] = diag_val;
      }
    }

    // copy to NPU
    framework::TensorCopy(
        cpu_tensor, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), tensor);
    ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(uniform_random,
                       paddle::operators::NPUUniformRandomKernel<float>);
