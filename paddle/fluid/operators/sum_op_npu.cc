/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using SelectedRows = phi::SelectedRows;

template <typename DeviceContext, typename T>
class SumNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<phi::DenseTensor>()) {
      auto *out = out_var->GetMutable<phi::DenseTensor>();
      auto x = ctx.MultiInput<phi::DenseTensor>("X");
      out->mutable_data<T>(ctx.GetPlace());

      auto place = ctx.GetPlace();

      int n = static_cast<int>(x.size());
      if (n == 1) {
        paddle::framework::TensorCopy(*x[0], place, out);
        return;
      }

      std::vector<phi::DenseTensor> inputs;
      std::vector<std::string> names;
      for (int i = 0; i < n; ++i) {
        if (x[i] && x[i]->numel() > 0) {
          inputs.push_back(*x[i]);
          names.push_back("x" + std::to_string(i));
        } else {
          continue;
        }
      }

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      NpuOpRunner runner{"AddN", {inputs}, {*out}, {{"N", n}}};
      runner.AddInputNames(names);
      runner.Run(stream);
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      auto in_vars = ctx.MultiInputVar("X");
      bool in_place = out_var == in_vars[0];
      auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE_EQ(in_vars[i]->IsType<framework::LoDTensorArray>(),
                          true,
                          platform::errors::InvalidArgument(
                              "Only support all inputs are TensorArray, "
                              "but inputs[%d] is not TensorArray.",
                              i));
        auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].IsInitialized() && (in_array[i].numel() != 0)) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (!out_array[i].IsInitialized() || (out_array[i].numel() == 0)) {
              framework::TensorCopy(in_array[i],
                                    in_array[i].place(),
                                    ctx.device_context(),
                                    &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE_EQ(
                  out_array[i].lod(),
                  in_array[i].lod(),
                  platform::errors::InvalidArgument(
                      "The lod message between inputs[%d] and"
                      " outputs[%d] must be same, but now is not same.",
                      i,
                      i));
              auto stream = ctx.template device_context<
                                   paddle::platform::NPUDeviceContext>()
                                .stream();
              NpuOpRunner runner{
                  "Add", {out_array[i], in_array[i]}, {out_array[i]}, {}};
              runner.Run(stream);
            }
          }
        }
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) must be phi::DenseTensor or "
          "LoDTensorArray. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    sum,
    ops::SumNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SumNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
