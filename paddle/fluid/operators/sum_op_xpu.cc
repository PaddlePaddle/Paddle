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

#ifdef PADDLE_WITH_XPU

#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {
using framework::Tensor;
using SelectedRows = phi::SelectedRows;
using LoDTensor = framework::LoDTensor;
template <typename DeviceContext, typename T>
class SumXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    auto out_var = context.OutputVar("Out");

    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = context.Output<LoDTensor>("Out");
      bool in_place = out_var == in_vars[0];
      int N = in_vars.size();

      if (!in_place) {
        out->mutable_data<T>(context.GetPlace());
      }
      auto &dev_ctx = context.template device_context<DeviceContext>();
      std::vector<const XPUType *> ptrs;
      for (int i = 0; i < N; ++i) {
        PADDLE_ENFORCE_EQ(
            in_vars[i]->IsType<framework::LoDTensor>(),
            true,
            platform::errors::InvalidArgument("XPU only support LodTensor"));
        auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
        if (in_t.numel() == 0) {
          continue;
        }
        ptrs.push_back(reinterpret_cast<const XPUType *>(in_t.data<T>()));
      }
      int r = xpu::sum(dev_ctx.x_context(),
                       ptrs,
                       reinterpret_cast<XPUType *>(out->data<T>()),
                       out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "sum");
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
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
                                    context.device_context(),
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

              std::vector<const XPUType *> ptrs;
              ptrs.push_back(
                  reinterpret_cast<const XPUType *>(in_array[i].data<T>()));
              ptrs.push_back(
                  reinterpret_cast<const XPUType *>(out_array[i].data<T>()));

              auto &dev_ctx = context.template device_context<DeviceContext>();
              // int sum(Context* ctx, const std::vector<const T*>& x_list, T*
              // y, int len);
              int r =
                  xpu::sum(dev_ctx.x_context(),
                           ptrs,
                           reinterpret_cast<XPUType *>(out_array[i].data<T>()),
                           out_array[i].numel());
              PADDLE_ENFORCE_XDNN_SUCCESS(r, "sum");
            }
          }
        }
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) must be Tensor or "
          "LoDTensorArray. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    sum,
    ops::SumXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SumXPUKernel<paddle::platform::XPUDeviceContext,
                      paddle::platform::float16>);
#endif
