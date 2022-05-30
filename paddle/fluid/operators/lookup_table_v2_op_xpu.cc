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

#include "paddle/fluid/operators/lookup_table_v2_op.h"
#include <memory>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LookupTableV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");      // int
    auto *output_t = context.Output<LoDTensor>("Out");  // float
    auto *table_var = context.InputVar("W");
    PADDLE_ENFORCE_EQ(
        (std::is_same<DeviceContext, platform::XPUDeviceContext>::value), true,
        platform::errors::PreconditionNotMet("Unsupported place! only support "
                                             "xpu place , please check your "
                                             "place."));

    PADDLE_ENFORCE_EQ(table_var->IsType<LoDTensor>(), true,
                      platform::errors::PermissionDenied(
                          "Unsupported Variable Type , idx in "
                          "LookupTableV2XPUKernel should be LoDTensor."));

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t ids_numel = ids_t->numel();

    auto *table_t = context.Input<LoDTensor>("W");
    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto *table = table_t->data<T>();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    const int64_t *ids = ids_t->data<int64_t>();

    PADDLE_ENFORCE_EQ(
        ids_numel <= std::numeric_limits<int32_t>::max(), true,
        platform::errors::OutOfRange(
            "Number of ids greater than int32_t::max , please check "
            "number of ids in LookupTableV2XPUKernel."));

    int ym = static_cast<int>(ids_numel);

    size_t xm = table_t->dims()[0];
    size_t n = table_t->dims()[1];

    int r =
        xpu::embedding<T, int64_t>(dev_ctx.x_context(), table, ids, output, xm,
                                   n, ym, static_cast<int>(padding_idx));

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding");
  }
};

template <typename DeviceContext, typename T>
class LookupTableV2GradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    PADDLE_ENFORCE_EQ(table_var->IsType<LoDTensor>(), true,
                      platform::errors::PermissionDenied(
                          "Unsupported Variable Type , idx in "
                          "LookupTableV2GradXPUKernel should be LoDTensor."));
    table_dim = context.Input<LoDTensor>("W")->dims();

    bool is_sparse = context.Attr<bool>("is_sparse");
    PADDLE_ENFORCE_EQ(
        is_sparse, false,
        platform::errors::InvalidArgument(
            "LookupTableV2GradXPUKernel dose NOT support is_sparse = True."));

    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto d_table_t = context.Output<LoDTensor>(framework::GradVarName("W"));

    int64_t ids_numel = ids_t->numel();
    PADDLE_ENFORCE_EQ(
        ids_numel <= std::numeric_limits<int32_t>::max(), true,
        platform::errors::OutOfRange(
            "Number of ids greater than int32_t::max , please check "
            "number of ids in LookupTableV2GradXPUKernel."));

    auto &dev_ctx = context.template device_context<DeviceContext>();
    const int64_t *ids_data = ids_t->data<int64_t>();
    const T *d_output_data = d_output_t->data<T>();
    T *d_table_data = d_table_t->mutable_data<T>(context.GetPlace());
    int xm = d_table_t->dims()[0];
    int ym = static_cast<int>(ids_numel);
    int n = d_table_t->dims()[1];
    int padding_idx = context.Attr<int64_t>("padding_idx");

    int r = xpu::embedding_grad<T, int64_t>(dev_ctx.x_context(), d_output_data,
                                            ids_data, d_table_data, xm, n, ym,
                                            padding_idx);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_grad");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    lookup_table_v2,
    ops::LookupTableV2XPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    lookup_table_v2_grad,
    ops::LookupTableV2GradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
