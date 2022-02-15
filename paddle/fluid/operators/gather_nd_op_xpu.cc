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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/gather_nd_op.h"

namespace paddle {
namespace operators {

template <typename T>
class GatherNdXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *out = ctx.Output<Tensor>("Out");

    out->template mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;

    if (index->numel() == 0) {
      framework::TensorCopy(*x, ctx.GetPlace(), ctx.device_context(), out);
      return;
    }

    const auto &index_type = framework::TransToProtoVarType(index->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s],"
                          "but desires to be [%s] or [%s]",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto x_shape = paddle::framework::vectorize<int>(x->dims());
    auto index_shape = paddle::framework::vectorize<int>(index->dims());
    if (index_shape.size() == 1) {
      index_shape.insert(index_shape.begin(), 1);
    }
    xpu::VectorParam<int> x_vec = {x_shape.data(),
                                   static_cast<int>(x_shape.size()), nullptr};

    auto &dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int ret = XPU_SUCCESS;
    if (index_type == framework::proto::VarType::INT32) {
      ret = xpu::gather_nd<T, int>(dev_ctx.x_context(), x->data<T>(),
                                   index->data<int>(), out->data<T>(), x_vec,
                                   index_shape);
    } else {
      ret = xpu::gather_nd<T, int64_t>(dev_ctx.x_context(), x->data<T>(),
                                       index->data<int64_t>(), out->data<T>(),
                                       x_vec, index_shape);
    }
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU gather_nd kernel return wrong value[%d %s]", ret,
                          XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(gather_nd, ops::GatherNdXPUKernel<int>,
                       ops::GatherNdXPUKernel<int64_t>,
                       ops::GatherNdXPUKernel<float>);

#endif
