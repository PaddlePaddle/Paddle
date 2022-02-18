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
#include <memory>
#include <string>

#include "paddle/fluid/operators/scatter_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ScatterOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Ids");
    auto *updates = ctx.Input<Tensor>("Updates");
    auto *out = ctx.Output<Tensor>("Out");
    bool overwrite = ctx.Attr<bool>("overwrite");

    // In place output: Out = X, Out[ids] = Updates
    framework::TensorCopy(*x, ctx.GetPlace(), out);
    // Apply ScatterUpdate: Out[index] = Updates[:]
    const auto &index_type = framework::TransToProtoVarType(index->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s],"
                          "but desires to be [%s] or [%s].",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    // check index of shape 1-D
    PADDLE_ENFORCE_EQ(
        index->dims().size() == 1 ||
            (index->dims().size() == 2 && index->dims()[1] == 1),
        true, platform::errors::InvalidArgument(
                  "index's shape is error, "
                  "expect index'dims shape is 1 or 2 and index.dims[1] is 1"
                  "but got index'dims shape is %d",
                  index->dims().size()));

    int index_size = static_cast<int>(index->dims()[0]);
    auto x_dims = x->dims();
    auto update_dims = updates->dims();
    for (int i = 1; i < x_dims.size(); i++)
      PADDLE_ENFORCE_EQ(
          x_dims[i], update_dims[i],
          platform::errors::InvalidArgument(
              "The dimensions of the source tensor and target tensor should"
              " match, but received source tensor's %d-th dimension is %d,"
              "target tensor's %d-th dimension is %d.",
              i, x_dims[i], i, update_dims[i]));

    int dim0 = static_cast<int>(x->dims()[0]);
    int dim1 = static_cast<int>(
        framework::product(framework::slice_ddim(x_dims, 1, x_dims.size())));
    T *out_data = out->data<T>();
    const T *updates_data = updates->data<T>();

    auto &dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;

    Tensor indices_cpu(index->type());
    framework::TensorCopy(*index, platform::CPUPlace(), &indices_cpu);

    if (index_type == framework::proto::VarType::INT32) {
      auto index_data = const_cast<int *>(index->data<int>());
      xpu::VectorParam<int> indices{indices_cpu.data<int>(), index_size,
                                    index_data};
      r = xpu::scatter(dev_ctx.x_context(), updates_data, out_data, indices,
                       dim0, dim1, overwrite);
    } else {
      auto index_data = const_cast<int64_t *>(index->data<int64_t>());
      xpu::VectorParam<int64_t> indices{indices_cpu.data<int64_t>(), index_size,
                                        index_data};
      r = xpu::scatter(dev_ctx.x_context(), updates_data, out_data, indices,
                       dim0, dim1, overwrite);
    }
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU scatter kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(scatter, ops::ScatterOpXPUKernel<float>,
                       ops::ScatterOpXPUKernel<int64_t>);
#endif
