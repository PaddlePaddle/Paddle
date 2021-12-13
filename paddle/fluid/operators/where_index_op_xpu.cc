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

#include "paddle/fluid/operators/where_index_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class WhereIndexXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");

    const T* cond_data = condition->data<T>();
    auto numel = condition->numel();
    auto dims = condition->dims();
    const int rank = dims.size();

    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* true_num = RAII_GUARD.alloc_l3_or_gm<int32_t>(1);
    int true_num_cpu;
    int ret =
        xpu::nonzero_count(dev_ctx.x_context(), cond_data, true_num, numel);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU nonzero_count kernel return wrong value[%d %s] in WhereIndex",
            ret, XPUAPIErrorMsg[ret]));

    memory::Copy(platform::CPUPlace(), static_cast<void*>(&true_num_cpu),
                 BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                 static_cast<void*>(true_num), sizeof(int32_t));

    out->Resize(
        framework::make_ddim({static_cast<int64_t>(true_num_cpu), rank}));
    auto out_data = out->mutable_data<int64_t>(context.GetPlace());
    if (true_num_cpu == 0) {
      return;
    }

    auto condition_shape = framework::vectorize<int>(dims);
    ret = xpu::where(dev_ctx.x_context(), cond_data, out_data, condition_shape,
                     true_num_cpu);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU masked_select kernel return wrong value[%d %s]",
                          ret, XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(where_index, ops::WhereIndexXPUKernel<int>,
                       ops::WhereIndexXPUKernel<bool>,
                       ops::WhereIndexXPUKernel<float>);
#endif
