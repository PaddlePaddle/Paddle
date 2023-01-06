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

#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class MLUWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<phi::DenseTensor>("Condition");
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto dims = condition->dims();
    const int rank = dims.size();

    phi::DenseTensor num_true;
    num_true.mutable_data<int>({1}, context.GetPlace());
    MLUCnnlTensorDesc con_desc(*condition);
    MLUCnnlTensorDesc num_true_desc(num_true);
    MLUCnnl::NumTrue(context,
                     con_desc.get(),
                     GetBasePtr(condition),
                     num_true_desc.get(),
                     GetBasePtr(&num_true));

    phi::DenseTensor local_true_num;
    paddle::framework::TensorCopySync(
        num_true, platform::CPUPlace(), &local_true_num);
    auto true_num = *local_true_num.data<int>();

    out->Resize(phi::make_ddim({true_num, rank}));
    out->mutable_data<int64_t>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    auto& dev_ctx = context.template device_context<MLUDeviceContext>();
    phi::DenseTensor out_int32 =
        context.AllocateTmpTensor<int32_t, MLUDeviceContext>(out->dims(),
                                                             dev_ctx);
    MLUCnnlTensorDesc out_int32_desc(out_int32);
    MLUCnnlTensorDesc out_desc(*out);
    bool as_tuple = false;
    MLUCnnl::Where(context,
                   con_desc.get(),
                   GetBasePtr(condition),
                   num_true_desc.get(),
                   GetBasePtr(&num_true),
                   as_tuple,
                   out_int32_desc.get(),
                   GetBasePtr(&out_int32));
    cnnlCastDataType_t cast_type = GetCastDataType(VT::INT32, VT::INT64);
    MLUCnnl::Cast(context,
                  cast_type,
                  out_int32_desc.get(),
                  GetBasePtr(&out_int32),
                  out_desc.get(),
                  GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(where_index,
                       ops::MLUWhereIndexKernel<int>,
                       ops::MLUWhereIndexKernel<bool>,
                       ops::MLUWhereIndexKernel<float>);
