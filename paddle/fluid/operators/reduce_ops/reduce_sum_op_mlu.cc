// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_ops/reduce_op_mlu.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceSumMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    MLUReduceOp<T>(context, "reduce_sum");
  }
};

template <typename T>
class ReduceSumGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<phi::DenseTensor>("X");
    auto* out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* in_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    in_grad->mutable_data<T>(context.GetPlace());

    bool reduce_all = context.Attr<bool>("reduce_all");
    auto reduce_dims = context.Attr<std::vector<int>>("dim");
    auto in_dims = phi::vectorize(in->dims());

    if (reduce_all) {
      reduce_dims.clear();
      for (size_t d = 0; d < in_dims.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }
    for (auto& d : reduce_dims) {
      if (d < 0) {
        d = d + in_dims.size();
      }
    }

    Tensor tmp_out(out_grad->dtype());
    auto tmp_output_dims = in_dims;
    for (auto d : reduce_dims) {
      tmp_output_dims[d] = 1;
    }
    tmp_out.ShareDataWith(*out_grad);
    tmp_out.Resize(phi::make_ddim(tmp_output_dims));

    MLUCnnlTensorDesc out_desc(tmp_out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
    MLUCnnlTensorDesc in_grad_desc(
        *in_grad, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

    MLUCnnl::BroadcastTo(context,
                         out_desc.get(),
                         GetBasePtr(&tmp_out),
                         in_grad_desc.get(),
                         GetBasePtr(in_grad));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(reduce_sum,
                       ops::ReduceSumMLUKernel<float>,
                       ops::ReduceSumMLUKernel<int>,
                       ops::ReduceSumMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(reduce_sum_grad,
                       ops::ReduceSumGradMLUKernel<float>,
                       ops::ReduceSumGradMLUKernel<plat::float16>);
