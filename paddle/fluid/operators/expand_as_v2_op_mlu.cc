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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/expand_as_v2_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ExpandAsV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<phi::DenseTensor>("X")->dims().size();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto target_rank = target_shape.size();
    PADDLE_ENFORCE_GE(target_rank,
                      rank,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be greater than or equal to "
                          "the rank (%d) of the input 'x'.",
                          target_rank,
                          rank));
    PADDLE_ENFORCE_GE(
        rank,
        1,
        platform::errors::InvalidArgument("The rank (%d) of the input 'x' for "
                                          "expand_as_v2 op must be positive.",
                                          rank));
    PADDLE_ENFORCE_LE(target_rank,
                      MAX_RANK_SUPPORTED,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be less than or equal to %d.",
                          target_rank,
                          MAX_RANK_SUPPORTED));
    ExpandAs(context);
  }

 protected:
  void ExpandAs(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<phi::DenseTensor>("X");
    auto in_dims = in0->dims();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto vec_in_dims = phi::vectorize<int>(in_dims);
    auto diff = target_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);

    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(target_shape[i],
                        0,
                        platform::errors::InvalidArgument(
                            "The value of target shape cannot be zero."));
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            target_shape[i],
            platform::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in "
                "target tensor for expand_as_v2 op.",
                vec_in_dims[i],
                target_shape[i]));
      }
    }
    auto* out0 = context.Output<phi::DenseTensor>("Out");

    framework::DDim out_dims = phi::make_ddim(target_shape);

    out0->Resize(out_dims);
    out0->mutable_data<T>(context.GetPlace());

    MLUCnnlTensorDesc x_desc(*in0);
    MLUCnnlTensorDesc out_desc(*out0);

    MLUCnnl::BroadcastTo(context,
                         x_desc.get(),
                         GetBasePtr(in0),
                         out_desc.get(),
                         GetBasePtr(out0));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(expand_as_v2,
                       ops::ExpandAsV2MLUKernel<float>,
                       ops::ExpandAsV2MLUKernel<int>,
                       ops::ExpandAsV2MLUKernel<int64_t>,
                       ops::ExpandAsV2MLUKernel<int8_t>,
                       ops::ExpandAsV2MLUKernel<uint8_t>,
                       ops::ExpandAsV2MLUKernel<bool>,
                       ops::ExpandAsV2MLUKernel<paddle::platform::float16>);
