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

#ifdef PADDLE_WITH_MLU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/expand_v2_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ExpandV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<phi::DenseTensor>("X");
    auto* Out = ctx.Output<phi::DenseTensor>("Out");
    auto in_dims = X->dims();
    auto expand_shape = get_expand_shape(ctx);
    auto vec_in_dims = phi::vectorize<int>(in_dims);
    auto diff = expand_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> final_expand_shape(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(expand_shape[i],
                        0,
                        platform::errors::InvalidArgument(
                            "The expanded size cannot be zero."));
      if (i < diff) {  // expand_shape = [3,4,-1,-1], X = [10,2] -->
                       // final_expand_shape = [3,4,10,2]
        PADDLE_ENFORCE_GT(
            expand_shape[i],
            0,
            platform::errors::InvalidArgument(
                "The expanded size (%d) for non-existing dimensions must be "
                "positive for expand_v2 op.",
                expand_shape[i]));
        final_expand_shape[i] = expand_shape[i];
      } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], X =
                                         // [10,1] --> final_expand_shape =
                                         // [3,4,10,4]
        if (vec_in_dims[i] != 1) {
          PADDLE_ENFORCE_EQ(
              vec_in_dims[i],
              expand_shape[i],
              platform::errors::InvalidArgument(
                  "The value (%d) of the non-singleton dimension does not match"
                  " the corresponding value (%d) in shape for expand_v2 op.",
                  vec_in_dims[i],
                  expand_shape[i]));
          final_expand_shape[i] = expand_shape[i];
        } else {
          final_expand_shape[i] = expand_shape[i];
        }
      } else {  // expand_shape = [3,4,-1,-1], X = [10,2] --> final_expand_shape
                // = [3,4,10,2]
        PADDLE_ENFORCE_EQ(
            expand_shape[i],
            -1,
            platform::errors::InvalidArgument(
                "When the value in shape is negative for expand_v2 op, "
                "only -1 is supported, but the value received is %d.",
                expand_shape[i]));
        final_expand_shape[i] = vec_in_dims[i];
      }
    }

    auto rank = X->dims().size();
    PADDLE_ENFORCE_GE(
        rank,
        1,
        platform::errors::InvalidArgument(
            "The rank of the input 'X' for expand_v2_mlu op must be positive, "
            "but the value received is %d.",
            rank));
    auto shape_size = final_expand_shape.size();
    PADDLE_ENFORCE_GE(
        shape_size,
        rank,
        platform::errors::InvalidArgument(
            "The number (%d) of elements of 'shape' for expand_v2_mlu op must "
            "be "
            "greater than or equal to the rank (%d) of the input 'X'.",
            shape_size,
            rank));

    framework::DDim out_dims = phi::make_ddim(final_expand_shape);
    Out->Resize(out_dims);
    auto place = ctx.GetPlace();
    Out->mutable_data<T>(place);
    MLUCnnlTensorDesc x_desc(*X);
    MLUCnnlTensorDesc out_desc(*Out);
    MLUCnnl::BroadcastTo(
        ctx, x_desc.get(), GetBasePtr(X), out_desc.get(), GetBasePtr(Out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(expand_v2,
                       ops::ExpandV2MLUKernel<float>,
                       ops::ExpandV2MLUKernel<paddle::platform::float16>,
                       ops::ExpandV2MLUKernel<bool>,
                       ops::ExpandV2MLUKernel<int>,
                       ops::ExpandV2MLUKernel<int64_t>);

#endif
