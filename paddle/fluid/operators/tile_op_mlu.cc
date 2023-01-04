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
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/tile_op_functor.h"

namespace paddle {
namespace operators {

template <typename T>
class TileMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<phi::DenseTensor>("X")->dims().size();
    PADDLE_ENFORCE_GE(
        rank,
        1,
        platform::errors::InvalidArgument(
            "The rank of the input 'x' for tile op must be a positive "
            "integer, but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank,
        MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of the input 'x' for tile op "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED,
            rank));
    auto repeat_times = get_repeat_times(context);
    int repeat_times_size = repeat_times.size();
    PADDLE_ENFORCE_GE(
        repeat_times_size,
        1,
        platform::errors::InvalidArgument(
            "The number of elements of the input 'repeat_times' for tile "
            "op must be positive, but the value received is %d.",
            repeat_times_size));
    PADDLE_ENFORCE_LE(
        repeat_times_size,
        MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The number of elements of the input 'repeat_times' for tile op "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED,
            repeat_times_size));

    auto* in0 = context.Input<phi::DenseTensor>("X");
    auto in_dims = in0->dims();
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      PADDLE_ENFORCE_GT(
          repeat_times[i],
          0,
          platform::errors::InvalidArgument(
              "All elements of the input 'repeat_times' for tile op must "
              "be positive integers, but the value received is %d.",
              repeat_times[i]));
    }
    auto vec_in_dims = phi::vectorize<int>(in_dims);
    if (repeat_times.size() < vec_in_dims.size()) {
      int diff = vec_in_dims.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, 1);
    } else {
      int diff = repeat_times.size() - vec_in_dims.size();
      vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    }
    PADDLE_ENFORCE_EQ(
        repeat_times.size(),
        vec_in_dims.size(),
        platform::errors::InvalidArgument(
            "The rank (%d) of the input 'x' and the rank (%d) of the input "
            "'repeat_times' for tile op must match after promotion.",
            vec_in_dims.size(),
            repeat_times.size()));

    auto* out0 = context.Output<phi::DenseTensor>("Out");
    bool repeat_one_times = true;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      if (repeat_times[i] != 1) {
        repeat_one_times = false;
      }
    }
    if (repeat_one_times) {
      paddle::framework::TensorCopy(*in0, context.GetPlace(), out0);
    } else {
      framework::DDim new_in_dims = phi::make_ddim(vec_in_dims);
      framework::DDim out_dims(new_in_dims);
      for (size_t i = 0; i < repeat_times.size(); ++i) {
        out_dims[i] *= repeat_times[i];
      }
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
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(tile,
                       ops::TileMLUKernel<bool>,
                       ops::TileMLUKernel<int>,
                       ops::TileMLUKernel<int64_t>,
                       ops::TileMLUKernel<float>);

#endif
