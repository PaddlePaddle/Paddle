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

#ifdef PADDLE_WITH_ASCEND_CL
#include <iostream>
#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/expand_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ExpandNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    std::cout << "enter" << std::endl;
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'x' for Op(expand) "
            "must be greater than or equal to 1, but the value received is %d.",
            rank));
    std::cout << "enter" << std::endl;
    PADDLE_ENFORCE_LE(
        rank, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'x' for Op(expand) "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, rank));
    std::cout << "enter" << std::endl;
    switch (rank) { REP_EXPAND_TEMPLATE(MAX_RANK_SUPPORTED) }
    std::cout << "out" << std::endl;
  }

 protected:
  template <int Rank>
  void Expand(const framework::ExecutionContext& context) const {
    std::cout << "enter expand " << Rank << std::endl;
    auto* in0 = context.Input<framework::LoDTensor>("X");
    auto in_dims = in0->dims();
    auto expand_times = get_expand_times(context);
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(in_dims.size()), expand_times.size(),
        platform::errors::InvalidArgument(
            "The number of elements (%d) of 'expand_times' for "
            "Op(expand) must be equal to the number "
            "of dimensions (%d) of the input.",
            expand_times.size(), static_cast<size_t>(in_dims.size())));
    auto* out0 = context.Output<framework::LoDTensor>("Out");
    framework::DDim out_dims(in_dims);
    for (size_t i = 0; i < expand_times.size(); ++i) {
      out_dims[i] *= expand_times[i];
      std::cout << "out_dims " << out_dims[i] << std::endl;
    }
    out0->Resize(out_dims);

    // auto* expand_times_t2 =
    // context.Input<framework::LoDTensor>("ExpandTimes");
    framework::LoDTensor expand_times_t;
    TensorFromVector(expand_times, context.device_context(), &expand_times_t);
    std::cout << "expand_times_t:size " << expand_times.size() << std::endl;
    for (auto& t : expand_times) {
      std::cout << "expand_times_t " << t << std::endl;
    }

    auto runner = NpuOpRunner("Tile", {*in0, expand_times_t}, {*out0});
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    expand, ops::ExpandNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ExpandNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>,
    ops::ExpandNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ExpandNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::ExpandNPUKernel<paddle::platform::NPUDeviceContext, int8_t>);

#endif
