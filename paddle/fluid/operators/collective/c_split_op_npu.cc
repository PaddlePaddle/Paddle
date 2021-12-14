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

#include "paddle/fluid/operators/collective/c_split_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CSplitOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_GE(rank, 0, platform::errors::PreconditionNotMet(
                                   "The value of rank (%d) for c_split must be "
                                   "greater than or equal to 0.",
                                   rank));
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank, nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank, nranks));

    auto dims = x->dims();
    int axis = dims.size() - 1;
    auto dims_size = dims.size();
    PADDLE_ENFORCE_EQ(
        dims[dims_size - 1] % nranks, 0,
        platform::errors::PreconditionNotMet("The last dim size (%d) must be "
                                             "devided by nranks (%d).",
                                             dims[dims_size - 1], nranks));
    dims[dims_size - 1] /= nranks;
    out->mutable_data<T>(dims, place);

    Tensor out_invalid(out->type());
    out_invalid.mutable_data<T>(dims, place);

    std::vector<Tensor> outputs(nranks);
    for (int j = 0; j < nranks; ++j) {
      if (j == rank) {
        outputs[j].ShareDataWith(*out);
      } else {
        outputs[j].ShareDataWith(out_invalid);
      }
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner runner;
    framework::NPUAttributeMap attr_input = {{"num_split", nranks},
                                             {"split_dim", axis}};
    runner.SetType("SplitD").AddInputs({*x}).AddOutputs(outputs).AddAttrs(
        attr_input);
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(c_split, ops::CSplitOpNPUKernel<float>,
                       ops::CSplitOpNPUKernel<int>,
                       ops::CSplitOpNPUKernel<int8_t>,
                       ops::CSplitOpNPUKernel<plat::float16>);
