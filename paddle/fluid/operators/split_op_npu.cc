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

#include <memory>
#include <string>

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/split_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SplitNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int num = ctx.Attr<int>("num");
    std::vector<int> sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");

    auto in_dims = in->dims();
    auto outs_number = outs.size();

    bool need_resize_outs_dims = false;

    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor(axis_tensor)[0];
      need_resize_outs_dims = true;
    }

    PADDLE_ENFORCE_EQ(
        axis >= 0 && axis < in->dims().size(), true,
        platform::errors::InvalidArgument(
            "axis(%d) must satisfy 0 <= axis < input.dims(%d) and ", axis,
            static_cast<int>(in->dims().size())));

    auto sections_tensor_list =
        ctx.MultiInput<framework::Tensor>("SectionsTensorList");
    if (sections_tensor_list.size() > 0) {
      sections = GetDataFromTensorList(sections_tensor_list);
      need_resize_outs_dims = true;
    }

    if (need_resize_outs_dims) {
      std::vector<framework::DDim> outs_dims =
          UpdateOutsDims(true, true, in_dims, num, sections, axis, outs_number);
      for (size_t j = 0; j < outs.size(); ++j) {
        outs[j]->Resize(outs_dims[j]);
      }
    }

    std::vector<Tensor> outputs;
    auto place = ctx.GetPlace();
    for (size_t j = 0; j < outs.size(); ++j) {
      outs[j]->mutable_data<T>(ctx.GetPlace());
      outputs.push_back(*outs[j]);
    }
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner runner;
    if (sections.size() == 0) {
      framework::NPUAttributeMap attr_input = {{"num_split", num},
                                               {"split_dim", axis}};
      runner.SetType("SplitD").AddInputs({*in}).AddOutputs(outputs).AddAttrs(
          attr_input);
    } else {
      framework::NPUAttributeMap attr_input = {
          {"size_splits", sections},
          {"split_dim", axis},
          {"num_split", static_cast<int32_t>(sections.size())}};
      runner.SetType("SplitVD").AddInput(*in).AddOutputs(outputs).AddAttrs(
          attr_input);
    }

    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(split, ops::SplitNPUKernel<float>,
                       ops::SplitNPUKernel<int>,
                       ops::SplitNPUKernel<plat::float16>);
