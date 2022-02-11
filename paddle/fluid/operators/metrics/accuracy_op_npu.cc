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

#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/metrics/accuracy_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AccuracyNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Out");
    auto* label = ctx.Input<Tensor>("Label");
    auto* indices = ctx.Input<Tensor>("Indices");

    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    auto* correct = ctx.Output<Tensor>("Correct");
    auto* total = ctx.Output<Tensor>("Total");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    int num_samples = inference->dims()[0];
    if (num_samples == 0) {
      return;
    }

    // cast `indices` or `label` if their type is not consistent
    Tensor cast_indices(experimental::DataType::INT32);
    Tensor cast_label(experimental::DataType::INT32);
    if (indices->dtype() != label->dtype()) {
      auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::INT32);
      if (framework::TransToProtoVarType(indices->dtype()) !=
          framework::proto::VarType::INT32) {
        cast_indices.Resize(indices->dims());
        cast_indices.mutable_data<int>(ctx.GetPlace());
        const auto& runner_cast_indices =
            NpuOpRunner("Cast", {*indices}, {cast_indices},
                        {{"dst_type", static_cast<int>(dst_dtype)}});
        runner_cast_indices.Run(stream);
      } else {
        cast_indices.ShareDataWith(*indices);
      }
      if (framework::TransToProtoVarType(label->dtype()) !=
          framework::proto::VarType::INT32) {
        cast_label.Resize(label->dims());
        cast_label.mutable_data<int>(ctx.GetPlace());
        const auto& runner_cast_label =
            NpuOpRunner("Cast", {*label}, {cast_label},
                        {{"dst_type", static_cast<int>(dst_dtype)}});
        runner_cast_label.Run(stream);
      } else {
        cast_label.ShareDataWith(*label);
      }
    } else {
      cast_indices.ShareDataWith(*indices);
      cast_label.ShareDataWith(*label);
    }

    // equal
    Tensor tmp_equal(experimental::DataType::BOOL);
    tmp_equal.Resize(inference->dims());
    tmp_equal.mutable_data<bool>(ctx.GetPlace());
    const auto& runner_equal =
        NpuOpRunner("Equal", {cast_indices, cast_label}, {tmp_equal}, {});
    runner_equal.Run(stream);

    // cast equal
    Tensor tmp_equal_cast(experimental::DataType::FLOAT32);
    tmp_equal_cast.Resize(inference->dims());
    tmp_equal_cast.mutable_data<float>(ctx.GetPlace());
    const auto& runner_cast_equal = NpuOpRunner(
        "Cast", {tmp_equal}, {tmp_equal_cast},
        {{"dst_type",
          static_cast<int>(ConvertToNpuDtype(
              framework::TransToProtoVarType(tmp_equal_cast.dtype())))}});
    runner_cast_equal.Run(stream);

    // [correct]
    // reduce_max
    Tensor tmp_correct_max(experimental::DataType::FLOAT32);
    tmp_correct_max.Resize(framework::make_ddim({num_samples}));
    tmp_correct_max.mutable_data<float>(ctx.GetPlace());
    const auto& runner_reduce_max =
        NpuOpRunner("ReduceMaxD", {tmp_equal_cast}, {tmp_correct_max},
                    {{"axes", std::vector<int>{1}}, {"keep_dims", false}});
    runner_reduce_max.Run(stream);

    // reduce_sum
    Tensor tmp_correct(experimental::DataType::FLOAT32);
    tmp_correct.Resize(correct->dims());
    tmp_correct.mutable_data<float>(ctx.GetPlace());
    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {tmp_correct_max}, {tmp_correct},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", false}});
    runner_reduce_sum.Run(stream);

    // cast to int
    correct->mutable_data<int>(ctx.GetPlace());
    const auto& runner_cast_correct = NpuOpRunner(
        "Cast", {tmp_correct}, {*correct},
        {{"dst_type", static_cast<int>(ConvertToNpuDtype(
                          framework::TransToProtoVarType(correct->dtype())))}});
    runner_cast_correct.Run(stream);

    // [total]
    total->mutable_data<int>(ctx.GetPlace());
    FillNpuTensorWithConstant<int>(total, static_cast<int>(num_samples));

    // use `total` of type `float32` for calculating accuracy
    Tensor tmp_total(experimental::DataType::FLOAT32);
    tmp_total.Resize(total->dims());
    tmp_total.mutable_data<float>(ctx.GetPlace());
    FillNpuTensorWithConstant<float>(&tmp_total,
                                     static_cast<float>(num_samples));

    // [accuracy]
    accuracy->mutable_data<float>(ctx.GetPlace());
    const auto& runner_accuracy =
        NpuOpRunner("Div", {tmp_correct, tmp_total}, {*accuracy}, {});
    runner_accuracy.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    accuracy, ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
