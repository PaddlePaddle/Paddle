/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/top_k_v2_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TopkV2NPUKernel : public framework::OpKernel<T> {
 public:
  // Use CANN TopKV2 operator to implement paddle TopKV2Op
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = framework::Tensor;

    // Read message from context
    auto* input = context.Input<Tensor>("X");
    auto* k_tensor = context.Input<Tensor>("K");
    auto* output = context.Output<Tensor>("Out");
    auto* indices = context.Output<Tensor>("Indices");

    int k = static_cast<int>(context.Attr<int>("k"));
    int axis = static_cast<int>(context.Attr<int>("axis"));
    const bool sorted = static_cast<bool>(context.Attr<bool>("sorted"));
    const bool largest = static_cast<bool>(context.Attr<bool>("largest"));

    // Calculate the real value of axis and k
    if (axis < 0) {
      axis += input->dims().size();
    }

    if (k_tensor != nullptr) {
      // seems complicated, but I really don't know how to assign a NPU value to
      // a CPU variable by an elegant way
      std::vector<int> v_tmp(1);
      TensorToVector(
          *k_tensor,
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          &v_tmp);
      k = v_tmp[0];
    }

    // Allocate space for output tensors on NPU
    framework::DDim output_dims = input->dims();
    output_dims[axis] = k;

    output->Resize(output_dims);
    indices->Resize(output_dims);

    output->mutable_data<T>(context.GetPlace());
    indices->mutable_data<int64_t>(context.GetPlace());

    // Construct the input tensor x of CANN TopKV2 operator
    // as CANN TopKV2 operator does not support setting 'axis'(defaults to the
    // last dimension) and 'largest'(defaults to true) parameter yet,
    // 1. when the 'axis' is not the last dimension, we use CANN Transpose
    // operator to permutes the dimension 'axis' to the last dimension
    // 2. when the 'largest' is false, we use CANN Neg operator to negate the
    // input tensor element-wise, which convert descending to ascending order
    // once the functino of the parameter 'dim' and 'largest' is further
    // improved, these additional actions can be removed
    Tensor* input_transpose = nullptr;
    Tensor* input_neg = nullptr;
    const Tensor* x_cann =
        input;  // the input tensor "x" of CANN TopKV2 operator
    std::vector<int> perm;
    const int last_axis = static_cast<int>(
        input->dims().size() -
        1);  // attention: there may be bugs when the input tensor is empty

    if (axis !=
        last_axis) {  // in this case, the 'input' tensor should be transposed
      // compute perm vector
      perm.resize(last_axis + 1);
      for (int i = 0; i <= last_axis; ++i) {
        perm[i] = i;
      }
      std::swap(perm[axis], perm[last_axis]);

      // construct 'input_transpose'
      input_transpose = new Tensor(input->type());

      framework::DDim input_transpose_dims = input->dims();
      std::swap(input_transpose_dims[axis], input_transpose_dims[last_axis]);

      input_transpose->Resize(input_transpose_dims);
      input_transpose->mutable_data<T>(context.GetPlace());

      // run CANN Transpose operator
      NpuOpRunner npu_op_runner_transpose;
      auto npu_stream_transpose =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      npu_op_runner_transpose.SetType("Transpose")
          .AddInput(*input)
          .AddInput(std::move(perm))
          .AddOutput(*input_transpose)
          .Run(npu_stream_transpose);

      x_cann = input_transpose;
    }

    if (!largest) {  // in this case, the 'input' tensor should be negated
                     // element-wise
      // construct 'input_neg'
      auto* input_tensor =
          (input_transpose == nullptr ? input : input_transpose);
      input_neg = new Tensor(input_tensor->type());
      input_neg->Resize(input_tensor->dims());
      input_neg->mutable_data<T>(context.GetPlace());

      // run CANN Neg operator
      const auto& npu_op_runner_neg =
          NpuOpRunner("Neg", {*input_tensor}, {*input_neg});
      auto npu_stream_neg =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      npu_op_runner_neg.Run(npu_stream_neg);

      x_cann = input_neg;
    }

    // Construct the input and output tensors of CANN TopKV2 operator (except x)
    // input k: a 0D tensor of type int32, Number of top elements to look for
    // along the last dimension (along each row for matrices)
    Tensor* k_cann = new Tensor(framework::proto::VarType::INT32);
    k_cann->mutable_data<int32_t>({1}, context.GetPlace());
    FillNpuTensorWithConstant<int32_t>(k_cann, static_cast<int32_t>(k));

    // output values: a tensor specifying the sorted data, which has the same
    // type as 'x'
    Tensor* values_cann = nullptr;
    if (axis == last_axis && largest) {  // in this case, the CANN TopKV2 result
                                         // will directly output to the 'output'
                                         // tensor, which save an operation of
                                         // tensor copy
      values_cann = output;
    } else {
      values_cann = new Tensor(x_cann->type());
      framework::DDim values_cann_dims = x_cann->dims();
      values_cann_dims[last_axis] = k;
      values_cann->Resize(values_cann_dims);
      values_cann->mutable_data<T>(context.GetPlace());
    }

    // output indices: a tensor of type int32 specifying the indices of sorted
    // data
    Tensor* indices_cann = new Tensor(framework::proto::VarType::INT32);
    indices_cann->Resize(values_cann->dims());
    indices_cann->mutable_data<int32_t>(context.GetPlace());

    // Run CANN TopKV2 operator
    const auto& npu_op_runner_topkv2 =
        NpuOpRunner("TopKV2", {*x_cann, *k_cann}, {*values_cann, *indices_cann},
                    {{"sorted", sorted}});
    auto npu_stream_topkv2 =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner_topkv2.Run(npu_stream_topkv2);

    // Convert the computing result into paddle's output tensors
    // 'values_cann' to 'output' and 'indices_cann' to 'indices_transpose'
    Tensor* values_cann_neg = nullptr;
    Tensor* indices_cann_transpose = nullptr;

    if (!largest) {
      // run CANN Neg operator
      if (axis == last_axis) {
        values_cann_neg = output;  // in this case, the CANN Neg result will
                                   // directly output to the 'output' tensor
      } else {
        values_cann_neg = input_neg;  // as the 'input_neg' tensor is no longer
                                      // needed, we reuse its resources to
                                      // 'values_cann_neg' tensor
        values_cann_neg->Resize(values_cann->dims());
      }
      const auto& npu_op_runner_neg =
          NpuOpRunner("Neg", {*values_cann}, {*values_cann_neg});
      auto npu_stream_neg =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      npu_op_runner_neg.Run(npu_stream_neg);
    }

    if (axis != last_axis) {
      // run CANN Transpose operator
      // transpose values
      Tensor* input_tensor = (largest ? values_cann : values_cann_neg);
      NpuOpRunner npu_op_runner_transpose_values;
      auto npu_stream_transpose_values =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      npu_op_runner_transpose_values.SetType("Transpose")
          .AddInput(*input_tensor)
          .AddInput(std::move(perm))
          .AddOutput(*output)
          .Run(npu_stream_transpose_values);

      // transpose indices
      indices_cann_transpose = new Tensor(indices_cann->type());
      indices_cann_transpose->Resize(indices->dims());
      indices_cann_transpose->mutable_data<int32_t>(context.GetPlace());

      NpuOpRunner npu_op_runner_transpose_indices;
      auto npu_stream_transpose_indices =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      npu_op_runner_transpose_indices.SetType("Transpose")
          .AddInput(*indices_cann)
          .AddInput(std::move(perm))
          .AddOutput(*indices_cann_transpose)
          .Run(npu_stream_transpose_indices);
    } else {
      indices_cann_transpose = indices_cann;
    }

    // 'indices_cann_transpose' to 'indices', from INT32 to INT64
    auto dst_dtype = ConvertToNpuDtype(indices->type());
    const auto& npu_op_runner_cast =
        NpuOpRunner("Cast", {*indices_cann_transpose}, {*indices},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    auto npu_stream_cast =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner_cast.Run(npu_stream_cast);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    top_k_v2, ops::TopkV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TopkV2NPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::TopkV2NPUKernel<paddle::platform::NPUDeviceContext, int32_t>,
    ops::TopkV2NPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
