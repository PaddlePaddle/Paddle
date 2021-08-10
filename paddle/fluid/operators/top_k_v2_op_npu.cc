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

template <typename T>
class TopkV2NPUKernel : public framework::OpKernel<T> {
 public:
  // Use Ascend TopKV2 operator to implement paddle TopKV2Op
  void Compute(const framework::ExecutionContext& context) const override {
    // Read message from context
    auto* input = context.Input<Tensor>("X");
    auto* k_tensor = context.Input<Tensor>("K");
    auto* out = context.Output<Tensor>("Out");
    auto* indices = context.Output<Tensor>("Indices");  // type:INT64

    int32_t k = static_cast<int32_t>(context.Attr<int>("k"));
    int axis = static_cast<int>(context.Attr<int>("axis"));
    const bool sorted = static_cast<bool>(context.Attr<bool>("sorted"));
    const bool largest = static_cast<bool>(context.Attr<bool>("largest"));

    // Calculate the real value of axis and k
    if (axis < 0) {
      axis += input->dims().size();
    }

    if (k_tensor != nullptr) {
      std::vector<int> v_tmp(1);
      TensorToVector(
          *k_tensor,
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          &v_tmp);
      k = static_cast<int32_t>(v_tmp[0]);
    }

    // Allocate space for output tensors of Paddle topKV2 operator
    framework::DDim output_dims = input->dims();
    output_dims[axis] = k;

    out->Resize(output_dims);
    indices->Resize(output_dims);

    out->mutable_data<T>(context.GetPlace());
    indices->mutable_data<int64_t>(context.GetPlace());

    // Allocate space for output indices of Ascend topkV2 operator
    framework::Tensor* indices_int32 =
        new Tensor(framework::proto::VarType::INT32);
    indices_int32->Resize(output_dims);
    indices_int32->mutable_data<int32_t>(context.GetPlace());
    VLOG(4) << "input:" << *input;
    // Run Ascend TopKV2 operator
    NpuOpRunner npu_op_runner_topkv2;
    auto npu_stream_topkv2 =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner_topkv2.SetType("TopKV2")
        .AddInput(*input)
        .AddInput(std::vector<int32_t>{k})
        .AddOutput(*out)
        .AddOutput(*indices_int32)
        .AddAttr("sorted", sorted)
        .AddAttr("dim", axis)
        .AddAttr("largest", largest)
        .Run(npu_stream_topkv2);
    VLOG(4) << "output:" << *out;
    // Cast 'indices_int32' to 'indices', from INT32 to INT64
    auto dst_dtype = ConvertToNpuDtype(indices->type());
    const auto& npu_op_runner_cast =
        NpuOpRunner("Cast", {*indices_int32}, {*indices},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    auto npu_stream_cast =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner_cast.Run(npu_stream_cast);

    VLOG(4) << "indices: " << *indices;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(top_k_v2, ops::TopkV2NPUKernel<float>,
                       ops::TopkV2NPUKernel<double>,
                       ops::TopkV2NPUKernel<int32_t>,
                       ops::TopkV2NPUKernel<int64_t>);
