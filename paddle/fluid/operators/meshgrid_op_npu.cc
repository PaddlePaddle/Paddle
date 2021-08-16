/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/meshgrid_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeshgridNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto outs = context.MultiOutput<framework::Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        (ins.size() > 1) && (ins.size() < 7), true,
        platform::errors::InvalidArgument(
            "Excepted Tensor numbers between 2 and 6, but only received d% .",
            ins.size()));

    int64_t size = ins.size();
    std::vector<int64_t> shape(size);

    for (int64_t i = 0; i < size; i++) {
      switch (ins[i]->dims().size()) {
        case 0:
          shape[i] = 1;
          break;
        case 1:
          shape[i] = ins[i]->dims()[0];
          break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Expected scalar or 1D tensor in the tensor list but got tensor "
              "%d: ",
              i));
      }
    }

    for (int64_t i = 0; i < size; i++) {
      std::vector<int64_t> view_shape(size, 1);
      view_shape[i] = shape[i];

      framework::DDim out_dims_reshape = framework::make_ddim(view_shape);
      framework::Tensor reshape_ins_tensor(ins[i]->type());
      reshape_ins_tensor.ShareDataWith(*ins[i]);
      reshape_ins_tensor.Resize(out_dims_reshape);

      framework::DDim out_dims = framework::make_ddim(shape);
      outs[i]->Resize(out_dims);
      outs[i]->mutable_data<T>(context.GetPlace());

      auto stream =
          context.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      const auto& runner = NpuOpRunner("BroadcastToD", {reshape_ins_tensor},
                                       {*(outs[i])}, {{"shape", shape}});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    meshgrid, ops::MeshgridNPUKernel<plat::NPUDeviceContext, float>,
    ops::MeshgridNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::MeshgridNPUKernel<plat::NPUDeviceContext, int32_t>);
