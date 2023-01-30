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

<<<<<<< HEAD
#include <string>

#include "paddle/fluid/framework/op_registry.h"
=======
#include "paddle/fluid/operators/load_op.h"

#include <string>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
<<<<<<< HEAD
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
=======
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::proto::VarType::FP32, ctx.GetPlace());
    return kt;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
<<<<<<< HEAD
    AddOutput("Out", "The phi::DenseTensor / SelectedRows need to be loaded");
=======
    AddOutput("Out", "The LoDTensor / SelectedRows need to be loaded");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    AddAttr<bool>(
        "load_as_fp16",
        "If true, the tensor will be first loaded and then "
        "converted to float16 data type. Otherwise, the tensor will be "
        "directly loaded without data type conversion. Default is false.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddAttr<int64_t>("seek", "(int64_t) Starting for load tensor from seek pos")
        .SetDefault(-1);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddComment(
<<<<<<< HEAD
        "Load operator will load a phi::DenseTensor / SelectedRows variable "
        "from "
=======
        "Load operator will load a LoDTensor / SelectedRows variable from "
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        "disk "
        "file.");
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);
<<<<<<< HEAD
=======

REGISTER_OP_CPU_KERNEL(
    load,
    ops::LoadOpKernel<phi::CPUContext, float>,
    ops::LoadOpKernel<phi::CPUContext, double>,
    ops::LoadOpKernel<phi::CPUContext, paddle::platform::bfloat16>,
    ops::LoadOpKernel<phi::CPUContext, int>,
    ops::LoadOpKernel<phi::CPUContext, int8_t>,
    ops::LoadOpKernel<phi::CPUContext, int64_t>);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
