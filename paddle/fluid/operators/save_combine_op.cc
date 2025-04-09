/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/save_combine_op.h"

#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle::operators {

class SaveCombineOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
  // TODO(lujun): The override here is just to bypass transform
  //  in operator impl, which is not elegant enough.
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    return phi::KernelKey(tensor.place(),
                          phi::DataLayout::ALL_LAYOUT,
                          expected_kernel_type.dtype());
  }
};

class SaveCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(vector) Input DenseTensors that need to be saved together in a file.")
        .AsDuplicable();
    AddComment(R"DOC(
SaveCombine operator

This operator will serialize and write a list of input phi::DenseTensor variables
to a file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if it exists.")
        .SetDefault(true);
    AddAttr<bool>("save_as_fp16",
                  "(boolean, default false)"
                  "If true, the tensor will be converted to float16 data "
                  "type and then saved. Otherwise, the tensor will be "
                  "directly saved without data type conversion.")
        .SetDefault(false);
    AddAttr<std::string>(
        "file_path",
        "(string)"
        "The \"file_path\" where the phi::DenseTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string& path) { return !path.empty(); });
    AddAttr<bool>("save_to_memory",
                  "(boolean, default false)"
                  "If true, the variables will be saved to binary strings.")
        .SetDefault(false);
    AddOutput("Y",
              "(RAW, default empty)."
              "This output is used when saving variables to binary strings.")
        .AsDispensable();
  }
};

class SaveCombineOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType(
        "Y", framework::proto::VarType::RAW, framework::ALL_ELEMENTS);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(save_combine,
                  ops::SaveCombineOp,
                  ops::SaveCombineOpProtoMaker,
                  ops::SaveCombineOpInferVarType);
