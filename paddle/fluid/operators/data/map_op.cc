/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/data/map_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class MapOp : public framework::OperatorBase {
 public:
  // using framework::OperatorWithKernel::OperatorWithKernel;
  MapOp(const std::string& type,
        const framework::VariableNameMap& inputs,
        const framework::VariableNameMap& outputs,
        const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInputs("In"), "Input", "In", "MapOp");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "MapOp");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

 private:
  void RunImpl(const framework::Scope& scope,
      const platform::Place& dev_place) const override {
    // LOG(ERROR) << "MapOpKernel RunImpl enter";
    // Step1: get output vars and attrs
    auto inputs = Inputs("In");
    std::vector<Variable*> input_vars;
    input_vars.reserve(inputs.size());
    for (auto& input : inputs) {
      input_vars.emplace_back(scope.FindVar(input));
    }

    auto outputs = Outputs("Out");
    std::vector<Variable*> output_vars;
    output_vars.reserve(outputs.size());
    for (auto& output : outputs) {
      output_vars.emplace_back(scope.FindVar(output));
    }

    CheckInputQueueStatus(input_vars);
    CheckAndInitOutputQueue(output_vars, /*capacity=*/2);

    auto input_var_names = Attr<std::vector<std::string>>("input_var_names");
    auto output_var_names = Attr<std::vector<std::string>>("output_var_names");
    auto* map_block = Attr<BlockDesc*>("map_block");
    auto program_id = Attr<int64_t>("program_id");

    auto input_queues = GetQueueVecFromVariableVec(input_vars);
    auto output_queues = GetQueueVecFromVariableVec(output_vars);
    data::MapRunnerManager::Instance()->StartMapRunner(
                    map_block, program_id, &scope, dev_place,
                    input_var_names, output_var_names,
                    input_queues, output_queues);
    // LOG(ERROR) << "MapOpKernel RunImpl finish";
  }
};

class MapInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("In"), "Input", "In", "MapOp");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "MapOp");
  }
};

class MapInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};

class MapOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("In",
             "(LoDTensorBlockingQueueHolder)"
              "The output tensors of Map operator")
        .AsDuplicable();
    AddOutput("Out",
              "(LoDTensorBlockingQueueHolder)"
              "The output tensors of Map operator")
        .AsDuplicable();
    AddAttr<BlockDesc*>("map_block",
                        "(BlockDesc *)"
                        "The global block of executed map program "
                        "desc.");
    AddAttr<int64_t>("program_id",
                     "(int64_t)"
                     "The unique hash id used as cache key for "
                     "ExecutorInfoCache");
    AddAttr<std::vector<std::string>>("input_var_names",
                     "(list of string)"
                     "input variable names for map program");
    AddAttr<std::vector<std::string>>("output_var_names",
                     "(list of string)"
                     "output variable names for map program");
    AddComment(R"DOC(
        Map Op
         )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(map, ops::MapOp, ops::MapOpMaker,
                  ops::MapInferShape, ops::MapInferVarType);
REGISTER_OP_CPU_KERNEL(map, ops::MapOpKernel<paddle::platform::CPUDeviceContext, float>);
