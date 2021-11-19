// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/cinn_launch_op.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(cudnn_deterministic);

namespace paddle {
namespace operators {

namespace details {

const ::cinn::common::Target& PlaceToCinnTarget(const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return ::cinn::common::DefaultHostTarget();
  } else if (platform::is_gpu_place(place)) {
    return ::cinn::common::DefaultNVGPUTarget();
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "CINN is not supported on current place:%s", place));
  return ::cinn::common::UnkTarget();
}

void DebugCinnCompiledResult(const CinnCompiledObject& result) {
  if (!VLOG_IS_ON(4)) {
    return;
  }
  const auto& cinn_runtime_program = result.runtime_program;
  const auto& cinn_scope = *(result.scope);
  const auto& paddle2cinn_varmap = result.paddle2cinn_varmap;

  VLOG(4) << "Compiled runtime_program instrunction size:["
          << cinn_runtime_program->size() << "]";

  std::vector<std::string> infos;
  auto cinn_var_names = cinn_scope.var_names();
  infos.reserve(cinn_var_names.size());
  std::transform(cinn_var_names.begin(), cinn_var_names.end(),
                 std::back_inserter(infos),
                 [](const auto& name_view) { return name_view.data(); });
  VLOG(4) << "Compiled scope variable names:["
          << string::join_strings(infos, ',') << "]";

  infos.clear();
  infos.reserve(paddle2cinn_varmap.size());
  std::transform(paddle2cinn_varmap.begin(), paddle2cinn_varmap.end(),
                 std::back_inserter(infos), [](const auto& paddle2cinn) {
                   return paddle2cinn.first + "->" + paddle2cinn.second;
                 });
  VLOG(4) << "Compiled paddle2cinn_varmap:[" << string::join_strings(infos, ',')
          << "]";
}

void LaunchCinnExecution(const CinnCompiledObject& compiled_obj,
                         const CinnLaunchContext& context) {
  compiled_obj.runtime_program->Execute(&context.FinalizeArguments());
}

void SetCinnRuntimeFlags() {
  VLOG(4) << "Set FLAGS_cinn_cudnn_deterministic to "
          << FLAGS_cudnn_deterministic;
  ::cinn::runtime::SetCinnCudnnDeterministic(FLAGS_cudnn_deterministic);
}

CinnLaunchContext::CinnLaunchContext(const CinnCompiledObject& compiled_obj)
    : paddle2cinn_varmap_(compiled_obj.paddle2cinn_varmap),
      cinn_scope_(compiled_obj.scope) {
  auto var_names = cinn_scope_->var_names();
  cinn_variable_names_.reserve(var_names.size());
  std::transform(
      var_names.begin(), var_names.end(),
      std::inserter(cinn_variable_names_, cinn_variable_names_.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });
}

bool CinnLaunchContext::IsVariableUsed(const std::string& paddle_name) {
  return paddle2cinn_varmap_.count(paddle_name) > 0 &&
         cinn_variable_names_.count(paddle2cinn_varmap_.at(paddle_name)) > 0;
}

CinnTensor CinnLaunchContext::GetCinnTensor(const std::string& var_name) {
  PADDLE_ENFORCE_GT(cinn_variable_names_.count(var_name), 0,
                    platform::errors::NotFound(
                        "Variable(%s) not found in cinn scope.", var_name));
  return cinn_scope_->GetTensor(var_name);
}

std::unordered_set<std::string> CinnLaunchContext::GetInternalVariableNames() {
  std::unordered_set<std::string> all_parameters(cinn_variable_names_);
  std::for_each(name2argument_.begin(), name2argument_.end(),
                [&all_parameters](const auto& name2arg) {
                  all_parameters.erase(name2arg.first);
                });
  return all_parameters;
}

void CinnLaunchContext::MutableTensorData(const std::string& var_name,
                                          const platform::Place& place,
                                          LoDTensor* paddle_tensor,
                                          bool is_internal_var) {
  auto cinn_name = var_name;
  if (!is_internal_var) {
    PADDLE_ENFORCE_EQ(IsVariableUsed(var_name), true,
                      platform::errors::InvalidArgument(
                          "Paddle variable(%s) not used by cinn", var_name));
    cinn_name = paddle2cinn_varmap_.at(var_name);
  }

  auto cinn_tensor = GetCinnTensor(cinn_name);
  // TODO(CtfGo): support mutable corresponding c++ type after CINN ready
  paddle_tensor->mutable_data<float>(
      framework::make_ddim(cinn_tensor->shape().data()), place);
}

void CinnLaunchContext::CheckTensorEquivalent(const std::string& paddle_name,
                                              const LoDTensor& paddle_tensor,
                                              const CinnTensor& cinn_tensor) {
  PADDLE_ENFORCE_EQ(
      paddle_tensor.IsInitialized(), true,
      platform::errors::InvalidArgument(
          "Tensor in variable(%s) is not initialized.", paddle_name));

  // check dimension
  auto cinn_dims = framework::make_ddim(cinn_tensor->shape().data());
  PADDLE_ENFORCE_EQ(paddle_tensor.dims(), cinn_dims,
                    platform::errors::PreconditionNotMet(
                        "Tensors' shape in variable(%s) are not equivalent, "
                        "paddle's shape = [%s], but cinn's shape = [%s].",
                        paddle_name, paddle_tensor.dims(), cinn_dims));

  // TODO(CtfGo): check the underlying data type after CINN ready
}

void CinnLaunchContext::AssignExternalVariable(const std::string& paddle_name,
                                               LoDTensor* paddle_tensor) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(paddle_name), true,
                    platform::errors::InvalidArgument(
                        "Paddle variable(%s) not used by cinn", paddle_name));

  const auto& cinn_name = paddle2cinn_varmap_.at(paddle_name);
  CheckTensorEquivalent(paddle_name, *paddle_tensor, GetCinnTensor(cinn_name));
  return SetArgument(cinn_name, paddle_tensor);
}

void CinnLaunchContext::AssignInternalVariable(const std::string& cinn_name,
                                               LoDTensor* paddle_tensor) {
  PADDLE_ENFORCE_GT(cinn_variable_names_.count(cinn_name), 0,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not found in cinn socpe.", cinn_name));
  CheckTensorEquivalent(cinn_name, *paddle_tensor, GetCinnTensor(cinn_name));
  return SetArgument(cinn_name, paddle_tensor);
}

std::unique_ptr<cinn_buffer_t> CinnLaunchContext::ShareTensorWithCinnBuffer(
    LoDTensor* tensor) {
  // convert paddle dimensions array to cinn format
  std::vector<cinn_dimension_t> cinn_dims(tensor->dims().size());
  for (auto i = 0; i < tensor->dims().size(); ++i) {
    cinn_dims[i] = static_cast<cinn_dimension_t>(tensor->dims().at(i));
  }

  auto cinn_buffer = std::make_unique<cinn_buffer_t>();
  // assign size and memory
  cinn_buffer->resize(cinn_dims.data(), cinn_dims.size());
  cinn_buffer->memory = reinterpret_cast<uint8_t*>(tensor->data<float>());
  return cinn_buffer;
}

void CinnLaunchContext::SetArgument(const std::string& cinn_name,
                                    LoDTensor* paddle_tensor) {
  auto buffer = ShareTensorWithCinnBuffer(paddle_tensor);
  name2argument_.emplace(cinn_name, buffer.get());
  hold_buffers_.emplace_back(std::move(buffer));
  VLOG(4) << "SetArgument-" << name2argument_.size() << ": "
          << "name(" << cinn_name << "), "
          << "type(" << framework::DataTypeToString(paddle_tensor->type())
          << "), dims(" << paddle_tensor->dims() << ").";
}

const std::map<std::string, cinn_pod_value_t>&
CinnLaunchContext::FinalizeArguments() const {
  // Check all execution parameters are assigned valued.
  std::for_each(cinn_variable_names_.begin(), cinn_variable_names_.end(),
                [this](const auto& var_name) {
                  PADDLE_ENFORCE_GT(name2argument_.count(var_name), 0,
                                    platform::errors::InvalidArgument(
                                        "Variable(%s) is missed for launching "
                                        "compiled program execution",
                                        var_name));
                });
  return name2argument_;
}

}  // namespace details

class CinnLaunchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs(kX), "Input", kX, "CinnLaunchOp");
    OP_INOUT_CHECK(ctx->HasOutputs(kOutputs), "Output", kOutputs,
                   "CinnLaunchOp");
  }

 protected:
  /* [Why use single type kernel]:
   *
   * This op is similar to a control flow op, it doses not need
   * a op kernel, but in order to make it execute under dynamic
   * graph mode, implement it with op kernel.
   *
   * So whether the kernel data type is int, float or other type,
   * which has no effect on its execution logic, so directly
   * specified a data type here.
   *
   * Of course, the data type here is also not important.
   */

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class CinnLaunchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "(vector<LoDTensor>)"
             "which are the input of graph inside the CinnLaunchOp.")
        .AsDuplicable();
    AddOutput(kOutputs,
              "(vector<LoDTensor>)"
              "which are the output of graph inside the CinnLaunchOp.")
        .AsDuplicable();
    AddAttr<std::string>(
        kCompilationKey,
        "(string)"
        "a hash key used to get the graph object or its computation result.");
    AddComment(R"DOC(
CinnLaunch Operator.

This operator is used to launch CINN(https://github.com/PaddlePaddle/CINN/blob/develop/README.md)
to compile a graph and execute the compiled object.

Both input and output of this operator are a set of variables
which are input and output of the graph respectively that will be
compiled and executed in this operator.
In addition, there is an attribute named 'compilation_key' should be
set necessarily to get corresponding ir::Graph object of the graph
or its computation result.

It accomplishes the computation of graph following several steps:
  1. Fetch ir::Graph object from CinnCompiler using kCompilationKey
  2. Compile the graph to a compiled object, and insert it to the
     global cache so that we can directly query it from this cache next time
     when shape of input variables are not changed at all.
  3. Create and instantiate all variables used to execute compiled runtime program
     if necessary according to the info(type,shape) included in the return scope.
  4. Pack each tensor buffer of all above variables as execution arguments.
  5. Launch execution of the runtime program with above arguments, then
     the result would be output by writing value on underlying buffer address.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cinn_launch, ops::CinnLaunchOp, ops::CinnLaunchOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
/* see [Why use single type kernel] */
REGISTER_OP_CPU_KERNEL(
    cinn_launch,
    ops::CinnLaunchOpKernel<paddle::platform::CPUDeviceContext, float>);
