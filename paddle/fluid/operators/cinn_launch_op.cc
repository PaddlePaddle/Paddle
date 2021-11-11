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

void CheckTensorEquivalent(const std::string& paddle_name,
                           const LoDTensor* paddle_tensor,
                           const CinnTensor& cinn_tensor) {
  PADDLE_ENFORCE_EQ(
      paddle_tensor->IsInitialized(), true,
      platform::errors::InvalidArgument(
          "The tensor in variable(%s) is not initialized.", paddle_name));

  // check dimension
  auto cinn_dims = framework::make_ddim(cinn_tensor->shape().data());
  PADDLE_ENFORCE_EQ(paddle_tensor->dims(), cinn_dims,
                    platform::errors::InvalidArgument(
                        "The tensor dimension in variable(%s) "
                        "is not equivalent, paddle is [%s] "
                        "but cinn is [%s].",
                        paddle_name, paddle_tensor->dims(), cinn_dims));

  // TODO(CtfGo): check the underlying data type after CINN ready
}

void TensorMutableDataWithCinnInfo(const platform::Place& place,
                                   const CinnTensor& cinn_tensor,
                                   LoDTensor* paddle_tensor) {
  // TODO(CtfGo): support mutable corresponding c++ type after CINN ready
  paddle_tensor->mutable_data<float>(
      framework::make_ddim(cinn_tensor->shape().data()), place);
}

CinnLaunchContext::CinnLaunchContext(const CinnCompiledObject& compiled_obj)
    : paddle2cinn_varmap_(compiled_obj.paddle2cinn_varmap),
      cinn_scope_(compiled_obj.cinn_scope) {}

bool CinnLaunchContext::IsVariableUsed(const std::string& var_name) {
  return paddle2cinn_varmap_.count(var_name) > 0 &&
         cinn_scope_.FindVar(paddle2cinn_varmap_.at(var_name)) != nullptr;
}

CinnTensor CinnLaunchContext::GetCinnTensor(const std::string& paddle_name) {
  PADDLE_ENFORCE_GT(paddle2cinn_varmap_.count(paddle_name), 0,
                    platform::errors::NotFound(
                        "Not found the corresponding cinn variable "
                        "of paddle variable(%s) in compilation result.",
                        paddle_name));
  const auto& cinn_name = paddle2cinn_varmap_.at(paddle_name);
  PADDLE_ENFORCE_NOT_NULL(
      cinn_scope_.FindVar(cinn_name),
      platform::errors::NotFound("Variable(%s) not found in cinn scope",
                                 cinn_name));
  return cinn_scope_.GetTensor(cinn_name);
}

std::vector<std::string> SeperateTempVar(
    const CinnScope& cinn_scope,
    const std::map<std::string, cinn_pod_value_t>& processed_name2argument) {
  auto cinn_var_names = cinn_scope.var_names();
  std::unordered_set<std::string> all_cinn_names;
  all_cinn_names.reserve(cinn_var_names.size());
  std::transform(
      cinn_var_names.begin(), cinn_var_names.end(),
      std::inserter(all_cinn_names, all_cinn_names.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });

  std::for_each(processed_name2argument.begin(), processed_name2argument.end(),
                [&all_cinn_names](const auto& name2arg) {
                  all_cinn_names.erase(name2arg.first);
                });
  return {all_cinn_names.begin(), all_cinn_names.end()};
}

std::unique_ptr<cinn_buffer_t> ShareTensorWithCinnBuffer(LoDTensor* tensor) {
  // convert paddle dimensions array to cinn format
  std::vector<cinn_dimension_t> cinn_dims(tensor->dims().size());
  for (auto i = 0; i < tensor->dims().size(); ++i) {
    cinn_dims[i] = static_cast<cinn_dimension_t>(tensor->dims().at(i));
  }

  auto cinn_buffer = std::make_unique<cinn_buffer_t>();
  // assign size and memory
  cinn_buffer->resize(cinn_dims.data(), cinn_dims.size());
  cinn_buffer->memory = reinterpret_cast<uint8_t*>(tensor->data<float>());
  VLOG(4) << "ShareTensorWithCinnBuffer:num_elements:"
          << cinn_buffer->num_elements() << ", memory_address:"
          << reinterpret_cast<void*>(cinn_buffer->memory);
  return cinn_buffer;
}

void CheckArgumentsNotMissed(
    const CinnScope& cinn_scope,
    const std::map<std::string, cinn_pod_value_t>& name2argument) {
  auto cinn_var_names = cinn_scope.var_names();
  std::for_each(cinn_var_names.begin(), cinn_var_names.end(),
                [&name2argument](const auto& name_view) {
                  PADDLE_ENFORCE_GT(
                      name2argument.count(name_view.data()), 0,
                      platform::errors::InvalidArgument(
                          "Parameter(%s) is not assgined.", name_view.data()));
                });
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
