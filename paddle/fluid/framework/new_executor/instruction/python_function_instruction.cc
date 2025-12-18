// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/python_function_instruction.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

COMMON_DECLARE_bool(check_cuda_error);

namespace paddle::framework {

void PythonFunctionInstruction::BuildPythonFunctionContext(
    const paddle::dialect::OpYamlInfoParser& op_yaml_info) {
  PADDLE_ENFORCE_NOT_NULL(
      python_op_meta_,
      common::errors::PreconditionNotMet(
          "PythonFunctionInstruction: python_op_meta_ is null"));

  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(*python_op_meta_);
  VLOG(6) << "op_inplace_map.size(): " << op_inplace_map.size();

  // check inplace
  for (auto const& pair : op_inplace_map) {
    pir::Value output_value =
        op_->result(op_yaml_info.OutputName2Id().at(pair.second));
    if (paddle::framework::detail::IsDuplicableVar(pair.first) &&
        !IsInvalid(output_value)) {
      // make sure ctx has valid inplace optional outputs
      PADDLE_ENFORCE(
          paddle::framework::detail::IsOptionalVar(pair.second),
          common::errors::InvalidArgument(
              "Python operator couldn't find output name for %s. If "
              "you are using inplace optional inputs & outputs, please "
              "check your InplaceMap and `Outputs` again and make sure %s is "
              "wrapped by `paddle::Optional`",
              pair.second,
              pair.second));
    }
  }

  Scope* inner_scope = value_exec_info_.GetScope();

  auto attr_map = op_->attributes();

  // EmplaceBackInputs
  auto& vec_input_tensor_params = op_yaml_info.TensorParams(true);
  auto& name2id = op_yaml_info.InputName2Id();
  auto inplace_id_map = op_yaml_info.GetInplaceIdMap();
  int input_index = 0;
  int vec_input_index = 0;

  for (const std::string& t : vec_input_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        common::errors::NotFound("param [%s] MUST in name2id map", t));

    pir::Value ptr = op_->operand_source(op_yaml_info.InputName2Id().at(t));
    if (!IsInvalid(ptr)) {
      if (op_yaml_info.GetInputType(op_yaml_info.InputName2Id().at(t)) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        PADDLE_THROW(common::errors::Unimplemented(
            "Only support Tensor input type for now in "
            "PythonFunctionInstruction, "
            "does not support VectorType<DenseTensorType>."));
      } else {
        input_index++;
        python_function_ctx_.EmplaceBackInput(paddle::Tensor());
      }
      continue;
    }
    auto in_var_name = value_exec_info_.GetVarName(ptr);

    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      auto dense_tensor_in = var->GetMutable<phi::DenseTensor>();

      std::shared_ptr<phi::DenseTensor> tensor_in(
          dense_tensor_in, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      input_index++;
      paddle::Tensor python_in;
      python_in.set_impl(tensor_in);
      python_function_ctx_.EmplaceBackInput(std::move(python_in));
    } else if (var->IsType<VariableRefArray>()) {
      PADDLE_THROW(common::errors::Unimplemented(
          "Only support Tensor input type for "
          "now in PythonFunctionInstruction, "
          "does not support Vector<DenseTensor>."));
    } else {
      PADDLE_THROW(common::errors::Unimplemented("Not support var type [%d].",
                                                 var->Type()));
    }
  }

  // EmplaceBackAttributes
  const std::vector<std::string>& vec_attr_params =
      op_yaml_info.AttrParams(true);
  for (auto& t : vec_attr_params) {
    PADDLE_ENFORCE_NE(attr_map.find(t),
                      attr_map.end(),
                      common::errors::NotFound(
                          "Not found %s in attr_map, it maybe need mapping "
                          "it in OpTranslator.",
                          t));
    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "pir::Int32Attribute") {
      python_function_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int32Attribute>().data());
    } else if (attr_type_name == "pir::Int64Attribute") {
      python_function_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int64Attribute>().data());
    } else if (attr_type_name == "pir::PointerAttribute") {
      python_function_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::PointerAttribute>().data());
    } else {
      PADDLE_THROW(common::errors::Unimplemented("attr type not support [%s] ",
                                                 attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // EmplaceBackOutputs
  VLOG(8) << "ctx->EmplaceBackOutput: ";
  for (size_t i = 0; i < op_->num_results(); ++i) {
    pir::Value out_ptr = op_->result(i);
    auto out_name = op_yaml_info.OutputNames()[i];
    if (!IsInvalid(out_ptr)) {
      PADDLE_ENFORCE(
          paddle::framework::detail::IsOptionalVar(out_name) &&
              !inplace_id_map.empty(),
          common::errors::InvalidArgument(
              "Python operator couldn't find python output for name %s. If you "
              "are using inplace optional inputs & outputs, please check your "
              "InplaceMap and `Outputs` again and make sure %s is wrapped by "
              "`paddle::Optional`",
              out_name,
              out_name));
      VLOG(3) << "Python Operator: BuildContext - inplace optional outputs : "
              << out_name << " is None.";
      python_function_ctx_.EmplaceBackOutput(paddle::Tensor());

      VLOG(8) << "ctx->EmplaceBackOutput : an optional output";
      continue;
    }
    if (out_ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto dense_tensor_out =
          inner_scope->FindVar(value_exec_info_.GetVarName(out_ptr))
              ->GetMutable<phi::DenseTensor>();
      cache_out_ptrs_.push_back(dense_tensor_out);
      std::shared_ptr<phi::DenseTensor> tensor_out(
          dense_tensor_out, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      paddle::Tensor python_out;
      // here only can copy the output tensor into context
      python_out.set_impl(tensor_out);

      python_function_ctx_.EmplaceBackOutput(std::move(python_out));
      VLOG(8) << "ctx->EmplaceBackOutput DenseTensor: "
              << value_exec_info_.GetVarName(out_ptr);
    } else if (out_ptr.type().isa<pir::VectorType>()) {
      PADDLE_THROW(
          common::errors::Unimplemented("Only support DenseTensor output type "
                                        "for now in PythonFunctionInstruction, "
                                        "does not support VectorType."));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Only support DenseTensor and VectorType output types in "
          "PythonFunctionInstruction."));
    }
  }

  auto& op_inputs = OpMetaInfoHelper::GetInputs(*python_op_meta_);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(*python_op_meta_);

  // handle inplace map
  python_function_ctx_.UpdatePlainOutputs(
      op_inputs, op_outputs, op_inplace_map);
}

PythonFunctionInstruction::PythonFunctionInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place),
      cache_out_ptrs_(),
      value_exec_info_(value_exec_info) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  python_op_name_ = op_name;
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  VLOG(6) << "construct python kernel instruction for: " << op_name;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      common::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", op_name));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(op_name),
      /*is_legacy_op=*/false);
  VLOG(6) << "finish process yaml_info_parser";
  const auto& op_meta =
      paddle::framework::detail::GetPythonOperatorInfoByPirName(op_name);
  python_op_meta_ = &op_meta;

  py_func_ptr_ = &(OpMetaInfoHelper::GetPythonOperatorFunction(op_meta));
  py_func_infer_meta_ptr_ =
      &(OpMetaInfoHelper::GetPythonOperatorInferMetaFunction(op_meta));

  BuildPythonFunctionContext(yaml_info_parser);
  VLOG(6) << "finish process python context";
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(op_meta);
  for (auto const& pair : op_inplace_map) {
    pir::Value input_value =
        op->operand_source(yaml_info_parser.InputName2Id().at(pair.first));
    pir::Value output_value =
        op->result(yaml_info_parser.OutputName2Id().at(pair.second));
    if (IsInvalid(output_value) && IsInvalid(input_value)) {
      this->AddInplace(value_exec_info_.GetVarByValue(input_value),
                       value_exec_info_.GetVarByValue(output_value));
    }
  }

  InitInputsOutputsIds(op, value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

void PythonFunctionInstruction::Run() {
  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("PythonFunctionInstruction " + python_op_name_ + " begin");
  }

  for (auto& pair : this->InplaceInfo()) {
    ShareVarBuffer(pair.first, pair.second);
  }

  PADDLE_ENFORCE_NOT_NULL(
      py_func_ptr_,
      common::errors::InvalidArgument("Python function pointer is nullptr."));

  std::vector<Tensor> vec_dense_inputs;
  size_t num = op_->num_operands();
  for (size_t i = 0; i < num; ++i) {
    vec_dense_inputs.push_back(python_function_ctx_.InputAt(i));
  }

  auto out = (*py_func_ptr_)(vec_dense_inputs);
  python_function_ctx_.ValidateAndAssignOutputs(out);
  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("PythonFunctionInstruction " + python_op_name_ + " finish");
  }
}
}  // namespace paddle::framework
