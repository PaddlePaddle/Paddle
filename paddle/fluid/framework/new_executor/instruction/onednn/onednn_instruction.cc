// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace paddle::framework {

static phi::Attribute ConvertPirAttribute2RuntimeAttribute(
    pir::Attribute attr,
    const std::string& attr_name,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info) {
  auto& attr_type_name = op_yaml_info.AttrTypeName(attr_name);
  if (attr_type_name == "pir::Int32Attribute") {
    return attr.dyn_cast<pir::Int32Attribute>().data();
  } else if (attr_type_name == "pir::FloatAttribute") {
    return attr.dyn_cast<pir::FloatAttribute>().data();
  } else if (attr_type_name == "pir::BoolAttribute") {
    return attr.dyn_cast<pir::BoolAttribute>().data();
  } else if (attr_type_name == "pir::StrAttribute") {
    return attr.dyn_cast<pir::StrAttribute>().AsString();
  } else if (attr_type_name == "pir::ArrayAttribute<pir::Int32Attribute>") {
    auto array_list = attr.dyn_cast<pir::ArrayAttribute>().AsVector();
    std::vector<int32_t> vec_res;
    if (array_list.size() > 0) {
      PADDLE_ENFORCE_EQ(array_list[0].isa<pir::Int32Attribute>(),
                        true,
                        common::errors::Unimplemented(
                            "the 0th elementwise MUST be pir::Int32Attribute"));
      for (size_t i = 0; i < array_list.size(); ++i) {
        vec_res.push_back(array_list[i].dyn_cast<pir::Int32Attribute>().data());
      }
    }
    return vec_res;
  } else if (attr_type_name == "pir::ArrayAttribute<pir::Int64Attribute>") {
    auto array_list = attr.dyn_cast<pir::ArrayAttribute>().AsVector();
    std::vector<int64_t> vec_res;
    if (array_list.size() > 0) {
      PADDLE_ENFORCE_EQ(array_list[0].isa<pir::Int64Attribute>(),
                        true,
                        common::errors::Unimplemented(
                            "the 0th elementwise MUST be pir::Int64Attribute"));
      for (size_t i = 0; i < array_list.size(); ++i) {
        vec_res.push_back(array_list[i].dyn_cast<pir::Int64Attribute>().data());
      }
    }
    return vec_res;
  } else if (attr_type_name == "pir::ArrayAttribute<pir::FloatAttribute>") {
    auto array_list = attr.dyn_cast<pir::ArrayAttribute>().AsVector();
    std::vector<float> vec_res;
    if (array_list.size() > 0) {
      if (array_list[0].isa<pir::FloatAttribute>()) {
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::FloatAttribute>().data());
        }

      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "ConvertPirAttribute2RuntimeAttribute not support [%s] ",
            attr_type_name));
      }
    }
    return vec_res;
  } else if (attr_type_name == "paddle::dialect::IntArrayAttribute") {
    std::vector<int64_t> int_array =
        attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
    return int_array;
  } else if (attr_type_name == "paddle::dialect::DataTypeAttribute") {
    phi::DataType dtype =
        attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
    return dtype;
  } else if (attr_type_name == "paddle::dialect::ScalarAttribute") {
    return attr.dyn_cast<dialect::ScalarAttribute>().data();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ConvertPirAttribute2RuntimeAttribute not support [%s] ",
        attr_type_name));
  }
}

void TensorNameMap(pir::Operation* op,
                   const ValueExecutionInfo& value_exec_info,
                   const paddle::dialect::OpYamlInfoParser& op_yaml_info,
                   std::map<std::string, std::vector<std::string>>&
                       inputs_tensor_name_map,  // NOLINT
                   std::map<std::string, std::vector<std::string>>&
                       outputs_tensor_name_map) {  // NOLINT
  static int unique_id = 0;
  const Scope* inner_scope = value_exec_info.GetScope();
  VLOG(6) << "TensorNameMap in scope[" << inner_scope << "]";

  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(true);

  auto& name2id = op_yaml_info.InputName2Id();

  std::string fluid_op_name =
      phi::TransToFluidOpName(op_yaml_info.OpRuntimeInfo().kernel_func);

  auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();

  for (auto& name : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(name),
        true,
        common::errors::NotFound("param [%s] MUST in name2id map", name));
    auto index = name2id.at(name);
    pir::Value ptr = op->operand_source(index);

    if (!IsInvalid(ptr)) {
      continue;
    }

    auto legacy_arg_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);
    auto in_var_name = value_exec_info.GetVarName(ptr);
    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));

    auto type = ptr.type();
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>() ||
        type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
      inputs_tensor_name_map[legacy_arg_name] = {in_var_name +
                                                 std::to_string(unique_id++)};
    } else if (type.isa<pir::VectorType>()) {
      auto var = inner_scope->FindVar(in_var_name);
      auto var_ref = var->Get<VariableRefArray>();
      std::vector<std::string> vec_tmp;
      vec_tmp.reserve(var_ref.size());
      for (size_t k = 0; k < var_ref.size(); ++k) {
        vec_tmp.push_back(value_exec_info.GetVarName(var_ref[k]) +
                          std::to_string(unique_id++));
      }
      inputs_tensor_name_map[legacy_arg_name] = vec_tmp;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType  and "
          "pir::vector type"));
    }
  }

  auto& output_name_list = op_yaml_info.OutputNames();
  for (size_t i = 0; i < output_name_list.size(); ++i) {
    auto name = output_name_list[i];
    pir::Value ptr = op->result(i);
    auto legacy_arg_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);

    if (!IsInvalid(ptr)) {
      continue;
    }

    auto out_var_name = value_exec_info.GetVarName(ptr);

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(out_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", out_var_name));

    auto type = ptr.type();
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>() ||
        type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
      outputs_tensor_name_map[legacy_arg_name] = {out_var_name +
                                                  std::to_string(unique_id++)};
    } else if (type.isa<pir::VectorType>()) {
      auto var = inner_scope->FindVar(out_var_name);
      auto var_ref = var->Get<VariableRefArray>();
      std::vector<std::string> vec_tmp;
      vec_tmp.reserve(var_ref.size());
      for (size_t k = 0; k < var_ref.size(); ++k) {
        vec_tmp.push_back(value_exec_info.GetVarName(var_ref[k]) +
                          std::to_string(unique_id++));
      }
      outputs_tensor_name_map[legacy_arg_name] = vec_tmp;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType  and "
          "pir::vector type"));
    }
  }
}

OneDNNPhiKernelInstruction::OneDNNPhiKernelInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  // Step1: build phi kernel instruction as PhiKernelInstruction
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  phi_op_name_ = op_name;
  VLOG(6) << "construct phi kernel instruction for: " << phi_op_name_;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  infer_meta_interface_ =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  VLOG(6) << "finish process infer_meta_interface_";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      common::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", phi_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));
  VLOG(6) << "finish process yaml_info_parser";

  if (infer_meta_interface_) {
    BuildPhiContext<
        phi::InferMetaContext,
        phi::MetaTensor,
        phi::MetaTensor,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        false>(op, *value_exec_info_, yaml_info_parser, &infer_meta_context_);
  }
  VLOG(6) << "finish process infer meta context";

  auto kernel_name_ =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_key_ = op_attributes.at("kernel_key")
                         .dyn_cast<paddle::dialect::KernelAttribute>()
                         .data();

  phi_kernel_ = new phi::Kernel(
      phi::KernelFactory::Instance().SelectKernel(kernel_name_, kernel_key_));
  PADDLE_ENFORCE_EQ(
      phi_kernel_->IsValid(), true, "not found kernel for [%s]", kernel_name_);
  VLOG(6) << "finish process select kernel";

  BuildPhiContext<phi::KernelContext,
                  const phi::TensorBase*,
                  phi::TensorBase*,
                  paddle::small_vector<const phi::TensorBase*>,
                  paddle::small_vector<phi::TensorBase*>,
                  true>(
      op, *value_exec_info_, yaml_info_parser, &kernel_context_);

  kernel_context_.SetDeviceContext(phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key_.backend())));
  VLOG(6) << "finish process kernel context";

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key_.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  InitInputsOutputsIds(op, *value_exec_info);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";

  // Step2: build layout_transform information
  if (op_attributes.count("data_format_tensors")) {
    VLOG(6) << "data_format_tensors is not empty";
    if (op_attributes.count("data_format")) {
      auto data_layout = op_attributes.at("data_format")
                             .dyn_cast<pir::StrAttribute>()
                             .AsString();
      input_layout_ = common::StringToDataLayout(data_layout);
    } else {
      input_layout_ = phi::OneDNNContext::tls().get_cur_paddle_data_layout();
    }
    VLOG(6) << "input_layout = " << input_layout_;

    std::vector<pir::Attribute> data_format_tensors_attr =
        op->attributes()
            .at("data_format_tensors")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();

    for (auto& attr : data_format_tensors_attr) {
      auto data_format_tensor = attr.dyn_cast<pir::StrAttribute>().AsString();
      auto pair = kernel_context_.InputRangeAt(
          yaml_info_parser.InputName2Id().at(data_format_tensor));
      VLOG(6) << "data_format_tensor = " << data_format_tensor;
      for (int i = pair.first; i < pair.second; ++i) {
        data_format_tensors_.insert(i);
        VLOG(6) << data_format_tensor << " index = " << i;
      }
    }
  }

  // Step3: build extra attr information
  if (op_attributes.count("extra_args")) {
    std::vector<pir::Attribute> extra_args_attr =
        op->attributes()
            .at("extra_args")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();
    auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();
    std::string fluid_op_name =
        phi::TransToFluidOpName(yaml_info_parser.OpRuntimeInfo().kernel_func);

    for (auto& attr : extra_args_attr) {
      auto attr_name = attr.dyn_cast<pir::StrAttribute>().AsString();
      extra_attr_[attr_name] = ConvertPirAttribute2RuntimeAttribute(
          op_attributes.at(attr_name), attr_name, yaml_info_parser);
      auto legacy_attr_name =
          op_normalizer.GetLegacyAttrName(fluid_op_name, attr_name);
      if (legacy_attr_name != attr_name) {
        extra_attr_[legacy_attr_name] = extra_attr_[attr_name];
      }
    }
    auto attr_name_list = yaml_info_parser.AttrParams(true);
    for (auto& attr : attr_name_list) {
      auto attr_name = attr;
      if (!op_attributes.count(attr_name)) {
        // In PIR, IntArray attr will be input, but not attr.
        continue;
      }
      ctx_attr_[attr_name] = ConvertPirAttribute2RuntimeAttribute(
          op_attributes.at(attr_name), attr_name, yaml_info_parser);
      auto legacy_attr_name =
          op_normalizer.GetLegacyAttrName(fluid_op_name, attr_name);
      if (legacy_attr_name != attr_name) {
        ctx_attr_[legacy_attr_name] = ctx_attr_[attr_name];
      }
    }
  }
  TensorNameMap(op, *value_exec_info_, yaml_info_parser, inputs_, outputs_);

  // Step4: Mark is_run_mkldnn_kernel=true
  phi::MetaConfig new_config = infer_meta_context_.GetMetaConfig();
  new_config.is_run_mkldnn_kernel = true;
  infer_meta_context_.SetMetaConfig(new_config);

  // Step5: Handle skip_transform_inputs
  if (op_attributes.count("skip_transform_inputs")) {
    std::vector<pir::Attribute> skip_transform_inputs =
        op->attributes()
            .at("skip_transform_inputs")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();

    for (auto& input : skip_transform_inputs) {
      auto input_name = input.dyn_cast<pir::StrAttribute>().AsString();
      auto pair = kernel_context_.InputRangeAt(
          yaml_info_parser.InputName2Id().at(input_name));
      VLOG(6) << "skip_transform_input = " << input_name;
      for (int i = pair.first; i < pair.second; ++i) {
        skip_format_tensors_.insert(i);
        VLOG(6) << input_name << " index = " << i;
      }
    }
  }
}

OneDNNPhiKernelInstruction::~OneDNNPhiKernelInstruction() {
  delete phi_kernel_;
}

void OneDNNPhiKernelInstruction::Run() {
  std::vector<std::shared_ptr<phi::DenseTensor>> tmp_holders;
  auto tmp_kernel_context = kernel_context_;
  auto tmp_infer_meta_context_ = infer_meta_context_;
  // Step1. TransLayout
  auto inputs = tmp_kernel_context.InputsBetween<phi::DenseTensor>(
      size_t(0), tmp_kernel_context.InputsSize());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (input == nullptr) {
      continue;
    }
    if (!input->initialized()) {
      continue;
    }
    if (skip_format_tensors_.count(i)) {
      continue;
    }
    VLOG(6) << "input[" << i << "].layout() = " << input->layout()
            << ", shape = " << input->dims();
    if (input->layout() != phi::DataLayout::ONEDNN) {
      phi::DataLayout from_layout = input->layout();
      tmp_holders.emplace_back(std::make_shared<phi::DenseTensor>(*input));
      auto transed_tensor = tmp_holders.back().get();

      std::set<std::string> elementwise_kernels = {"onednn_op.add",
                                                   "onednn_op.subtract",
                                                   "onednn_op.multiply",
                                                   "onednn_op.divide"};

      if (elementwise_kernels.count(phi_op_name_)) {
        if (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
                phi::DataLayout::kNHWC &&
            !(kernel_key_.dtype() == phi::DataType::COMPLEX64 ||
              kernel_key_.dtype() == phi::DataType::COMPLEX128)) {
          from_layout = phi::DataLayout::kNHWC;
          phi::funcs::MatchShapeToLayout(
              transed_tensor, from_layout, phi::DataLayout::ONEDNN);
        }
      } else {
        //  Handle 'layout_transform' in
        //  ops_onednn_extra.yaml(GetKernelTypeForVar)
        if (data_format_tensors_.count(i) &&
            input_layout_ != phi::DataLayout::kAnyLayout) {
          from_layout = input_layout_;
        }
        VLOG(6) << "from_layout = " << from_layout;

        if (from_layout == DataLayout::kNHWC ||
            from_layout == DataLayout::kNDHWC) {
          phi::funcs::MatchShapeToLayout(
              transed_tensor, from_layout, phi::DataLayout::ONEDNN);
          // We register only NHWC assuming that model is consistent e.g. either
          // NHWC or NCHW
          phi::OneDNNContext::tls().set_cur_paddle_data_layout(from_layout);
        }

        if (from_layout == DataLayout::kAnyLayout) {
          from_layout = phi::OneDNNContext::tls().get_cur_paddle_data_layout();
        }
      }

      dnnl::memory::desc out_mem_desc =
          phi::funcs::make_memory_desc(*transed_tensor, from_layout);
      transed_tensor->set_mem_desc(out_mem_desc);
      tmp_kernel_context.UpdataInput(i, transed_tensor);
      auto meta_tensor = phi::MetaTensor(transed_tensor);
      auto input_meta_tensor = phi::MetaTensor(input);
      if (tmp_infer_meta_context_.InputsSize() > i &&
          tmp_infer_meta_context_.InputAt(i).is_same_tensor(
              input_meta_tensor)) {
        tmp_infer_meta_context_.UpdataInput(i, meta_tensor);
      } else {
        for (size_t j = 0; j < tmp_infer_meta_context_.InputsSize(); ++j) {
          if (tmp_infer_meta_context_.InputAt(j).is_same_tensor(
                  input_meta_tensor)) {
            tmp_infer_meta_context_.UpdataInput(j, meta_tensor);
            break;
          }
        }
      }
    }
  }

  // Step2. Append extra information into ctx
  // SetDnnAttrIntoDeviceContext
  // SetInputsName SetOutputsName
  auto one_dnn_ctx = const_cast<phi::OneDNNContext*>(
      &tmp_kernel_context.GetDeviceContext<phi::OneDNNContext>());
  for (auto& attr : extra_attr_) {
    one_dnn_ctx->SetDnnAttr(attr.first, attr.second);
  }
  for (auto& attr : ctx_attr_) {
    one_dnn_ctx->SetDnnAttr(attr.first, attr.second);
  }
  one_dnn_ctx->SetInputsName(inputs_);
  one_dnn_ctx->SetOutputsName(outputs_);

  // Step3. InferMeta
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(tmp_infer_meta_context_));
  }

  // Step4. Run kernel
  VLOG(6) << "Run op " << phi_op_name_ << " infer meta.";
  (*(phi_kernel_))(&(tmp_kernel_context));
  VLOG(6) << "Run op " << phi_op_name_ << " kernel.";

  // Step5. ClearDnnAttr
  one_dnn_ctx->ClearDnnAttr();
}

}  // namespace paddle::framework
