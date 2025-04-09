// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_legacy_instruction.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"

#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace paddle::framework {

static paddle::framework::Attribute ConvertPirAttribute2FrameworkAttribute(
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
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ConvertPirAttribute2RuntimeAttribute not support [%s] ",
        attr_type_name));
  }
}

OneDNNLegacyKernelInstruction::OneDNNLegacyKernelInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  // Step1: build phi kernel instruction as PhiKernelInstruction
  auto& op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  legacy_op_name_ = op_name;
  VLOG(6) << "construct onednn phi kernel instruction for: " << legacy_op_name_;

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
          "can not find OpYamlInfoInterface from [%s]", legacy_op_name_));
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

  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  auto kernel_result =
      phi::KernelFactory::Instance().SelectKernel(kernel_name, kernel_key);
  phi_kernel_ = new phi::Kernel(kernel_result);
  PADDLE_ENFORCE_EQ(
      phi_kernel_->IsValid(), true, "not found kernel for [%s]", kernel_name);
  VLOG(6) << "finish process select kernel: " << kernel_name;

  const Scope* inner_scope = value_exec_info_->GetScope();

  operator_base_ = BuildOperatorBase(op, *value_exec_info_, yaml_info_parser);

  // build extra attr information
  if (op_attributes.count("extra_args")) {
    std::vector<pir::Attribute> extra_args_attr =
        op->attributes()
            .at("extra_args")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();
    AttributeMap attr_map = operator_base_->RuntimeAttrs();
    for (auto& attr : extra_args_attr) {
      auto attr_name = attr.dyn_cast<pir::StrAttribute>().AsString();
      attr_map[attr_name] = ConvertPirAttribute2FrameworkAttribute(
          op_attributes.at(attr_name), attr_name, yaml_info_parser);
    }
    operator_base_->SetRuntimeAttributeMap(attr_map);
  }

  paddle::framework::VariableValueMap in_map;
  paddle::framework::VariableValueMap out_map;
  auto dev_ctx = phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key.backend()));

  runtime_context_ = std::make_shared<paddle::framework::RuntimeContext>(
      paddle::framework::RuntimeContext(in_map, out_map));
  BuildRuntimeContext(
      op, *value_exec_info, yaml_info_parser, runtime_context_.get());

  kernel_context_ = new paddle::framework::ExecutionContext(
      *operator_base_, *inner_scope, *dev_ctx, *(runtime_context_.get()));

  VLOG(6) << "finish process kernel context";
  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
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
    if (op_attributes.count("data_format")) {
      auto data_layout = op_attributes.at("data_format")
                             .dyn_cast<pir::StrAttribute>()
                             .AsString();
      input_layout_ = common::StringToDataLayout(data_layout);
    } else {
      input_layout_ = phi::OneDNNContext::tls().get_cur_paddle_data_layout();
    }

    std::vector<pir::Attribute> data_format_tensors_attr =
        op->attributes()
            .at("data_format_tensors")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();

    auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();
    std::string fluid_op_name =
        phi::TransToFluidOpName(yaml_info_parser.OpRuntimeInfo().kernel_func);
    for (auto& attr : data_format_tensors_attr) {
      auto input_name = attr.dyn_cast<pir::StrAttribute>().AsString();
      data_format_tensors_.insert(
          op_normalizer.GetLegacyArgName(fluid_op_name, input_name));
    }
  }

  // Step3: Mark is_run_mkldnn_kernel=true
  phi::MetaConfig new_config = infer_meta_context_.GetMetaConfig();
  new_config.is_run_mkldnn_kernel = true;
  infer_meta_context_.SetMetaConfig(new_config);

  // Step4: Handle skip_transform_inputs
  if (op_attributes.count("skip_transform_inputs")) {
    std::vector<pir::Attribute> skip_transform_inputs =
        op->attributes()
            .at("skip_transform_inputs")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();

    auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();
    std::string fluid_op_name =
        phi::TransToFluidOpName(yaml_info_parser.OpRuntimeInfo().kernel_func);

    for (auto& input : skip_transform_inputs) {
      auto input_name = input.dyn_cast<pir::StrAttribute>().AsString();
      skip_format_tensors_.insert(
          op_normalizer.GetLegacyArgName(fluid_op_name, input_name));
    }
  }
}

OneDNNLegacyKernelInstruction::~OneDNNLegacyKernelInstruction() {
  delete kernel_context_;
  delete phi_kernel_;
}

void OneDNNLegacyKernelInstruction::Run() {
  // Step1. TransLayout
  auto inputs = kernel_context_->InNameList();
  for (auto& input_name : inputs) {
    if (skip_format_tensors_.count(*input_name)) {
      continue;
    }
    auto input_vars = kernel_context_->MultiInputVar(*input_name);
    for (auto& var : input_vars) {
      if (var->IsType<phi::DenseTensor>()) {
        auto input = var->GetMutable<phi::DenseTensor>();
        if (input->layout() != phi::DataLayout::ONEDNN) {
          phi::DataLayout from_layout = input->layout();

          //  Handle 'layout_transform' in
          //  ops_onednn_extra.yaml(GetKernelTypeForVar)
          if (data_format_tensors_.count(*input_name) &&
              input_layout_ != phi::DataLayout::kAnyLayout) {
            from_layout = input_layout_;
          }

          auto transed_tensor = const_cast<phi::DenseTensor*>(input);

          if (from_layout == DataLayout::kNHWC ||
              from_layout == DataLayout::kNDHWC) {
            phi::funcs::MatchShapeToLayout(
                transed_tensor, from_layout, phi::DataLayout::ONEDNN);
            // We register only NHWC assuming that model is consistent e.g.
            // either NHWC or NCHW
            phi::OneDNNContext::tls().set_cur_paddle_data_layout(from_layout);
          }

          if (from_layout == DataLayout::kAnyLayout) {
            from_layout =
                phi::OneDNNContext::tls().get_cur_paddle_data_layout();
          }

          dnnl::memory::desc out_mem_desc =
              phi::funcs::make_memory_desc(*input, from_layout);
          transed_tensor->set_mem_desc(out_mem_desc);
        }
      }
    }
  }

  // Step2. InferMeta
  VLOG(6) << "Run op " << legacy_op_name_ << " infer meta.";
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  }

  // Step3. Run kernel
  VLOG(6) << "Run op " << legacy_op_name_ << " kernel.";
  (*(phi_kernel_))((kernel_context_));
}
}  // namespace paddle::framework
