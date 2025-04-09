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

#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_mixed_instruction.h"

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
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace paddle::framework {

OneDNNMixedPhiKernelInstruction::OneDNNMixedPhiKernelInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : OneDNNPhiKernelInstruction(id, place, op, value_exec_info) {
  auto op_attributes = op->attributes();
  kernel_name_ =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  kernel_key_ = op_attributes.at("kernel_key")
                    .dyn_cast<paddle::dialect::KernelAttribute>()
                    .data();
}

void OneDNNMixedPhiKernelInstruction::Run() {
  std::vector<std::shared_ptr<phi::DenseTensor>> tmp_holders;
  // Step1. Mixed Dynamic Choose Kernel
  if (!has_choose_kernel_) {
    has_choose_kernel_ = true;
    use_onednn_kernel_ =
        phi_kernel_->check_if_onednn_kernel_support_(&kernel_context_);
    if (!use_onednn_kernel_) {
      auto kernel_result =
          phi::KernelFactory::Instance().SelectKernelOrThrowError(kernel_name_,
                                                                  kernel_key_);
      delete phi_kernel_;
      phi_kernel_ = new phi::Kernel(kernel_result.kernel);
    }
  }

  // Step2. Run Kernel
  if (use_onednn_kernel_) {
    OneDNNPhiKernelInstruction::Run();
  } else {
    auto tmp_kernel_context = kernel_context_;
    auto tmp_infer_meta_context_ = infer_meta_context_;
    // TransLayout first
    auto inputs = tmp_kernel_context.InputsBetween<phi::DenseTensor>(
        size_t(0), tmp_kernel_context.InputsSize());

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto input = inputs[i];
      if (input->layout() == phi::DataLayout::ONEDNN) {
        DataLayout tmp_layout =
            phi::OneDNNContext::tls().get_cur_paddle_data_layout();

        // NOTE(zhiqiu): to handle the special case in ApplyDataTransform() in
        // data_transfer.cc
        if (!input->IsInitialized() && tmp_layout == DataLayout::NHWC) {
          tmp_holders.emplace_back(std::make_shared<phi::DenseTensor>(*input));
          auto transed_tensor = tmp_holders.back().get();
          transed_tensor->set_layout(tmp_layout);
          phi::funcs::MatchShapeToLayout(
              transed_tensor, phi::DataLayout::ONEDNN, tmp_layout);
          dnnl::memory::desc out_mem_desc =
              phi::funcs::make_memory_desc(*transed_tensor, tmp_layout);
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
        } else {
          tmp_holders.emplace_back(std::make_shared<phi::DenseTensor>());
          auto transed_tensor = tmp_holders.back().get();
          transed_tensor->set_meta(input->meta());
          phi::funcs::TransDataLayoutFromOneDNN(phi::DataLayout::ONEDNN,
                                                tmp_layout,
                                                *input,
                                                transed_tensor,
                                                phi::CPUPlace());
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
    }

    VLOG(6) << "Begin run op " << phi_op_name_ << " infer meta.";
    if (infer_meta_interface_) {
      infer_meta_interface_->infer_meta_(&(tmp_infer_meta_context_));
    }
    VLOG(6) << "End run op " << phi_op_name_ << " infer meta.";
    VLOG(6) << "Begin run op " << phi_op_name_ << " kernel.";
    (*(phi_kernel_))(&(tmp_kernel_context));
    VLOG(6) << "End run op " << phi_op_name_ << " kernel.";
  }
}

}  // namespace paddle::framework
