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

#include "paddle/infrt/dialect/phi/pass/phi_op_cvt_pass.h"

#include <glog/logging.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <list>
#include <unordered_set>
#include <vector>

#include "paddle/infrt/dialect/infrt/infrt_dialect.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/pass/kernel_op_desc.h"
#include "paddle/infrt/dialect/phi/pass/proto_arg_map_context.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/ops/compat/signatures.h"
namespace infrt {
// Implementation of the phiOpCvtPass.
void phiOpCvtPass::runOnFunction() {
  convertStage();
  diapatchStage();
}
void phiOpCvtPass::convertStage() {
  mlir::Block &body = getFunction().front();
  std::vector<mlir::Operation *> worklist;
  for (auto &op : body.without_terminator()) {
    worklist.push_back(&op);
  }
  mlir::OpBuilder builder(&body, body.begin());
  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    if (op == nullptr) continue;

    std::string op_name = op->getName().getIdentifier().str();

    // only convert op in pd dialect.
    if (op_name.substr(0, 3) != "pd.") continue;
    op_name = op_name.substr(3);
    if (pd_dialect_inputs_info_map_.find(op_name) ==
            pd_dialect_inputs_info_map_.end() ||
        pd_dialect_outputs_info_map_.find(op_name) ==
            pd_dialect_outputs_info_map_.end()) {
      // Todo: print log
      continue;
    }

    ::phi::KernelSignature kernel_sign =
        ::phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_name)(
            ProtoArgumentMappingContext(op));
    // resort input&output according to kernel_sign
    ::llvm::SmallVector<mlir::Value, 4> inputs, ori_output;
    ::llvm::SmallVector<mlir::Type, 4> output_types;
    for (const std::string &str : std::get<0>(kernel_sign.args)) {
      if (pd_dialect_inputs_info_map_.at(op_name).count(str) == 0) {
        // Todo: print error log
        return;
      }
      uint8_t index = pd_dialect_inputs_info_map_.at(op_name).at(str);
      inputs.push_back(op->getOperands()[index]);
    }

    for (const std::string &str : std::get<2>(kernel_sign.args)) {
      if (pd_dialect_outputs_info_map_.at(op_name).count(str) == 0) {
        // Todo: print error log
        return;
      }
      uint8_t index = pd_dialect_outputs_info_map_.at(op_name).at(str);
      output_types.push_back(op->getResultTypes()[index]);
      ori_output.push_back(op->getResult(index));
    }

    auto loc = getFunction().getLoc();
    builder.setInsertionPoint(op);
    auto kernel_op = builder.create<infrt::KernelOp>(
        loc, output_types, inputs, kernel_sign.name, op->getAttrDictionary());
    for (size_t index = 0; index < ori_output.size(); ++index) {
      ori_output[index].replaceAllUsesWith(kernel_op.getResult(index));
    }
    if (!op->use_empty()) {
      // Todo: print error log
      return;
    }
    op->erase();
  }
}
void phiOpCvtPass::diapatchStage() {
  std::vector<infrt::KernelOp> worklist;
  mlir::Block &block = getFunction().front();
  for (auto &op : block) {
    infrt::KernelOp kernel_op = ::llvm::dyn_cast_or_null<infrt::KernelOp>(&op);
    if (nullptr != kernel_op) worklist.push_back(kernel_op);
  }

  mlir::OpBuilder builder(&block, block.begin());
  std::map<TargetType, mlir::Value> phi_context;
  for (infrt::KernelOp kernel_op : worklist) {
    std::string kernel_name = kernel_op.name().str();
    std::vector<PhiKernelDesc> candidates =
        getCandidateKernels(kernel_name, valid_places_);
    if (candidates.empty()) {
      LOG(FATAL) << "No candidate kernels for op:" << kernel_name;
      continue;
    }
    builder.setInsertionPoint(kernel_op);

    // Todo: Implimentation the concrete pass pick strategy
    const PhiKernelDesc &phi_kernel_desc = candidates.front();

    kernel_name = getPhiTargetPrefix(phi_kernel_desc.kernelType.target) +
                  kernel_name +
                  getPhiLayoutSuffix(phi_kernel_desc.kernelType.layout) +
                  getPhiPrecisionSuffix(phi_kernel_desc.kernelType.precision);

    // mlir::OperationName operation_name = kernel_op.getOperation()->getName();

    mlir::OperationName operation_name(kernel_name, kernel_op.getContext());
    mlir::OperationState operation_state(kernel_op.getLoc(), operation_name);

    if (phi_context.find(phi_kernel_desc.kernelType.target) ==
        phi_context.end()) {
      switch (phi_kernel_desc.kernelType.target) {
        case TargetType::CPU: {
          auto alloctor_value =
              builder
                  .create<infrt::phi::CreateAllocatorOp_cpu>(
                      kernel_op.getLoc(),
                      phi::AllocatorType::get(kernel_op.getContext(),
                                              TargetType::CPU))
                  .output();
          auto context_value =
              builder
                  .create<infrt::phi::CreateContextOp_cpu>(
                      kernel_op.getLoc(),
                      phi::ContextType::get(kernel_op.getContext(),
                                            TargetType::CPU),
                      alloctor_value)
                  .output();
          phi_context[TargetType::CPU] = context_value;
        } break;
        case TargetType::GPU:
        case TargetType::UNK:
        default:
          LOG(FATAL) << "Unsupported TargetType";
          break;
      }
    }
    operation_state.addOperands(
        phi_context.at(phi_kernel_desc.kernelType.target));
    for (size_t index = 0; index < phi_kernel_desc.inputsType.size(); ++index) {
      mlir::Value input = kernel_op.getOperand(index);
      auto cvt_tensor_type_op = builder.create<CvtTensorOp>(
          kernel_op.getLoc(),
          DenseTensorType::get(kernel_op.getContext(),
                               phi_kernel_desc.inputsType[index].target,
                               phi_kernel_desc.inputsType[index].precision,
                               phi_kernel_desc.inputsType[index].layout),
          input);
      operation_state.addOperands(cvt_tensor_type_op.output());
    }
    for (size_t index = 0; index < phi_kernel_desc.outputsType.size();
         ++index) {
      operation_state.addTypes(
          DenseTensorType::get(kernel_op.getContext(),
                               phi_kernel_desc.outputsType[index].target,
                               phi_kernel_desc.outputsType[index].precision,
                               phi_kernel_desc.outputsType[index].layout));
    }
    operation_state.addAttributes(kernel_op.attrsAttr().getValue());
    mlir::Operation *phi_operation = builder.createOperation(operation_state);
    for (size_t index = 0; index < phi_kernel_desc.outputsType.size();
         ++index) {
      mlir::Value input = phi_operation->getResult(index);
      auto cvt_tensor_type_op = builder.create<CvtTensorOp>(
          kernel_op.getLoc(), kernel_op.getResultTypes()[index], input);
      kernel_op.getResult(index).replaceAllUsesWith(
          cvt_tensor_type_op.output());
    }
    kernel_op.erase();
  }
}
}  // namespace infrt
