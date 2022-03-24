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

#include "paddle/infrt/dialect/phi/pass/phi_op_convert_pass.h"

#include <glog/logging.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <list>
#include <unordered_set>
#include <vector>

#include "paddle/infrt/common/string.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/dialect/phi/ir/phi_kernels.h"
#include "paddle/infrt/dialect/phi/pass/kernel_op_desc.h"
#include "paddle/infrt/dialect/phi/pass/proto_arg_map_context.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/ops/compat/signatures.h"

namespace {
class PhiOpConvertPass
    : public mlir::PassWrapper<PhiOpConvertPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "PhiOpConvertPass"; }
  void runOnFunction() override;
  PhiOpConvertPass();
  explicit PhiOpConvertPass(const std::vector<infrt::Place> &valid_places)
      : valid_places_(valid_places) {}

  PhiOpConvertPass(const PhiOpConvertPass &other)
      : mlir::PassWrapper<PhiOpConvertPass, mlir::FunctionPass>(*this),
        valid_places_(other.valid_places_) {}

  ::llvm::StringRef getArgument() const override { return "phi-op-convert"; }
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

 private:
  void convertStage();
  void dispatchStage();

  // Force a specified data format for all layout sensitive operations.
  Option<std::string> valid_places_options_{
      *this,
      "valid-targets",
      llvm::cl::desc("Set the valid target, [CPU-FP32-NCHW]")};

  std::vector<infrt::Place> valid_places_;
};
// Implementation of the PhiOpConvertPass.
void PhiOpConvertPass::runOnFunction() {
  convertStage();
  dispatchStage();
}

void PhiOpConvertPass::convertStage() {
  mlir::Block &body = getFunction().front();
  std::vector<mlir::Operation *> worklist;
  for (auto &op : body.without_terminator()) {
    worklist.push_back(&op);
  }
  mlir::OpBuilder builder(&body, body.begin());
  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    if (!op) continue;

    auto op_name = op->getName().getIdentifier().str();

    // only convert op in pd dialect.
    if (op_name.substr(0, 3) != "pd.") continue;
    op_name = op_name.substr(3);
    if (pd_dialect_inputs_info_map_.find(op_name) ==
            pd_dialect_inputs_info_map_.end() ||
        pd_dialect_outputs_info_map_.find(op_name) ==
            pd_dialect_outputs_info_map_.end()) {
      LOG(WARNING) << "No op info found for " << op_name;
      // Todo: print log
      continue;
    }
    auto loc = getFunction().getLoc();
    builder.setInsertionPoint(op);

    if (!::phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_name)) {
      op_name = phi::TransToPhiKernelName(op_name);
      auto kernel_op = builder.create<infrt::KernelOp>(loc,
                                                       op->getResultTypes(),
                                                       op->getOperands(),
                                                       op_name,
                                                       op->getAttrDictionary());
      op->replaceAllUsesWith(kernel_op.getResults());
    } else {
      ::phi::KernelSignature kernel_sign =
          ::phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_name)(
              infrt::ProtoArgumentMappingContext(op));
      // resort input&output according to kernel_sign
      ::llvm::SmallVector<mlir::Value, 4> inputs, ori_output;
      ::llvm::SmallVector<mlir::Type, 4> output_types;
      for (const std::string &str : std::get<0>(kernel_sign.args)) {
        if (pd_dialect_inputs_info_map_.at(op_name).count(str) == 0) {
          LOG(ERROR) << "No input info for Op " << op_name << " and argument "
                     << str;
          return;
        }
        uint8_t index = pd_dialect_inputs_info_map_.at(op_name).at(str);
        inputs.push_back(op->getOperands()[index]);
      }

      for (const std::string &str : std::get<2>(kernel_sign.args)) {
        if (pd_dialect_outputs_info_map_.at(op_name).count(str) == 0) {
          LOG(ERROR) << "No output info for Op " << op_name << " and argument "
                     << str;
          return;
        }
        uint8_t index = pd_dialect_outputs_info_map_.at(op_name).at(str);
        output_types.push_back(op->getResultTypes()[index]);
        ori_output.push_back(op->getResult(index));
      }
      auto kernel_op = builder.create<infrt::KernelOp>(
          loc, output_types, inputs, kernel_sign.name, op->getAttrDictionary());
      for (size_t index = 0; index < ori_output.size(); ++index) {
        ori_output[index].replaceAllUsesWith(kernel_op.getResult(index));
      }
    }
    CHECK(op->use_empty());
    op->erase();
  }
}

void PhiOpConvertPass::dispatchStage() {
  std::vector<infrt::KernelOp> worklist;
  mlir::Block &block = getFunction().front();
  for (auto &op : block) {
    infrt::KernelOp kernel_op = ::llvm::dyn_cast_or_null<infrt::KernelOp>(&op);
    if (nullptr != kernel_op) worklist.push_back(kernel_op);
  }

  mlir::OpBuilder builder(&block, block.begin());
  std::map<infrt::TargetType, mlir::Value> phi_context;
  for (infrt::KernelOp kernel_op : worklist) {
    std::string kernel_name = kernel_op.name().str();
    std::vector<infrt::PhiKernelDesc> candidates =
        GetCandidateKernels(kernel_name, valid_places_);
    if (candidates.empty()) {
      LOG(FATAL) << "No candidate kernels for op:" << kernel_name;
      continue;
    }
    builder.setInsertionPoint(kernel_op);

    // Todo: Implimentation the concrete pass pick strategy
    const infrt::PhiKernelDesc &phi_kernel_desc = candidates.front();

    kernel_name =
        infrt::getPhiTargetPrefix(phi_kernel_desc.kernel_type.target) +
        kernel_name +
        infrt::getPhiPrecisionSuffix(phi_kernel_desc.kernel_type.precision) +
        infrt::getPhiLayoutSuffix(phi_kernel_desc.kernel_type.layout);

    mlir::OperationName operation_name(kernel_name, kernel_op.getContext());
    mlir::OperationState operation_state(kernel_op.getLoc(), operation_name);

    if (phi_context.find(phi_kernel_desc.kernel_type.target) ==
        phi_context.end()) {
      switch (phi_kernel_desc.kernel_type.target) {
        case infrt::TargetType::CPU: {
          auto context_value =
              builder
                  .create<infrt::phi::CreateCPUContextOp>(
                      kernel_op.getLoc(),
                      infrt::phi::ContextType::get(kernel_op.getContext(),
                                                   infrt::TargetType::CPU))
                  .output();
          phi_context[infrt::TargetType::CPU] = context_value;
        } break;
        case infrt::TargetType::GPU:
        case infrt::TargetType::UNK:
        default:
          LOG(FATAL) << "Unsupported TargetType";
          break;
      }
    }
    operation_state.addOperands(
        phi_context.at(phi_kernel_desc.kernel_type.target));

    for (size_t index = 0; index < phi_kernel_desc.input_types.size();
         ++index) {
      mlir::Value input = kernel_op.getOperand(index);
      auto cvt_tensor_type_op = builder.create<infrt::TensorCastOp>(
          kernel_op.getLoc(),
          infrt::DenseTensorType::get(
              kernel_op.getContext(),
              phi_kernel_desc.input_types[index].target,
              phi_kernel_desc.input_types[index].precision,
              phi_kernel_desc.input_types[index].layout),
          input);
      operation_state.addOperands(cvt_tensor_type_op.output());
    }

    for (size_t index = 0; index < phi_kernel_desc.output_types.size();
         ++index) {
      operation_state.addTypes(infrt::DenseTensorType::get(
          kernel_op.getContext(),
          phi_kernel_desc.output_types[index].target,
          phi_kernel_desc.output_types[index].precision,
          phi_kernel_desc.output_types[index].layout));
    }
    operation_state.addAttributes(kernel_op.attrsAttr().getValue());
    mlir::Operation *phi_operation = builder.createOperation(operation_state);
    for (size_t index = 0; index < phi_kernel_desc.output_types.size();
         ++index) {
      mlir::Value input = phi_operation->getResult(index);
      auto cvt_tensor_type_op = builder.create<infrt::TensorCastOp>(
          kernel_op.getLoc(), kernel_op.getResultTypes()[index], input);
      kernel_op.getResult(index).replaceAllUsesWith(
          cvt_tensor_type_op.output());
    }
    kernel_op.erase();
  }
}

PhiOpConvertPass::PhiOpConvertPass() {
  if (!valid_places_options_.hasValue()) {
    valid_places_.emplace_back(infrt::TargetType::CPU,
                               infrt::PrecisionType::FLOAT32,
                               infrt::LayoutType::NCHW);
    return;
  }

  LOG(FATAL) << "To be done for specifying places in command line";
}

void PhiOpConvertPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<infrt::InfrtDialect>();
  registry.insert<infrt::phi::PHIDialect>();
  registry.insert<infrt::phi::PHIDenseTensorDialect>();
  registry.insert<infrt::phi::PHICPUKernelDialect>();
  registry.insert<infrt::phi::PHIGPUKernelDialect>();
}

}  // namespace

mlir::PassRegistration<PhiOpConvertPass> phi_op_convert;

std::unique_ptr<mlir::Pass> infrt::createPhiOpCvtPass(
    std::vector<Place> valid_places) {
  return std::make_unique<PhiOpConvertPass>(valid_places);
}

std::unique_ptr<mlir::Pass> infrt::createPhiOpCvtPass() {
  return std::make_unique<PhiOpConvertPass>();
}
