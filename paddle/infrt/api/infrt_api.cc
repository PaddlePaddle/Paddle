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

#include "paddle/infrt/api/infrt_api.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/DynamicLibrary.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Parser.h>

#include <unordered_map>
#include <vector>

#include "mlir/Pass/PassManager.h"
#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse_pass.h"
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/dialect/phi/pass/phi_op_convert_pass.h"
#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"
#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"
#include "paddle/infrt/host_context/op_executable.h"
#include "paddle/infrt/host_context/paddle_mlir.h"
#include "paddle/infrt/host_context/value.h"
#include "paddle/infrt/kernel/basic_kernels.h"
#include "paddle/infrt/kernel/control_flow_kernels.h"
#include "paddle/infrt/kernel/phi/dense_tensor_kernels.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/registry.h"
#include "paddle/infrt/kernel/tensor_kernels.h"
#include "paddle/infrt/kernel/tensor_shape_kernels.h"
#include "paddle/infrt/kernel/test_kernels.h"
#include "paddle/infrt/tensor/tensor_map.h"

using namespace infrt::host_context;  // NOLINT
using namespace infrt::tensor;        // NOLINT
using namespace infrt::tensor;        // NOLINT

namespace infrt {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

struct MlirToRuntimeTranslator::Impl {
  mlir::ModuleOp module;
  // The runtime for a function call.
  CoreRuntimeBuilder* runtime{};

  // The current working op, the translator process the ops one by one, each
  // time it updates `cur_op` here to current op
  // working on.
  OpExecutableBuilder* cur_op{};

  // record the current function name.
  std::string cur_func_name;

  // Name to function definitions.
  std::unordered_map<std::string, mlir::FuncOp> func_defs;

  // Map from an operation to its results.
  std::unordered_map<const mlir::Operation*, std::vector<ValueRef>> op_results;
  llvm::DenseMap<mlir::Value, ValueRef> value_map;
};

/**
 * Execute the mlir program in predict mode.
 */
class PredictExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder core_runtime;

  PredictExecutor(mlir::ModuleOp module,
                  KernelRegistry* registry,
                  ::infrt::phi::DenseTensorMap&& map)
      : MlirToRuntimeTranslator(module, &core_runtime),
        core_runtime(registry),
        registry_(registry) {
    CHECK(registry_);
    Init(std::move(map));
  }

  void Run() {
    auto arguments = llvm::makeArrayRef(arguments_);
    auto results = llvm::makeMutableArrayRef(results_.begin(), results_.size());
    function_executable_->Execute(arguments, results);
  }

  int GetInputNum() { return inputs_.size(); }

  ::phi::DenseTensor* GetInput(int i) { return inputs_[i]; }

  int GetOutputNum() { return outputs_.size(); }

  ::phi::DenseTensor* GetOutput(int i) { return outputs_[i]; }

 private:
  void Init(::infrt::phi::DenseTensorMap&& map) {
    EmitFunctions();
    llvm::Optional<mlir::FuncOp> predict_func_ = llvm::None;
    for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
      if (func_op.getName().str() != "main_graph") continue;
      predict_func_ = func_op;
      break;
    }
    if (!predict_func_) {
      std::cout << "ERROR: init failed, no predict function found in mlir."
                << std::endl;
      return;
    }
    auto& predict_func = predict_func_.getValue();
    function_executable_ =
        new MlirFunctionExecutable(predict_func, registry_, impl_->func_defs);

    // process parammeters
    VLOG(3) << "Arguments num of predict func: "
            << predict_func.getNumArguments();
    for (size_t i = 0; i < predict_func.getNumArguments(); ++i) {
      auto arg = predict_func.getArgument(i);
      auto type = arg.getType();
      // this param is TensorMap
      if (type.isa<::infrt::phi::DenseTensorMapType>()) {
        auto* value = new host_context::Value(std::move(map));
        arguments_.push_back(value);
        AddValue(predict_func.getArgument(i), value);
      } else if (type.isa<::infrt::DenseTensorType>()) {
        // this param is an input Tensor
        auto dht = ::phi::DenseTensor();
        auto* value = new host_context::Value(std::move(dht));
        arguments_.push_back(value);
        inputs_.push_back(&(value->get<::phi::DenseTensor>()));
      } else {
        llvm_unreachable("The input type has not been supported by predictor.");
      }
    }

    // process results
    auto& last_op = predict_func.front().back();
    if (last_op.getName().getStringRef() == "infrt.return") {
      for (size_t i = 0; i < last_op.getNumOperands(); ++i) {
        auto operand = last_op.getOperand(i);
        if (operand.getType().isa<::infrt::DenseTensorType>()) {
          auto r = impl_->value_map.try_emplace(
              operand, ValueRef(new host_context::Value(::phi::DenseTensor())));
          CHECK(r.second) << "Duplicate add mlir value ["
                          << DumpToString(operand) << "]";
          auto* value = r.first->second.get();
          results_.push_back(ValueRef(value));
          outputs_.push_back(&(value->get<::phi::DenseTensor>()));
        } else {
          llvm_unreachable("infrt.return only supports DenseTensor now.");
        }
      }
    }
  }

 protected:
  std::unordered_map<std::string, mlir::FuncOp> func_def_table;

  void EmitFunction(mlir::FuncOp op) override {
    CHECK(!impl_->func_defs.count(op.getName().str()))
        << "Duplicate function defition found for function ["
        << op.getName().str();
    impl_->func_defs.emplace(op.getName().str(), op);
  }

 private:
  KernelRegistry* registry_{};
  MlirFunctionExecutable* function_executable_;
  llvm::SmallVector<::phi::DenseTensor*, 1> inputs_;
  llvm::SmallVector<host_context::Value*, 2> arguments_;
  llvm::SmallVector<::phi::DenseTensor*, 1> outputs_;
  llvm::SmallVector<ValueRef, 1> results_;
};

std::unique_ptr<InfRtPredictor> CreateInfRtPredictor(
    const InfRtConfig& config) {
  auto x = std::make_unique<InfRtPredictor>();
  x->Init(config);
  return x;
}

struct InfRtPredictor::Impl {
  std::unique_ptr<PredictExecutor> executor;
  MLIRModelGenImpl module_gen_;
};

InfRtPredictor::InfRtPredictor() : impl_(new Impl) {}
InfRtPredictor::~InfRtPredictor() {}

void InfRtPredictor::Run() { impl_->executor->Run(); }

int InfRtPredictor::Init(const InfRtConfig& config) {
  mlir::MLIRContext* context = ::infrt::Global::getMLIRContext();

  KernelRegistry* registry = new KernelRegistry();

  kernel::RegisterBasicKernels(registry);
  kernel::RegisterTestKernels(registry);
  kernel::RegisterTensorShapeKernels(registry);
  kernel::RegisterTensorKernels(registry);
  kernel::RegisterControlFlowKernels(registry);
#ifdef INFRT_WITH_PHI
  kernel::RegisterPhiKernels(registry);
  kernel::RegisterInferShapeLaunchers(registry);
#if defined(INFRT_WITH_GPU) && defined(INFRT_WITH_TRT)
  kernel::RegisterTrtKernels(registry);
#endif  // INFRT_WITH_GPU && INFRT_WITH_TRT
#endif

  auto module_op = impl_->module_gen_.ImportPaddleModel(config.model_dir(),
                                                        config.param_dir());

  context->loadAllAvailableDialects();
  ::mlir::PassManager pm(context);
  ::mlir::OpPassManager& phi_pass_manager = pm.nest<::mlir::FuncOp>();
  std::vector<::infrt::Place> valid_places = {{::infrt::TargetType::CPU,
                                               ::infrt::PrecisionType::FLOAT32,
                                               ::infrt::LayoutType::NCHW}};
  phi_pass_manager.addPass(::infrt::createPhiOpCvtPass(valid_places));
  phi_pass_manager.addPass(::infrt::createInfrtOpFusePass());
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "\npass failed!\n" << std::endl;
    return 4;
  }
#ifndef NDEBUG
  module_op.dump();
#endif  // NDEBUG

  // load extra shared library
  for (const std::string& lib_path : config.shared_libs()) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      return 1;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func = reinterpret_cast<void (*)(KernelRegistry*)>(reg_sym);
      reg_func(registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path
                   << "\". Skip.\n";
    }
  }

  // Load params
  auto tensor_map = ::infrt::kernel::phi::LoadCombinedParameters(
      config.model_dir(), config.param_dir());

  // Create PredictExecutor
  impl_->executor.reset(
      new PredictExecutor(module_op, registry, std::move(tensor_map)));
  return 0;
}

int InfRtPredictor::GetInputNum() { return impl_->executor->GetInputNum(); }

::phi::DenseTensor* InfRtPredictor::GetInput(int i) {
  return impl_->executor->GetInput(i);
}

int InfRtPredictor::GetOutputNum() { return impl_->executor->GetOutputNum(); }

::phi::DenseTensor* InfRtPredictor::GetOutput(int i) {
  return impl_->executor->GetOutput(i);
}

}  // namespace infrt
