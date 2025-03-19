// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/llvm/llvm_optimizer.h"

#include <glog/logging.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Vectorize.h>

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "llvm/Support/CodeGen.h"

namespace cinn::backends {

namespace {
template <typename PassManagerT>
class CustomPassManager : public PassManagerT {
 public:
  template <typename... Ts>
  explicit CustomPassManager(bool print_passes, Ts &&...ts)
      : PassManagerT(std::forward<Ts>(ts)...), print_passes_(print_passes) {}

  void add(llvm::Pass *pass) override {
    if (print_passes_) {
      if (is_function_pass_manager_) {
        VLOG(4) << "llvm run function pass[" << std::string(pass->getPassName())
                << "]";
      }

      if (is_module_pass_manager_) {
        VLOG(4) << "llvm run module pass[" << std::string(pass->getPassName())
                << "]";
      }
    }
    // static bool add_pass = true;
    // if (add_pass) {
    //  PassManagerT::add(pass);
    //}

    // if (std::string(pass->getPassName()) == "Loop Vectorization") {
    //  return;
    //}
    PassManagerT::add(pass);
  }

  void run(llvm::Function &f) {  // NOLINT
    if (is_function_pass_manager_) {
      PassManagerT::run(f);
    }
  }

  void run(llvm::Module &m) {  // NOLINT
    if (is_module_pass_manager_) {
      PassManagerT::run(m);
    }
  }

 private:
  static constexpr bool is_function_pass_manager_ =
      std::is_same<llvm::legacy::FunctionPassManager, PassManagerT>::value;
  static constexpr bool is_module_pass_manager_ =
      std::is_same<llvm::legacy::PassManager, PassManagerT>::value;
  bool print_passes_;
};

using CustomFunctionPassManager =
    CustomPassManager<llvm::legacy::FunctionPassManager>;
using CustomModulePassManager = CustomPassManager<llvm::legacy::PassManager>;
}  // namespace

LLVMModuleOptimizer::LLVMModuleOptimizer(llvm::TargetMachine *machine,
                                         int opt_level,
                                         llvm::FastMathFlags fast_math_flags,
                                         bool print_passes)
    : opt_level_(opt_level), print_passes_(print_passes), machine_(machine) {}

void LLVMModuleOptimizer::operator()(llvm::Module *m) {
  auto machine = std::move(llvm::cantFail(
      llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost())
          .createTargetMachine()));
  auto fpm = std::make_unique<CustomFunctionPassManager>(print_passes_, m);
  // fpm->add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
  // fpm->add(llvm::createInstructionCombiningPass());
  // fpm->add(llvm::createReassociatePass());
  // fpm->add(llvm::createGVNPass());
  // fpm->add(llvm::createCFGSimplificationPass());
  // fpm->add(llvm::createSROAPass());
  // fpm->add(llvm::createEarlyCSEPass());
  // fpm->add(llvm::createLowerExpectIntrinsicPass());
  // fpm->add(llvm::createCallSiteSplittingPass());
  // fpm->add(llvm::createLoopVectorizePass());
  // fpm->add(llvm::createSLPVectorizerPass());
  // fpm->add(llvm::createLoadStoreVectorizerPass());
  // fpm->add(llvm::createLoopUnrollPass());

  auto mpm = std::make_unique<CustomModulePassManager>(print_passes_);
  // mpm->add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
  // LOG(INFO) << "llvm run pass: target machine: name[" <<
  // machine_->getTarget().getName() << "]"; LOG(INFO) << "llvm run pass: target
  // machine: cpu[" << machine_->getTargetCPU().str() << "]";
  fpm->add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));
  mpm->add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));
  auto builder = std::make_unique<llvm::PassManagerBuilder>();
  builder->OptLevel = opt_level_;
  builder->Inliner = llvm::createFunctionInliningPass();
  builder->LoopVectorize = true;
  builder->SLPVectorize = true;
#if LLVM_VERSION_MAJOR >= 11
  machine->adjustPassManager(*builder);
#endif
  builder->populateFunctionPassManager(*fpm);
  builder->populateModulePassManager(*mpm);

  fpm->doInitialization();
  std::for_each(m->begin(), m->end(), [&fpm](auto &fn) { fpm->run(fn); });
  fpm->doFinalization();

  mpm->run(*m);
}

}  // namespace cinn::backends
