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

#include "paddle/cinn/backends/llvm/simple_jit.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <string>
#include <utility>

#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

void SimpleJIT::AddModule(std::unique_ptr<llvm::Module> module, bool optimize) {
  /*
  for (auto &fn : module->functions()) {
    LOG(INFO) << "fn:\n" << DumpToString(fn);
  }
   */
  PADDLE_ENFORCE_EQ(
      !llvm::verifyModule(*module, &llvm::errs()),
      true,
      ::common::errors::InvalidArgument(
          "Transformation resulted in an invalid module\n\nmodule:\n"));

  bool debug = false;
  if (optimize) {
    llvm::PassBuilder pass_builder;
    llvm::LoopAnalysisManager loop_analysis_manager(debug);
    llvm::FunctionAnalysisManager function_analysis_manager(debug);
    llvm::CGSCCAnalysisManager cgscc_analysis_manager(debug);
    llvm::ModuleAnalysisManager module_analysis_manager(debug);

    pass_builder.registerModuleAnalyses(module_analysis_manager);
    pass_builder.registerCGSCCAnalyses(cgscc_analysis_manager);
    pass_builder.registerFunctionAnalyses(function_analysis_manager);
    pass_builder.registerLoopAnalyses(loop_analysis_manager);
    pass_builder.crossRegisterProxies(loop_analysis_manager,
                                      function_analysis_manager,
                                      cgscc_analysis_manager,
                                      module_analysis_manager);

    llvm::ModulePassManager module_pass_manager =
        pass_builder.buildPerModuleDefaultPipeline(
            llvm::PassBuilder::OptimizationLevel::O3);
    module_pass_manager.run(*module, module_analysis_manager);
  }

  VLOG(3) << "jit target: " << jit_->getDataLayout().getStringRepresentation();
  VLOG(3) << "module target: "
          << module->getDataLayout().getStringRepresentation();

  llvm::orc::ThreadSafeModule tsm(std::move(module), context_);
  llvm::cantFail(jit_->addIRModule(std::move(tsm)));

  if (debug) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    jit_->getExecutionSession().dump(os);
    os.flush();
    VLOG(3) << "compiled jit:\n" << buffer;
  }
}

SimpleJIT::SimpleJIT() : context_(std::make_unique<llvm::LLVMContext>()) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  jit_ = llvm::cantFail(llvm::orc::LLJITBuilder().create());
  PADDLE_ENFORCE_NOT_NULL(
      jit_, ::common::errors::InvalidArgument("JIT creation failed."));

  auto proc_symbols_generator = llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit_->getDataLayout().getGlobalPrefix()));
  jit_->getMainJITDylib().addGenerator(std::move(proc_symbols_generator));

  llvm::orc::MangleAndInterner mangle(jit_->getExecutionSession(),
                                      jit_->getDataLayout());

  for (auto &item : GlobalSymbolRegistry::Global().All()) {
    VLOG(2) << "Insert [" << item.first << "] to SimpleJIT";
    llvm::cantFail(jit_->define(llvm::orc::absoluteSymbols(
        {{mangle(item.first),
          {llvm::pointerToJITTargetAddress(item.second),
           llvm::JITSymbolFlags::None}}})));
  }
}

template <typename CodeGenT>
void SimpleJIT::Link(ir::Module module, bool optimize) {
  std::string runtime_ir(backends::kRuntimeLlvmIr);
  llvm::SMDiagnostic error;
  auto m = llvm::parseAssemblyString(runtime_ir, error, context());
  m->setDataLayout(jit_->getDataLayout());
  auto b = std::make_unique<llvm::IRBuilder<>>(context());

  auto ir_emitter = std::make_unique<CodeGenT>(m.get(), b.get());
  ir_emitter->Compile(module);

  PADDLE_ENFORCE_EQ(!llvm::verifyModule(*m, &llvm::errs()),
                    true,
                    ::common::errors::InvalidArgument("Invalid module found."));

  AddModule(std::move(m), optimize);
}

template void SimpleJIT::Link<CodeGenLLVM>(ir::Module module, bool optimize);
template void SimpleJIT::Link<CodeGenGpuHost>(ir::Module module, bool optimize);

}  // namespace backends

}  // namespace cinn
