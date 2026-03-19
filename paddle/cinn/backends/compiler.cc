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

#include "paddle/cinn/backends/compiler.h"

#include <sys/stat.h>
#include <fstream>
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/arch_device.h"
#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/cinn/utils/string.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/flags.h"
#endif
#ifdef CINN_WITH_CUSTOM_DEVICE
#include "paddle/cinn/backends/custom_device/codegen_custom_device_dev.h"
#include "paddle/cinn/backends/custom_device/compiler_custom_device.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/phi/backends/device_manager.h"
#endif
#ifdef CINN_WITH_HIP
#include "paddle/cinn/backends/hip/codegen_hip_dev.h"
#include "paddle/cinn/backends/hip/compiler_hip.h"
#include "paddle/cinn/runtime/hip/hip_module.h"
#endif
#ifdef CINN_WITH_SYCL
#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include "paddle/cinn/backends/sycl/compiler_sycl.h"
#include "paddle/cinn/runtime/sycl/sycl_module.h"
#endif
#include "paddle/cinn/adt/adt.h"

PD_DECLARE_string(cinn_source_code_save_path);
PD_DECLARE_string(cinn_dump_group_lowered_func);
PD_DECLARE_string(cinn_dump_group_source_code);
PD_DECLARE_string(cinn_dump_group_ptx);
PD_DECLARE_string(cinn_dump_group_instruction);
PD_DECLARE_string(cinn_debug_custom_code_path);
COMMON_DECLARE_bool(enable_cinn_kernel_cache);
COMMON_DECLARE_string(cinn_kernel_cache_save_path);

namespace {

bool MakeDirectory(const std::string& dirname, mode_t mode) {
  struct stat st;
  std::string path;
  for (int i = 0; i < dirname.size(); ++i) {
    path.push_back(dirname[i]);
    if (!(dirname[i] == '/' || i + 1 == dirname.size())) {
      continue;
    }
    if (stat(path.c_str(), &st) == 0) {
      if (S_ISDIR(st.st_mode)) {
        continue;
      } else {
        LOG(WARNING) << path << " is not a directory, please check your path.";
        return false;
      }
    } else {
      if (mkdir(path.c_str(), mode) == 0) {
        continue;
      } else {
        LOG(WARNING) << "Make directory fail: " << path;
        return false;
      }
    }
  }
  return true;
}
}  // namespace

namespace cinn {
namespace backends {
using ir::Module;
using CompilationStatus = hlir::framework::CompilationStatus;

static constexpr int DebugLogMaxLen = 30000;

void CompilationInfoDumper::DumpLoweredFuncByGroupIndex(
    const ir::LoweredFunc& lowered_func, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_lowered_func.empty() ||
      lowered_func.get() == nullptr) {
    return;
  }
  std::stringstream content;
  content << lowered_func;
  Dump(FLAGS_cinn_dump_group_lowered_func,
       gidx,
       device_id,
       "lowered_function.txt",
       content.str());
}

void CompilationInfoDumper::DumpSourceCodeByGroupIndex(
    const std::string& source_code, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  Dump(FLAGS_cinn_dump_group_source_code,
       gidx,
       device_id,
       "source_code.cu",
       source_code);
}

void CompilationInfoDumper::DumpPtxCodeByGroupIndex(
    const std::string& source_ptx, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  Dump(
      FLAGS_cinn_dump_group_ptx, gidx, device_id, "source_ptx.ptx", source_ptx);
}

void CompilationInfoDumper::DumpLoweredFunc() {
  if (FLAGS_cinn_dump_group_lowered_func.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::stringstream content;
    if (info_.Status(idx) > CompilationStatus::LOWERING_FAIL) {
      content << info_.LoweredFuncs(idx).front();
    } else {
      content << "[No lowered func generated]\n\n" << info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_lowered_func,
         idx,
         device_id_,
         "lowered_function.txt",
         content.str());
  }
}

void CompilationInfoDumper::DumpSourceCode() {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourceCode(idx);
    } else {
      dump_str = "[No source code generated]\n\n" + info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_source_code,
         idx,
         device_id_,
         "source_code.cu",
         dump_str);
  }
}

void CompilationInfoDumper::DumpPtxCode() {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourcePtx(idx);
    } else {
      dump_str = "[No source ptxs generated]\n\n" + info_.Message(idx);
    }
    Dump(
        FLAGS_cinn_dump_group_ptx, idx, device_id_, "source_ptx.ptx", dump_str);
  }
}

void CompilationInfoDumper::Dump(const std::string& base_path,
                                 const int idx,
                                 const int device_id,
                                 const std::string& file_name,
                                 const std::string& content) {
  auto dump_path = utils::StringFormat(
      "%s/device_%d/fusion_group_%d", base_path.c_str(), device_id, idx);
  if (!MakeDirectory(dump_path,
                     S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    LOG(WARNING) << "Failed to make directory: \"" << dump_path
                 << "\", the instruction for this group will not dump.";
  } else {
    auto dump_file =
        utils::StringFormat("%s/%s", dump_path.c_str(), file_name.c_str());
    VLOG(7) << "Dump instruction to: " << dump_file;
    std::ofstream of(dump_file, std::ios_base::out);
    if (of.is_open()) {
      of << content;
      of.close();
    } else {
      LOG(WARNING) << "Failed to open file: " << dump_file
                   << ", please check your path.";
    }
  }
}

SourceCodePrint::SourceCodePrint() {
  if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(INFO)
        << "The CINN auto generated source code will writing into file: \""
        << FLAGS_cinn_source_code_save_path << "\"";
    of.open(FLAGS_cinn_source_code_save_path, std::ios_base::out);
  }
}

SourceCodePrint::~SourceCodePrint() {
  if (of.is_open()) {
    of.close();
  }
}

void SourceCodePrint::write(const std::string& source_code) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (of.is_open()) {
    of << source_code << std::endl;
  } else if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(WARNING) << "Failed to open \"" << FLAGS_cinn_source_code_save_path
                 << "\", source code will print.";
    if (source_code.size() > DebugLogMaxLen) {
      LOG(INFO) << "[CUDA] source code-0:\n"
                << source_code.substr(0, DebugLogMaxLen);
      for (int i = 1; i * DebugLogMaxLen < source_code.size(); ++i) {
        LOG(INFO) << "[CUDA] source code-" << i << ":\n"
                  << source_code.substr(DebugLogMaxLen * i, DebugLogMaxLen);
      }
    } else {
      LOG(INFO) << "[CUDA] source code:\n" << source_code;
    }
  }
}

void Compiler::Build(const Module& module, const std::string& code) {
  target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { CompileX86Module(module); },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) { CompileCudaModule(module, code); },
      [&](common::HygonDCUArchHIP) { CompileHipModule(module, code); },
      [&](common::HygonDCUArchSYCL) { CompileSyclModule(module, code); },
      [&](common::CustomDeviceArch) {
        CompileCustomDeviceModule(module, code);
      });
}

void Compiler::AppendCX86(const Module& module) {
  VLOG(3) << "Start Compiler::BuildCX86" << module;
  CompileX86Module(module);
  VLOG(3) << "Over Compiler::BuildCX86";
}

void Compiler::AppendBroadcastSwitchModule(const ir::Module& module) {
  engine_->Link<CodeGenSwitchHost>(module);
}

void Compiler::EndCompile() {
  RegisterDeviceModuleSymbol();
  std::vector<std::string> cinn_runtime_include_path = {
      Context::Global().runtime_include_dir()};
  engine_->AddSelfModule(GetFusionHash(), cinn_runtime_include_path);
}

void Compiler::LoadAndRegisterFromCache() {
#ifdef CINN_WITH_CUDA
  std::string cache_so_path = GetCachePath() + CINN_CACHE_SO;
  // 1. Load metadata (restore Kernel name list)
  LoadKernelNamesFromMeta();

  // 2. Load shared library (.so)
  void* handle = dlopen(cache_so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    LOG(FATAL) << "Failed to dlopen shared library: " << cache_so_path
               << " Error: " << dlerror();
  }

  // 3. Load CUDA Fatbin (Device Code)
  std::string fatbin_path = GetCachePath() + CINN_CUDA_KERNEL_FATBIN;
  CUmodule cu_module;
  if (cuModuleLoad(&cu_module, fatbin_path.c_str()) != CUDA_SUCCESS) {
    LOG(FATAL) << "Failed to load CUDA Module from " << fatbin_path;
  }
  // Store the CUmodule handle in a member variable for subsequent
  // cuModuleUnload
  this->cuda_module_handle_ = cu_module;

  RuntimeSymbols symbols;
  // 4. Iterate and register symbols
  for (const auto& kernel_fn_name : device_fn_name_) {
    // 4A. Find Host Wrapper function pointer
    void* fn_kernel = dlsym(handle, kernel_fn_name.c_str());
    if (!fn_kernel) {
      LOG(FATAL) << "Failed to dlsym kernel symbol: " << kernel_fn_name
                 << " from " << cache_so_path << " Error: " << dlerror();
    }

    // 4B. Register to ExecutionEngine (for runtime lookup of Host Wrapper
    // address)
    fn_ptr_.push_back(fn_kernel);
    symbols.RegisterVar(kernel_fn_name + "_ptr_", fn_kernel);

    // 4C. Get Device handle and override Host-side pointer
    CUfunction cu_kernel_func;
    if (cuModuleGetFunction(&cu_kernel_func,
                            cu_module,
                            kernel_fn_name.c_str()) != CUDA_SUCCESS) {
      LOG(FATAL) << "Failed to get CUfunction handle for " << kernel_fn_name;
    }
    void* kernel_ptr_host_addr =
        dlsym(handle, (kernel_fn_name + "_ptr_").c_str());
    if (!kernel_ptr_host_addr) {
      LOG(FATAL) << "Failed to dlsym kernel pointer variable: "
                 << kernel_fn_name + "_ptr_";
    }
    *static_cast<void**>(kernel_ptr_host_addr) =
        reinterpret_cast<void*>(cu_kernel_func);
  }

  // 5. Register all runtime symbols
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));

  // 6. Store handles and paths for subsequent dlclose
  dynamic_library_path_ = cache_so_path;
  dynamic_library_handle_ = handle;
#else
  CINN_NOT_IMPLEMENTED
#endif
}

std::string Compiler::GetSourceCode(const ir::Module& module) {
  return target_.arch.Match(
      [&](common::UnknownArch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::ARMArch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::CustomDeviceArch) -> std::string {
#ifdef CINN_WITH_CUSTOM_DEVICE
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        custom_device::CodeGenCustomDevice codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      },
      [&](common::NVGPUArch) -> std::string {
#ifdef CINN_WITH_CUDA
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        CodeGenCudaDev codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      },
      [&](common::HygonDCUArchHIP) -> std::string {
#ifdef CINN_WITH_HIP
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        hip::CodeGenHipDevice codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      },
      [&](common::HygonDCUArchSYCL) -> std::string {
#ifdef CINN_WITH_SYCL
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        sycl::CodeGenSyclDevice codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      });
}

void Compiler::BuildDefault(const Module& module) {
  target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { CompileX86Module(module); },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::CustomDeviceArch) { CompileCustomDeviceModule(module); },
      [&](common::NVGPUArch) { CompileCudaModule(module); },
      [&](common::HygonDCUArchHIP) { CompileHipModule(module); },
      [&](common::HygonDCUArchSYCL) { CompileSyclModule(module); });
}

namespace {
std::string GetFileContent(const std::string& path) {
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << path << std::endl;
    return "";
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();

  file.close();
  return content;
}
}  // namespace

void Compiler::RegisterDeviceModuleSymbol() {
  return target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { return; },
      [&](common::ARMArch) { return; },
      [&](common::CustomDeviceArch) { RegisterCustomDeviceModuleSymbol(); },
      [&](common::NVGPUArch) { RegisterCudaModuleSymbol(); },
      [&](common::HygonDCUArchHIP) { RegisterHipModuleSymbol(); },
      [&](common::HygonDCUArchSYCL) { RegisterSyclModuleSymbol(); });
}

std::string Compiler::GetDeviceId() const {
  const auto device_id = cinn::runtime::GetArchDevice(target_);
  return std::to_string(device_id.value());
}
std::string Compiler::GetCachePath() const {
  return FLAGS_cinn_kernel_cache_save_path + "/" + GetDeviceId() + "/" +
         std::to_string(GetFusionHash()) + "/";
}
void Compiler::RegisterCudaModuleSymbol() {
#ifdef CINN_WITH_CUDA
  nvrtc::Compiler compiler;
  std::string source_code =
      (FLAGS_enable_cinn_kernel_cache ? CodeGenCudaDev::GetGeneralSourceHeader()
                                      : CodeGenCudaDev::GetSourceHeader()) +
      device_fn_code_;

  if (FLAGS_enable_cinn_kernel_cache) {
    std::string cache_so_path = GetCachePath() + CINN_CACHE_SO;

    // Check if cache file exists
    if (std::ifstream(cache_so_path).good()) {
      LOG(FATAL) << "cinn_cache.so exists! Should not walk in!!! "
                 << "Should already redirect to kernel cache mechanism in "
                    "PIRCompiler.";
      LoadAndRegisterFromCache();
      return;
    } else {  // .so doesn't exist, compile new CINN_CUDA_KERNEL_OBJ and
              // CINN_CUDA_KERNEL_FATBIN
      // We must define in C++ (Host) code the [kernel_name]_ptr_ global
      // variables that LLVM IR (module.o) expects to link. These variables must
      // be compiled together with the CUDA Kernel functions themselves.
      std::string host_symbol_definitions = "\n\nextern \"C\" {\n";
      for (const auto& kernel_fn_name : device_fn_name_) {
        // This will generate C++ code like:
        // void* fn_name..._kernel_ptr_ = (void*)fn_name...;
        host_symbol_definitions += "  void* " + kernel_fn_name +
                                   "_ptr_ = (void*)" + kernel_fn_name + ";\n";
      }
      host_symbol_definitions += "}\n";

      // Append C++ pointer definitions to CUDA Kernel source code
      std::string full_source_to_compile =
          source_code + host_symbol_definitions;

      dynamic_library_path_ =
          GenerateObjectWithoutCache(full_source_to_compile);
      GenerateFatbinWithoutCache();
      SaveKernelNamesToMeta();

      // Register to JIT in normal way
      auto ptx = compiler(source_code);
      PADDLE_ENFORCE_EQ(!ptx.empty(),
                        true,
                        ::common::errors::InvalidArgument(
                            "Compile PTX failed from source code\n"));
      using runtime::cuda::CUDAModule;
      cuda_module_.reset(new CUDAModule(ptx,
                                        compiler.compile_to_cubin()
                                            ? CUDAModule::Kind::CUBIN
                                            : CUDAModule::Kind::PTX));

      RuntimeSymbols symbols;
      for (const auto& kernel_fn_name : device_fn_name_) {
        auto fn_kernel = cuda_module_->GetFunction(kernel_fn_name);
        PADDLE_ENFORCE_NOT_NULL(fn_kernel,
                                ::common::errors::InvalidArgument(
                                    "Fail to get CUfunction kernel_fn_name"));
        fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
        symbols.RegisterVar(kernel_fn_name + "_ptr_",
                            reinterpret_cast<void*>(fn_kernel));
      }
      engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
    }
  } else {
    // Register to JIT in normal way
    auto ptx = compiler(source_code);
    PADDLE_ENFORCE_EQ(!ptx.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Compile PTX failed from source code\n"));
    using runtime::cuda::CUDAModule;
    cuda_module_.reset(new CUDAModule(ptx,
                                      compiler.compile_to_cubin()
                                          ? CUDAModule::Kind::CUBIN
                                          : CUDAModule::Kind::PTX));

    RuntimeSymbols symbols;
    for (const auto& kernel_fn_name : device_fn_name_) {
      auto fn_kernel = cuda_module_->GetFunction(kernel_fn_name);
      PADDLE_ENFORCE_NOT_NULL(fn_kernel,
                              ::common::errors::InvalidArgument(
                                  "Fail to get CUfunction kernel_fn_name"));
      fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
      symbols.RegisterVar(kernel_fn_name + "_ptr_",
                          reinterpret_cast<void*>(fn_kernel));
    }
    engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
  }
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::RegisterCustomDeviceModuleSymbol() {
#ifdef CINN_WITH_CUSTOM_DEVICE
  // 1. Get the plugin instance (needed for LoadModule later)
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  PADDLE_ENFORCE_EQ(!dev_types.empty(),
                    true,
                    ::common::errors::NotFound(
                        "No custom device registered in DeviceManager."));
  std::string dev_type = dev_types[0];
  auto place = phi::CustomPlace(dev_type, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);

  // 2. Invoke cdrtc::Compiler to compile source → shared lib
  common::Target target = common::DefaultCustomDeviceTarget();
  cdrtc::Compiler compiler(target);
  std::string lib_path = compiler(device_fn_code_);

  PADDLE_ENFORCE_EQ(
      !lib_path.empty(),
      true,
      ::common::errors::External("Custom Device Toolchain compile failed."));

  // 3. Invoke the plugin runtime to load the module
  this->device_module_ = plugin.GetRuntime()->LoadModule(lib_path);
  PADDLE_ENFORCE_NOT_NULL(
      this->device_module_,
      ::common::errors::External(
          "Custom Device Runtime failed to load module from %s",
          lib_path.c_str()));

  // 4. Register Kernel symbols
  // Retrieve the device function pointers (handles) and register them
  // as [kernel_name]_ptr_
  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    void* fn_kernel = this->device_module_->GetFunction(kernel_fn_name);

    PADDLE_ENFORCE_NOT_NULL(fn_kernel,
                            ::common::errors::NotFound(
                                "Custom Device Runtime cannot find kernel: %s",
                                kernel_fn_name.c_str()));

    // 5. Store the pointer for use by the ExecutionEngine
    fn_ptr_.push_back(fn_kernel);
    symbols.RegisterVar(kernel_fn_name + "_ptr_", fn_kernel);
  }

  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::RegisterHipModuleSymbol() {
#ifdef CINN_WITH_HIP
  hiprtc::Compiler compiler;
  std::string source_code =
      hip::CodeGenHipDevice::GetSourceHeader() + device_fn_code_;
  std::string hsaco = compiler(source_code);
  PADDLE_ENFORCE_EQ(
      !hsaco.empty(),
      true,
      ::common::errors::Fatal("Compile hsaco failed from source code:\n%s",
                              source_code));
  using runtime::hip::HIPModule;
  hip_module_.reset(new HIPModule(hsaco));
  // get device id
  using cinn::runtime::BackendAPI;
  int device_id = BackendAPI::get_backend(target_)->get_device();
  // register kernel
  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    auto fn_kernel = hip_module_->GetFunction(device_id, kernel_fn_name);
    PADDLE_ENFORCE_NOT_NULL(
        fn_kernel,
        ::common::errors::Fatal("HIP GetFunction Error: get valid kernel."));
    fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::RegisterSyclModuleSymbol() {
#ifdef CINN_WITH_SYCL
  syclrtc::Compiler compiler;
  std::string source_code =
      sycl::CodeGenSyclDevice::GetSourceHeader() + device_fn_code_;
  std::string hsaco = compiler(source_code);
  PADDLE_ENFORCE_EQ(
      !hsaco.empty(),
      true,
      ::common::errors::Fatal("Compile hsaco failed from source code:\n%s",
                              source_code));
  using runtime::sycl::SYCLModule;
  sycl_module_.reset(new SYCLModule(source_code, hsaco, SYCLModule::Kind::so));
  // get device id
  using cinn::runtime::BackendAPI;
  int device_id = BackendAPI::get_backend(target_)->get_device();
  // register kernel
  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    auto fn_kernel = sycl_module_->GetFunction(kernel_fn_name);
    PADDLE_ENFORCE_NOT_NULL(
        fn_kernel,
        ::common::errors::Fatal("HIP GetFunction Error: get valid kernel."));
    fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileCudaModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_CUDA
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[CUDA] host module:\n" << host_module;

  VLOG(3) << "[CUDA] device module:\n" << device_module;
  std::string source_code;

  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    CodeGenCudaDev codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }

  PADDLE_ENFORCE_EQ(!source_code.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Compile CUDA C code failed from device module"));
  VLOG(3) << "[CUDA] C:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;

  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileCustomDeviceModule(const Module& module,
                                         const std::string& code) {
#ifdef CINN_WITH_CUSTOM_DEVICE
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[CustomDevice] host module:\n" << host_module;

  VLOG(3) << "[CustomDevice] device module:\n" << device_module;
  std::string source_code;

  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    custom_device::CodeGenCustomDevice codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }

  PADDLE_ENFORCE_EQ(!source_code.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Compile CustomDevice code failed from device module"));
  VLOG(1) << "[CustomDevice] Source:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;

  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileHipModule(const Module& module, const std::string& code) {
#ifdef CINN_WITH_HIP
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[HIP] host module:\n" << host_module;
  VLOG(3) << "[HIP] device module:\n" << device_module;
  std::string source_code;
  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    hip::CodeGenHipDevice codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  PADDLE_ENFORCE_EQ(
      !source_code.empty(),
      true,
      ::common::errors::Fatal("Compile HIP code failed from device module:\n%s",
                              device_module));
  VLOG(3) << "[HIP]:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileSyclModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_SYCL
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[SYCL] host module:\n" << host_module;
  VLOG(3) << "[SYCL] device module:\n" << device_module;
  std::string source_code;
  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    sycl::CodeGenSyclDevice codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  PADDLE_ENFORCE_EQ(
      !source_code.empty(),
      true,
      ::common::errors::Fatal(
          "Compile SYCL code failed from device module:\n%s", device_module));
  VLOG(3) << "[SYCL]:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileX86Module(const Module& module) {
  engine_->Link<CodeGenX86>(module);
}

void Compiler::ExportObject(const std::string& path) {
  engine_->ExportObject(path);
}

void* Compiler::Lookup(std::string_view fn_name) {
  // 1. Check if the dynamic library has already been loaded via Cache.
  // This dynamic_library_handle_ is only assigned when LoadAndRegisterFromCache
  // is called. it will only be non-null in the "cache hit" path.
  if (FLAGS_enable_cinn_kernel_cache && dynamic_library_handle_) {
    void* func_ptr = dlsym(dynamic_library_handle_, fn_name.data());
    if (func_ptr) {
      VLOG(5) << "Lookup symbol " << fn_name << " from cached .so success.";
      return func_ptr;
    }
    LOG(FATAL) << "Kernel cache is enabled but symbol " << fn_name
               << " not found in .so";
    return nullptr;
  }

  PADDLE_ENFORCE_NOT_NULL(
      engine_, ::common::errors::InvalidArgument("Sorry, engine_ is nullptr"));
  if (engine_->Lookup(fn_name) != nullptr) {
    return engine_->Lookup(fn_name);
  }
  return nullptr;
}

#ifdef CINN_WITH_CUDA
std::string Compiler::GetDeviceArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
          cudaSuccess) {
    return "sm_" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to sm_80.";
    return "sm_80";
  }
}

std::string Compiler::GetComputeArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
          cudaSuccess) {
    return "compute_" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_80.";
    return "compute_80";
  }
}

std::string Compiler::GenerateObjectWithoutCache(
    const std::string& source_code) {
  std::string library_path = GetCachePath();
  llvm::sys::fs::create_directories(library_path);

  // Generate a temporary .cu file, then compile it to .o file using nvcc
  std::string cuda_source_file = library_path + CINN_CUDA_KERNEL;
  std::ofstream source_file(cuda_source_file);

  // Check if file opened successfully
  if (!source_file.is_open()) {
    LOG(FATAL) << "Failed to open CUDA source file for writing: "
               << cuda_source_file << ". Check file permissions.";
    return "";
  }

  source_file << source_code;
  source_file.flush();
  source_file.close();

  // Check file status
  if (!source_file.good()) {
    LOG(FATAL) << "Failed to write or close the CUDA source file: "
               << cuda_source_file << ". Check disk space or permissions.";
    return "";
  }

  if (!llvm::sys::fs::exists(cuda_source_file)) {
    LOG(FATAL)
        << "File successfully written but immediately missing/unreadable: "
        << cuda_source_file;
    return "";
  }

  // Create .o file
  std::string cuda_source_o = library_path + CINN_CUDA_KERNEL_OBJ;

  // If cuda_source_o already exists, report an error
  if (llvm::sys::fs::exists(cuda_source_o)) {
    LOG(FATAL)
        << "Internal error: Object file already exists. "
        << "This indicates a logic error in hash or kernel naming. File: "
        << cuda_source_o;
  }

  std::vector<std::string> cinn_runtime_include_path = {
      Context::Global().runtime_include_dir()};
  std::string include_dir_str = "";
  for (const auto& dir : cinn_runtime_include_path) {
    include_dir_str += "-I" + dir + " ";
  }

  std::string compile_cmd =
      "nvcc -c -Xcompiler -fPIC -o " + cuda_source_o + " " + cuda_source_file +
      " -arch=" + GetDeviceArch() + " --std=c++14 --expt-relaxed-constexpr " +
      include_dir_str + "-I/usr/local/cuda/include -include cuda_fp16.h " +
      "-DCINN_CUDA_FP16 -include cuda_fp8.h -DCINN_CUDA_FP8 " +
      "-include cuda_bf16.h -DCINN_CUDA_BF16 " +
      "-DCUDA_VERSION=" + std::to_string(CUDA_VERSION) + " " +
      "-Wno-deprecated-gpu-targets " +
      "--generate-code=arch=" + GetComputeArch() + ",code=" + GetDeviceArch();

  int result = std::system(
      (compile_cmd + " > " + GetCachePath() + "compile_o.log 2>&1").c_str());
  if (result != 0) {
    std::ifstream log_file(GetCachePath() + "compile_o.log");
    std::string log_content((std::istreambuf_iterator<char>(log_file)),
                            std::istreambuf_iterator<char>());
    LOG(ERROR) << "Compilation failed with output:\n"
               << compile_cmd << "\n"
               << log_content;
    return "";
  }
  return cuda_source_o;
}

std::string Compiler::GenerateFatbinWithoutCache() {
  std::string library_path = GetCachePath();
  llvm::sys::fs::create_directories(library_path);

  std::string cuda_source_file = library_path + CINN_CUDA_KERNEL;

  if (!llvm::sys::fs::exists(cuda_source_file)) {
    LOG(FATAL) << "CUDA source file is missing. Expected file: "
               << cuda_source_file
               << ". Was GenerateObjectWithoutCache called first?";
    return "";
  }

  // Create fatbin file
  std::string cuda_fatbin = library_path + CINN_CUDA_KERNEL_FATBIN;

  // If cuda_source_o already exists, report an error
  if (llvm::sys::fs::exists(cuda_fatbin)) {
    LOG(FATAL)
        << "Internal error: Object file already exists. "
        << "This indicates a logic error in hash or kernel naming. File: "
        << cuda_fatbin;
  }

  std::vector<std::string> cinn_runtime_include_path = {
      Context::Global().runtime_include_dir()};
  std::string include_dir_str = "";
  for (const auto& dir : cinn_runtime_include_path) {
    include_dir_str += "-I" + dir + " ";
  }

  std::string compile_cmd =
      "nvcc --fatbin -o " + cuda_fatbin + " " + cuda_source_file +
      " -arch=" + GetDeviceArch() + "   --std=c++14 --expt-relaxed-constexpr " +
      include_dir_str + "-I/usr/local/cuda/include -include cuda_fp16.h " +
      "-DCINN_CUDA_FP16 -include cuda_fp8.h -DCINN_CUDA_FP8 " +
      "-include cuda_bf16.h -DCINN_CUDA_BF16 " +
      "-DCUDA_VERSION=" + std::to_string(CUDA_VERSION) + " " +
      "-Wno-deprecated-gpu-targets " +
      "--generate-code=arch=" + GetComputeArch() + ",code=" + GetDeviceArch();

  int result = std::system(
      (compile_cmd + " > " + GetCachePath() + "compile_fatbin.log 2>&1")
          .c_str());
  if (result != 0) {
    std::ifstream log_file(GetCachePath() + "compile_fatbin.log");
    std::string log_content((std::istreambuf_iterator<char>(log_file)),
                            std::istreambuf_iterator<char>());
    LOG(ERROR) << "Compilation failed with output:\n"
               << compile_cmd << "\n"
               << log_content;
    return "";
  }
  return cuda_fatbin;
}

void Compiler::SaveKernelNamesToMeta() {
  // 1. Get metadata file path
  std::string meta_path = GetCachePath();
  llvm::sys::fs::create_directories(meta_path);
  std::string meta_file = meta_path + CINN_CUDA_KERNEL_META;

  // 2. Open file
  std::ofstream outfile(meta_file);
  if (!outfile.is_open()) {
    // Theoretically the directory was created in GenerateFatbinWithoutCache,
    // should only check permissions here
    LOG(FATAL) << "Failed to open meta file for writing: " << meta_file;
  }

  // 3. Write data
  // Write one Kernel name per line
  for (const auto& name : device_fn_name_) {
    outfile << name << "\n";
    VLOG(8) << "Saving kernel name " << name;
  }

  // 4. Check write status
  if (outfile.fail()) {
    LOG(FATAL) << "Error writing to meta file: " << meta_file;
  }
}

void Compiler::LoadKernelNamesFromMeta() {
  // 1. Get metadata file path
  std::string meta_path = GetCachePath() + CINN_CUDA_KERNEL_META;
  VLOG(5) << "Loading CINN kernel names from meta file: " << meta_path;

  // 2. Open file
  std::ifstream infile(meta_path);
  if (!infile.is_open()) {
    // If file does not exist, this is a cache logic error
    LOG(FATAL) << "Failed to open meta file for reading during cache hit: "
               << meta_path;
  }

  // 3. Clear old data and read new data
  device_fn_name_.clear();  // Clear old data to prevent data contamination
  std::string line;

  while (std::getline(infile, line)) {
    // Check and ignore empty lines
    if (!line.empty()) {
      device_fn_name_.push_back(line);
      VLOG(5) << "Loaded kernel name: " << line;
    }
  }

  // 4. Check if at least one Kernel name was successfully loaded
  if (device_fn_name_.empty()) {
    LOG(FATAL) << "Meta file is empty or corrupted: " << meta_path;
  }
}

#endif

}  // namespace backends
}  // namespace cinn
