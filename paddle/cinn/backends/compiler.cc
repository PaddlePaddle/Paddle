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
#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/cinn/utils/string.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/flags.h"
#include <dlfcn.h>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
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
      [&](common::HygonDCUArchSYCL) { CompileSyclModule(module, code); });
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
  std::vector<std::string> cinn_runtime_include_path = {Context::Global().runtime_include_dir()};
  engine_->AddSelfModule(GetFusionHash(), cinn_runtime_include_path);
}

std::string Compiler::GetSourceCode(const ir::Module& module) {
  return target_.arch.Match(
      [&](common::UnknownArch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::ARMArch) -> std::string { CINN_NOT_IMPLEMENTED; },
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
      [&](common::NVGPUArch) { RegisterCudaModuleSymbol(); },
      [&](common::HygonDCUArchHIP) { RegisterHipModuleSymbol(); },
      [&](common::HygonDCUArchSYCL) { RegisterSyclModuleSymbol(); });
}


void Compiler::RegisterCudaModuleSymbol() {
#ifdef CINN_WITH_CUDA
  VLOG(3) << "RegisterCudaModuleSymbol with kernel cache: " << cinn_kernel_cache_;
  nvrtc::Compiler compiler;
  std::string source_code = CodeGenCudaDev::GetSourceHeader() + device_fn_code_;

  // 我们必须在 C++ (Host) 代码中定义 LLVM IR (module.o) 期望链接的
  // [kernel_name]_ptr_ 全局变量。
  // 这些变量必须与 CUDA Kernel 函数本身一起编译。

  std::string host_symbol_definitions = "\n\nextern \"C\" {\n";
  for (const auto& kernel_fn_name : device_fn_name_) {
      // 这将生成如下 C++ 代码：
      // void* fn_mul_sum_scale..._kernel_ptr_ = (void*)fn_mul_sum_scale...;
      host_symbol_definitions += "  void* " + kernel_fn_name + "_ptr_ = (void*)" + kernel_fn_name + ";\n";
  }
  host_symbol_definitions += "}\n";

  // 2. 将 C++ 指针定义附加到 CUDA Kernel 源代码
  std::string full_source_to_compile = source_code + host_symbol_definitions;

  // ----------------------------------------------------------------------
  
  if (cinn_kernel_cache_) {
    // 缓存模式逻辑：检查/tmp/cinn/source_hash/cinn_cuda_kernel.o是否存在
    std::string kernel_name = ExtractKernelName(source_code);
    std::string source_hash = ComputeSourceHash();
    std::string cache_so_path = "/tmp/cinn/" + source_hash + "/" + "cinn_cache.so";
    VLOG(3) << "YUHAN!!! source code is : " << source_code;
    VLOG(3) << "YUHAN!!! kernel name is : " << kernel_name;
    VLOG(3) << "YUHAN!!! source hash is : " << source_hash;
    
    // 检查缓存文件是否存在
    if (std::ifstream(cache_so_path).good()) {
      VLOG(3) << "YUHAN!!! " << cache_so_path << " already exist!!";
      // Find Loading symbols from .so...
      // Step 1: 加载共享库 (dlopen)
      // RTLD_LAZY: 延迟解析符号。RTLD_LOCAL: 符号不导出给其他 dlopen 的库。
      void* handle = dlopen(cache_so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (!handle) {
        // 严重错误，dlerror() 给出详细信息
        LOG(FATAL) << "Failed to dlopen shared library: " << cache_so_path << " Error: " << dlerror();
      }
      
      // 加载 CUDA Fatbin (Device Code)
      std::string fatbin_path = "/tmp/cinn/" + source_hash + "/cinn_cuda_kernel.fatbin"; 
      CUmodule cu_module;
      // 假设您的 CUDA Driver API 函数名是 cuModuleLoad
      if (cuModuleLoad(&cu_module, fatbin_path.c_str()) != CUDA_SUCCESS) {
          LOG(FATAL) << "Failed to load CUDA Module from " << fatbin_path;
      }
      // 🚨 TODO: 务必将 cu_module 句柄保存到 ExecutionEngine 或 Compiler 成员变量中，
      // 以便在 CINN 销毁时调用 cuModuleUnload。

      RuntimeSymbols symbols;
      // Step 2: 查找 CUDA Kernel 符号 (dlsym)
      // 我们查找 device_fn_name_ 列表中的所有 CUDA Kernel 函数
      LoadKernelNamesFromMeta();
      for (const auto& kernel_fn_name : device_fn_name_) {
        // 在 .so 中查找符号
        void* fn_kernel = dlsym(handle, kernel_fn_name.c_str());
        if (!fn_kernel) {
            // 找不到符号，可能是 .so 损坏或编译失败
            LOG(FATAL) << "Failed to dlsym kernel symbol: " << kernel_fn_name << " from " << cache_so_path << " Error: " << dlerror();
        }
        
        // Step 3: 注册符号 (注册到 CINN ExecutionEngine)
        // 注册给运行时使用的函数指针
        fn_ptr_.push_back(fn_kernel); 
        // 注册到 ExecutionEngine 的符号表，通常以 "_ptr_" 结尾供运行时识别
        symbols.RegisterVar(kernel_fn_name + "_ptr_", fn_kernel);

        // ===================================================================
        // 🚨 修复步骤 B: 获取 Device 句柄，并覆盖 .so 中的全局变量
        // ===================================================================
        
        // B1. 从已加载的 CUDA Module 中获取 Device 侧 Kernel 句柄 (CUfunction)
        CUfunction cu_kernel_func;
        if (cuModuleGetFunction(&cu_kernel_func, cu_module, kernel_fn_name.c_str()) != CUDA_SUCCESS) {
            LOG(FATAL) << "Failed to get CUfunction handle for " << kernel_fn_name;
        }

        // B2. 查找 .so 中全局变量 [kernel_name]_ptr_ 的 Host 内存地址
        // 这是 Host 函数读取的变量
        void* kernel_ptr_host_addr = dlsym(handle, (kernel_fn_name + "_ptr_").c_str());
        if (!kernel_ptr_host_addr) {
             LOG(FATAL) << "Failed to dlsym kernel pointer variable: " << kernel_fn_name + "_ptr_";
        }
        
        // B3. 写入：将 Device 句柄的值写入 Host 变量的地址中
        // 覆盖掉它初始化时错误的 Host 函数地址
        *static_cast<void**>(kernel_ptr_host_addr) = reinterpret_cast<void*>(cu_kernel_func); 
        // ===================================================================
      }

      // 注册所有运行时符号到 ExecutionEngine
      engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
      VLOG(3) << "Successfully registered symbols from cache .so: " << cache_so_path;

      // 存储句柄和路径，以便在 Compiler 或 ExecutionEngine 销毁时调用 dlclose
      // TODO:: 确认是否要销毁
      dynamic_library_path_ = cache_so_path; // 存储路径
      dynamic_library_handle_ = handle;      // 假设 Compiler 类中有一个成员 `void* dynamic_library_handle_` 存储句柄
    } else {
      // .so不存在，编译生成新的cinn_cuda_kernel.o和cinn_cuda_kernel.fatbin
      VLOG(3) << "Creating new kernel cache file";
      dynamic_library_path_ = GenerateObjectWithoutCache(full_source_to_compile);
      GenerateFatbinWithoutCache(full_source_to_compile);
      SaveKernelNamesToMeta();


      // 按照正常方式注册到JIT
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
    // 按照正常方式注册到JIT
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
  if (cinn_kernel_cache_) {
    // 存在场景.so已经存在，但是还没有dlopen所以没有dynamic_library_handle_
    // 这里需要
    if (dynamic_library_handle_ ==  nullptr) {
      std::string cache_so_path = "/tmp/cinn/" + std::to_string(fusion_hash_) + "/" + "cinn_cache.so";
      if (std::ifstream(cache_so_path).good()) {
        VLOG(3) << "YUHAN!!! " << cache_so_path << " already exist!!";
        // Find Loading symbols from .so...
        // Step 1: 加载共享库 (dlopen)
        // RTLD_LAZY: 延迟解析符号。RTLD_LOCAL: 符号不导出给其他 dlopen 的库。
        void* handle = dlopen(cache_so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
          // 严重错误，dlerror() 给出详细信息
          LOG(FATAL) << "Failed to dlopen shared library: " << cache_so_path << " Error: " << dlerror();
        }
        dynamic_library_path_ = cache_so_path; // 存储路径
        dynamic_library_handle_ = handle; 
      } else {
        LOG(FATAL) << "dynamic_library_handle_ is nullptr and " << cache_so_path << " not exist!";
        return nullptr;
      }
    }
    VLOG(3) << "Lookup symbol in cached .so: " << fn_name;
    
    // 使用 dlsym 从共享库句柄中查找主入口 Host 函数
    void* func_ptr = dlsym(dynamic_library_handle_, fn_name.data());
    
    if (!func_ptr) {
      // 找不到主入口函数是致命错误
      LOG(FATAL) << "Failed to dlsym host function: " << fn_name 
                << " from " << dynamic_library_path_ << ". Error: " << dlerror();
    }
    
    VLOG(3) << "Successfully looked up host function: " << fn_name;
    return func_ptr;

  }
  

  PADDLE_ENFORCE_NOT_NULL(
      engine_, ::common::errors::InvalidArgument("Sorry, engine_ is nullptr"));
  if (engine_->Lookup(fn_name) != nullptr) {
    return engine_->Lookup(fn_name);
  }
  return nullptr;
}

#ifdef CINN_WITH_CUDA
std::string Compiler::ComputeSourceHash() {
  return std::to_string(fusion_hash_);
}
std::string Compiler::ExtractKernelName(const std::string& source_code) {
  // 从CUDA源码中提取kernel函数名
  // 查找"__global__"关键字后面的函数名
  size_t global_pos = source_code.find("__global__");
  if (global_pos == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  // 跳过__global__关键字
  size_t pos = global_pos + 10; // "__global__"的长度是10
  
  // 跳过空格和换行符
  pos = source_code.find_first_not_of(" \t\n", pos);
  if (pos == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  // 跳过返回类型（如void、float等），找到函数名
  // 查找函数名的开始位置（在返回类型之后）
  size_t func_name_start = source_code.find_first_not_of(" \t\n", pos);
  if (func_name_start == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  // 跳过返回类型单词
  size_t return_type_end = source_code.find_first_of(" \t\n", func_name_start);
  if (return_type_end == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  // 找到__launch_bounds__开始位置
  size_t launch_bounds_start = source_code.find_first_not_of(" \t\n", return_type_end);
  if (launch_bounds_start == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  // 找到__launch_bounds__结束的位置
  size_t launch_bounds_end = source_code.find_first_of(" \t\n", launch_bounds_start);
  if (launch_bounds_end == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  
  // 处理可能存在的 __launch_bounds__ 属性
  std::string potential_attribute = source_code.substr(launch_bounds_start,
                                                       launch_bounds_end - launch_bounds_start);
  if (potential_attribute.find("__launch_bounds__") == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
    
  // 找到 __launch_bounds__ 后面的位置
  size_t real_func_start = source_code.find_first_not_of(" \t\n", launch_bounds_end + 1);
  if (real_func_start == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  // 找到函数名结束位置（空格、制表符、换行符或左括号）
  size_t func_end = source_code.find_first_of(" \t\n(", real_func_start);
  if (func_end == std::string::npos) {
    return "unknown_kernel_" + std::to_string(std::time(nullptr));
  }
  
  std::string kernel_name = source_code.substr(real_func_start, func_end - real_func_start);
  
  // 验证提取的函数名是否有效（不能是C++关键字或返回类型）
  static const std::set<std::string> invalid_names = {
    "void", "int", "float", "double", "char", "bool", "short", "long", 
    "signed", "unsigned", "const", "volatile", "static", "extern", "auto",
    "register", "inline", "virtual", "explicit", "friend", "typedef",
    "__launch_bounds__"
  };
  
  if (invalid_names.find(kernel_name) != invalid_names.end() || kernel_name.empty()) {
    // 如果提取到的是无效名称，尝试更精确的解析
    // 查找第一个有效的函数名（跳过所有C++关键字）
    size_t current_pos = real_func_start;
    while (current_pos < source_code.length()) {
      size_t next_space = source_code.find_first_of(" \t\n(", current_pos);
      if (next_space == std::string::npos) {
        break;
      }
      
      std::string candidate = source_code.substr(current_pos, next_space - current_pos);
      
      // 检查候选名称是否有效
      if (invalid_names.find(candidate) == invalid_names.end() && 
          !candidate.empty()) {
        return candidate;
      }
      
      current_pos = source_code.find_first_not_of(" \t\n", next_space);
      if (current_pos == std::string::npos) {
        break;
      }
    }
    
    // 如果所有尝试都失败，使用默认名称
    return "cinn_kernel_" + std::to_string(std::time(nullptr));
  }
  
  return kernel_name;
}

std::string Compiler::GenerateObjectWithoutCache(const std::string& source_code) {

  std::string kernel_name = ExtractKernelName(source_code);
  std::string library_path = "/tmp/cinn/" + std::to_string(fusion_hash_) + "/";
  llvm::sys::fs::create_directories(library_path);

  // 生成一个临时.cu文件，然后使用nvcc编译为.o文件
  std::string cuda_source_file = library_path + "cinn_cuda_kernel.cu";
  std::ofstream source_file(cuda_source_file);

  // 检查文件是否成功打开
  if (!source_file.is_open()) {
      LOG(FATAL) << "Failed to open CUDA source file for writing: " 
                 << cuda_source_file 
                 << ". Check file permissions.";
      return "";
  }

  source_file << source_code;
  source_file.close();

  // 检查文件状态
  if (!source_file.good()) {
      LOG(FATAL) << "Failed to write or close the CUDA source file: " 
                 << cuda_source_file 
                 << ". Check disk space or permissions.";
      return "";
  }

  // 创建.o
  std::string cuda_source_o = library_path + "cinn_cuda_kernel.o";
  
  std::string compile_cmd = "nvcc -c -Xcompiler -fPIC -o " + cuda_source_o + " " + cuda_source_file + 
                           " -arch=sm_90 --std=c++14 --expt-relaxed-constexpr " +
                           "-I/workspace/xuyuhan/env3.10/lib/python3.10/site-packages/paddle/libs " +
                           "-I/usr/local/cuda/include -include cuda_fp16.h " +
                           "-DCINN_CUDA_FP16 -include cuda_fp8.h -DCINN_CUDA_FP8 " +
                           "-DCUDA_VERSION=12030 " +
                           "-Wno-deprecated-gpu-targets " +
                           "--generate-code=arch=compute_90,code=sm_90";

  VLOG(3) << "Yuhan!!! GenerateObjectWithoutCache FusionHash \n" << std::to_string(fusion_hash_);
  VLOG(3) << "Yuhan!!! GenerateObjectWithoutCache compile_cmd \n" << compile_cmd;

  int result = std::system((compile_cmd + " > compile.log 2>&1").c_str());
  if (result != 0) {
    std::ifstream log_file("compile.log");
    std::string log_content((std::istreambuf_iterator<char>(log_file)), 
                           std::istreambuf_iterator<char>());
    LOG(ERROR) << "Compilation failed with output:\n" << log_content;
    return "";
  }
  VLOG(3) << "Yuhan!!! GenerateObjectWithoutCache " << cuda_source_o;
  std::remove(cuda_source_file.c_str());
  return cuda_source_o;
}

std::string Compiler::GenerateFatbinWithoutCache(const std::string& source_code) {
  
  std::string kernel_name = ExtractKernelName(source_code);
  std::string library_path = "/tmp/cinn/" + std::to_string(fusion_hash_) + "/";
  llvm::sys::fs::create_directories(library_path);
  
  // 生成一个临时.cu文件，然后使用nvcc编译为fatbin文件
  std::string cuda_source_file = library_path + "cinn_cuda_kernel.cu";
  std::ofstream source_file(cuda_source_file);

  // 检查文件是否成功打开
  if (!source_file.is_open()) {
      LOG(FATAL) << "Failed to open CUDA source file for writing: " 
                 << cuda_source_file 
                 << ". Check file permissions.";
      return "";
  }

  source_file << source_code;
  source_file.close();

  // 检查文件状态
  if (!source_file.good()) {
      LOG(FATAL) << "Failed to write or close the CUDA source file: " 
                 << cuda_source_file 
                 << ". Check disk space or permissions.";
      return "";
  }
  
  // 创建fatbin
  std::string cuda_fatbin = library_path + "cinn_cuda_kernel.fatbin";

  std::string compile_cmd = "nvcc --fatbin -o " + cuda_fatbin + " " + cuda_source_file + 
                           " -arch=sm_90 --std=c++14 --expt-relaxed-constexpr " +
                           "-I/workspace/xuyuhan/env3.10/lib/python3.10/site-packages/paddle/libs " +
                           "-I/usr/local/cuda/include -include cuda_fp16.h " +
                           "-DCINN_CUDA_FP16 -include cuda_fp8.h -DCINN_CUDA_FP8 " +
                           "-DCUDA_VERSION=12030 " +
                           "-Wno-deprecated-gpu-targets " +
                           "--generate-code=arch=compute_90,code=sm_90";

  VLOG(3) << "Yuhan!!! GenerateFatbinWithoutCache FusionHash \n" << std::to_string(fusion_hash_);;
  VLOG(3) << "Yuhan!!! GenerateFatbinWithoutCache compile_cmd \n" << compile_cmd;
  VLOG(3) << "Yuhan!!! GenerateFatbinWithoutCache " << cuda_fatbin;
  int result = std::system((compile_cmd + " > compile_fatbin.log 2>&1").c_str());
  if (result != 0) {
    std::ifstream log_file("compile_fatbin.log");
    std::string log_content((std::istreambuf_iterator<char>(log_file)), 
                           std::istreambuf_iterator<char>());
    LOG(ERROR) << "Compilation failed with output:\n" << log_content;
    return "";
  }
  VLOG(3) << "Yuhan!!! GenerateFatbinWithoutCache " << cuda_fatbin;
  std::remove(cuda_source_file.c_str());
  return cuda_fatbin;
}

void Compiler::SaveKernelNamesToMeta() {
  // 1. 获取元数据文件路径
  std::string meta_path = "/tmp/cinn/" + std::to_string(fusion_hash_) + "/cinn_cuda_kernel.meta";
  VLOG(3) << "Saving CINN kernel names to meta file: " << meta_path;

  // 2. 打开文件
  // 使用 std::ofstream 自动处理文件关闭
  std::ofstream outfile(meta_path);
  if (!outfile.is_open()) {
    // 理论上目录在 GenerateFatbinWithoutCache 中已创建，此处应仅检查权限
    LOG(FATAL) << "Failed to open meta file for writing: " << meta_path;
  }

  // 3. 写入数据
  // 每行写入一个 Kernel 名称
  for (const auto& name : device_fn_name_) {
    outfile << name << "\n";
    VLOG(3) << "Saving kernel name " << name;
  }

  // 4. 检查写入状态
  if (outfile.fail()) {
    LOG(FATAL) << "Error writing to meta file: " << meta_path;
  }
}

void Compiler::LoadKernelNamesFromMeta() {
  // 1. 获取元数据文件路径
  std::string meta_path = "/tmp/cinn/" + std::to_string(fusion_hash_) + "/cinn_cuda_kernel.meta";
  VLOG(3) << "Loading CINN kernel names from meta file: " << meta_path;

  // 2. 打开文件
  std::ifstream infile(meta_path);
  if (!infile.is_open()) {
    // 如果文件不存在，这是一个缓存逻辑错误
    LOG(FATAL) << "Failed to open meta file for reading during cache hit: " << meta_path;
  }

  // 3. 清空旧数据并读取新数据
  device_fn_name_.clear(); // 清空旧数据，防止数据污染
  std::string line;
  
  while (std::getline(infile, line)) {
    // 检查并忽略空行
    if (!line.empty()) {
      device_fn_name_.push_back(line);
      VLOG(4) << "Loaded kernel name: " << line;
    }
  }

  // 4. 检查是否成功加载了至少一个 Kernel 名称
  if (device_fn_name_.empty()) {
    LOG(FATAL) << "Meta file is empty or corrupted: " << meta_path;
  }
}

std::pair<bool, std::string> Compiler::FindKernelInCache(const std::string& so_path, 
                                                        const std::string& kernel_name) {
  // 从缓存文件中查找kernel信息
  std::ifstream so_file(so_path, std::ios::binary);
  if (!so_file.is_open()) {
    VLOG(3) << "Unable to open file: " << so_path;
    return {false, ""};
  }
  
  // 读取缓存文件头
  struct CacheHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_kernels;  // 应该是kernel数量，不是fatbin大小
    uint64_t timestamp;
  };
  
  CacheHeader header;
  so_file.read(reinterpret_cast<char*>(&header), sizeof(header));
  
  if (std::string(header.magic, 4) != "CINN") {
    VLOG(3) << "Invalid cache file format: " << so_path;
    return {false, ""};
  }
  
  VLOG(3) << "Cache header: magic=" << std::string(header.magic, 4) 
          << ", version=" << header.version 
          << ", num_kernels=" << header.num_kernels;
  
  // 遍历kernel条目
  for (uint32_t i = 0; i < header.num_kernels; i++) {
    struct KernelEntry {
      char name[256];
      char hash[64];
      uint64_t fatbin_offset;
      uint64_t fatbin_size;
    };
    
    KernelEntry entry;
    so_file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    
    // 正确提取kernel名称，确保以null结尾
    std::string entry_name(entry.name, strnlen(entry.name, sizeof(entry.name)));
    std::string entry_hash(entry.hash, strnlen(entry.hash, sizeof(entry.hash)));
    
    VLOG(3) << "Entry name is: " << entry_name;
    VLOG(3) << "Looking for kernel: " << kernel_name;
    
    if (entry_name == kernel_name) {
      VLOG(3) << "Found kernel in cache! Hash is: " << entry_hash;
      return {true, entry_hash};
    }
  }
  
  VLOG(3) << "Kernel '" << kernel_name << "' not found in cache";
  return {false, ""};
}

std::string Compiler::UpdateKernelInCache(const std::string& so_path, 
                                         const std::string& kernel_name,
                                         const std::string& source_code,
                                         const std::string& source_hash) {
  // 重新生成fatbin
  std::string temp_fatbin = GenerateObjectWithoutCache(source_code);
  if (temp_fatbin.empty()) {
    return "";
  }
  
  // 读取新的fatbin数据
  std::ifstream fatbin_file(temp_fatbin, std::ios::binary);
  std::vector<char> new_fatbin_data((std::istreambuf_iterator<char>(fatbin_file)),
                                   std::istreambuf_iterator<char>());
  fatbin_file.close();
  std::remove(temp_fatbin.c_str());
  
  // 读取现有缓存文件
  std::ifstream old_so_file(so_path, std::ios::binary);
  if (!old_so_file.is_open()) {
    LOG(ERROR) << "Failed to open existing cache file: " << so_path;
    return "";
  }
  
  // 读取缓存文件头
  struct CacheHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_kernels;
    uint64_t timestamp;
  };
  
  CacheHeader header;
  old_so_file.read(reinterpret_cast<char*>(&header), sizeof(header));
  
  if (std::string(header.magic, 4) != "CINN") {
    LOG(ERROR) << "Invalid cache file format: " << so_path;
    old_so_file.close();
    return "";
  }
  
  // 读取所有kernel条目，跳过要更新的那个
  struct KernelEntry {
    char name[256];
    char hash[64];
    uint64_t fatbin_offset;
    uint64_t fatbin_size;
  };
  
  std::vector<KernelEntry> remaining_entries;
  std::vector<std::vector<char>> remaining_fatbins;
  
  for (uint32_t i = 0; i < header.num_kernels; i++) {
    KernelEntry entry;
    old_so_file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    
    std::string entry_name(entry.name, strnlen(entry.name, sizeof(entry.name)));
    
    if (entry_name != kernel_name) {
      // 保留其他kernel
      remaining_entries.push_back(entry);
      
      // 读取对应的fatbin数据
      old_so_file.seekg(entry.fatbin_offset);
      std::vector<char> fatbin_data(entry.fatbin_size);
      old_so_file.read(fatbin_data.data(), entry.fatbin_size);
      remaining_fatbins.push_back(fatbin_data);
    }
  }
  old_so_file.close();
  
  // 创建新的缓存文件
  std::string new_so_path = so_path + ".new";
  std::ofstream new_so_file(new_so_path, std::ios::binary);
  if (!new_so_file.is_open()) {
    LOG(ERROR) << "Failed to create new cache file: " << new_so_path;
    return "";
  }
  
  // 写入新的缓存文件头
  CacheHeader new_header;
  strncpy(new_header.magic, "CINN", 4);
  new_header.version = header.version;
  new_header.num_kernels = remaining_entries.size() + 1;  // 保留的kernel + 新kernel
  new_header.timestamp = std::time(nullptr);
  
  new_so_file.write(reinterpret_cast<char*>(&new_header), sizeof(new_header));
  
  // 写入kernel条目
  uint64_t current_offset = sizeof(new_header) + (remaining_entries.size() + 1) * sizeof(KernelEntry);
  
  // 先写入保留的kernel条目
  for (size_t i = 0; i < remaining_entries.size(); i++) {
    KernelEntry entry = remaining_entries[i];
    entry.fatbin_offset = current_offset;
    entry.fatbin_size = remaining_fatbins[i].size();
    new_so_file.write(reinterpret_cast<char*>(&entry), sizeof(entry));
    current_offset += entry.fatbin_size;
  }
  
  // 写入新kernel的条目
  KernelEntry new_entry;
  strncpy(new_entry.name, kernel_name.c_str(), sizeof(new_entry.name) - 1);
  new_entry.name[sizeof(new_entry.name) - 1] = '\0';
  strncpy(new_entry.hash, source_hash.c_str(), sizeof(new_entry.hash) - 1);
  new_entry.hash[sizeof(new_entry.hash) - 1] = '\0';
  new_entry.fatbin_offset = current_offset;
  new_entry.fatbin_size = new_fatbin_data.size();
  
  new_so_file.write(reinterpret_cast<char*>(&new_entry), sizeof(new_entry));
  
  // 写入fatbin数据
  for (size_t i = 0; i < remaining_fatbins.size(); i++) {
    new_so_file.write(remaining_fatbins[i].data(), remaining_fatbins[i].size());
  }
  
  // 写入新fatbin数据
  new_so_file.write(new_fatbin_data.data(), new_fatbin_data.size());
  new_so_file.close();
  
  // 替换旧文件
  std::remove(so_path.c_str());
  std::rename(new_so_path.c_str(), so_path.c_str());
  
  VLOG(3) << "Updated cache file: " << so_path << ", removed old kernel: " << kernel_name;
  return so_path;
}

std::string Compiler::AddKernelToCache(const std::string& so_path,
                                      const std::string& kernel_name,
                                      const std::string& source_code,
                                      const std::string& source_hash) {
  // 生成新的fatbin
  std::string temp_fatbin = GenerateObjectWithoutCache(source_code);
  if (temp_fatbin.empty()) {
    return "";
  }
  
  // 读取新的fatbin数据
  std::ifstream fatbin_file(temp_fatbin, std::ios::binary);
  std::vector<char> new_fatbin_data((std::istreambuf_iterator<char>(fatbin_file)),
                                   std::istreambuf_iterator<char>());
  fatbin_file.close();
  std::remove(temp_fatbin.c_str());
  
  // 读取现有缓存文件
  std::ifstream old_so_file(so_path, std::ios::binary);
  if (!old_so_file.is_open()) {
    LOG(ERROR) << "Failed to open existing cache file: " << so_path;
    return "";
  }
  
  // 读取缓存文件头
  struct CacheHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_kernels;
    uint64_t timestamp;
  };
  
  CacheHeader header;
  old_so_file.read(reinterpret_cast<char*>(&header), sizeof(header));
  
  if (std::string(header.magic, 4) != "CINN") {
    LOG(ERROR) << "Invalid cache file format: " << so_path;
    old_so_file.close();
    return "";
  }
  
  // 读取所有现有的kernel条目和fatbin数据
  struct KernelEntry {
    char name[256];
    char hash[64];
    uint64_t fatbin_offset;
    uint64_t fatbin_size;
  };
  
  std::vector<KernelEntry> existing_entries;
  std::vector<std::vector<char>> existing_fatbins;
  
  for (uint32_t i = 0; i < header.num_kernels; i++) {
    KernelEntry entry;
    old_so_file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    existing_entries.push_back(entry);
    
    // 读取对应的fatbin数据
    old_so_file.seekg(entry.fatbin_offset);
    std::vector<char> fatbin_data(entry.fatbin_size);
    old_so_file.read(fatbin_data.data(), entry.fatbin_size);
    existing_fatbins.push_back(fatbin_data);
  }
  old_so_file.close();
  
  // 创建新的缓存文件
  std::string new_so_path = so_path + ".new";
  std::ofstream new_so_file(new_so_path, std::ios::binary);
  if (!new_so_file.is_open()) {
    LOG(ERROR) << "Failed to create new cache file: " << new_so_path;
    return "";
  }
  
  // 写入新的缓存文件头
  CacheHeader new_header;
  strncpy(new_header.magic, "CINN", 4);
  new_header.version = header.version;
  new_header.num_kernels = existing_entries.size() + 1;  // 现有kernel + 新kernel
  new_header.timestamp = std::time(nullptr);
  
  new_so_file.write(reinterpret_cast<char*>(&new_header), sizeof(new_header));
  
  // 写入kernel条目
  uint64_t current_offset = sizeof(new_header) + (existing_entries.size() + 1) * sizeof(KernelEntry);
  
  // 写入现有的kernel条目
  for (size_t i = 0; i < existing_entries.size(); i++) {
    KernelEntry entry = existing_entries[i];
    entry.fatbin_offset = current_offset;
    entry.fatbin_size = existing_fatbins[i].size();
    new_so_file.write(reinterpret_cast<char*>(&entry), sizeof(entry));
    current_offset += entry.fatbin_size;
  }
  
  // 写入新kernel的条目
  KernelEntry new_entry;
  strncpy(new_entry.name, kernel_name.c_str(), sizeof(new_entry.name) - 1);
  new_entry.name[sizeof(new_entry.name) - 1] = '\0';
  strncpy(new_entry.hash, source_hash.c_str(), sizeof(new_entry.hash) - 1);
  new_entry.hash[sizeof(new_entry.hash) - 1] = '\0';
  new_entry.fatbin_offset = current_offset;
  new_entry.fatbin_size = new_fatbin_data.size();
  
  new_so_file.write(reinterpret_cast<char*>(&new_entry), sizeof(new_entry));
  
  // 写入fatbin数据
  for (size_t i = 0; i < existing_fatbins.size(); i++) {
    new_so_file.write(existing_fatbins[i].data(), existing_fatbins[i].size());
  }
  
  // 写入新fatbin数据
  new_so_file.write(new_fatbin_data.data(), new_fatbin_data.size());
  new_so_file.close();
  
  // 替换旧文件
  std::remove(so_path.c_str());
  std::rename(new_so_path.c_str(), so_path.c_str());
  
  VLOG(3) << "Added kernel to cache: " << so_path << ", new kernel: " << kernel_name;
  return so_path;
}

std::string Compiler::CreateNewCache(const std::string& so_path,
                                    const std::string& kernel_name,
                                    const std::string& source_code,
                                    const std::string& source_hash) {
  // 生成新的fatbin
  std::string temp_fatbin = GenerateObjectWithoutCache(source_code);
  if (temp_fatbin.empty()) {
    return "";
  }
  
  // 读取fatbin文件内容
  std::ifstream fatbin_file(temp_fatbin, std::ios::binary);
  if (!fatbin_file.is_open()) {
    LOG(ERROR) << "Failed to open fatbin file: " << temp_fatbin;
    return "";
  }
  
  std::vector<char> fatbin_data((std::istreambuf_iterator<char>(fatbin_file)),
                              std::istreambuf_iterator<char>());
  fatbin_file.close();
  
  // 创建新的缓存文件
  std::ofstream so_file(so_path, std::ios::binary);
  if (!so_file.is_open()) {
    LOG(ERROR) << "Failed to create cache file: " << so_path;
    return "";
  }
  
  // 写入缓存文件头
  struct CacheHeader {
    char magic[4] = {'C', 'I', 'N', 'N'};
    uint32_t version = 1;
    uint32_t num_kernels = 1;  // 使用正确的字段名
    uint64_t timestamp = std::time(nullptr);
  };
  
  CacheHeader header;
  so_file.write(reinterpret_cast<char*>(&header), sizeof(header));
  
  // 写入kernel条目
  struct KernelEntry {
    char name[256] = {0};
    char hash[64] = {0};
    uint64_t fatbin_offset = 0;
    uint64_t fatbin_size = 0;
  };
  
  KernelEntry entry;
  // 确保字符串正确null终止
  strncpy(entry.name, kernel_name.c_str(), sizeof(entry.name) - 1);
  entry.name[sizeof(entry.name) - 1] = '\0';  // 确保null终止
  strncpy(entry.hash, source_hash.c_str(), sizeof(entry.hash) - 1);
  entry.hash[sizeof(entry.hash) - 1] = '\0';  // 确保null终止
  entry.fatbin_offset = sizeof(header) + sizeof(entry);
  entry.fatbin_size = fatbin_data.size();
  
  so_file.write(reinterpret_cast<char*>(&entry), sizeof(entry));
  
  // 写入fatbin数据
  so_file.write(fatbin_data.data(), fatbin_data.size());
  so_file.close();
  
  // 清理临时文件
  std::remove(temp_fatbin.c_str());
  
  VLOG(3) << "Created new cache file: " << so_path << " with kernel: " << kernel_name;
  return so_path;
}
void* Compiler::CreateLibraryInfo(const std::string& library_path, const std::string& function_name) {
  // 创建库信息结构，包含库路径和函数名
  struct LibraryInfo {
    std::string path;
    std::string function;
  };
  
  LibraryInfo* info = new LibraryInfo{library_path, function_name};
  return reinterpret_cast<void*>(info);
}

std::vector<char> Compiler::ExtractFatbinFromSo(const std::string& so_path) {
  std::ifstream so_file(so_path, std::ios::binary);
  VLOG(3) << "Trying to open .so file: " << so_path;
  if (!so_file.is_open()) {
    LOG(ERROR) << "Failed to open .so file: " << so_path;
    return {};
  }
  VLOG(3) << "Successfully open .so file: " << so_path;
  
  // 读取头部 - 使用新的缓存格式
  struct CacheHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_kernels;  // 使用正确的字段名
    uint64_t timestamp;
  };
  
  CacheHeader header;
  so_file.read(reinterpret_cast<char*>(&header), sizeof(header));
  
  // 验证魔术字
  if (std::string(header.magic, 4) != "CINN") {
    LOG(ERROR) << "Invalid .so file format: " << so_path;
    return {};
  }
  VLOG(3) << "In .so file: \t version is " << header.version;
  VLOG(3) << "\t num_kernels is " << header.num_kernels;
  VLOG(3) << "\t timestamp is " << header.timestamp;
  
  // 跳过kernel条目，直接读取fatbin数据
  // 计算fatbin数据的位置：header + 所有kernel条目
  struct KernelEntry {
    char name[256];
    char hash[64];
    uint64_t fatbin_offset;
    uint64_t fatbin_size;
  };
  
  // 跳过所有kernel条目
  so_file.seekg(sizeof(header) + header.num_kernels * sizeof(KernelEntry));
  
  // 读取fatbin数据 - 需要从文件末尾计算大小
  so_file.seekg(0, std::ios::end);
  size_t file_size = so_file.tellg();
  size_t fatbin_start = sizeof(header) + header.num_kernels * sizeof(KernelEntry);
  size_t fatbin_size = file_size - fatbin_start;
  
  so_file.seekg(fatbin_start);
  std::vector<char> fatbin_data(fatbin_size);
  so_file.read(fatbin_data.data(), fatbin_size);
  so_file.close();
  
  LOG(INFO) << "Successfully extracted fatbin from .so file, size: " << fatbin_data.size();
  return fatbin_data;
}
#endif

}  // namespace backends
}  // namespace cinn
