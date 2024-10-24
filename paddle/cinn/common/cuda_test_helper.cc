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

#include "paddle/cinn/common/cuda_test_helper.h"

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"

namespace cinn {
namespace common {

#ifdef CINN_WITH_CUDA
void CudaModuleTester::Compile(const ir::Module& m,
                               const std::string& rewrite_cuda_code) {
  auto _host_module_device_module_ =
      backends::SplitDeviceAndHostModule(m);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  PADDLE_ENFORCE_EQ(host_module.functions().empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "host_module.functions() should not be empty"));
  PADDLE_ENFORCE_EQ(device_module.functions().empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "device_module.functions() should not be empty"));

  backends::CodeGenCudaDev codegen(DefaultHostTarget());
  auto source_code = codegen.Compile(device_module);

  // compile CUDA kernel.
  backends::nvrtc::Compiler compiler;

  std::string ptx;
  if (rewrite_cuda_code.empty())
    ptx = compiler(source_code);
  else
    ptx = compiler(rewrite_cuda_code);

  cuda_module_ =
      new runtime::cuda::CUDAModule(ptx, runtime::cuda::CUDAModule::Kind::PTX);

  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    auto fn_kernel = reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_)
                         ->GetFunction(0, kernel_fn_name);
    PADDLE_ENFORCE_EQ(fn_kernel,
                      true,
                      ::common::errors::InvalidArgument("%s should not be null",
                                                        kernel_fn_name));
    kernel_handles_.push_back(fn_kernel);

    backends::GlobalSymbolRegistry::Global().RegisterFn(
        kernel_fn_name + "_ptr_",
        reinterpret_cast<void*>(&kernel_handles_.back()));
  }

  jit_ = backends::SimpleJIT::Create();

  // compile host module
  jit_->Link<backends::CodeGenGpuHost>(host_module, false);
}

void* CudaModuleTester::CreateDeviceBuffer(const cinn_buffer_t* host_buffer) {
  PADDLE_ENFORCE_EQ(host_buffer->memory,
                    true,
                    ::common::errors::InvalidArgument(
                        "host_buffer->memory should not be null"));
  int num_bytes = host_buffer->num_elements() * sizeof(float);
  CUdeviceptr data;
  cuMemAlloc(&data, num_bytes);

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_buffer->memory,
                       num_bytes,
                       cudaMemcpyHostToDevice));
  return reinterpret_cast<void*>(data);
}

CudaModuleTester::CudaModuleTester() {}

void CudaModuleTester::operator()(const std::string& fn_name,
                                  void* args,
                                  int arg_num) {
  auto fn = jit_->Lookup(fn_name);
  auto fnp = reinterpret_cast<lower_func_ptr_g>(fn);
  (*fnp)(args, arg_num, stream_);
}

void* CudaModuleTester::LookupKernel(const std::string& name) {
  return reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_)
      ->GetFunction(0, name);
}

CudaModuleTester::~CudaModuleTester() {
  if (cuda_module_) {
    delete reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_);
  }
}

#endif

}  // namespace common
}  // namespace cinn
