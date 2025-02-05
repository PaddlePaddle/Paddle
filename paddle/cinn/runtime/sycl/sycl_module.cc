// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <dlfcn.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include "paddle/cinn/runtime/sycl/sycl_module.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace sycl {

SYCLModule::SYCLModule(const std::string& source_code,
                       const std::string& shared_library,
                       Kind kind)
    : source_code_(source_code), shared_library_(shared_library), kind_(kind) {
  PADDLE_ENFORCE_NE(
      shared_library.empty(),
      true,
      ::common::errors::InvalidArgument("sharede library is not empty !"));
}

SYCLModule::~SYCLModule() { VLOG(3) << "destructor for SYCLModule"; }

void* SYCLModule::GetFunction(const std::string& func_name) {
  if (so_handler_ == nullptr) {
    so_handler_ = dlopen(shared_library_.c_str(), RTLD_NOW | RTLD_GLOBAL);
  }
  VLOG(5) << "getting function " << func_name;
  PADDLE_ENFORCE_NE(
      so_handler_,
      nullptr,
      ::common::errors::InvalidArgument(
          "Errors: SYCL failed to load shared library %s", dlerror()));
  void (*kernel_func)(::sycl::queue & Q,
                      ::sycl::range<3> k0_dimGrid,
                      ::sycl::range<3> k0_dimBlock,
                      void** void_args) =
      (void (*)(::sycl::queue & Q,
                ::sycl::range<3> k0_dimGrid,
                ::sycl::range<3> k0_dimBlock,
                void** void_args)) dlsym(so_handler_, func_name.c_str());
  PADDLE_ENFORCE_NE(
      kernel_func,
      nullptr,
      ::common::errors::InvalidArgument(
          "Errors: Sycl failed to get function %s", dlerror(), ":dlsym\n"));
  return reinterpret_cast<void*>(kernel_func);
}

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
