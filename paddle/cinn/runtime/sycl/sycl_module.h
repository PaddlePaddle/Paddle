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

#pragma once

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace cinn {
namespace runtime {
namespace Sycl {
/**
 * The SYCL module, helps to compile SYCL codes and fetch symbols.
 * Currently, it is a wrapper of NVRTC.
 */
class SYCLModule {
 public:
  enum class Kind {
    so = 0,
  };
  SYCLModule(const std::string& source_code,
             const std::string& shared_library,
             Kind kind);
  void* GetFunction(const std::string& func_name);
  ~SYCLModule();

 private:
  //! sycl source code
  std::string source_code_;

  std::string shared_library_;
  // handler of the shared library
  void* so_handler_ = nullptr;
  //! Kind of the input.
  Kind kind_;
  std::mutex mutex_;
};

/**
 * Call a SYCL compiled kernel.
 *
 * @param kernel_fn the func pointer.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream);

}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn
