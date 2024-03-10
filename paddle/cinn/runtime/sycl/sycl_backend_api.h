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

#include <sycl/sycl.hpp>
#include "paddle/cinn/runtime/backend_api.h"
#include <vector>
#include "paddle/cinn/common/target.h"
using cinn::common::Target;

namespace cinn {
namespace runtime {
namespace Sycl {

inline const char* SYCLGetErrorString(std::error_code error_code) {
  sycl::errc error_code_value = static_cast<sycl::errc>(error_code.value());
  switch(error_code_value){
    case sycl::errc::success:
      return "SUCCESS";
    case sycl::errc::runtime:
      return "RUNTIME ERROR";
    case sycl::errc::kernel:
      return "KERNEL ERROR";
    case sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case sycl::errc::nd_range:
      return "NDRANGE ERROR";
    case sycl::errc::event:
      return "EVENT ERROR";
    case sycl::errc::kernel_argument:
      return "KERNEL ARGUMNET ERROR";
    case sycl::errc::build:
      return "BUILD ERROR";
    case sycl::errc::invalid:
      return "INVALID ERROR";
    case sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION";
    case sycl::errc::platform:
      return "PLATFORM ERROR";
    case sycl::errc::profiling:
      return "PROFILING ERROR";
    case sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case sycl::errc::kernel_not_supported:
      return "kERNEL NOT SUPPORTED";
    case sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";
    default:
        return "";
  }
}

/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CALL(func)                                                       \
  {                                                                           \
    try{                                                                      \
      func;                                                                   \
    }catch(const sycl::exception &e){                                         \
      CHECK(e.code() == sycl::errc::success) << "SYCL Error, code=" << ": " << SYCLGetErrorString(e.code()) <<", message:"<< e.what();;\
    }                                                                         \
  }

class SYCLBackendAPI final : public BackendAPI {
 public:
  SYCLBackendAPI(){};
  ~SYCLBackendAPI(){};
  static SYCLBackendAPI* Global();
  /*!
   * \brief
   * \param arch
   * \return return device0's_arch : arch if arch is Unk.
   */
  Target::Arch Init(Target::Arch arch);
  void set_device(int device_id) final;
  int get_device() final;
  std::variant<int, std::array<int, 3>> get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id = std::nullopt) final;
  void* malloc(size_t numBytes) final;
  // void set_active_devices(std::vector<int> device_ids) final;
  void free(void* data) final;
  void memset(void* data, int value, size_t numBytes) final;
  void memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type) final;
  void device_sync() final;
  void stream_sync(void* stream) final;
  sycl::queue* get_now_queue();
  std::string GetGpuVersion();

 private:
  Target::Arch arch;
  // all devices
  std::vector<sycl::device> devices;
  // all contexts
  std::vector<sycl::context*> contexts;
  // all queues in all devices
  std::vector<std::vector<sycl::queue*>> queues;
  // now_device_id, change by set_device()
  int now_device_id = -1;
  // whether the BackendAPI is initialized.
  bool initialized_{false};
};
}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn