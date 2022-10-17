// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"

#include <set>
#include <thread>

#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"

DECLARE_bool(new_executor_serial_run);

namespace paddle {
namespace framework {
namespace interpreter {

static constexpr size_t kHostNumThreads = 4;
static constexpr size_t kDeviceNumThreads = 1;
static constexpr size_t kNumGcThreads = 1;

// By default, one interpretercore contains:
// 1-size thread pool for device kernel launch (or 0 for cpu execution),
// 1-size thread pool for host kernel launch (or more if the system contains
// enough processors).

// Note that the purpose of the config is to limit the total 'possible'
// threads introduced by interpretercore to avoid hurting performance.

inline std::tuple<int, int> GetThreadPoolConfig(const phi::Place& place,
                                                size_t op_num) {
  int num_device_threads = kDeviceNumThreads,
      num_host_threads = kHostNumThreads;

  int device_count = 0, processor_count = 0;
  if (platform::is_cpu_place(place)) {
    num_device_threads = 0;
    num_host_threads = 4;
  } else {
    processor_count = std::thread::hardware_concurrency();
    if (processor_count) {
      if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        device_count = phi::backends::gpu::GetGPUDeviceCount();
#endif
      }
      if (platform::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
        device_count = phi::backends::xpu::GetXPUDeviceCount();
#endif
      }
      if (platform::is_npu_place(place)) {
#if defined(PADDLE_WITH_ASCEND_CL)
        device_count = platform::GetNPUDeviceCount();
#endif
      }
      if (platform::is_ipu_place(place)) {
#if defined(PADDLE_WITH_IPU)
        device_count = platform::GetIPUDeviceCount();
#endif
      }
      if (platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
        device_count =
            phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
#endif
      }

      // Tricky implementation.
      // In multi-card training, each card may set env like
      // CUDA_VISIBLE_DEVICE=0 In that case, device_count is set to 8.
      if (device_count == 1) {
        device_count = 8;  // in many case, the accelerator has 8 cards.
      }

      // We expect processor_count = 2 * (the possible total threads when doing
      // multi-card training), to make sure that the system will not slow down
      // because of too many threads. Here, 2 is experience value. Since each
      // device has one interpretercore, the possible total threads when doing
      // multi-card training = device_count * (the possible total threads in one
      // interpretercore).

      if (device_count) {
        auto num = processor_count / device_count / 2 -
                   (kNumGcThreads + num_device_threads);
        num_host_threads =
            num > 0 ? (num > kHostNumThreads ? kHostNumThreads : num) : 1;
      }
    }
  }

  // In serial run, only one 1-size thread pool is used
  if (FLAGS_new_executor_serial_run) {
    num_host_threads = 0;
    num_device_threads = 1;
  }

  VLOG(4) << "place:" << place << ", processor_count:" << processor_count
          << ", device_count:" << device_count
          << ", serial_run:" << FLAGS_new_executor_serial_run
          << ", num_host_threads:" << num_host_threads
          << ", num_device_threads:" << num_device_threads;

  return std::make_tuple(num_host_threads, num_device_threads);
}

ExecutionConfig::ExecutionConfig(const phi::Place& place, size_t op_num) {
  std::tie(host_num_threads, deivce_num_threads) =
      GetThreadPoolConfig(place, op_num);
}

void ExecutionConfig::Log(int log_level) {
  VLOG(log_level) << "ExecutionConfig:";
  VLOG(log_level) << "used_for_jit = " << used_for_jit;
  VLOG(log_level) << "create_local_scope = " << create_local_scope;
  VLOG(log_level) << "host_num_threads = " << host_num_threads;
  VLOG(log_level) << "deivce_num_threads = " << deivce_num_threads;
  VLOG(log_level) << "skip_gc_vars = ";
  for (const std::string& var : skip_gc_vars) {
    VLOG(log_level) << var;
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
