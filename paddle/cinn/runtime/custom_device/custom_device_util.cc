// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef CINN_WITH_CUSTOM_DEVICE
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"
#include <vector>
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/phi/backends/device_manager.h"

namespace cinn {
namespace runtime {
namespace custom_device {

void cinn_call_custom_device_kernel(void *kernel_fn,
                                    void *v_args,
                                    int num_args,
                                    int grid_x,
                                    int grid_y,
                                    int grid_z,
                                    int block_x,
                                    int block_y,
                                    int block_z,
                                    int shared_memory_bytes,
                                    void *stream) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  PADDLE_ENFORCE_EQ(dev_types.empty(),
                    false,
                    phi::errors::NotFound("No Custom Device type registered."));

  std::string dev_type = dev_types[0];
  int device_id = phi::DeviceManager::GetDevice(dev_type);
  auto place = phi::CustomPlace(dev_type, device_id);

  auto &plugin = CinnCustomDevicePlugin::GetInstance(place);
  auto *runtime_strategy = plugin.GetRuntime();

  VLOG(3) << "Launching kernel on " << dev_type << ":" << device_id << " Grid("
          << grid_x << "," << grid_y << "," << grid_z << ")"
          << " Block(" << block_x << "," << block_y << "," << block_z << ")";

  std::vector<void *> kernel_args;
  kernel_args.reserve(num_args);

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
        cinn_buffer_t *buffer = static_cast<cinn_buffer_t *>(args[idx]);
        kernel_args.emplace_back(&(buffer->memory));
      } else {
        kernel_args.emplace_back(args[idx].data_addr());
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("plugin_launch_kernel",
                                        cinn::utils::EventType::kInstruction);

    runtime_strategy->LaunchKernel(kernel_fn,
                                   "",  // func_name
                                   kernel_args.data(),
                                   num_args,
                                   grid_x,
                                   grid_y,
                                   grid_z,
                                   block_x,
                                   block_y,
                                   block_z,
                                   shared_memory_bytes,
                                   stream);
  }
}

void infer_shape_set_value(int row, int col, int64_t value, int64_t **v) {
  v[row][col] = value;
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
