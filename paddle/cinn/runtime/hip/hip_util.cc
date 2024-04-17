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

#include "paddle/cinn/runtime/hip/hip_util.h"
#include <glog/logging.h>
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace hip {

void cinn_call_hip_kernel(void *kernel_fn,
                          void *v_args,
                          int num_args,
                          int grid_x,
                          int grid_y,
                          int grid_z,
                          int block_x,
                          int block_y,
                          int block_z,
                          void *stream) {
  VLOG(3) << "cinn_call_hip_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args
          << ", stream=" << stream;

  std::vector<void *> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
        kernel_args.emplace_back(
            &((cinn_buffer_t *)(args[idx]))->memory);  // NOLINT
      } else {
        kernel_args.emplace_back(args[idx].data_addr());
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("hipLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    HIP_DRIVER_CALL(hipModuleLaunchKernel(static_cast<hipFunction_t>(kernel_fn),
                                          grid_x,
                                          grid_y,
                                          grid_z,
                                          block_x,
                                          block_y,
                                          block_z,
                                          0,  // share memory
                                          static_cast<hipStream_t>(stream),
                                          kernel_args.data(),
                                          nullptr))
  }
}

void cinn_call_hip_memset(
    void *v_args, int num_args, int value, size_t count, void *stream) {
  PADDLE_ENFORCE_EQ(num_args,
                    1,
                    phi::errors::PreconditionNotMet(
                        "The cinn_call_hip_memset only accept a output."));
  VLOG(4) << "call cinn_call_hip_memset with value=" << value
          << ", count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *output = args[0].operator cinn_buffer_t *()->memory;

  hipStream_t custream = static_cast<hipStream_t>(stream);

  HIP_CALL(hipMemsetAsync(output, value, count, custream));
}

void cinn_call_hip_memcpy(void *v_args,
                          int num_args,
                          size_t count,
                          void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      phi::errors::PreconditionNotMet(
          "The cinn_call_hip_memset only accept a input and a output."));
  VLOG(4) << "call cinn_call_hip_memcpy with count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *input = args[0].operator cinn_buffer_t *()->memory;
  void *output = args[1].operator cinn_buffer_t *()->memory;

  hipStream_t hipstream = static_cast<hipStream_t>(stream);
  HIP_CALL(
      hipMemcpyAsync(output, input, count, hipMemcpyDeviceToDevice, hipstream));
}

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
