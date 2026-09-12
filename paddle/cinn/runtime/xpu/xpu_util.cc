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

#include "paddle/cinn/runtime/xpu/xpu_util.h"

#include <glog/logging.h>

#ifdef CINN_WITH_XPU
#include "xpu/xpurtc.h"
#endif

#include "paddle/cinn/runtime/xpu/xpu_module.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace xpu {

// ---------------------------------------------------------------------------
// Parameter serialisation helpers (mirrors xpurtc::detail::SafeParamSerializer)
//
// xpurtc uses the following alignment rules for the param buffer:
//   XPU arch <= 2: min_align=4, max_align=4, min_size=4
//   XPU arch >= 3: min_align=4, max_align=8, min_size=4
//
// For M100 (arch=4) we use max_align=8.
// ---------------------------------------------------------------------------

namespace {

static constexpr uint32_t kMinAlign = 4;
static constexpr uint32_t kMaxAlign = 8;  // XPU arch >= 3

inline uint32_t AlignUp(uint32_t offset, uint32_t align) {
  return (offset + align - 1) & ~(align - 1);
}

// Returns the alignment for a value of byte size `size`.
inline uint32_t ParamAlign(uint32_t size) {
  uint32_t a = std::min(std::max(size, kMinAlign), kMaxAlign);
  return a;
}

// Returns the slot size (padded to at least 4 bytes).
inline uint32_t ParamSlotSize(uint32_t size) {
  return std::max(size, kMinAlign);
}

}  // namespace

void cinn_call_xpu_kernel(void* kernel_fn,
                          void* v_args,
                          int num_args,
                          int grid_x,
                          int grid_y,
                          int grid_z,
                          int block_x,
                          int block_y,
                          int block_z,
                          int shared_memory_bytes,
                          void* stream) {
  // `kernel_fn` is a pointer to an XpuModule, registered via
  // xpu_intrinsics.cc: RegisterVar(kernel_fn_name + "_ptr_", module_ptr).
  XpuModule* module = static_cast<XpuModule*>(kernel_fn);

  int current_device_id = 0;
#ifdef CINN_WITH_XPU
  cudaGetDevice(&current_device_id);
#endif
  VLOG(3) << "cinn_call_xpu_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args
          << ", shared_memory_bytes=" << shared_memory_bytes
          << ", stream=" << stream << ", module=" << module << " on device "
          << current_device_id;

  // ---------------------------------------------------------------------------
  // Serialise arguments into a flat byte buffer.
  // This mirrors xpurtc::detail::SafeParamSerializer for arch >= 3.
  //
  // Each argument is stored at an offset aligned to
  // min(max(sizeof(arg), kMinAlign), kMaxAlign), with a slot of at least
  // kMinAlign bytes.
  //
  // For pointer arguments (cinn_buffer_t*), we pass the device memory pointer
  // stored inside the buffer (8 bytes on 64-bit).
  // For scalar arguments, we pass the raw value.
  // ---------------------------------------------------------------------------

  // First pass: compute total buffer size
  cinn_pod_value_t* args = static_cast<cinn_pod_value_t*>(v_args);
  uint32_t total_size = 0;
  for (int idx = 0; idx < num_args; ++idx) {
    uint32_t arg_size;
    if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
      arg_size = sizeof(void*);  // device pointer
    } else {
      // Scalar: use the raw data size stored in the pod value.
      // All CINN scalars are at most 8 bytes; use 8 as a safe upper bound.
      arg_size = 8;
    }
    uint32_t align = ParamAlign(arg_size);
    total_size = AlignUp(total_size, align);
    total_size += ParamSlotSize(arg_size);
  }

  std::vector<uint8_t> param_buf(total_size, 0);

  // Second pass: write values
  {
    cinn::utils::RecordEvent record_run("prepare_xpu_args",
                                        cinn::utils::EventType::kInstruction);
    uint32_t offset = 0;
    for (int idx = 0; idx < num_args; ++idx) {
      void* src = nullptr;
      void* dev_ptr = nullptr;
      uint32_t arg_size;

      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
        // Extract the device memory pointer from the buffer.
        dev_ptr = reinterpret_cast<cinn_buffer_t*>(
                      static_cast<cinn_buffer_t*>(args[idx]))
                      ->memory;
        src = &dev_ptr;
        arg_size = sizeof(void*);
      } else {
        src = args[idx].data_addr();
        arg_size = 8;
      }

      uint32_t align = ParamAlign(arg_size);
      offset = AlignUp(offset, align);
      std::memcpy(&param_buf[offset], src, arg_size);
      offset += ParamSlotSize(arg_size);
    }
  }

  {
    cinn::utils::RecordEvent record_run("xpurtc_launch_kernel",
                                        cinn::utils::EventType::kInstruction);
    // For M100: ncluster = grid_x (1-D grid), ncore = block_x * block_y *
    // block_z. CINN currently generates 1-D or 3-D grids; pass grid_x as
    // ncluster and block_x * block_y * block_z as ncore to match the xpurtc
    // convention.
    int ncluster = grid_x;
    int ncore = block_x * block_y * block_z;
    module->Launch(ncluster,
                   ncore,
                   stream,
                   param_buf.data(),
                   static_cast<uint32_t>(param_buf.size()));
  }
}

void infer_shape_set_value(int row, int col, int64_t value, int64_t** v) {
  v[row][col] = value;
}

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
