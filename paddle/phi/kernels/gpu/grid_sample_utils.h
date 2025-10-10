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

#pragma once

#include <limits.h>

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif
namespace phi {

enum class Mode {
  bilinear,
  nearest,
};

template <typename T>
__forceinline__ __device__ T SafeDownGradeToIntRange(T x) {
  bool unsafe_cond =
      x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x));
  return unsafe_cond ? static_cast<T>(-100.0) : x;
}

enum class PaddingMode { zeros, border, reflect };

static __forceinline__ __device__ bool InBounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ bool InBounds3D(
    int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

inline bool cudnnIsAvailable() {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  // Get all custom device types
  auto custom_device_types = phi::DeviceManager::GetAllCustomDeviceTypes();

  // Use the first custom device type
  if (!custom_device_types.empty()) {
    const std::string& device_type = custom_device_types[0];
    // Get current device ID for this device type
    int device_id = phi::DeviceManager::GetDevice(device_type);
    // Create place for the current device
    phi::Place place(phi::CustomPlace(device_type, device_id));
    // Check if this device has DNN support
    return phi::DeviceManager::IsDnnAvailable(place);
  }
  return false;
#elif defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // cuDNN/MIOpen version > 0 means DNN lib loaded; require v7+ for sampler
  return phi::backends::gpu::DnnVersion() >= 7000;
#else
  return false;
#endif
}

inline bool isGpuTensor(const phi::DenseTensor& x) {
  return phi::is_gpu_place(x.place());
}

inline bool canUse32bitIndexMath(const phi::DenseTensor& x) {
  auto elements = x.numel();
  int64_t max_elem = static_cast<int64_t>(std::numeric_limits<int>::max());

  if (elements > max_elem) {
    return false;
  }

  auto dims = x.dims();
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] > max_elem) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline bool condCudnnGridSampler(const phi::DenseTensor& input,
                                 const phi::DenseTensor& grid) {
  if (!cudnnIsAvailable()) return false;
  if (!isGpuTensor(input) || !isGpuTensor(grid)) return false;
  if (!(std::is_same<T, float>::value || std::is_same<T, double>::value))
    return false;
  if (!canUse32bitIndexMath(input) || !canUse32bitIndexMath(grid)) return false;

  // Only 4-D NCHW input is supported by cuDNN sampler path here
  auto in_dims = input.dims();
  if (in_dims.size() != 4) return false;

  // Channel constraint to match PyTorch guard: C <= 1024
  if (in_dims[1] > 1024) return false;

  return true;
}
}  // namespace phi
