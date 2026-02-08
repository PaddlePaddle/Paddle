// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

namespace c10 {
/**
 * This legacy enum class defines the set of backends supported by old school,
 * code generated Type-based ATen.  A "backend" in this sense roughly
 * corresponds to the cartesian product of (device type, layout), but restricted
 * only to combinations which we actually have kernels for.  Backend does NOT
 * include dtype.
 *
 * The reason we are sunsetting this enum class is because it doesn't allow for
 * open registration; e.g., if you want to add SparseXLA, you'd have to
 * edit this enum; you wouldn't be able to do it out of tree.  DispatchKey is
 * the replacement for Backend which supports open registration.
 *
 * NB: The concept of 'Backend' here disagrees with the notion of backend
 * exposed to users in torch.backends.  Backend here is something like "CPU"
 * or "SparseCUDA"; backend in torch.backends is something like "MKL" or
 * "CUDNN".
 */

enum class Backend {
  CPU,
  CUDA,
  HIP,
  VE,
  FPGA,
  IPU,
  XPU,
  SparseCPU,
  SparseCUDA,
  SparseCsrCPU,
  SparseCsrCUDA,
  SparseCsrMPS,
  SparseMPS,
  SparseHIP,
  SparseVE,
  SparseXPU,
  SparsePrivateUse1,
  SparseCsrHIP,
  SparseCsrVE,
  SparseCsrXPU,
  SparseCsrPrivateUse1,
  MAIA,
  XLA,
  Vulkan,
  Metal,
  Meta,
  QuantizedCPU,
  QuantizedCUDA,
  QuantizedXPU,
  QuantizedPrivateUse1,
  Undefined,
  MkldnnCPU,
  MPS,
  HPU,
  Lazy,
  MTIA,
  PrivateUse1,
  NumOptions
};

}  // namespace c10
