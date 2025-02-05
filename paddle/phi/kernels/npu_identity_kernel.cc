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

#include "paddle/phi/kernels/npu_identity_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void NPUIdentityKernel(const Context& dev_ctx UNUSED,
                       const DenseTensor& x,
                       const int format UNUSED,
                       DenseTensor* out) {
  VLOG(4) << "npu_identity op is only for NPU, please avoid using this kernel!";
  out->ShareDataWith(x);
}

}  // namespace phi

/** [ Why need npu_identity op? ]
 *
 * 1. Ascend CANN use internal storage format for high performance
 * computing, for example if run BatchNorm2D op with CANN internal
 * storage format ACL_FORMAT_NC1HWC0, time costs in transdata will
 * be removed, and at will gain 2x performance improvement.
 *
 * 2.The internal storage format will use storage_properties_ in
 * DenseTensor, and will change the size and layout of denser, and
 * finally it should be called when change tensor to numpy and restore
 * original size and format by calling CANN Identity OP.
 *
 * TODO(qili93): remove this op after custom op and custom device
 * integrated and then move this op along with its code to plugin.
 */

PD_REGISTER_KERNEL(npu_identity,
                   CPU,
                   ALL_LAYOUT,
                   phi::NPUIdentityKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(npu_identity,
                   GPU,
                   ALL_LAYOUT,
                   phi::NPUIdentityKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
#endif
