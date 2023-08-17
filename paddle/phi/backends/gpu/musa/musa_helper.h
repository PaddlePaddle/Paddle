// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace backends {
namespace gpu {

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (0 >= ((major)*1000 + (minor)*100 + (patch)))

#define CUDA_KERNEL_LOOP_TYPE(i, num, index_type)                    \
  int64_t __index__ =                                                \
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;   \
  int64_t __stride__ = static_cast<int64_t>(blockDim.x) * gridDim.x; \
  for (index_type i = __index__; __index__ < (num);                  \
       __index__ += __stride__, i = __index__)

}  // namespace gpu
}  // namespace backends
}  // namespace phi
