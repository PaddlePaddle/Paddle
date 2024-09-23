// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void FusedSeqpoolCVMOpCPUKernel(const Context &dev_ctx,
                                const std::vector<const DenseTensor *> &x,
                                const DenseTensor &cvm,
                                const std::string &pooltype,
                                float pad_value,
                                bool use_cvm,
                                int cvm_offset,
                                std::vector<DenseTensor *> out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Unimplemented CPU kernel for FusedSeqpoolCVMOp, only support GPU "
      "now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_seqpool_cvm,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedSeqpoolCVMOpCPUKernel,
                   float) {}
