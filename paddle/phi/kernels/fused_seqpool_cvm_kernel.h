// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

#pragma once

#include <string>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedSeqpoolCVMCUDAKernel(const Context &dev_ctx,
                               const std::vector<const DenseTensor *> &x,
                               const DenseTensor &cvm,
                               const std::string &pooltype,
                               float pad_value,
                               bool use_cvm,
                               int cvm_offset,
                               std::vector<DenseTensor *> out);

}  // namespace fusion
}  // namespace phi
