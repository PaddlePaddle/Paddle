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

#include <string>

#include "paddle/phi/backends/c_comm_lib.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace detail {

// FIXME(paddle-dev): Since the singleton of ProcessGroup in fluid is used in
// SyncBN, the fluid symbol will be dependent on external hardware access.
// Here, the part that depends on the fluid symbol is individually encapsulated
// as a temporary function to isolate external symbol dependencies.
// In the future, the dependence on the singleton in fluid in SyncBN needs
// to be removed.
// In principle, the PHI Kernel cannot use the global singleton internally,
// and the required members need to be passed in from the eucalyptus tree.
ccl::CCLComm GetCCLComm(const Place& place, int global_gid = 0);

}  // namespace detail

template <typename T, typename Context>
void SyncBatchNormKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& scale,
                         const DenseTensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
                         const std::string& data_layout,
                         bool use_global_stats,
                         bool trainable_statistics,
                         DenseTensor* y,
                         DenseTensor* mean_out,
                         DenseTensor* variance_out,
                         DenseTensor* saved_mean,
                         DenseTensor* saved_variance,
                         DenseTensor* reserve_space);
}  // namespace phi
