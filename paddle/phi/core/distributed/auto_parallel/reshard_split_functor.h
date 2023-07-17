// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <vector>
#include "paddle/phi/api/include/tensor.h"

namespace phi {
class DeviceContext;
class DenseTensor;
class KernelKey;

namespace distributed {
namespace auto_parallel {

class ReshardSplitFunctor final {
 public:
  using SPLIT_KERNEL_SIG = void (*)(const DeviceContext&,
                                    const DenseTensor&,
                                    const phi::IntArray&,
                                    const phi::Scalar&,
                                    std::vector<DenseTensor*>);

  ReshardSplitFunctor(const KernelKey& kernel_key,
                      const IntArray& sections,
                      int64_t axis);

  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& input,
                  std::vector<DenseTensor>* output);

 private:
  IntArray sections_;
  int64_t axis_;
  SPLIT_KERNEL_SIG functor_;

  void PrepareOutput(const DenseTensor& input,
                     const std::vector<DenseTensor*>& output);
};

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
