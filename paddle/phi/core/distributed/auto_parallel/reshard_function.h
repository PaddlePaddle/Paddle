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
#include <memory>

namespace phi {
class DeviceContext;

namespace distributed {
namespace auto_parallel {
class TensorDistAttr;
}  // namespace auto_parallel

class DistTensor;
using auto_parallel::TensorDistAttr;

class ReshardFunction {
 public:
  ReshardFunction() = default;
  virtual ~ReshardFunction() = default;

  virtual bool IsSuitable(
      const DistTensor& in,
      const std::shared_ptr<TensorDistAttr>& out_dist_attr) = 0;

  virtual std::shared_ptr<DistTensor> Eval(
      const DeviceContext& dev_ctx,
      const DistTensor& in,
      const std::shared_ptr<TensorDistAttr>& out_dist_attr) = 0;
};

}  // namespace distributed
}  // namespace phi
