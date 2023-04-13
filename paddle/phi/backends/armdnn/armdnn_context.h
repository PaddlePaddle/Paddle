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

#ifdef PADDLE_WITH_ARM_DNN_LIBRARY

#include <memory>
#include "paddle/phi/backends/armdnn/armdnn_type_traits.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"

namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen

namespace phi {

class ArmDNNContext : public CPUContext {
 public:
  explicit ArmDNNContext(const Place& place);
  ~ArmDNNContext();

  void* context() const;

  static const char* name() { return "ArmDNNContext"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi

#endif
