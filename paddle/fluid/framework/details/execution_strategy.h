// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstddef>  // for size_t

namespace paddle {
namespace framework {
namespace details {

struct ExecutionStrategy {
  enum ExecutorType { kDefault = 0, kExperimental = 1 };

  size_t num_threads_{0};
  bool use_cuda_{true};
  bool allow_op_delay_{false};
  size_t num_iteration_per_drop_scope_{1};
  ExecutorType type_{kDefault};
  bool dry_run_{false};
};

}  //  namespace details
}  //  namespace framework
}  //  namespace paddle
