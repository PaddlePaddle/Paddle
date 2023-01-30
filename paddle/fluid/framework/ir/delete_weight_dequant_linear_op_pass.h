<<<<<<< HEAD
/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/ir/pass.h"
=======
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
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace framework {
namespace ir {

<<<<<<< HEAD
class Graph;

class DeleteWeightDequantLinearOpPass : public Pass {
=======
class DeleteWeightQuantDequantLinearOpPass : public FusePassBase {
 public:
  DeleteWeightQuantDequantLinearOpPass();
  virtual ~DeleteWeightQuantDequantLinearOpPass() {}

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
