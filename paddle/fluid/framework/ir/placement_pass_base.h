/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Specifies which operators should use cuDNN.
 */
class Graph;

class PlacementPassBase : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  virtual const std::string GetPlacementName() const = 0;
  virtual const std::string GetAttrName() const = 0;
  virtual const std::unordered_set<std::string> GetOpTypesList() const = 0;

 private:
  bool IsSupport(const std::string& op_type) const;
  bool IsDefaultOpTypes(const std::string& op_type) const;

#if PADDLE_WITH_TESTING
  friend class PlacementPassTest;
#endif
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
