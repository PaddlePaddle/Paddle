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

#include <glog/logging.h>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr char kGraphvizPath[] = "graph_viz_path";

class SSAGraphPrinter {
 public:
  virtual ~SSAGraphPrinter() {}
  virtual void Print(const ir::Graph& graph, std::ostream& sout) const = 0;
};

class GraphvizSSAGraphPrinter : public SSAGraphPrinter {
 public:
  void Print(const ir::Graph& graph, std::ostream& sout) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
