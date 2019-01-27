// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <fstream>
#include <memory>
#include <unordered_map>

#include "paddle/fluid/framework/details/multi_devices_helper.h"

namespace paddle {
namespace framework {
namespace details {

constexpr char kGraphvizPath[] = "debug_graphviz_path";
constexpr char kGraphviz[] = "graphviz";

class GraphvizNode {
 public:
  GraphvizNode(ir::Node* n, const int& i) : node_(n), id_(i) {}
  virtual ~GraphvizNode() = default;

 protected:
  ir::Node* node_;
  int id_;
};
class GraphvizNode;
typedef std::unordered_set<std::unique_ptr<GraphvizNode>> GraphvizNodes;

class SSAGraphPrinter {
 public:
  virtual ~SSAGraphPrinter() {}
  virtual void Print(const ir::Graph& graph, std::ostream& sout) const = 0;
};

class SSAGraphPrinterImpl : public SSAGraphPrinter {
 public:
  void Print(const ir::Graph& graph, std::ostream& sout) const override;

 private:
  std::unordered_map<ir::Node*, int> ToGraphvizNode(
      const ir::Graph& graph) const;
};

class SSAGraphPrintPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  mutable std::unique_ptr<SSAGraphPrinter> printer_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
