/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file create an DFG_GraphvizDrawPass which helps to draw a data flow
 * graph's structure using graphviz.
 */

#pragma once

#include <fstream>
#include <string>
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Output a dot file and write to some place.
 */
class DFG_GraphvizDrawPass : public DataFlowGraphPass {
 public:
  DFG_GraphvizDrawPass(const std::string& dir, const std::string& id)
      : dir_(dir), id_(id) {}

  bool Initialize() override { return Pass::Initialize(); }
  void Run(DataFlowGraph* graph) override {
    auto content = Draw(graph);
    std::ofstream file(GenDotPath());
    file.write(content.c_str(), content.size());
    file.close();
    LOG(INFO) << "draw dot to " << GenDotPath();
  }

  bool Finalize() override { return Pass::Finalize(); }

  Pass* CreatePrinterPass(std::ostream& os,
                          const std::string& banner) const override {
    return nullptr;
  }

 private:
  // Path of the dot file to output.
  std::string GenDotPath() const {
    return dir_ + "/" + "graph_" + id_ + ".dot";
  }

  std::string Draw(DataFlowGraph* graph) { return graph->DotString(); }

  std::string dir_;
  std::string id_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
