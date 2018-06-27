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
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Output a dot file and write to some place.
 */
class DFG_GraphvizDrawPass : public DataFlowGraphPass {
 public:
  struct Config {
    Config(const std::string &dir, const std::string &id,
           bool display_deleted_node = false)
        : dir(dir), id(id), display_deleted_node(display_deleted_node) {}

    // The directory to store the .dot or .png files.
    const std::string dir;
    // The identifier for this dot file.
    const std::string id;
    // Whether to display deleted nodes, default false.
    const bool display_deleted_node;
  };

  DFG_GraphvizDrawPass(const Config &config) : config_(config) {}

  bool Initialize(Argument *argument) override { return true; }
  void Run(DataFlowGraph *graph) override;
  bool Finalize() override { return Pass::Finalize(); }

  std::string repr() const override { return "DFG graphviz drawer"; }
  std::string description() const override {
    return "Debug a DFG by draw with graphviz";
  }

 private:
  // Path of the dot file to output.
  std::string GenDotPath() const {
    return config_.dir + "/" + "graph_" + config_.id + ".dot";
  }

  std::string Draw(DataFlowGraph *graph);

  Config config_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
