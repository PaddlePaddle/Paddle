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

#pragma once
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {

// Read ProgramDesc from a __model__ file, defined in io.cc
extern void ReadBinaryFile(const std::string& filename, std::string* contents);

namespace analysis {

DEFINE_string(inference_model_dir, "", "inference test model dir");

static DataFlowGraph ProgramDescToDFG(
    const framework::proto::ProgramDesc& desc) {
  DataFlowGraph graph;
  FluidToDataFlowGraphPass pass;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  argument.origin_program_desc.reset(new framework::proto::ProgramDesc(desc));
  pass.Initialize(&argument);
  pass.Run(&graph);
  pass.Finalize();
  return graph;
}

class DFG_Tester : public ::testing::Test {
 protected:
  void SetUp() override {
    auto desc = LoadProgramDesc(FLAGS_inference_model_dir + "/__model__");
    argument.origin_program_desc.reset(new framework::proto::ProgramDesc(desc));
  }

  Argument argument;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
