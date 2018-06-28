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
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {

// Read ProgramDesc from a __model__ file, defined in io.cc
extern void ReadBinaryFile(const std::string& filename, std::string* contents);

namespace analysis {

DEFINE_string(inference_model_dir, "", "inference test model dir");

static framework::proto::ProgramDesc LoadProgramDesc(
    const std::string& model_dir = FLAGS_inference_model_dir) {
  std::string msg;
  std::string net_file = FLAGS_inference_model_dir + "/__model__";
  std::ifstream fin(net_file, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", net_file);
  fin.seekg(0, std::ios::end);
  msg.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(msg.at(0)), msg.size());
  fin.close();
  framework::proto::ProgramDesc program_desc;
  program_desc.ParseFromString(msg);
  return program_desc;
}

static DataFlowGraph ProgramDescToDFG(
    const framework::proto::ProgramDesc& desc) {
  DataFlowGraph graph;
  FluidToDataFlowGraphPass pass;
  Argument argument;
  argument.origin_program_desc.reset(new framework::proto::ProgramDesc(desc));
  pass.Initialize(&argument);
  pass.Run(&graph);
  pass.Finalize();
  return graph;
}

class DFG_Tester : public ::testing::Test {
 protected:
  void SetUp() override {
    auto desc = LoadProgramDesc(FLAGS_inference_model_dir);
    argument.origin_program_desc.reset(new framework::proto::ProgramDesc(desc));
  }

  Argument argument;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
