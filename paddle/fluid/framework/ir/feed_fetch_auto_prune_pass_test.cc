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

#include "paddle/fluid/framework/ir/feed_fetch_auto_prune_pass.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>

DEFINE_string(modelfile, "", "model file");

namespace paddle {
namespace framework {
namespace ir {

TEST(FeedFetchAutoPrunePass, Basic) {
  // Read protobuf from file
  std::ifstream file(FLAGS_modelfile, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0);
  std::string buf(size, ' ');
  file.read(&buf[0], size);

  // Load into graph
  proto::ProgramDesc program;
  program.ParseFromString(buf);

  ProgramDesc program_desc(program);
  std::unique_ptr<Graph> graph(new Graph(program_desc));
  graph->Set(kFeedsAttr, new std::vector<std::string>(
                             {"firstw", "secondw", "thirdw", "forthw"}));
  graph->Set(kFetchesAttr, new std::vector<std::string>({"fc_1.tmp_0"}));

  auto pass = PassRegistry::Instance().Get("feed_fetch_auto_prune_pass");
  auto clean_pass = PassRegistry::Instance().Get("infer_clean_graph_pass");
  auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  auto graph_viz_pass1 = PassRegistry::Instance().Get("graph_viz_pass");

  graph_viz_pass->Set("graph_viz_path", new std::string("0.dot"));
  graph = clean_pass->Apply(std::move(graph));
  graph = graph_viz_pass->Apply(std::move(graph));
  graph = pass->Apply(std::move(graph));
  graph_viz_pass1->Set("graph_viz_path", new std::string("1.dot"));
  graph = graph_viz_pass1->Apply(std::move(graph));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(feed_fetch_auto_prune_pass);
USE_PASS(infer_clean_graph_pass);
USE_PASS(graph_viz_pass);
