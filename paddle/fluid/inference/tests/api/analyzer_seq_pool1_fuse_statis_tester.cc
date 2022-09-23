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

#include <algorithm>
#include <fstream>
#include <iostream>

#include "paddle/fluid/inference/tests/api/analyzer_seq_pool1_tester_helper.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace seq_pool1_tester {

// Check the fuse status
TEST(Analyzer_seq_pool1_fuse_statis, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(predictor.get(), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  ASSERT_TRUE(fuse_statis.count("seqpool_concat_fuse"));
  ASSERT_TRUE(fuse_statis.count("squared_mat_sub_fuse"));
  ASSERT_TRUE(fuse_statis.count("repeated_fc_relu_fuse"));
  ASSERT_EQ(fuse_statis.at("fc_fuse"), 10);
  EXPECT_EQ(fuse_statis.at("seqpool_concat_fuse"), 2);
  EXPECT_EQ(fuse_statis.at("squared_mat_sub_fuse"), 0);
  EXPECT_EQ(fuse_statis.at("repeated_fc_relu_fuse"), 2);
  LOG(INFO) << "num_ops: " << num_ops;
  EXPECT_EQ(num_ops, 183);
}

}  // namespace seq_pool1_tester
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
