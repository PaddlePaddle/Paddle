/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/trieconcat_op.h"
#include "gtest/gtest.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/place.h"

TEST(TrieConcatOp, RemoveFromEnd) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;

  BeamHelper helper;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* b2 = new BeamNode(2, 2);
  BeamNode* b3 = new BeamNode(3, 3);

  b1->AppendTo(root);
  b2->AppendTo(root);
  b3->AppendTo(b1);

  helper.RemoveFromEnd(b3);
  helper.RemoveFromEnd(b2);
}

TEST(TrieConcatOp, AppendBeamNodeToResult) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* end = new BeamNode(2, 2);
  b1->AppendTo(root);
  end->AppendTo(b1);

  BeamHelper helper;

  helper.AppendBeamNodeToResult(0, end);

  ASSERT_EQ(helper.result_prob.at(0).size(), 1UL);
  for (size_t i = 0; i < helper.result_prob.at(0).at(0).size(); ++i) {
    float prob = helper.result_prob.at(0).at(0).at(i);
    ASSERT_EQ(prob, static_cast<float>(i));
    ASSERT_EQ(static_cast<float>(helper.result_id.at(0).at(0).at(i)), prob);
    std::cout << prob << " ";
    std::cout << std::endl;
  }
}

TEST(TrieConcatOp, InitFirstStepBeamNodes) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;
  using LoD = paddle::framework::LoD;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;

  CPUPlace place;

  std::vector<BeamNode*> prefixes;
  prefixes.push_back(new BeamNode(0, 0));
  prefixes.push_back(new BeamNode(1, 1));
  prefixes.push_back(new BeamNode(2, 2));

  std::vector<LoDTensor> beam_out_Ids;
  std::vector<LoDTensor> beam_out_Probs;

  LoD lod;
  lod.push_back(std::vector<size_t>{0, 3, 6});
  lod.push_back(std::vector<size_t>{0, 1, 2, 3, 4, 5, 6});

  // Ids
  LoDTensor tensor_id;
  tensor_id.set_lod(lod);
  tensor_id.Resize({6});
  // malloc memory
  int64_t* id_ptr = tensor_id.mutable_data<int64_t>(place);
  for (int64_t i = 0; i < 6; ++i) {
    id_ptr[i] = i;
  }

  // Probs
  LoDTensor tensor_prob;
  tensor_prob.set_lod(lod);
  tensor_prob.Resize({6});
  // malloc memory
  float* prob_ptr = tensor_prob.mutable_data<float>(place);
  for (int i = 0; i < 6; ++i) {
    prob_ptr[i] = static_cast<float>(i);
  }

  BeamHelper helper;
  std::unordered_map<size_t, std::vector<BeamNode*>> result;
  helper.InitFirstStepBeamNodes(tensor_id, tensor_prob, &result);

  ASSERT_EQ(result.size(), 2UL);
  ASSERT_EQ(result.at(0).size(), 3UL);
  for (size_t i = 0; i < result.at(0).size(); ++i) {
    auto* beam_node = result.at(0).at(i);
    ASSERT_EQ(beam_node->word_id_, static_cast<int64_t>(i));
    ASSERT_EQ(beam_node->prob_, static_cast<float>(i));
  }
  ASSERT_EQ(result.at(1).size(), 3UL);
  for (size_t i = 0; i < result.at(1).size(); ++i) {
    auto* beam_node = result.at(1).at(i);
    ASSERT_EQ(beam_node->word_id_, static_cast<int64_t>(i + 3));
    ASSERT_EQ(beam_node->prob_, static_cast<float>(i + 3));
  }
}

TEST(TrieConcatOp, PackAllSteps) {}
