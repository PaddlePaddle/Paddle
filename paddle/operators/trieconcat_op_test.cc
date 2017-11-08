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
  using BeamNode = paddle::operators::BeamNode;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* b2 = new BeamNode(2, 2);
  BeamNode* b3 = new BeamNode(3, 3);

  b1->AppendTo(root);
  b2->AppendTo(root);
  b3->AppendTo(b1);

  BeamNode::RemoveFromEnd(b3);
  BeamNode::RemoveFromEnd(b2);
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
  std::unordered_map<size_t, std::vector<std::vector<int64_t>>> result_id;
  std::unordered_map<size_t, std::vector<std::vector<float>>> result_score;

  helper.AppendBeamNodeToResult(0, end, &result_id, &result_score);

  ASSERT_EQ(result_score.at(0).size(), 1UL);
  for (size_t i = 0; i < result_score.at(0).at(0).size(); ++i) {
    float score = result_score.at(0).at(0).at(i);
    ASSERT_EQ(score, static_cast<float>(i));
    ASSERT_EQ(static_cast<float>(result_id.at(0).at(0).at(i)), score);
  }
}

TEST(TrieConcatOp, InitFirstStepBeamNodes) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;
  using LoD = paddle::framework::LoD;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;

  CPUPlace place;

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

  // Scores
  LoDTensor tensor_score;
  tensor_score.set_lod(lod);
  tensor_score.Resize({6});
  // malloc memory
  float* score_ptr = tensor_score.mutable_data<float>(place);
  for (int i = 0; i < 6; ++i) {
    score_ptr[i] = static_cast<float>(i);
  }

  BeamHelper helper;
  std::unordered_map<size_t, std::vector<BeamNode*>> result;
  helper.InitFirstStepBeamNodes(tensor_id, tensor_score, &result);

  ASSERT_EQ(result.size(), 2UL);
  ASSERT_EQ(result.at(0).size(), 3UL);
  for (size_t i = 0; i < result.at(0).size(); ++i) {
    auto* beam_node = result.at(0).at(i);
    ASSERT_EQ(beam_node->word_id_, static_cast<int64_t>(i));
    ASSERT_EQ(beam_node->score_, static_cast<float>(i));
  }
  ASSERT_EQ(result.at(1).size(), 3UL);
  for (size_t i = 0; i < result.at(1).size(); ++i) {
    auto* beam_node = result.at(1).at(i);
    ASSERT_EQ(beam_node->word_id_, static_cast<int64_t>(i + 3));
    ASSERT_EQ(beam_node->score_, static_cast<float>(i + 3));
  }
}

TEST(TrieConcatOp, PackTwoBeamStepOut) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;
  using LoD = paddle::framework::LoD;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;

  CPUPlace place;

  // we have 2 source prefix to handle
  LoD lod;
  lod.push_back(std::vector<size_t>{0, 3, 5});
  // the first source prefix have 4 candidate, the second have 2 candidate
  lod.push_back(std::vector<size_t>{0, 1, 1, 4, 4, 6});

  // Ids
  LoDTensor tensor_id;
  tensor_id.set_lod(lod);
  tensor_id.Resize({6});
  // malloc memory
  int64_t* id_ptr = tensor_id.mutable_data<int64_t>(place);
  for (int64_t i = 0; i < 6; ++i) {
    id_ptr[i] = i;
  }

  // Scores
  LoDTensor tensor_score;
  tensor_score.set_lod(lod);
  tensor_score.Resize({6});
  // malloc memory
  float* score_ptr = tensor_score.mutable_data<float>(place);
  for (int i = 0; i < 6; ++i) {
    score_ptr[i] = static_cast<float>(i);
  }

  // result should be:
  // [1 0]
  // vector<BeamNode*> should be:
  // [3 1] [3 2] [3 3]

  // three prefix
  std::vector<BeamNode*> prefixes1;
  prefixes1.push_back(new BeamNode(1, 1));
  prefixes1.push_back(new BeamNode(2, 2));
  prefixes1.push_back(new BeamNode(3, 3));

  BeamHelper helper1;
  std::unordered_map<size_t, std::vector<std::vector<int64_t>>> result_id_1;
  std::unordered_map<size_t, std::vector<std::vector<float>>> result_score_1;
  std::vector<BeamNode*> vec = helper1.PackTwoBeamStepOut(
      0, prefixes1, tensor_id, tensor_score, &result_id_1, &result_score_1);
  ASSERT_EQ(vec.size(), 3UL);
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_EQ(vec.at(i)->word_id_, static_cast<int64_t>(i + 1));
    ASSERT_EQ(vec.at(i)->score_, static_cast<float>(i + 1));
  }

  ASSERT_EQ(result_id_1.at(0).size(), 1UL);
  std::vector<int64_t> id_res = {1, 0};
  ASSERT_EQ(result_id_1.at(0).at(0), id_res);

  ASSERT_EQ(result_score_1.at(0).size(), 1UL);
  std::vector<float> score_res = {1, 0};
  ASSERT_EQ(result_score_1.at(0).at(0), score_res);

  // two prefix
  // result should be:
  // [2 4] [2 5]
  std::vector<BeamNode*> prefixes2;
  prefixes2.push_back(new BeamNode(1, 1));
  prefixes2.push_back(new BeamNode(2, 2));

  BeamHelper helper2;
  std::unordered_map<size_t, std::vector<std::vector<int64_t>>> result_id_2;
  std::unordered_map<size_t, std::vector<std::vector<float>>> result_score_2;
  std::vector<BeamNode*> vec2 = helper2.PackTwoBeamStepOut(
      1, prefixes2, tensor_id, tensor_score, &result_id_2, &result_score_2);
  ASSERT_EQ(vec2.size(), 2UL);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(vec2.at(i)->word_id_, static_cast<int64_t>(i + 4));
    ASSERT_EQ(vec2.at(i)->father_->word_id_, static_cast<int64_t>(2));
    ASSERT_EQ(vec2.at(i)->score_, static_cast<float>(i + 4));
    ASSERT_EQ(vec2.at(i)->father_->score_, static_cast<float>(2));
  }
  ASSERT_EQ(result_id_2.size(), 0UL);
  ASSERT_EQ(result_score_2.size(), 0UL);
}

namespace paddle {
namespace test {
using LoD = paddle::framework::LoD;
using CPUPlace = paddle::platform::CPUPlace;
using LoDTensor = paddle::framework::LoDTensor;

void GenerateExample(const std::vector<size_t>& level_0,
                     const std::vector<size_t>& level_1,
                     const std::vector<int>& data, std::vector<LoDTensor>* ids,
                     std::vector<LoDTensor>* scores) {
  PADDLE_ENFORCE_EQ(level_0.back(), level_1.size() - 1, "");
  PADDLE_ENFORCE_EQ(level_1.back(), data.size(), "");

  CPUPlace place;

  LoD lod;
  lod.push_back(level_0);
  lod.push_back(level_1);

  // Ids
  LoDTensor tensor_id;
  tensor_id.set_lod(lod);
  tensor_id.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  int64_t* id_ptr = tensor_id.mutable_data<int64_t>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    id_ptr[i] = static_cast<int64_t>(data.at(i));
  }

  // Scores
  LoDTensor tensor_score;
  tensor_score.set_lod(lod);
  tensor_score.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  float* score_ptr = tensor_score.mutable_data<float>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    score_ptr[i] = static_cast<float>(data.at(i));
  }

  ids->push_back(tensor_id);
  scores->push_back(tensor_score);
}

}  // namespace test
}  // namespace paddle

TEST(TrieConcatOp, PackAllSteps) {
  using BeamHelper = paddle::operators::BeamHelpter;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;
  using LoD = paddle::framework::LoD;

  CPUPlace place;

  // we will constuct a sample data with 3 steps and 2 source sentences
  std::vector<LoDTensor> ids;
  std::vector<LoDTensor> scores;

  paddle::test::GenerateExample(
      std::vector<size_t>{0, 3, 6}, std::vector<size_t>{0, 1, 2, 3, 4, 5, 6},
      std::vector<int>{1, 2, 3, 4, 5, 6}, &ids, &scores);
  paddle::test::GenerateExample(
      std::vector<size_t>{0, 3, 6}, std::vector<size_t>{0, 1, 1, 3, 5, 5, 6},
      std::vector<int>{0, 1, 2, 3, 4, 5}, &ids, &scores);
  paddle::test::GenerateExample(std::vector<size_t>{0, 2, 5},
                                std::vector<size_t>{0, 1, 2, 3, 4, 5},
                                std::vector<int>{0, 1, 2, 3, 4}, &ids, &scores);

  ASSERT_EQ(ids.size(), 3UL);
  ASSERT_EQ(scores.size(), 3UL);

  BeamHelper helper;

  LoDTensor id_tensor;
  LoDTensor score_tensor;
  helper.PackAllSteps(ids, scores, &id_tensor, &score_tensor);

  LoD lod = id_tensor.lod();
  std::vector<size_t> expect_source_lod = {0, 3, 6};
  EXPECT_EQ(lod[0], expect_source_lod);
  std::vector<size_t> expect_sentence_lod = {0, 2, 5, 8, 11, 14, 17};
  EXPECT_EQ(lod[1], expect_sentence_lod);
  std::vector<int> expect_data = {1, 0, 3, 1, 0, 3, 2, 1, 4,
                                  3, 2, 4, 4, 3, 6, 5, 4};
  ASSERT_EQ(id_tensor.dims()[0], static_cast<int64_t>(expect_data.size()));
  for (size_t i = 0; i < expect_data.size(); ++i) {
    ASSERT_EQ(id_tensor.data<int64_t>()[i],
              static_cast<int64_t>(expect_data[i]));
  }

  for (size_t level = 0; level < 2; ++level) {
    for (size_t i = 0; i < lod[level].size(); ++i) {
      std::cout << lod[level][i] << ", ";
    }
    std::cout << std::endl;
  }
  for (int64_t i = 0; i < id_tensor.dims()[0]; ++i) {
    std::cout << id_tensor.data<int64_t>()[i] << ", ";
  }
  std::cout << std::endl;
  for (int64_t i = 0; i < score_tensor.dims()[0]; ++i) {
    std::cout << score_tensor.data<float>()[i] << ", ";
  }
  std::cout << std::endl;
}
