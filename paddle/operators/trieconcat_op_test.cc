/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

namespace paddle {
namespace test {
using LoD = paddle::framework::LoD;
using CPUPlace = paddle::platform::CPUPlace;
using LoDTensor = paddle::framework::LoDTensor;

void GenerateExample(const std::vector<size_t>& level_0,
                     const std::vector<size_t>& level_1,
                     const std::vector<int>& data, std::vector<LoDTensor>* ids,
                     std::vector<LoDTensor>* scores) {
  PADDLE_ENFORCE_EQ(level_0.back(), level_1.size() - 1,
                    "source level is used to describe candidate set");
  PADDLE_ENFORCE_EQ(level_1.back(), data.size(),
                    "the lowest level is used to describe data"
                    ", so it's last element should be data length");

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

TEST(TrieConcatOp, DeleteBeamNode) {
  using BeamNode = paddle::operators::BeamNode;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* b2 = new BeamNode(2, 2);
  BeamNode* b3 = new BeamNode(3, 3);

  b1->AppendTo(root);
  b2->AppendTo(root);
  b3->AppendTo(b1);

  delete b3;
  delete b2;
}

TEST(TrieConcatOp, MakeSentence) {
  using BeamHelper = paddle::operators::BeamHelper;
  using BeamNode = paddle::operators::BeamNode;
  using Sentence = paddle::operators::Sentence;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* end = new BeamNode(2, 2);
  b1->AppendTo(root);
  end->AppendTo(b1);

  BeamHelper helper;
  Sentence sentence = helper.MakeSentence(end);
  delete end;

  std::vector<int64_t> expect_ids = {0, 1, 2};
  ASSERT_EQ(sentence.word_ids, expect_ids);

  std::vector<float> expect_scores = {0, 1, 2};
  ASSERT_EQ(sentence.scores, expect_scores);
}

TEST(TrieConcatOp, PackTwoStepsFistStep) {
  using BeamHelper = paddle::operators::BeamHelper;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;
  using BeamNodeVector = paddle::operators::BeamNodeVector;
  using SentenceVector = paddle::operators::SentenceVector;

  CPUPlace place;

  std::vector<LoDTensor> ids;
  std::vector<LoDTensor> scores;

  paddle::test::GenerateExample(
      std::vector<size_t>{0, 2, 6}, std::vector<size_t>{0, 1, 2, 3, 4, 5, 6},
      std::vector<int>{1, 2, 3, 4, 5, 6}, &ids, &scores);

  std::vector<BeamNodeVector> beamnode_vector_list;
  std::vector<SentenceVector> sentence_vector_list(2, SentenceVector());

  BeamHelper helper;
  beamnode_vector_list = helper.PackTwoSteps(
      ids[0], scores[0], beamnode_vector_list, &sentence_vector_list);
  ASSERT_EQ(beamnode_vector_list.size(), 2UL);
  ASSERT_EQ(beamnode_vector_list[0].size(), 2UL);
  ASSERT_EQ(beamnode_vector_list[1].size(), 4UL);
}

TEST(TrieConcatOp, PackTwoSteps) {
  using BeamHelper = paddle::operators::BeamHelper;
  using BeamNode = paddle::operators::BeamNode;
  using BeamNodeVector = paddle::operators::BeamNodeVector;
  using SentenceVector = paddle::operators::SentenceVector;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;

  CPUPlace place;

  // first source has three prefix
  BeamNodeVector source0_prefixes;
  source0_prefixes.push_back(new BeamNode(1, 1));
  source0_prefixes.push_back(new BeamNode(0, 0));
  source0_prefixes.push_back(new BeamNode(3, 3));

  // second source has two prefix
  BeamNodeVector source1_prefixes;
  source1_prefixes.push_back(new BeamNode(4, 4));
  source1_prefixes.push_back(new BeamNode(5, 5));

  std::vector<BeamNodeVector> beamnode_vector_list;
  std::vector<SentenceVector> sentence_vector_list(2, SentenceVector());

  beamnode_vector_list.push_back(source0_prefixes);
  beamnode_vector_list.push_back(source1_prefixes);

  // generate data for one step
  std::vector<LoDTensor> ids;
  std::vector<LoDTensor> scores;

  paddle::test::GenerateExample(std::vector<size_t>{0, 3, 5},
                                std::vector<size_t>{0, 1, 1, 3, 4, 5},
                                std::vector<int>{0, 1, 2, 3, 4}, &ids, &scores);

  BeamHelper helper1;
  beamnode_vector_list = helper1.PackTwoSteps(
      ids[0], scores[0], beamnode_vector_list, &sentence_vector_list);

  ASSERT_EQ(sentence_vector_list[0].size(), 1UL);
  ASSERT_EQ(sentence_vector_list[1].size(), 0UL);
  ASSERT_EQ(beamnode_vector_list[0].size(), 3UL);
  ASSERT_EQ(beamnode_vector_list[1].size(), 2UL);
}

TEST(TrieConcatOp, PackAllSteps) {
  using BeamHelper = paddle::operators::BeamHelper;
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
  paddle::test::GenerateExample(std::vector<size_t>{0, 3, 6},
                                std::vector<size_t>{0, 0, 1, 2, 3, 4, 5},
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
  for (int64_t i = 0; i < id_tensor.dims()[0]; ++i) {
    ASSERT_EQ(score_tensor.data<float>()[i],
              static_cast<float>(id_tensor.data<int64_t>()[i]));
  }
}
