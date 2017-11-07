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

TEST(TrieConcatOp, AppendVector) {
  using LoD = paddle::framework::LoD;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;
  using BeamHelper = paddle::operators::BeamHelpter;

  CPUPlace place;

  LoD lod;
  lod.push_back(std::vector<size_t>{});
  lod.push_back(std::vector<size_t>{0, 3, 6});

  LoDTensor tensor;
  tensor.set_lod(lod);

  tensor.Resize({10});
  float* dst_ptr = tensor.mutable_data<float>(place);
  for (int i = 0; i < 6; ++i) {
    dst_ptr[i] = i;
  }

  std::vector<float> vec = {6.0, 7.0, 8.0, 9.0};

  BeamHelper helper;
  helper.AppendVector<float>(vec, &tensor);

  ASSERT_EQ(tensor.lod()[1][3], 10UL);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tensor.data<float>()[i], static_cast<float>(i));
  }
}

TEST(TrieConcatOp, AppendBeamNodeToLoDTensor) {
  using LoD = paddle::framework::LoD;
  using CPUPlace = paddle::platform::CPUPlace;
  using LoDTensor = paddle::framework::LoDTensor;
  using BeamHelper = paddle::operators::BeamHelpter;
  using BeamNode = paddle::operators::BeamNode;

  BeamNode* root = new BeamNode(0, 0);
  BeamNode* b1 = new BeamNode(1, 1);
  BeamNode* end = new BeamNode(2, 2);
  b1->AppendTo(root);
  end->AppendTo(b1);

  BeamHelper helper;

  CPUPlace place;

  LoD lod;
  lod.push_back(std::vector<size_t>{});
  lod.push_back(std::vector<size_t>{0, 3, 6});

  // id tensor
  LoDTensor tensor_ids;
  tensor_ids.set_lod(lod);

  tensor_ids.Resize({10});
  int64_t* ids_ptr = tensor_ids.mutable_data<int64_t>(place);
  for (int i = 0; i < 6; ++i) {
    ids_ptr[i] = static_cast<int64_t>(i);
  }

  // probs tensor
  LoDTensor tensor_probs;
  tensor_probs.set_lod(lod);

  tensor_probs.Resize({10});
  float* probs_ptr = tensor_probs.mutable_data<float>(place);
  for (int i = 0; i < 6; ++i) {
    probs_ptr[i] = static_cast<float>(i);
  }

  helper.AppendBeamNodeToLoDTensor(end, &tensor_ids, &tensor_probs);

  // debug string tensor_ids
  for (auto& item : tensor_ids.lod()[1]) {
    std::cout << item << " ";
  }
  std::cout << std::endl;

  for (size_t i = 0; i < tensor_ids.lod()[1].back(); ++i) {
    std::cout << ids_ptr[i] << " ";
  }
  std::cout << std::endl;

  // make sure ids and probs are the same
  auto sentence_ids = tensor_ids.lod().at(1);
  auto sentences_probs = tensor_probs.lod().at(1);
  ASSERT_EQ(sentence_ids.size(), sentences_probs.size());
  for (size_t i = 0; i < sentence_ids.size(); ++i) {
    ASSERT_EQ(sentence_ids.at(i), sentences_probs.at(i));
  }
  for (size_t i = 0; i < sentence_ids.back(); ++i) {
    ASSERT_EQ(static_cast<float>(ids_ptr[i]), probs_ptr[i]);
  }

  ASSERT_EQ(tensor_ids.lod()[1].size(), 4UL);
  ASSERT_EQ(tensor_ids.lod()[1].back(), 9UL);
  size_t sentence_len = tensor_ids.lod()[1].size() - 1;
  ASSERT_EQ(sentence_len, 3UL);
  size_t added_sentence_start = tensor_ids.lod().at(1).at(sentence_len - 1);
  ASSERT_EQ(added_sentence_start, 6UL);
  for (size_t i = added_sentence_start; i < tensor_ids.lod().at(1).back();
       ++i) {
    ASSERT_EQ(ids_ptr[i], static_cast<int64_t>(i - added_sentence_start));
  }
}

// TEST(TrieConcatOp, CPU) {
//  using LoD = paddle::framework::LoD;
//  using CPUPlace = paddle::platform::CPUPlace;
//  using LoDTensor = paddle::framework::LoDTensor;
//
//  CPUPlace place;
//
//  size_t step_num = 3;
//  size_t beam_size = 3;
//  size_t batch_size = 2;
//
//  std::vector<LoDTensor> beam_out_Ids;
//  std::vector<LoDTensor> beam_out_Probs;
//
//  LoD lod_step_0;
//  lod_step_0.push_back(std::vector<size_t>{0, 3, 6});
//  lod_step_0.push_back(std::vector<size_t>{0, 1, 2, 3, 4, 5, 6});
//
//  // Ids
//  LoDTensor ids_step_0;
//  ids_step_0.set_lod(lod_step_0);
//  ids_step_0.Resize({6});
//  // malloc memory
//  int64_t* dst_ptr = ids_step_0.mutable_data<int64_t>(place);
//  for (int i = 0; i < 6; ++i) {
//    dst_ptr[i] = i;
//  }
//
//  LoDTensor probs_step_0;
//  probs_step_0.set_lod(lod_step_0);
//  probs_step_0.Resize({6});
//  // malloc memory
//  float* dst_ptr = probs_step_0.mutable_data<float>(place);
//  for (int i = 0; i < 6; ++i) {
//    dst_ptr[i] = i;
//  }
//
//}