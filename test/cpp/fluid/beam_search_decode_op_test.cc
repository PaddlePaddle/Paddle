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

#include "paddle/fluid/operators/beam_search_decode_op.h"

#include "gtest/gtest.h"

using CPUPlace = phi::CPUPlace;
using LegacyLoD = phi::LegacyLoD;
using DenseTensorArray = phi::TensorArray;

template <typename T>
using BeamSearchDecoder = paddle::operators::BeamSearchDecoder<T>;
template <typename T>
using Sentence = paddle::operators::Sentence<T>;
template <typename T>
using SentenceVector = paddle::operators::SentenceVector<T>;

namespace paddle {
namespace test {

template <typename T>
void GenerateExample(const std::vector<size_t>& level_0,
                     const std::vector<size_t>& level_1,
                     const std::vector<int>& data,
                     DenseTensorArray* ids,
                     DenseTensorArray* scores) {
  PADDLE_ENFORCE_EQ(level_0.back(),
                    level_1.size() - 1,
                    common::errors::InvalidArgument(
                        "source level is used to describe candidate set"
                        ", so it's element should less than level_1 length. "
                        "And the value of source"
                        "level is %d. ",
                        level_1.size() - 1));
  PADDLE_ENFORCE_EQ(level_1.back(),
                    data.size(),
                    common::errors::InvalidArgument(
                        "the lowest level is used to describe data"
                        ", so it's last element should be data length %d. ",
                        data.size()));

  CPUPlace place;

  LegacyLoD lod;
  lod.push_back(level_0);
  lod.push_back(level_1);

  // Ids
  phi::DenseTensor tensor_id;
  tensor_id.set_lod(lod);
  tensor_id.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  int64_t* id_ptr = tensor_id.mutable_data<int64_t>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    id_ptr[i] = static_cast<int64_t>(data.at(i));
  }

  // Scores
  phi::DenseTensor tensor_score;
  tensor_score.set_lod(lod);
  tensor_score.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  T* score_ptr = tensor_score.mutable_data<T>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    score_ptr[i] = static_cast<T>(data.at(i));
  }

  ids->push_back(tensor_id);
  scores->push_back(tensor_score);
}

template <typename T>
void BeamSearchDecodeTestFrame() {
  CPUPlace place;

  // Construct sample data with 5 steps and 2 source sentences
  // beam_size = 2, start_id = 0, end_id = 1
  DenseTensorArray ids;
  DenseTensorArray scores;

  GenerateExample<T>(std::vector<size_t>{0, 1, 2},
                     std::vector<size_t>{0, 1, 2},
                     std::vector<int>{0, 0},
                     &ids,
                     &scores);  // start with start_id
  GenerateExample<T>(std::vector<size_t>{0, 1, 2},
                     std::vector<size_t>{0, 2, 4},
                     std::vector<int>{2, 3, 4, 5},
                     &ids,
                     &scores);
  GenerateExample<T>(std::vector<size_t>{0, 2, 4},
                     std::vector<size_t>{0, 2, 2, 4, 4},
                     std::vector<int>{3, 1, 5, 4},
                     &ids,
                     &scores);
  GenerateExample<T>(std::vector<size_t>{0, 2, 4},
                     std::vector<size_t>{0, 1, 2, 3, 4},
                     std::vector<int>{1, 1, 3, 5},
                     &ids,
                     &scores);
  GenerateExample<T>(
      std::vector<size_t>{0, 2, 4},
      std::vector<size_t>{0, 0, 0, 2, 2},  // the branches of the first source
                                           // sentence are pruned since finished
      std::vector<int>{5, 1},
      &ids,
      &scores);

  ASSERT_EQ(ids.size(), 5UL);
  ASSERT_EQ(scores.size(), 5UL);

  BeamSearchDecoder<T> helper(2, 1);  // beam_size = 2, end_id = 1

  phi::DenseTensor id_tensor;
  phi::DenseTensor score_tensor;
  helper.Backtrace(ids, scores, &id_tensor, &score_tensor);

  LegacyLoD lod = id_tensor.lod();
  std::vector<size_t> expect_source_lod = {0, 2, 4};
  EXPECT_EQ(lod[0], expect_source_lod);
  std::vector<size_t> expect_sentence_lod = {0, 4, 7, 12, 17};
  EXPECT_EQ(lod[1], expect_sentence_lod);
  std::vector<int> expect_data = {
      0, 2, 3, 1, 0, 2, 1, 0, 4, 5, 3, 5, 0, 4, 5, 3, 1};
  ASSERT_EQ(id_tensor.dims()[0], static_cast<int64_t>(expect_data.size()));
  for (size_t i = 0; i < expect_data.size(); ++i) {
    ASSERT_EQ(id_tensor.data<int64_t>()[i],
              static_cast<int64_t>(expect_data[i]));
  }
  for (int64_t i = 0; i < id_tensor.dims()[0]; ++i) {
    ASSERT_EQ(score_tensor.data<T>()[i],
              static_cast<T>(id_tensor.data<int64_t>()[i]));
  }
}

}  // namespace test
}  // namespace paddle

TEST(BeamSearchDecodeOp, Backtrace_CPU_Float) {
  paddle::test::BeamSearchDecodeTestFrame<float>();
}

TEST(BeamSearchDecodeOp, Backtrace_CPU_Float16) {
  paddle::test::BeamSearchDecodeTestFrame<phi::dtype::float16>();
}

TEST(BeamSearchDecodeOp, Backtrace_CPU_Double) {
  paddle::test::BeamSearchDecodeTestFrame<double>();
}

TEST(BeamSearchDecodeOp, Backtrace_CPU_Int) {
  paddle::test::BeamSearchDecodeTestFrame<int>();
}

TEST(BeamSearchDecodeOp, Backtrace_CPU_Int64) {
  paddle::test::BeamSearchDecodeTestFrame<int64_t>();
}
