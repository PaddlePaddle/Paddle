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

#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/beam_search_decode_xpu.h"

#include "gtest/gtest.h"

using CPUPlace = phi::CPUPlace;
using XPUPlace = phi::XPUPlace;
using LegacyLoD = phi::LegacyLoD;
using DenseTensorArray = phi::TensorArray;

template <typename T>
using BeamSearchDecoder = phi::funcs::BeamSearchDecoder<T>;
template <typename T>
using Sentence = phi::funcs::Sentence<T>;
template <typename T>
using SentenceVector = phi::funcs::SentenceVector<T>;

namespace paddle {
namespace test {

template <typename T>
void GenerateXPUExample(const std::vector<size_t>& level_0,
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
  int XPU_PlaceNo = phi::backends::xpu::GetXPUCurrentDeviceId();

  XPUPlace xpu_place(XPU_PlaceNo);

  LegacyLoD lod;
  lod.push_back(level_0);
  lod.push_back(level_1);

  // Ids
  phi::DenseTensor tensor_id_cpu;
  tensor_id_cpu.set_lod(lod);
  tensor_id_cpu.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  int64_t* id_cpu_ptr = tensor_id_cpu.mutable_data<int64_t>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    id_cpu_ptr[i] = static_cast<int64_t>(data.at(i));
  }

  phi::DenseTensor tensor_id;
  const phi::DenseTensorMeta meta_data_id(phi::DataType::INT64,
                                          tensor_id_cpu.dims());
  tensor_id.set_meta(meta_data_id);
  tensor_id.set_lod(lod);

  int64_t* id_ptr = tensor_id.mutable_data<int64_t>(xpu_place);
  phi::memory_utils::Copy(phi::XPUPlace(XPU_PlaceNo),
                          id_ptr,
                          phi::CPUPlace(),
                          id_cpu_ptr,
                          tensor_id_cpu.numel() * sizeof(int64_t));

  // Scores
  phi::DenseTensor tensor_score_cpu;
  tensor_score_cpu.set_lod(lod);
  tensor_score_cpu.Resize({static_cast<int64_t>(data.size())});
  // malloc memory
  T* score_cpu_ptr = tensor_score_cpu.mutable_data<T>(place);
  for (size_t i = 0; i < data.size(); ++i) {
    score_cpu_ptr[i] = static_cast<T>(data.at(i));
  }

  phi::DenseTensor tensor_score;

  if (std::is_same<float, T>::value) {
    const phi::DenseTensorMeta meta_data_score(phi::DataType::FLOAT32,
                                               tensor_score_cpu.dims());
    tensor_score.set_meta(meta_data_score);
  } else if (std::is_same<double, T>::value) {
    const phi::DenseTensorMeta meta_data_score(phi::DataType::FLOAT64,
                                               tensor_score_cpu.dims());
    tensor_score.set_meta(meta_data_score);
  } else if (std::is_same<phi::dtype::float16, T>::value) {
    const phi::DenseTensorMeta meta_data_score(phi::DataType::FLOAT16,
                                               tensor_score_cpu.dims());
    tensor_score.set_meta(meta_data_score);
  } else if (std::is_same<int, T>::value) {
    const phi::DenseTensorMeta meta_data_score(phi::DataType::INT32,
                                               tensor_score_cpu.dims());
    tensor_score.set_meta(meta_data_score);
  } else if (std::is_same<int64_t, T>::value) {
    const phi::DenseTensorMeta meta_data_score(phi::DataType::INT64,
                                               tensor_score_cpu.dims());
    tensor_score.set_meta(meta_data_score);
  }

  tensor_score.set_lod(lod);

  T* score_ptr = tensor_score.mutable_data<T>(xpu_place);

  phi::memory_utils::Copy(phi::XPUPlace(XPU_PlaceNo),
                          score_ptr,
                          phi::CPUPlace(),
                          score_cpu_ptr,
                          tensor_score_cpu.numel() * sizeof(T));

  ids->push_back(tensor_id);
  scores->push_back(tensor_score);
}

template <typename T>
void BeamSearchDecodeTestByXPUFrame() {
  CPUPlace place;

  // Construct sample data with 5 steps and 2 source sentences
  // beam_size = 2, start_id = 0, end_id = 1

  DenseTensorArray ids;
  DenseTensorArray scores;

  GenerateXPUExample<T>(std::vector<size_t>{0, 1, 2},
                        std::vector<size_t>{0, 1, 2},
                        std::vector<int>{0, 0},
                        &ids,
                        &scores);  // start with start_id
  GenerateXPUExample<T>(std::vector<size_t>{0, 1, 2},
                        std::vector<size_t>{0, 2, 4},
                        std::vector<int>{2, 3, 4, 5},
                        &ids,
                        &scores);
  GenerateXPUExample<T>(std::vector<size_t>{0, 2, 4},
                        std::vector<size_t>{0, 2, 2, 4, 4},
                        std::vector<int>{3, 1, 5, 4},
                        &ids,
                        &scores);
  GenerateXPUExample<T>(std::vector<size_t>{0, 2, 4},
                        std::vector<size_t>{0, 1, 2, 3, 4},
                        std::vector<int>{1, 1, 3, 5},
                        &ids,
                        &scores);
  GenerateXPUExample<T>(
      std::vector<size_t>{0, 2, 4},
      std::vector<size_t>{0, 0, 0, 2, 2},  // the branches of the first source
                                           // sentence are pruned since finished
      std::vector<int>{5, 1},
      &ids,
      &scores);

  ASSERT_EQ(ids.size(), 5UL);
  ASSERT_EQ(scores.size(), 5UL);

  phi::DenseTensor id_tensor_cpu;
  phi::DenseTensor score_tensor_cpu;

  phi::funcs::BeamSearchDecodeXPUFunctor bs_xpu(
      ids, scores, &id_tensor_cpu, &score_tensor_cpu, 2, 1);
  bs_xpu.apply_xpu<T>();

  LegacyLoD lod = id_tensor_cpu.lod();
  std::vector<size_t> expect_source_lod = {0, 2, 4};
  ASSERT_EQ(lod[0], expect_source_lod);

  std::vector<size_t> expect_sentence_lod = {0, 4, 7, 12, 17};
  ASSERT_EQ(lod[1], expect_sentence_lod);

  std::vector<int> expect_data = {
      0, 2, 3, 1, 0, 2, 1, 0, 4, 5, 3, 5, 0, 4, 5, 3, 1};
  ASSERT_EQ(id_tensor_cpu.dims()[0], static_cast<int64_t>(expect_data.size()));

  for (size_t i = 0; i < expect_data.size(); ++i) {
    ASSERT_EQ(id_tensor_cpu.data<int64_t>()[i],
              static_cast<int64_t>(expect_data[i]));
  }

  for (int64_t i = 0; i < id_tensor_cpu.dims()[0]; ++i) {
    ASSERT_EQ(score_tensor_cpu.data<T>()[i],
              static_cast<T>(id_tensor_cpu.data<int64_t>()[i]));
  }
}

}  // namespace test
}  // namespace paddle

TEST(BeamSearchDecodeOpXPU, Backtrace_XPU_Float) {
  paddle::test::BeamSearchDecodeTestByXPUFrame<float>();
}

TEST(BeamSearchDecodeOpXPU, Backtrace_XPU_Float16) {
  paddle::test::BeamSearchDecodeTestByXPUFrame<phi::dtype::float16>();
}

TEST(BeamSearchDecodeOpXPU, Backtrace_XPU_Int) {
  paddle::test::BeamSearchDecodeTestByXPUFrame<int>();
}

TEST(BeamSearchDecodeOpXPU, Backtrace_XPU_Int64) {
  paddle::test::BeamSearchDecodeTestByXPUFrame<int64_t>();
}

TEST(BeamSearchDecodeOpXPU, Backtrace_XPU_Double) {
  paddle::test::BeamSearchDecodeTestByXPUFrame<double>();
}
