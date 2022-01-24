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

#include "paddle/fluid/operators/math/beam_search.h"

#include <gtest/gtest.h>

void PrepareCPUTensors(paddle::framework::LoDTensor* ids,
                       paddle::framework::LoDTensor* scores,
                       paddle::framework::LoDTensor* pre_ids,
                       paddle::framework::LoDTensor* pre_scores) {
  // lod
  paddle::framework::LoD lod;
  std::vector<size_t> level0({0, 2, 4});
  std::vector<size_t> level1({0, 1, 2, 3, 4});
  lod.push_back(level0);
  lod.push_back(level1);
  ids->set_lod(lod);
  scores->set_lod(lod);

  auto dims = paddle::framework::make_ddim({4, 3});
  ids->Resize(dims);
  scores->Resize(dims);

  paddle::platform::CPUPlace place;
  auto* ids_data = ids->mutable_data<int64_t>(place);
  auto* scores_data = scores->mutable_data<float>(place);
  std::vector<int64_t> ids_vec_data({4, 2, 5, 2, 1, 3, 3, 5, 2, 8, 2, 1});
  std::vector<float> scores_vec_data(
      {0.6f, 0.3f, 0.5f, 0.2f, 0.3f, 0.1f, 0.9f, 0.5f, 0.1f, 0.7f, 0.5f, 0.1f});

  CHECK_EQ(static_cast<size_t>(ids->numel()), ids_vec_data.size());
  CHECK_EQ(static_cast<size_t>(ids->numel()), scores_vec_data.size());

  for (int i = 0; i < ids->numel(); i++) {
    ids_data[i] = ids_vec_data[i];
    scores_data[i] = scores_vec_data[i];
  }

  // pre_ids
  pre_ids->Resize(paddle::framework::make_ddim({4, 1}));
  for (int i = 0; i < 4; i++) {
    pre_ids->mutable_data<int64_t>(place)[i] = i + 1;
  }

  // pre_scores
  pre_scores->Resize(paddle::framework::make_ddim({4, 1}));
  for (int i = 0; i < 4; i++) {
    pre_scores->mutable_data<float>(place)[i] = 0.1 * (i + 1);
  }
}

template <typename DeviceContext, typename Place>
void TestBeamSearch() {
  paddle::framework::LoDTensor ids;
  paddle::framework::LoDTensor scores;
  paddle::framework::LoDTensor pre_ids;
  paddle::framework::LoDTensor pre_scores;

  auto* place = new Place();
  DeviceContext* context = new DeviceContext(*place);
  if (paddle::platform::is_cpu_place(*place)) {
    PrepareCPUTensors(&ids, &scores, &pre_ids, &pre_scores);
  } else {
    paddle::framework::LoDTensor cpu_ids;
    paddle::framework::LoDTensor cpu_scores;
    paddle::framework::LoDTensor cpu_pre_ids;
    paddle::framework::LoDTensor cpu_pre_scores;

    PrepareCPUTensors(&cpu_ids, &cpu_scores, &cpu_pre_ids, &cpu_pre_scores);

    paddle::framework::TensorCopySync(cpu_ids, *place, &ids);
    paddle::framework::TensorCopySync(cpu_scores, *place, &scores);
    paddle::framework::TensorCopySync(cpu_pre_ids, *place, &pre_ids);
    paddle::framework::TensorCopySync(cpu_pre_scores, *place, &pre_scores);

    ids.set_lod(cpu_ids.lod());
    scores.set_lod(cpu_scores.lod());
    pre_ids.set_lod(cpu_pre_ids.lod());
    pre_scores.set_lod(cpu_pre_scores.lod());
  }

  paddle::framework::LoDTensor selected_ids;
  paddle::framework::LoDTensor selected_scores;
  paddle::framework::LoDTensor parent_idx;

  size_t level = 0;
  size_t beam_size = 2;
  int end_id = 0;
  paddle::operators::math::BeamSearchFunctor<DeviceContext, float> beamsearch;
  beamsearch(*context, &pre_ids, &pre_scores, &ids, &scores, &selected_ids,
             &selected_scores, &parent_idx, level, beam_size, end_id, true);

  ASSERT_EQ(selected_ids.lod(), selected_scores.lod());

  paddle::framework::LoDTensor cpu_selected_ids;
  paddle::framework::LoDTensor cpu_selected_scores;
  if (paddle::platform::is_cpu_place(*place)) {
    cpu_selected_ids = selected_ids;
    cpu_selected_scores = selected_scores;
  } else {
    paddle::framework::TensorCopySync(
        selected_ids, paddle::platform::CPUPlace(), &cpu_selected_ids);
    paddle::framework::TensorCopySync(
        selected_scores, paddle::platform::CPUPlace(), &cpu_selected_scores);
    cpu_selected_ids.set_lod(selected_ids.lod());
    cpu_selected_scores.set_lod(selected_scores.lod());
  }

  std::vector<int64_t> expected_ids({4, 5, 3, 8});
  std::vector<float> expected_scores({0.6f, 0.5f, 0.9f, 0.7f});
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(expected_ids[i], cpu_selected_ids.data<int64_t>()[i]);
    ASSERT_EQ(expected_scores[i], cpu_selected_scores.data<float>()[i]);
  }

  delete place;
  delete context;
}

TEST(BeamSearch, CPU) {
  TestBeamSearch<paddle::platform::CPUDeviceContext,
                 paddle::platform::CPUPlace>();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(BeamSearch, GPU) {
  TestBeamSearch<paddle::platform::CUDADeviceContext,
                 paddle::platform::CUDAPlace>();
}
#endif
