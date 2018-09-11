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

#include "paddle/fluid/operators/beam_search_op.h"

#include <gtest/gtest.h>
#include <vector>

namespace paddle {
namespace test {

using std::vector;
using framework::LoDTensor;
using framework::LoD;
using operators::BeamSearch;
using paddle::platform::CPUPlace;
using std::cout;
using std::endl;

void CreateInput(LoDTensor* ids, LoDTensor* scores) {
  LoD lod;
  vector<size_t> level0({0, 2, 4});
  vector<size_t> level1({0, 1, 2, 3, 4});
  lod.push_back(level0);
  lod.push_back(level1);
  ids->set_lod(lod);
  scores->set_lod(lod);

  auto dims = framework::make_ddim(vector<int64_t>({4, 3}));
  ids->Resize(dims);
  scores->Resize(dims);
  CPUPlace place;

  auto* ids_data = ids->mutable_data<int64_t>(place);
  auto* scores_data = scores->mutable_data<float>(place);
  vector<int64_t> _ids({4, 2, 5, 2, 1, 3, 3, 5, 2, 8, 2, 1});
  vector<float> _scores(
      {0.5, 0.3, 0.2, 0.6, 0.3, 0.1, 0.9, 0.5, 0.1, 0.7, 0.5, 0.1});

  for (int i = 0; i < 12; i++) {
    ids_data[i] = _ids[i];
    scores_data[i] = _scores[i];
  }
}

TEST(beam_search_op, run) {
  CPUPlace place;
  LoDTensor ids, scores;
  CreateInput(&ids, &scores);

  LoDTensor pre_ids;
  pre_ids.Resize(framework::make_ddim(vector<int64_t>(4, 1)));
  for (int i = 0; i < 4; i++) {
    pre_ids.mutable_data<int64_t>(place)[i] = i + 1;
  }
  LoDTensor pre_scores;
  pre_scores.Resize(framework::make_ddim(vector<int64_t>(4, 1)));
  for (int i = 0; i < 4; i++) {
    pre_scores.mutable_data<float>(place)[i] = 0.1 * (i + 1);
  }

  BeamSearch beamsearch(ids, scores, (size_t)0, (size_t)2, 0);
  LoDTensor sids, sscores;
  beamsearch(pre_ids, pre_scores, &sids, &sscores);

  LOG(INFO) << "score: " << sscores << endl;

  ASSERT_EQ(sids.lod(), sscores.lod());

  vector<int> tids({4, 2, 3, 8});
  vector<float> tscores({0.5, 0.6, 0.9, 0.7});

  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(tids[i], sids.data<int64_t>()[i]);
    ASSERT_EQ(tscores[i], sscores.data<float>()[i]);
  }
}

}  // namespace test
}  // namespace paddle
