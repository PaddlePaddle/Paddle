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

#include "paddle/fluid/operators/distributed/async_sparse_param_update_recorder.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace paddle {
namespace operators {
namespace distributed {

TEST(ConcurrentSet, All) {
  ConcurrentSet concurrent_set;
  std::vector<int64_t> in1 = {1, 2, 3, 4};
  std::vector<int64_t> in2 = {2, 3, 5, 6};

  std::vector<std::future<void>> futures;
  futures.push_back(concurrent_set.Update(in1));
  futures.push_back(concurrent_set.Update(in2));

  for (auto &f : futures) {
    f.wait();
  }

  std::unordered_set<int64_t> in;
  std::copy(in1.begin(), in1.end(), std::inserter(in, in.begin()));
  std::copy(in2.begin(), in2.end(), std::inserter(in, in.begin()));

  std::vector<int64_t> ret;
  concurrent_set.GetAndClear(&ret).wait();

  std::unordered_set<int64_t> out;
  std::copy(ret.begin(), ret.end(), std::inserter(out, out.begin()));

  EXPECT_EQ(in, out);

  concurrent_set.GetAndClear(&ret).wait();
  EXPECT_EQ(ret.size(), 0);
}

TEST(AsyncSparseParamUpdateRecorder, All) {
  std::unordered_map<std::string, std::string> grad_to_param;
  grad_to_param["grad1"] = "param1";
  grad_to_param["grad2"] = "param2";

  int trainer_num = 10;

  AsyncSparseParamUpdateRecorder recorder(trainer_num, grad_to_param);
  std::vector<int64_t> in1 = {1, 2, 3, 4};
  std::vector<int64_t> in2 = {2, 3, 5, 6};

  std::unordered_set<int64_t> in;
  std::copy(in1.begin(), in1.end(), std::inserter(in, in.begin()));
  std::copy(in2.begin(), in2.end(), std::inserter(in, in.begin()));

  recorder.Update("grad1", in1);
  recorder.Update("grad1", in2);

  EXPECT_TRUE(recorder.HasParam("param1"));
  EXPECT_TRUE(recorder.HasParam("param2"));
  EXPECT_FALSE(recorder.HasParam("param3"));

  std::vector<int64_t> ret;
  EXPECT_ANY_THROW(recorder.GetAndClear("param1", trainer_num, &ret));

  for (int i = 0; i < trainer_num; ++i) {
    std::vector<int64_t> ret;
    std::unordered_set<int64_t> out;

    recorder.GetAndClear("param1", i, &ret);
    std::copy(ret.begin(), ret.end(), std::inserter(out, out.begin()));

    EXPECT_EQ(in, out);

    recorder.GetAndClear("param1", i, &ret);
    EXPECT_EQ(ret.size(), 0);
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
