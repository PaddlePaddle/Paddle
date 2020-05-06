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

#include "paddle/fluid/operators/distributed/large_scale_kv.h"

#include <algorithm>
#include <thread>  // NOLINT

#include "gtest/gtest.h"

namespace paddle {
namespace operators {
namespace distributed {

TEST(LargeScaleKV, All) {
  std::string varname = "embedding";
  std::vector<std::string> params;
  std::vector<int> dims;

  params.push_back("Param");
  params.push_back("Moment1");
  params.push_back("Moment2");

  dims.push_back(64);
  dims.push_back(64);
  dims.push_back(64);

  auto meta = std::make_pair(varname, params, dims);

  LargeScaleKV::Init(meta);

  auto* kv = LargeScaleKV::GetInstance();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
