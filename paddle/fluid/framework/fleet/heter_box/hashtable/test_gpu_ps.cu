// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_resource.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_ps.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"

using namespace paddle::framework;

TEST(TEST_FLEET, gpu_ps) {
  std::vector<int> dev_ids;
  dev_ids.push_back(0);
  std::shared_ptr<HeterBoxResource> resource = std::make_shared<HeterBoxResource>(dev_ids);
  size_t size = 10;
  auto gpu_ps = std::make_shared<GpuPs<FeatureKey, FeatureValue, FeaturePushValue> >(size, resource);
  FeatureKey keys[10];
  FeatureValue vals[10];
  for (int i = 0; i < 10; i++) {
    keys[i] = i;
    vals[i].lr_w = i;
    vals[i].mf_size = 0;
  }
  std::cout << vals[0] << std::endl;
  gpu_ps->build_ps(0, keys, vals, 10, 10, 1);
  gpu_ps->show_one_table(0);
}
