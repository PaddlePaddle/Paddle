/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/place.h"

#include "gtest/gtest.h"

namespace pten {
namespace tests {

TEST(PtenPlace, place) {
  pten::Place place;
  EXPECT_EQ(place.GetType(), pten::AllocationType::kUndef);

  place.Reset(pten::AllocationType::kGpu, 1);
  EXPECT_EQ(place.GetType(), pten::AllocationType::kGpu);
  EXPECT_EQ(place.GetDeviceId(), 1);
}

TEST(Place, cpu_place) {
  pten::CPUPlace place;
  EXPECT_EQ(place.GetType(), pten::AllocationType::kCpu);
}

TEST(Place, gpu_place) {
  pten::GPUPlace place;
  EXPECT_EQ(place.GetType(), pten::AllocationType::kGpu);
  EXPECT_EQ(place.GetDeviceId(), 0);

  pten::GPUPlace place1(2);
  EXPECT_EQ(place1.GetType(), pten::AllocationType::kGpu);
  EXPECT_EQ(place1.GetDeviceId(), 2);
  std::cout << "place string repr: " << place1 << std::endl;

  pten::GPUPinnedPlace place2;
  EXPECT_EQ(place2.GetType(), pten::AllocationType::kGpuPinned);
}

}  // namespace tests
}  // namespace pten
