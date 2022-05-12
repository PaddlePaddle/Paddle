/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/place.h"

#include <map>  // NOLINT
#include "gtest/gtest.h"

namespace phi {
namespace tests {

TEST(PhiPlace, place) {
  phi::Place place;
  EXPECT_EQ(place.GetType(), phi::AllocationType::UNDEFINED);

  place.Reset(phi::AllocationType::GPU, 1);
  EXPECT_EQ(place.GetType(), phi::AllocationType::GPU);
  EXPECT_EQ(place.GetDeviceId(), 1);
}

TEST(Place, cpu_place) {
  phi::CPUPlace place;
  EXPECT_EQ(place.GetType(), phi::AllocationType::CPU);
  std::cout << "cpu place repr: " << place << std::endl;
}

TEST(Place, gpu_place) {
  phi::GPUPlace place;
  EXPECT_EQ(place.GetType(), phi::AllocationType::GPU);
  EXPECT_EQ(place.GetDeviceId(), 0);

  phi::GPUPlace place1(2);
  EXPECT_EQ(place1.GetType(), phi::AllocationType::GPU);
  EXPECT_EQ(place1.GetDeviceId(), 2);
  std::cout << "gpu place repr: " << place1 << std::endl;

  phi::GPUPinnedPlace place2;
  EXPECT_EQ(place2.GetType(), phi::AllocationType::GPUPINNED);
  std::cout << "gpu pinned place repr: " << place2 << std::endl;

  EXPECT_NE(place2, phi::CPUPlace());
}

TEST(Place, convert_place) {
  phi::Place base_place(phi::AllocationType::CPU);
  phi::CPUPlace cpu_place = base_place;
  EXPECT_EQ(cpu_place.GetType(), base_place.GetType());
  base_place.Reset(phi::AllocationType::GPU, 2);
  phi::GPUPlace gpu_place = base_place;
  EXPECT_EQ(gpu_place.GetType(), base_place.GetType());
  EXPECT_EQ(gpu_place.GetDeviceId(), base_place.GetDeviceId());
  phi::Place place = gpu_place;
  EXPECT_EQ(gpu_place.GetType(), place.GetType());
  EXPECT_EQ(gpu_place.GetDeviceId(), place.GetDeviceId());
  place = cpu_place;
  EXPECT_EQ(cpu_place.GetType(), place.GetType());

  std::map<phi::Place, int> maps;
  maps[phi::CPUPlace()] = 1;
  maps[phi::GPUPlace(0)] = 2;
  maps[phi::GPUPlace(1)] = 3;
  maps[phi::GPUPlace(2)] = 4;
  maps[phi::GPUPlace(3)] = 5;
  maps[phi::GPUPinnedPlace()] = 6;
  for (auto iter = maps.begin(); iter != maps.end(); ++iter) {
    std::cout << iter->first << ":" << iter->second << std::endl;
  }
}

}  // namespace tests
}  // namespace phi
