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

#include <map>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/phi/common/place.h"

namespace phi {
namespace tests {

TEST(PhiPlace, place) {
  Place place;
  EXPECT_EQ(place.GetType(), AllocationType::UNDEFINED);

  place.Reset(AllocationType::GPU, 1);
  EXPECT_EQ(place.GetType(), AllocationType::GPU);
  EXPECT_EQ(place.GetDeviceId(), 1);
}

TEST(Place, cpu_place) {
  CPUPlace place;
  EXPECT_EQ(place.GetType(), AllocationType::CPU);
  std::cout << "cpu place repr: " << place << std::endl;
}

TEST(Place, gpu_place) {
  GPUPlace place;
  EXPECT_EQ(place.GetType(), AllocationType::GPU);
  EXPECT_EQ(place.GetDeviceId(), 0);

  GPUPlace place1(2);
  EXPECT_EQ(place1.GetType(), AllocationType::GPU);
  EXPECT_EQ(place1.GetDeviceId(), 2);
  std::cout << "gpu place repr: " << place1 << std::endl;

  GPUPinnedPlace place2;
  EXPECT_EQ(place2.GetType(), AllocationType::GPUPINNED);
  std::cout << "gpu pinned place repr: " << place2 << std::endl;

  EXPECT_NE(place2, CPUPlace());
}

TEST(Place, convert_place) {
  Place base_place(AllocationType::CPU);
  CPUPlace cpu_place = base_place;
  EXPECT_EQ(cpu_place.GetType(), base_place.GetType());
  base_place.Reset(AllocationType::GPU, 2);
  GPUPlace gpu_place = base_place;
  EXPECT_EQ(gpu_place.GetType(), base_place.GetType());
  EXPECT_EQ(gpu_place.GetDeviceId(), base_place.GetDeviceId());
  Place place = gpu_place;
  EXPECT_EQ(gpu_place.GetType(), place.GetType());
  EXPECT_EQ(gpu_place.GetDeviceId(), place.GetDeviceId());
  place = cpu_place;
  EXPECT_EQ(cpu_place.GetType(), place.GetType());

  std::map<Place, int> maps;
  maps[CPUPlace()] = 1;
  maps[GPUPlace(0)] = 2;
  maps[GPUPlace(1)] = 3;
  maps[GPUPlace(2)] = 4;
  maps[GPUPlace(3)] = 5;
  maps[GPUPinnedPlace()] = 6;
  for (auto& map_item : maps) {
    std::cout << map_item.first << ":" << map_item.second << std::endl;
  }
}

}  // namespace tests
}  // namespace phi
