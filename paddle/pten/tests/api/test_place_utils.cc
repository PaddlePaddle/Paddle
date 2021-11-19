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

#include "gtest/gtest.h"

#include "paddle/pten/api/lib/utils/place_utils.h"

namespace paddle {
namespace experimental {
namespace tests {

TEST(place_utils, cpu_place) {
  auto pd_place = platform::CPUPlace();
  Place pten_place = ConvertToPtenPlace(pd_place);
  CHECK_EQ(pten_place.device().id(), 0);
  CHECK(pten_place.device().type() == DeviceType::kHost);
  CHECK(pten_place.is_pinned() == false);

  auto pd_place_1 = ConvertToPlatformPlace(pten_place);
  CHECK(platform::is_cpu_place(pd_place_1));
  CHECK(pd_place == BOOST_GET_CONST(platform::CPUPlace, pd_place_1));
  CHECK(pten_place == ConvertToPtenPlace(pd_place_1));
}

TEST(place_utils, cuda_place) {
  auto pd_place = platform::CUDAPlace(1);
  Place pten_place = ConvertToPtenPlace(pd_place);
  CHECK_EQ(pten_place.device().id(), 1);
  CHECK(pten_place.device().type() == DeviceType::kCuda);
  CHECK(pten_place.is_pinned() == false);

  auto pd_place_1 = ConvertToPlatformPlace(pten_place);
  CHECK(platform::is_gpu_place(pd_place_1));
  CHECK(pd_place == BOOST_GET_CONST(platform::CUDAPlace, pd_place_1));
  CHECK(pten_place == ConvertToPtenPlace(pd_place_1));
}

TEST(place_utils, cuda_pinned_place) {
  auto pd_place = platform::CUDAPinnedPlace();
  Place pten_place = ConvertToPtenPlace(pd_place);
  CHECK_EQ(pten_place.device().id(), 0);
  CHECK(pten_place.device().type() == DeviceType::kCuda);
  CHECK(pten_place.is_pinned() == true);

  auto pd_place_1 = ConvertToPlatformPlace(pten_place);
  CHECK(platform::is_cuda_pinned_place(pd_place_1));
  CHECK(pd_place == BOOST_GET_CONST(platform::CUDAPinnedPlace, pd_place_1));
  CHECK(pten_place == ConvertToPtenPlace(pd_place_1));
}

TEST(place_utils, xpu_place) {
  auto pd_place = platform::XPUPlace(1);
  Place pten_place = ConvertToPtenPlace(pd_place);
  CHECK_EQ(pten_place.device().id(), 1);
  CHECK(pten_place.device().type() == DeviceType::kXpu);
  CHECK(pten_place.is_pinned() == false);

  auto pd_place_1 = ConvertToPlatformPlace(pten_place);
  CHECK(platform::is_xpu_place(pd_place_1));
  CHECK(pd_place == BOOST_GET_CONST(platform::XPUPlace, pd_place_1));
  CHECK(pten_place == ConvertToPtenPlace(pd_place_1));
}

}  // namespace tests
}  // namespace experimental
}  // namespace paddle
