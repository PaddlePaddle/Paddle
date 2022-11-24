// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/Common.h"

namespace paddle {
namespace distributed {

std::vector<Place> GetPlaceList(const std::vector<phi::DenseTensor>& tensors) {
  std::vector<Place> places;
  places.reserve(tensors.size());
  for (auto& tensor : tensors) {
    places.push_back(tensor.place());
  }
  return places;
}

std::string GetKeyFromPlaces(const std::vector<Place>& places) {
  std::string placeList;
  for (auto& place : places) {
    std::stringstream tmp;
    tmp << place;
    if (placeList.empty()) {
      placeList += tmp.str();
    } else {
      placeList += "," + tmp.str();
    }
  }
  return placeList;
}

std::string GetKeyFromPlace(const Place& place) { return place.DebugString(); }

bool CheckTensorsInCudaPlace(const std::vector<phi::DenseTensor>& tensors) {
  return std::all_of(
      tensors.cbegin(), tensors.cend(), [&](const phi::DenseTensor& t) {
        return platform::is_gpu_place(t.place());
      });
<<<<<<< HEAD
=======
}

bool CheckTensorsInCustomPlace(const std::vector<phi::DenseTensor>& tensors,
                               const std::string& dev_type) {
  return std::all_of(
      tensors.cbegin(), tensors.cend(), [&](const phi::DenseTensor& t) {
        return platform::places_are_same_class(
            t.place(), paddle::platform::CustomPlace(dev_type));
      });
}

bool CheckTensorsInXPUPlace(const std::vector<phi::DenseTensor>& tensors) {
  return std::all_of(
      tensors.cbegin(), tensors.cend(), [&](const phi::DenseTensor& t) {
        return platform::is_xpu_place(t.place());
      });
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}

}  //  namespace distributed
}  //  namespace paddle
