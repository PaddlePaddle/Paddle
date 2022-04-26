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

std::vector<Place> GetPlaceList(const std::vector<Tensor>& tensors) {
  std::vector<Place> places;
  places.reserve(tensors.size());
  for (auto& tensor : tensors) {
    places.push_back(tensor.inner_place());
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

static bool CheckTensorsInPlace(const std::vector<Tensor>& tensors,
                                const PlaceType type) {
  return std::all_of(tensors.cbegin(), tensors.cend(),
                     [&](const Tensor& t) { return t.place() == type; });
}

bool CheckTensorsInCudaPlace(const std::vector<Tensor>& tensors) {
  return CheckTensorsInPlace(tensors, PlaceType::kGPU);
}

}  //  namespace distributed
}  //  namespace paddle
