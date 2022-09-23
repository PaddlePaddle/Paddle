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

#pragma once

#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
// Get the list of devices from list of tensors
std::vector<Place> GetPlaceList(const std::vector<phi::DenseTensor>& tensors);
// Get the deviceList String from the list of devices
std::string GetKeyFromPlaces(const std::vector<Place>& places);

bool CheckTensorsInCudaPlace(const std::vector<phi::DenseTensor>& tensors);

bool CheckTensorsInCustomPlace(const std::vector<phi::DenseTensor>& tensors,
                               const std::string& dev_type);

}  //  namespace distributed
}  //  namespace paddle
