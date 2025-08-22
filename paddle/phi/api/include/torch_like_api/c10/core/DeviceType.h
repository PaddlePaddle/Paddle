// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <ostream>

#include "paddle/phi/common/place.h"

namespace c10 {

using DeviceType = phi::AllocationType;

constexpr DeviceType kCUDA = DeviceType::GPU;
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUSTOM = DeviceType::CUSTOM;

}  // namespace c10

namespace at {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
}  // namespace at

namespace torch {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
}  // namespace torch
