// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <string>
#include "paddle/common/macros.h"

namespace phi {
namespace distributed {
class Store;

PADDLE_API int64_t GetCurGlobalRank();

PADDLE_API std::string GetMasterAddr();

PADDLE_API int64_t GetGlobalWorldSize();

PADDLE_API uint16_t GetMasterPort();

PADDLE_API std::shared_ptr<Store> CreateOrGetGlobalTCPStore();

}  // namespace distributed
}  // namespace phi
