// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <optional>
#include <string>

namespace phi {

// Device count query interface.
// Backends register their device count provider at startup via static
// initialization, so that phi/core can query device counts without depending
// on upper-layer modules (e.g., fluid).
using DeviceCountFn = int (*)();

// Register a device count provider for the given device_type.
// PADDLE_ENFORCE: duplicate registration for the same device_type is fatal.
// PADDLE_ENFORCE: fn must not be null.
void RegisterDeviceCountProvider(const std::string& device_type,
                                 DeviceCountFn fn);

// Returns the number of devices for the given type, or std::nullopt if
// no provider is registered (caller decides whether to enforce or treat as 0).
std::optional<int> GetDeviceCount(const std::string& device_type);

}  // namespace phi
