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

#include <cstddef>
#include <string>

namespace paddle {
// Compute MD5 on GPU tensor data via D2H + CPU MD5.
// |data|: device pointer; |len|: byte count; |stream|: CUDA stream.
// Synchronizes stream before returning. Returns 32-char hex digest.
std::string md5_gpu(const void* data, size_t len, void* stream);

// Compute xxHash64 on GPU tensor data via D2H + CPU XXH64.
// Returns 16-char hex string of the 64-bit hash.
std::string xxhash64_gpu(const void* data, size_t len, void* stream);
}  // namespace paddle
