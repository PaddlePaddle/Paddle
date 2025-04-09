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
#include <cstddef>
#include <vector>

#include "paddle/utils/test_macros.h"

namespace phi {
using LegacyLoD = std::vector<std::vector<std::size_t>>;

/*
 * Transform an LegacyLoD from relative offsets to absolute offsets.
 */
LegacyLoD ToAbsOffset(const LegacyLoD& in);

TEST_API void AppendLegacyLoD(LegacyLoD* lod, const LegacyLoD& lod_length);

/*
 * Convert between length-based LegacyLoD and offset-based LegacyLoD.
 * The implementation of DenseTensor class use offset-based LegacyLoD.
 * However, we want to expose the more user-friendly length-based
 * LegacyLoD to the Python side instead.
 *
 * Example:
 * If offset_lod = [[0, 2, 3],[0, 3, 5, 9]]
 * then length_lod = [[2, 1], [3, 2, 4]]
 */
TEST_API LegacyLoD ConvertToLengthBasedLegacyLoD(const LegacyLoD& offset_lod);

}  // namespace  phi
