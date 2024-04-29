/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/pitch_linear_coord.h"

namespace cutlass {
namespace layout {

template <int RowsPerTile, int ColumnsInterleaved>
class ColumnMajorTileInterleave {
  static constexpr int kRowsPerTile = RowsPerTile;
  static constexpr int kColumnsInterleaved = ColumnsInterleaved;
};

template <class T>
struct IsColumnMajorTileInterleave {
  static constexpr bool value = false;
};

template <int U, int V>
struct IsColumnMajorTileInterleave<ColumnMajorTileInterleave<U, V>> {
  static constexpr bool value = true;
};

}  // namespace layout
}  // namespace cutlass
