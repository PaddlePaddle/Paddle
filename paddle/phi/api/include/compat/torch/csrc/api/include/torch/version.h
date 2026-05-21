// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

/// Indicates the major version of LibTorch.
#define TORCH_VERSION_MAJOR 2

/// Indicates the minor version of LibTorch.
#define TORCH_VERSION_MINOR 10

/// Indicates the patch version of LibTorch.
#define TORCH_VERSION_PATCH 0

/// Indicates the ABI version tag of LibTorch.
#define TORCH_VERSION_ABI_TAG 0

/// Indicates the version of LibTorch as a string literal.
#define TORCH_VERSION "2.10.0"

/// Indicates the ABI version of LibTorch as a single uint64.
/// [ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ]
/// [ MAJ  ][ MIN  ][ PATCH][                              ABI TAG ]
#define TORCH_ABI_VERSION                                                 \
  (((0ULL + TORCH_VERSION_MAJOR) << 56) |                                 \
   ((0ULL + TORCH_VERSION_MINOR) << 48) | /* NOLINT(whitespace/indent) */ \
   ((0ULL + TORCH_VERSION_PATCH) << 40) | /* NOLINT(whitespace/indent) */ \
   ((0ULL + TORCH_VERSION_ABI_TAG) << 0)) /* NOLINT(whitespace/indent) */
