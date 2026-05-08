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

// Placeholder header to satisfy PyTorch compatibility checks.
// Paddle does not use the same CUDA cmake macros as PyTorch,
// but the presence of this file allows downstream code to use
// __has_include(<c10/cuda/impl/cuda_cmake_macros.h>) for feature detection.
