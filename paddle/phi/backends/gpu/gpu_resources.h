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

#include <array>

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/common/place.h"

namespace phi {

void InitGpuProperties(Place place,
                       int* compute_capability,
                       int* runtime_version,
                       int* driver_version,
                       int* multi_process,
                       int* max_threads_per_mp,
                       int* max_threads_per_block,
                       std::array<int, 3>* max_grid_dim_size);

void InitStream(gpuStream_t* stream);
void DestoryStream(gpuStream_t stream);

void InitBlasHandle(blasHandle_t* blas_handle, gpuStream_t stream);
void DestroyBlasHandle(blasHandle_t handle);

void InitBlasLtHandle(blasLtHandle_t* blaslt_handle);
void DestroyBlasLtHandle(blasLtHandle_t handle);

void InitDnnHandle(dnnHandle_t* handle, gpuStream_t stream, Place place);
void DestroyDnnHandle(dnnHandle_t handle);

void InitSolverHandle(solverHandle_t* handle, gpuStream_t stream);
void DestroySolverHandle(solverHandle_t solver_handle);

void InitSparseHandle(sparseHandle_t* handle, gpuStream_t stream);
void DestroySparseHandle(sparseHandle_t handle);

// void InitDnnWorkspace();

}  // namespace phi
