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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <cstdint>
#include <map>
#include <shared_mutex>

// cublasLT was introduced in CUDA 10.1 but we enable only for 11.1 that also
// added bf16 support
#include <cublasLt.h>

#ifdef CUDART_VERSION
#include <cusolverDn.h>
#endif

#if defined(USE_CUDSS)
#include <cudss.h>
#endif

#if defined(USE_ROCM)
#include <hipsolver/hipsolver.h>
#endif

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAFunctions.h>

namespace c10 {
struct Allocator;
}

namespace at::cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() { return c10::cuda::device_count(); }

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() { return c10::cuda::device_count() > 0; }

cudaDeviceProp* getCurrentDeviceProperties();

int warp_size();

cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device);

c10::Allocator* getCUDADeviceAllocator();

/* Handles */
cusparseHandle_t getCurrentCUDASparseHandle();
cublasHandle_t getCurrentCUDABlasHandle();
cublasLtHandle_t getCurrentCUDABlasLtHandle();

void clearCublasWorkspaces();
struct WorkspaceMapWithMutex {
  std::map<std::tuple<void*, void*>, at::DataPtr> map;
  std::shared_mutex mutex;
};

WorkspaceMapWithMutex& cublas_handle_stream_to_workspace();
WorkspaceMapWithMutex& cublaslt_handle_stream_to_workspace();
size_t getChosenWorkspaceSize();
size_t getCUDABlasLtWorkspaceSize();
void* getCUDABlasLtWorkspace();

#if defined(CUDART_VERSION) || defined(USE_ROCM)
cusolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

#if defined(USE_CUDSS)
cudssHandle_t getCurrentCudssHandle();
#endif

}  // namespace at::cuda
