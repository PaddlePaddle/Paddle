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

// fake_nvgpu_device.h
//
// Purpose: Wrap NVIDIA GPU as a CustomDevice plugin so that the CINN
// CustomDevice code path (C_CinnInterface + C_DeviceInterface) can be fully
// exercised in CI without requiring WITH_GPU=ON in Paddle's cmake.
//
// Design:
//   All CUDA APIs are loaded at runtime via dlopen/dlsym from:
//     - libcuda.so    (CUDA Driver API: cuInit, cuModuleLoad,
//     cuLaunchKernel...)
//     - libcudart.so  (CUDA Runtime: cudaMalloc, cudaMemcpy, cudaStream...)
//     - libnvrtc.so   (NVRTC: nvrtcCompileProgram, nvrtcGetPTX...)
//
//   This allows the test to compile with WITH_GPU=OFF (Paddle CPU mode)
//   while still using real GPU hardware at runtime.

#pragma once

#include <dlfcn.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "paddle/phi/backends/device_ext.h"

// =============================================================================
// CUDA type definitions (replicate minimal set to avoid #include <cuda.h>)
// =============================================================================

// CUDA Driver API types
typedef int CUresult_t;
typedef void* CUmodule_t;
typedef void* CUfunction_t;
typedef void* CUstream_t;
typedef void* CUcontext_t;

// CUDA Runtime types
typedef int cudaError_t_;
typedef void* cudaStream_rt;
typedef void* cudaEvent_rt;

// NVRTC types
typedef int nvrtcResult_t;
typedef void* nvrtcProgram_t;

// cudaMemcpyKind
enum FakeMemcpyKind {
  kHostToHost = 0,
  kHostToDevice = 1,
  kDeviceToHost = 2,
  kDeviceToDevice = 3
};

// cudaDeviceAttr enum values we need
enum FakeCudaDeviceAttr {
  kMaxThreadsPerBlock = 1,
  kMaxBlockDimX = 2,
  kMaxBlockDimY = 3,
  kMaxBlockDimZ = 4,
  kMaxGridDimX = 5,
  kMaxGridDimY = 6,
  kMaxGridDimZ = 7,
  kMaxSharedMemoryPerBlock = 8,
  kMultiProcessorCount = 16,
  kMaxThreadsPerMultiProcessor = 39,
  kComputeCapabilityMajor = 75,
  kComputeCapabilityMinor = 76
};

// =============================================================================
// Dynamic loader for CUDA libraries
// =============================================================================

struct FakeNVGPU_CudaAPI {
  // Library handles
  void* libcuda = nullptr;
  void* libcudart = nullptr;
  void* libnvrtc = nullptr;

  // --- CUDA Driver API function pointers ---
  int (*cuInit)(unsigned int) = nullptr;
  int (*cuModuleLoad)(CUmodule_t*, const char*) = nullptr;
  int (*cuModuleLoadData)(CUmodule_t*, const void*) = nullptr;
  int (*cuModuleUnload)(CUmodule_t) = nullptr;
  int (*cuModuleGetFunction)(CUfunction_t*, CUmodule_t, const char*) = nullptr;
  int (*cuLaunchKernel)(CUfunction_t,
                        unsigned,
                        unsigned,
                        unsigned,
                        unsigned,
                        unsigned,
                        unsigned,
                        unsigned,
                        CUstream_t,
                        void**,
                        void**) = nullptr;
  int (*cuGetErrorString)(int, const char**) = nullptr;

  // --- CUDA Runtime API function pointers ---
  int (*cudaSetDevice)(int) = nullptr;
  int (*cudaGetDevice)(int*) = nullptr;
  int (*cudaGetDeviceCount)(int*) = nullptr;
  int (*cudaMalloc)(void**, size_t) = nullptr;
  int (*cudaFree)(void*) = nullptr;
  int (*cudaMallocHost)(void**, size_t) = nullptr;
  int (*cudaFreeHost)(void*) = nullptr;
  int (*cudaMemcpy)(void*, const void*, size_t, int) = nullptr;
  int (*cudaMemcpyAsync)(void*, const void*, size_t, int, cudaStream_rt) =
      nullptr;
  int (*cudaMemset)(void*, int, size_t) = nullptr;
  int (*cudaMemGetInfo)(size_t*, size_t*) = nullptr;
  int (*cudaStreamCreate)(cudaStream_rt*) = nullptr;
  int (*cudaStreamDestroy)(cudaStream_rt) = nullptr;
  int (*cudaStreamSynchronize)(cudaStream_rt) = nullptr;
  int (*cudaStreamWaitEvent)(cudaStream_rt, cudaEvent_rt, unsigned) = nullptr;
  int (*cudaEventCreate)(cudaEvent_rt*) = nullptr;
  int (*cudaEventDestroy)(cudaEvent_rt) = nullptr;
  int (*cudaEventRecord)(cudaEvent_rt, cudaStream_rt) = nullptr;
  int (*cudaEventSynchronize)(cudaEvent_rt) = nullptr;
  int (*cudaDeviceSynchronize)() = nullptr;
  int (*cudaDeviceGetAttribute)(int*, int, int) = nullptr;
  const char* (*cudaGetErrorString)(int) = nullptr;

  // --- NVRTC function pointers ---
  int (*nvrtcCreateProgram)(nvrtcProgram_t*,
                            const char*,
                            const char*,
                            int,
                            const char* const*,
                            const char* const*) = nullptr;
  int (*nvrtcDestroyProgram)(nvrtcProgram_t*) = nullptr;
  int (*nvrtcCompileProgram)(nvrtcProgram_t, int, const char* const*) = nullptr;
  int (*nvrtcGetPTXSize)(nvrtcProgram_t, size_t*) = nullptr;
  int (*nvrtcGetPTX)(nvrtcProgram_t, char*) = nullptr;
  int (*nvrtcGetCUBINSize)(nvrtcProgram_t, size_t*) = nullptr;
  int (*nvrtcGetCUBIN)(nvrtcProgram_t, char*) = nullptr;
  int (*nvrtcGetProgramLogSize)(nvrtcProgram_t, size_t*) = nullptr;
  int (*nvrtcGetProgramLog)(nvrtcProgram_t, char*) = nullptr;
  const char* (*nvrtcGetErrorString)(int) = nullptr;

  bool loaded = false;

  static FakeNVGPU_CudaAPI& Instance() {
    static FakeNVGPU_CudaAPI inst;
    return inst;
  }

  bool Load() {
    if (loaded) return true;

    // Try loading CUDA libraries
    libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!libcuda) {
      fprintf(stderr, "[FakeNVGPU] Cannot load libcuda.so: %s\n", dlerror());
      return false;
    }

    libcudart = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!libcudart) libcudart = dlopen("libcudart.so.11", RTLD_LAZY);
    if (!libcudart) libcudart = dlopen("libcudart.so", RTLD_LAZY);
    if (!libcudart) {
      fprintf(stderr, "[FakeNVGPU] Cannot load libcudart.so: %s\n", dlerror());
      return false;
    }

    libnvrtc = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!libnvrtc) libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (!libnvrtc) libnvrtc = dlopen("libnvrtc.so.11.2", RTLD_LAZY);
    if (!libnvrtc) {
      fprintf(stderr, "[FakeNVGPU] Cannot load libnvrtc.so: %s\n", dlerror());
      return false;
    }

    // Load Driver API symbols
    *reinterpret_cast<void**>(&cuInit) = dlsym(libcuda, "cuInit");
    *reinterpret_cast<void**>(&cuModuleLoad) = dlsym(libcuda, "cuModuleLoad");
    *reinterpret_cast<void**>(&cuModuleLoadData) =
        dlsym(libcuda, "cuModuleLoadData");
    *reinterpret_cast<void**>(&cuModuleUnload) =
        dlsym(libcuda, "cuModuleUnload");
    *reinterpret_cast<void**>(&cuModuleGetFunction) =
        dlsym(libcuda, "cuModuleGetFunction");
    *reinterpret_cast<void**>(&cuLaunchKernel) =
        dlsym(libcuda, "cuLaunchKernel");
    *reinterpret_cast<void**>(&cuGetErrorString) =
        dlsym(libcuda, "cuGetErrorString");

    // Load Runtime API symbols
    *reinterpret_cast<void**>(&cudaSetDevice) =
        dlsym(libcudart, "cudaSetDevice");
    *reinterpret_cast<void**>(&cudaGetDevice) =
        dlsym(libcudart, "cudaGetDevice");
    *reinterpret_cast<void**>(&cudaGetDeviceCount) =
        dlsym(libcudart, "cudaGetDeviceCount");
    *reinterpret_cast<void**>(&cudaMalloc) = dlsym(libcudart, "cudaMalloc");
    *reinterpret_cast<void**>(&cudaFree) = dlsym(libcudart, "cudaFree");
    *reinterpret_cast<void**>(&cudaMallocHost) =
        dlsym(libcudart, "cudaMallocHost");
    *reinterpret_cast<void**>(&cudaFreeHost) = dlsym(libcudart, "cudaFreeHost");
    *reinterpret_cast<void**>(&cudaMemcpy) = dlsym(libcudart, "cudaMemcpy");
    *reinterpret_cast<void**>(&cudaMemcpyAsync) =
        dlsym(libcudart, "cudaMemcpyAsync");
    *reinterpret_cast<void**>(&cudaMemset) = dlsym(libcudart, "cudaMemset");
    *reinterpret_cast<void**>(&cudaMemGetInfo) =
        dlsym(libcudart, "cudaMemGetInfo");
    *reinterpret_cast<void**>(&cudaStreamCreate) =
        dlsym(libcudart, "cudaStreamCreate");
    *reinterpret_cast<void**>(&cudaStreamDestroy) =
        dlsym(libcudart, "cudaStreamDestroy");
    *reinterpret_cast<void**>(&cudaStreamSynchronize) =
        dlsym(libcudart, "cudaStreamSynchronize");
    *reinterpret_cast<void**>(&cudaStreamWaitEvent) =
        dlsym(libcudart, "cudaStreamWaitEvent");
    *reinterpret_cast<void**>(&cudaEventCreate) =
        dlsym(libcudart, "cudaEventCreate");
    *reinterpret_cast<void**>(&cudaEventDestroy) =
        dlsym(libcudart, "cudaEventDestroy");
    *reinterpret_cast<void**>(&cudaEventRecord) =
        dlsym(libcudart, "cudaEventRecord");
    *reinterpret_cast<void**>(&cudaEventSynchronize) =
        dlsym(libcudart, "cudaEventSynchronize");
    *reinterpret_cast<void**>(&cudaDeviceSynchronize) =
        dlsym(libcudart, "cudaDeviceSynchronize");
    *reinterpret_cast<void**>(&cudaDeviceGetAttribute) =
        dlsym(libcudart, "cudaDeviceGetAttribute");
    *reinterpret_cast<void**>(&cudaGetErrorString) =
        dlsym(libcudart, "cudaGetErrorString");

    // Load NVRTC symbols
    *reinterpret_cast<void**>(&nvrtcCreateProgram) =
        dlsym(libnvrtc, "nvrtcCreateProgram");
    *reinterpret_cast<void**>(&nvrtcDestroyProgram) =
        dlsym(libnvrtc, "nvrtcDestroyProgram");
    *reinterpret_cast<void**>(&nvrtcCompileProgram) =
        dlsym(libnvrtc, "nvrtcCompileProgram");
    *reinterpret_cast<void**>(&nvrtcGetPTXSize) =
        dlsym(libnvrtc, "nvrtcGetPTXSize");
    *reinterpret_cast<void**>(&nvrtcGetPTX) = dlsym(libnvrtc, "nvrtcGetPTX");
    *reinterpret_cast<void**>(&nvrtcGetCUBINSize) =
        dlsym(libnvrtc, "nvrtcGetCUBINSize");
    *reinterpret_cast<void**>(&nvrtcGetCUBIN) =
        dlsym(libnvrtc, "nvrtcGetCUBIN");
    *reinterpret_cast<void**>(&nvrtcGetProgramLogSize) =
        dlsym(libnvrtc, "nvrtcGetProgramLogSize");
    *reinterpret_cast<void**>(&nvrtcGetProgramLog) =
        dlsym(libnvrtc, "nvrtcGetProgramLog");
    *reinterpret_cast<void**>(&nvrtcGetErrorString) =
        dlsym(libnvrtc, "nvrtcGetErrorString");

    // Verify critical symbols
    if (!cuInit || !cudaSetDevice || !cudaMalloc || !nvrtcCompileProgram) {
      fprintf(stderr, "[FakeNVGPU] Failed to load critical CUDA symbols\n");
      return false;
    }

    loaded = true;
    return true;
  }
};

// Convenience accessor
static inline FakeNVGPU_CudaAPI& cuda_api() {
  return FakeNVGPU_CudaAPI::Instance();
}

// ============================================================
// Section 1: C_DeviceInterface implementations
// ============================================================

static C_Status FakeNVGPU_Init() {
  if (!cuda_api().Load()) return C_FAILED;
  int result = cuda_api().cuInit(0);
  if (result != 0) {
    fprintf(stderr, "[FakeNVGPU] cuInit failed: %d\n", result);
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status FakeNVGPU_Finalize() { return C_SUCCESS; }

static C_Status FakeNVGPU_InitDevice(const C_Device device) {
  cuda_api().cudaSetDevice(device->id);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_SetDevice(const C_Device device) {
  cuda_api().cudaSetDevice(device->id);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetDevice(const C_Device device) {
  int id = 0;
  cuda_api().cudaGetDevice(&id);
  device->id = id;
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DestroyDevice(const C_Device /*device*/) {
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetDevicesCount(size_t* count) {
  int n = 0;
  cuda_api().cudaGetDeviceCount(&n);
  *count = static_cast<size_t>(n > 0 ? n : 0);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetDevicesList(size_t* devices) {
  int n = 0;
  cuda_api().cudaGetDeviceCount(&n);
  for (int i = 0; i < n; ++i) devices[i] = static_cast<size_t>(i);
  return C_SUCCESS;
}

// --- Memory ------------------------------------------------------------------

static C_Status FakeNVGPU_Allocate(const C_Device device,
                                   void** ptr,
                                   size_t size) {
  cuda_api().cudaSetDevice(device->id);
  int e = cuda_api().cudaMalloc(ptr, size);
  if (e != 0) {
    *ptr = nullptr;
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status FakeNVGPU_Deallocate(const C_Device /*device*/,
                                     void* ptr,
                                     size_t /*size*/) {
  cuda_api().cudaFree(ptr);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_HostAllocate(const C_Device /*device*/,
                                       void** ptr,
                                       size_t size) {
  int e = cuda_api().cudaMallocHost(ptr, size);
  return (e == 0) ? C_SUCCESS : C_FAILED;
}

static C_Status FakeNVGPU_HostDeallocate(const C_Device /*device*/,
                                         void* ptr,
                                         size_t /*size*/) {
  cuda_api().cudaFreeHost(ptr);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_MemCpyH2D(const C_Device device,
                                    void* dst,
                                    const void* src,
                                    size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpy(dst, src, size, kHostToDevice);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_MemCpyD2H(const C_Device device,
                                    void* dst,
                                    const void* src,
                                    size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpy(dst, src, size, kDeviceToHost);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_MemCpyD2D(const C_Device device,
                                    void* dst,
                                    const void* src,
                                    size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpy(dst, src, size, kDeviceToDevice);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_AsyncMemCpyH2D(const C_Device device,
                                         C_Stream stream,
                                         void* dst,
                                         const void* src,
                                         size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpyAsync(
      dst, src, size, kHostToDevice, reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_AsyncMemCpyD2H(const C_Device device,
                                         C_Stream stream,
                                         void* dst,
                                         const void* src,
                                         size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpyAsync(
      dst, src, size, kDeviceToHost, reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_AsyncMemCpyD2D(const C_Device device,
                                         C_Stream stream,
                                         void* dst,
                                         const void* src,
                                         size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemcpyAsync(
      dst, src, size, kDeviceToDevice, reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_MemSet(const C_Device device,
                                 void* ptr,
                                 unsigned char value,
                                 size_t size) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemset(ptr, static_cast<int>(value), size);
  return C_SUCCESS;
}

// --- Streams -----------------------------------------------------------------

static C_Status FakeNVGPU_CreateStream(const C_Device device,
                                       C_Stream* stream) {
  cuda_api().cudaSetDevice(device->id);
  cudaStream_rt s = nullptr;
  cuda_api().cudaStreamCreate(&s);
  *stream = reinterpret_cast<C_Stream>(s);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DestroyStream(const C_Device /*device*/,
                                        C_Stream stream) {
  cuda_api().cudaStreamDestroy(reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_SyncStream(const C_Device /*device*/,
                                     C_Stream stream) {
  cuda_api().cudaStreamSynchronize(reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

// --- Events ------------------------------------------------------------------

static C_Status FakeNVGPU_CreateEvent(const C_Device device, C_Event* event) {
  cuda_api().cudaSetDevice(device->id);
  cudaEvent_rt e = nullptr;
  cuda_api().cudaEventCreate(&e);
  *event = reinterpret_cast<C_Event>(e);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DestroyEvent(const C_Device /*device*/,
                                       C_Event event) {
  cuda_api().cudaEventDestroy(reinterpret_cast<cudaEvent_rt>(event));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_RecordEvent(const C_Device /*device*/,
                                      C_Stream stream,
                                      C_Event event) {
  cuda_api().cudaEventRecord(reinterpret_cast<cudaEvent_rt>(event),
                             reinterpret_cast<cudaStream_rt>(stream));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_SyncEvent(const C_Device /*device*/, C_Event event) {
  cuda_api().cudaEventSynchronize(reinterpret_cast<cudaEvent_rt>(event));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_StreamWaitEvent(const C_Device /*device*/,
                                          C_Stream stream,
                                          C_Event event) {
  cuda_api().cudaStreamWaitEvent(reinterpret_cast<cudaStream_rt>(stream),
                                 reinterpret_cast<cudaEvent_rt>(event),
                                 0);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_SyncDevice(const C_Device device) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaDeviceSynchronize();
  return C_SUCCESS;
}

// --- Memory stats ------------------------------------------------------------

static C_Status FakeNVGPU_DeviceMemStats(const C_Device device,
                                         size_t* total_memory,
                                         size_t* free_memory) {
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemGetInfo(free_memory, total_memory);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DeviceMinChunkSize(const C_Device /*device*/,
                                             size_t* size) {
  *size = 512;
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DeviceMaxChunkSize(const C_Device /*device*/,
                                             size_t* size) {
  *size = 256UL * 1024 * 1024;  // 256 MB
  return C_SUCCESS;
}

static C_Status FakeNVGPU_DeviceMaxAllocSize(const C_Device device,
                                             size_t* size) {
  size_t total = 0, free_mem = 0;
  cuda_api().cudaSetDevice(device->id);
  cuda_api().cudaMemGetInfo(&free_mem, &total);
  *size = static_cast<size_t>(total * 0.95);
  return C_SUCCESS;
}

// --- Device properties -------------------------------------------------------

static C_Status FakeNVGPU_GetComputeCapability(const C_Device device,
                                               size_t* compute_capability) {
  int major = 0, minor = 0;
  cuda_api().cudaDeviceGetAttribute(
      &major, kComputeCapabilityMajor, device->id);
  cuda_api().cudaDeviceGetAttribute(
      &minor, kComputeCapabilityMinor, device->id);
  *compute_capability = static_cast<size_t>(major * 10 + minor);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMaxSharedMemPerBlock(const C_Device device,
                                                  size_t* mem) {
  int val = 0;
  cuda_api().cudaDeviceGetAttribute(&val, kMaxSharedMemoryPerBlock, device->id);
  *mem = static_cast<size_t>(val);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMaxThreadsPerBlock(const C_Device device,
                                                size_t* threads) {
  int val = 0;
  cuda_api().cudaDeviceGetAttribute(&val, kMaxThreadsPerBlock, device->id);
  *threads = static_cast<size_t>(val);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMaxThreadsPerMultiprocessor(const C_Device device,
                                                         size_t* threads) {
  int val = 0;
  cuda_api().cudaDeviceGetAttribute(
      &val, kMaxThreadsPerMultiProcessor, device->id);
  *threads = static_cast<size_t>(val);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMultiprocessorCount(const C_Device device,
                                                 size_t* count) {
  int val = 0;
  cuda_api().cudaDeviceGetAttribute(&val, kMultiProcessorCount, device->id);
  *count = static_cast<size_t>(val);
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMaxGridDimSize(
    const C_Device device, std::array<unsigned int, 3>* grid_dim_size) {
  int attrs[3] = {kMaxGridDimX, kMaxGridDimY, kMaxGridDimZ};
  for (int i = 0; i < 3; ++i) {
    int val = 0;
    cuda_api().cudaDeviceGetAttribute(&val, attrs[i], device->id);
    (*grid_dim_size)[i] = static_cast<unsigned int>(val);
  }
  return C_SUCCESS;
}

static C_Status FakeNVGPU_GetMaxBlockDimSize(
    const C_Device device, std::array<unsigned int, 3>* block_dim_size) {
  int attrs[3] = {kMaxBlockDimX, kMaxBlockDimY, kMaxBlockDimZ};
  for (int i = 0; i < 3; ++i) {
    int val = 0;
    cuda_api().cudaDeviceGetAttribute(&val, attrs[i], device->id);
    (*block_dim_size)[i] = static_cast<unsigned int>(val);
  }
  return C_SUCCESS;
}

// ============================================================
// Section 2: C_CinnInterface implementations (via Driver API)
// ============================================================

static C_Status FakeNVGPU_CinnCompile(void* /*dev_ptr*/,
                                      const char* code,
                                      char* out_path,
                                      size_t len) {
  if (!code || !out_path || len == 0) return C_FAILED;
  auto& api = cuda_api();

  // Step 1: Create NVRTC program
  nvrtcProgram_t prog = nullptr;
  int r = api.nvrtcCreateProgram(
      &prog, code, "fake_nvgpu_kernel.cu", 0, nullptr, nullptr);
  if (r != 0) {
    fprintf(stderr, "[FakeNVGPU] nvrtcCreateProgram failed: %d\n", r);
    return C_FAILED;
  }

  // Determine target architecture
  int device_id = 0;
  api.cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  api.cudaDeviceGetAttribute(&major, kComputeCapabilityMajor, device_id);
  api.cudaDeviceGetAttribute(&minor, kComputeCapabilityMinor, device_id);
  char arch_flag[32];
  snprintf(
      arch_flag, sizeof(arch_flag), "--gpu-architecture=sm_%d%d", major, minor);

  const char* opts[] = {arch_flag, "--ftz=true", "--fmad=true"};
  int compile_res = api.nvrtcCompileProgram(prog, 3, opts);

  if (compile_res != 0) {
    size_t log_size = 0;
    api.nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    api.nvrtcGetProgramLog(prog, &log[0]);
    fprintf(stderr, "[FakeNVGPU] NVRTC compile error:\n%s\n", log.c_str());
    api.nvrtcDestroyProgram(&prog);
    return C_FAILED;
  }

  // Step 2: Get CUBIN (device-native binary, avoids PTX version mismatch)
  size_t cubin_size = 0;
  api.nvrtcGetCUBINSize(prog, &cubin_size);

  std::string binary;
  std::string path;
  if (cubin_size > 0) {
    // CUBIN available (sm_XX arch produces native code)
    binary.resize(cubin_size);
    api.nvrtcGetCUBIN(prog, &binary[0]);
    path = "/tmp/fake_nvgpu_kernel.cubin";
  } else {
    // Fallback to PTX if CUBIN not available
    size_t ptx_size = 0;
    api.nvrtcGetPTXSize(prog, &ptx_size);
    binary.resize(ptx_size);
    api.nvrtcGetPTX(prog, &binary[0]);
    path = "/tmp/fake_nvgpu_kernel.ptx";
  }
  api.nvrtcDestroyProgram(&prog);

  // Step 3: Write binary to file
  {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
      fprintf(stderr, "[FakeNVGPU] Cannot write to %s\n", path.c_str());
      return C_FAILED;
    }
    f.write(binary.data(), static_cast<std::streamsize>(binary.size()));
  }

  snprintf(out_path, len, "%s", path.c_str());
  return C_SUCCESS;
}

static const char* FakeNVGPU_CinnGetRuntimeSource(void* /*dev_ptr*/) {
  return "";
}

static C_Status FakeNVGPU_CinnModuleLoad(void* /*dev_ptr*/,
                                         const char* path,
                                         void** mod_out) {
  if (!path || !mod_out) return C_FAILED;

  // Read the binary file into memory
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    fprintf(stderr, "[FakeNVGPU] Cannot open module file: %s\n", path);
    return C_FAILED;
  }
  auto file_size = f.tellg();
  f.seekg(0);
  std::string data(static_cast<size_t>(file_size), '\0');
  f.read(&data[0], file_size);
  f.close();

  // Use cuModuleLoadData to load from memory (works for both CUBIN and PTX)
  CUmodule_t mod = nullptr;
  int result = cuda_api().cuModuleLoadData(&mod, data.data());
  if (result != 0) {
    const char* estr = nullptr;
    cuda_api().cuGetErrorString(result, &estr);
    fprintf(stderr,
            "[FakeNVGPU] cuModuleLoadData(%s) failed: %s\n",
            path,
            estr ? estr : "?");
    return C_FAILED;
  }
  *mod_out = mod;
  return C_SUCCESS;
}

static C_Status FakeNVGPU_CinnModuleUnload(void* /*dev_ptr*/,
                                           void* module_handle) {
  if (!module_handle) return C_SUCCESS;
  cuda_api().cuModuleUnload(reinterpret_cast<CUmodule_t>(module_handle));
  return C_SUCCESS;
}

static C_Status FakeNVGPU_CinnGetKernelAddress(void* /*dev_ptr*/,
                                               void* module_handle,
                                               const char* func_name,
                                               void** func_out) {
  if (!module_handle || !func_name || !func_out) return C_FAILED;
  CUfunction_t fn = nullptr;
  int result = cuda_api().cuModuleGetFunction(
      &fn, reinterpret_cast<CUmodule_t>(module_handle), func_name);
  if (result != 0) {
    const char* estr = nullptr;
    cuda_api().cuGetErrorString(result, &estr);
    fprintf(stderr,
            "[FakeNVGPU] cuModuleGetFunction(%s) failed: %s\n",
            func_name,
            estr ? estr : "?");
    return C_FAILED;
  }
  *func_out = fn;
  return C_SUCCESS;
}

static C_Status FakeNVGPU_CinnLaunchKernel(void* /*dev_ptr*/,
                                           void* func_ptr,
                                           void** args,
                                           int /*num_args*/,
                                           int gx,
                                           int gy,
                                           int gz,
                                           int bx,
                                           int by,
                                           int bz,
                                           int shm,
                                           void* stream) {
  int result =
      cuda_api().cuLaunchKernel(reinterpret_cast<CUfunction_t>(func_ptr),
                                static_cast<unsigned>(gx),
                                static_cast<unsigned>(gy),
                                static_cast<unsigned>(gz),
                                static_cast<unsigned>(bx),
                                static_cast<unsigned>(by),
                                static_cast<unsigned>(bz),
                                static_cast<unsigned>(shm),
                                reinterpret_cast<CUstream_t>(stream),
                                args,
                                nullptr);
  if (result != 0) {
    const char* estr = nullptr;
    cuda_api().cuGetErrorString(result, &estr);
    fprintf(
        stderr, "[FakeNVGPU] cuLaunchKernel failed: %s\n", estr ? estr : "?");
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status FakeNVGPU_CinnApplyCustomPass(void* /*dev_ptr*/,
                                              void* /*ir_module*/) {
  return C_SUCCESS;
}

// ============================================================
// Section 3: Public Init function
// ============================================================

#define FAKE_NVGPU_DEVICE_TYPE "FakeNVGPU"
#define FAKE_NVGPU_SUB_DEVICE_TYPE "CUDA"

static C_CinnInterface g_fake_nvgpu_cinn_interface;

static inline void InitFakeNVGPUDevice(CustomRuntimeParams* params) {
  params->device_type = const_cast<char*>(FAKE_NVGPU_DEVICE_TYPE);
  params->sub_device_type = const_cast<char*>(FAKE_NVGPU_SUB_DEVICE_TYPE);
  params->version.major = PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION;
  params->version.minor = PADDLE_CUSTOM_RUNTIME_MINOR_VERSION;
  params->version.patch = PADDLE_CUSTOM_RUNTIME_PATCH_VERSION;

  std::memset(params->interface, 0, sizeof(C_DeviceInterface));
  params->interface->size = sizeof(C_DeviceInterface);

  // --- Required fields ---
  params->interface->set_device = FakeNVGPU_SetDevice;
  params->interface->get_device = FakeNVGPU_GetDevice;
  params->interface->create_event = FakeNVGPU_CreateEvent;
  params->interface->record_event = FakeNVGPU_RecordEvent;
  params->interface->destroy_event = FakeNVGPU_DestroyEvent;
  params->interface->synchronize_event = FakeNVGPU_SyncEvent;
  params->interface->device_memory_allocate = FakeNVGPU_Allocate;
  params->interface->device_memory_deallocate = FakeNVGPU_Deallocate;
  params->interface->get_device_count = FakeNVGPU_GetDevicesCount;
  params->interface->get_device_list = FakeNVGPU_GetDevicesList;

  // --- Optional but exercised by CINN ---
  params->interface->initialize = FakeNVGPU_Init;
  params->interface->finalize = FakeNVGPU_Finalize;
  params->interface->init_device = FakeNVGPU_InitDevice;
  params->interface->deinit_device = FakeNVGPU_DestroyDevice;

  params->interface->create_stream = FakeNVGPU_CreateStream;
  params->interface->destroy_stream = FakeNVGPU_DestroyStream;
  params->interface->synchronize_stream = FakeNVGPU_SyncStream;
  params->interface->synchronize_device = FakeNVGPU_SyncDevice;
  params->interface->stream_wait_event = FakeNVGPU_StreamWaitEvent;

  params->interface->memory_copy_h2d = FakeNVGPU_MemCpyH2D;
  params->interface->memory_copy_d2h = FakeNVGPU_MemCpyD2H;
  params->interface->memory_copy_d2d = FakeNVGPU_MemCpyD2D;
  params->interface->async_memory_copy_h2d = FakeNVGPU_AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2h = FakeNVGPU_AsyncMemCpyD2H;
  params->interface->async_memory_copy_d2d = FakeNVGPU_AsyncMemCpyD2D;
  params->interface->host_memory_allocate = FakeNVGPU_HostAllocate;
  params->interface->host_memory_deallocate = FakeNVGPU_HostDeallocate;
  params->interface->device_memory_set = FakeNVGPU_MemSet;

  params->interface->device_memory_stats = FakeNVGPU_DeviceMemStats;
  params->interface->device_min_chunk_size = FakeNVGPU_DeviceMinChunkSize;
  params->interface->device_max_chunk_size = FakeNVGPU_DeviceMaxChunkSize;
  params->interface->device_max_alloc_size = FakeNVGPU_DeviceMaxAllocSize;

  params->interface->get_compute_capability = FakeNVGPU_GetComputeCapability;
  params->interface->get_max_shared_mem_per_block =
      FakeNVGPU_GetMaxSharedMemPerBlock;
  params->interface->get_max_threads_per_block =
      FakeNVGPU_GetMaxThreadsPerBlock;
  params->interface->get_max_threads_per_mp =
      FakeNVGPU_GetMaxThreadsPerMultiprocessor;
  params->interface->get_multi_process = FakeNVGPU_GetMultiprocessorCount;
  params->interface->get_max_grid_dim_size = FakeNVGPU_GetMaxGridDimSize;
  params->interface->get_max_block_dim_size = FakeNVGPU_GetMaxBlockDimSize;

  // --- C_CinnInterface ---
  std::memset(&g_fake_nvgpu_cinn_interface, 0, sizeof(C_CinnInterface));
  g_fake_nvgpu_cinn_interface.size = sizeof(C_CinnInterface);
  g_fake_nvgpu_cinn_interface.dev_ptr = nullptr;

  g_fake_nvgpu_cinn_interface.compile = FakeNVGPU_CinnCompile;
  g_fake_nvgpu_cinn_interface.get_runtime_source =
      FakeNVGPU_CinnGetRuntimeSource;
  g_fake_nvgpu_cinn_interface.module_load = FakeNVGPU_CinnModuleLoad;
  g_fake_nvgpu_cinn_interface.module_unload = FakeNVGPU_CinnModuleUnload;
  g_fake_nvgpu_cinn_interface.get_kernel_address =
      FakeNVGPU_CinnGetKernelAddress;
  g_fake_nvgpu_cinn_interface.launch_kernel = FakeNVGPU_CinnLaunchKernel;
  g_fake_nvgpu_cinn_interface.apply_custom_pass = FakeNVGPU_CinnApplyCustomPass;

  params->interface->cinn_interface = &g_fake_nvgpu_cinn_interface;
}
