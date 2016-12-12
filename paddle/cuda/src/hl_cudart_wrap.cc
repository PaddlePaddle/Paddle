/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_USE_DSO

#include <cuda_runtime.h>
#include <mutex>
#include "hl_dso_loader.h"

/**
 * cudart wrapper: for dynamic load libcudart.so.
 * When nvcc compile cuda kernels, it will insert
 * some build-in runtime routines, which must be
 * provided by us if PADDLE_USE_DSO is true. If
 * PADDLE_USE_DSO is false, all of them must be
 * ignored to avoid multiple definitions.
 */
namespace dynload {

extern std::once_flag cudart_dso_flag;
extern void *cudart_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cuda routine
 * via operator overloading.
 **/
#define DYNAMIC_LOAD_CUDART_WRAP(__name, __type)                               \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    __type operator()(Args... args) {                                          \
      typedef __type (*cudartFunc)(Args...);                                   \
      std::call_once(cudart_dso_flag, GetCudartDsoHandle, &cudart_dso_handle); \
      void *p_##__name = dlsym(cudart_dso_handle, #__name);                    \
      return reinterpret_cast<cudartFunc>(p_##__name)(args...);                \
    }                                                                          \
  } __name; /* struct DynLoad__##__name */

/* include all needed cuda functions in HPPL */
// clang-format off
#define CUDA_ROUTINE_EACH(__macro)          \
  __macro(cudaLaunch, cudaError_t)          \
  __macro(cudaSetupArgument, cudaError_t)   \
  __macro(cudaConfigureCall, cudaError_t)   \
  __macro(__cudaRegisterFatBinary, void**)  \
  __macro(__cudaUnregisterFatBinary, void)  \
  __macro(__cudaRegisterFunction, void)     \
  __macro(__cudaRegisterVar, void)          \
  __macro(__cudaRegisterManagedVar, void)   \
  __macro(__cudaInitModule, char)           \
  __macro(__cudaRegisterTexture, void)      \
  __macro(__cudaRegisterSurface, void)
// clang-format on

CUDA_ROUTINE_EACH(DYNAMIC_LOAD_CUDART_WRAP)

#if CUDART_VERSION >= 7000
DYNAMIC_LOAD_CUDART_WRAP(cudaLaunchKernel, cudaError_t)
#endif

#undef CUDA_ROUNTINE_EACH

} /* namespace dynload */

#if CUDART_VERSION >= 7000
__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func,
                                                dim3 gridDim,
                                                dim3 blockDim,
                                                void **args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  return dynload::cudaLaunchKernel(
      func, gridDim, blockDim, args, sharedMem, stream);
}
#endif /* CUDART_VERSION >= 7000 */

__host__ cudaError_t CUDARTAPI cudaLaunch(const void *func) {
  return dynload::cudaLaunch(func);
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg,
                                                 size_t size,
                                                 size_t offset) {
  return dynload::cudaSetupArgument(arg, size, offset);
}

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim,
                                                 dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  return dynload::cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

extern "C" {

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  return dynload::__cudaRegisterFatBinary(fatCubin);
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
  return dynload::__cudaUnregisterFatBinary(fatCubinHandle);
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun,
                                      char *deviceFun,
                                      const char *deviceName,
                                      int thread_limit,
                                      uint3 *tid,
                                      uint3 *bid,
                                      dim3 *bDim,
                                      dim3 *gDim,
                                      int *wSize) {
  return dynload::__cudaRegisterFunction(fatCubinHandle,
                                         hostFun,
                                         deviceFun,
                                         deviceName,
                                         thread_limit,
                                         tid,
                                         bid,
                                         bDim,
                                         gDim,
                                         wSize);
}

void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
                                 char *hostVar,
                                 char *deviceAddress,
                                 const char *deviceName,
                                 int ext,
                                 int size,
                                 int constant,
                                 int global) {
  return dynload::__cudaRegisterVar(fatCubinHandle,
                                    hostVar,
                                    deviceAddress,
                                    deviceName,
                                    ext,
                                    size,
                                    constant,
                                    global);
}

extern void CUDARTAPI __cudaRegisterManagedVar(void **fatCubinHandle,
                                               void **hostVarPtrAddress,
                                               char *deviceAddress,
                                               const char *deviceName,
                                               int ext,
                                               int size,
                                               int constant,
                                               int global) {
  return dynload::__cudaRegisterManagedVar(fatCubinHandle,
                                           hostVarPtrAddress,
                                           deviceAddress,
                                           deviceName,
                                           ext,
                                           size,
                                           constant,
                                           global);
}

char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
  return dynload::__cudaInitModule(fatCubinHandle);
}

void CUDARTAPI __cudaRegisterTexture(void **fatCubinHandle,
                                     const struct textureReference *hostVar,
                                     const void **deviceAddress,
                                     const char *deviceName,
                                     int dim,
                                     int norm,
                                     int ext) {
  return dynload::__cudaRegisterTexture(
      fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
}

void CUDARTAPI __cudaRegisterSurface(void **fatCubinHandle,
                                     const struct surfaceReference *hostVar,
                                     const void **deviceAddress,
                                     const char *deviceName,
                                     int dim,
                                     int ext) {
  return dynload::__cudaRegisterSurface(
      fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
}

} /* extern "C" */

#endif
