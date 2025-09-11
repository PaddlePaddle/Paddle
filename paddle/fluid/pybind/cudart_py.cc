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

#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/pybind/cudart_py.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "paddle/phi/core/platform/cuda_device_guard.h"

#if !defined(USE_ROCM)
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

namespace py = pybind11;
namespace paddle {
namespace pybind {
void BindCudaRt(py::module* m) {
  auto cudart = m->def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaOutputMode_t is used in cudaProfilerInitialize only. The latter is gone
  // in CUDA 12.
  py::enum_<cudaOutputMode_t>(cudart,
                              "cuda"
                              "OutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);
#endif

  py::enum_<cudaError_t>(cudart,
                         "cuda"
                         "Error")
      .value("success", cudaSuccess);

  cudart.def(
      "cuda"
      "GetErrorString",
      cudaGetErrorString);

  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStart
#endif
  );

  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStop
#endif
  );

  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> cudaError_t {
        py::gil_scoped_release no_gil;
        return cudaHostRegister(reinterpret_cast<void*>(ptr), size, flags);
      });

  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        return cudaHostUnregister(reinterpret_cast<void*>(ptr));
      });

  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        return cudaStreamCreate(reinterpret_cast<cudaStream_t*>(ptr));
      });

  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        return (cudaStreamDestroy((cudaStream_t)ptr));
      });

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaProfilerInitialize is no longer needed after CUDA 12:
  // https://forums.developer.nvidia.com/t/cudaprofilerinitialize-is-deprecated-alternative/200776/3
  cudart.def(
      "cuda"
      "ProfilerInitialize",
      cudaProfilerInitialize,
      py::call_guard<py::gil_scoped_release>());

#endif
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](int device) -> std::pair<size_t, size_t> {
        const auto& place = phi::GPUPlace(device);
        platform::CUDADeviceGuard cuda_guard(place);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        cudaMemGetInfo(&device_free, &device_total);
        return {device_free, device_total};
      });
}
}  // namespace pybind
}  // namespace paddle

#endif  // if defined(PADDLE_WITH_CUDA)
