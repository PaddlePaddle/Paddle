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

  struct PaddleCudaError {
    cudaError_t value;
    PaddleCudaError() : value(cudaSuccess) {}
    explicit PaddleCudaError(cudaError_t v) : value(v) {}
    explicit PaddleCudaError(int v) : value(static_cast<cudaError_t>(v)) {}
    operator cudaError_t() const { return value; }
    operator int() const { return static_cast<int>(value); }
    bool operator==(const PaddleCudaError& other) const {
      return value == other.value;
    }
    bool operator!=(const PaddleCudaError& other) const {
      return value != other.value;
    }
    bool operator==(cudaError_t other) const { return value == other; }
    bool operator!=(cudaError_t other) const { return value != other; }
    bool operator==(int other) const {
      return static_cast<int>(value) == other;
    }
    bool operator!=(int other) const {
      return static_cast<int>(value) != other;
    }
    int to_int() const { return static_cast<int>(value); }
    cudaError_t get_value() const { return value; }
  };

  py::class_<PaddleCudaError>(cudart, "cudaError")
      .def(py::init<int>(), "Create from integer value")
      .def(py::init<>(), "Default constructor")
      .def("__int__", &PaddleCudaError::to_int)
      .def("get_value",
           &PaddleCudaError::get_value,
           "Get the underlying cudaError_t value")
      .def("__eq__",
           [](const PaddleCudaError& a, const PaddleCudaError& b) {
             return a == b;
           })
      .def("__eq__", [](const PaddleCudaError& a, int b) { return a == b; })
      .def("__ne__",
           [](const PaddleCudaError& a, const PaddleCudaError& b) {
             return a != b;
           })
      .def("__ne__", [](const PaddleCudaError& a, int b) { return a != b; })
      .def("__repr__", [](const PaddleCudaError& error) -> std::string {
        switch (error.value) {
          case cudaSuccess:
            return "cudaError.success";
          default:
            return "cudaError(" +
                   std::to_string(static_cast<int>(error.value)) + ")";
        }
      });

  cudart.attr("cudaError").attr("success") = PaddleCudaError(cudaSuccess);

  cudart.def(
      "cudaGetErrorString",
      [](const PaddleCudaError& error) -> std::string {
        return std::string(cudaGetErrorString(error.value));
      },
      "Get error string for cuda error");

  cudart.def(
      "cudaGetErrorString",
      [](int error_code) -> std::string {
        return std::string(
            cudaGetErrorString(static_cast<cudaError_t>(error_code)));
      },
      "Get error string for cuda error code");

  cudart.def("cudaGetErrorString", cudaGetErrorString);

  cudart.def("cudaProfilerStart",
#ifdef USE_ROCM
             []() -> PaddleCudaError { return PaddleCudaError(hipSuccess); }
#else
      []() -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        return PaddleCudaError(cudaProfilerStart());
      }
#endif
  );

  cudart.def("cudaProfilerStop",
#ifdef USE_ROCM
             []() -> PaddleCudaError { return PaddleCudaError(hipSuccess); }
#else
      []() -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        return PaddleCudaError(cudaProfilerStop());
      }
#endif
  );

  cudart.def(
      "cudaHostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        cudaError_t result =
            cudaHostRegister(reinterpret_cast<void*>(ptr), size, flags);
        return PaddleCudaError(result);
      });

  cudart.def("cudaHostUnregister", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaHostUnregister(reinterpret_cast<void*>(ptr));
    return PaddleCudaError(result);
  });

  cudart.def("cudaStreamCreate", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(ptr));
    return PaddleCudaError(result);
  });

  cudart.def("cudaStreamDestroy", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ptr));
    return PaddleCudaError(result);
  });

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaProfilerInitialize is no longer needed after CUDA 12
  cudart.def("cudaProfilerInitialize",
             [](const char* configFile,
                const char* outputFile,
                cudaOutputMode_t outputMode) -> PaddleCudaError {
               py::gil_scoped_release no_gil;
               cudaError_t result =
                   cudaProfilerInitialize(configFile, outputFile, outputMode);
               return PaddleCudaError(result);
             });
#endif

  cudart.def("cudaMemGetInfo", [](int device) -> std::pair<size_t, size_t> {
    const auto& place = phi::GPUPlace(device);
    platform::CUDADeviceGuard cuda_guard(place);
    size_t device_free = 0;
    size_t device_total = 0;
    py::gil_scoped_release no_gil;
    cudaMemGetInfo(&device_free, &device_total);
    return {device_free, device_total};
  });

  cudart.def(
      "cudaMemcpy",
      [](py::int_ dst, py::int_ src, size_t count, int kind)
          -> PaddleCudaError {
        void* dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
        const void* src_ptr =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
        cudaError_t result = cudaMemcpy(
            dst_ptr, src_ptr, count, static_cast<cudaMemcpyKind>(kind));
        return PaddleCudaError(result);
      },
      "Copy memory");

  cudart.def(
      "cudaMemcpyAsync",
      [](py::int_ dst, py::int_ src, size_t count, int kind, py::int_ stream)
          -> PaddleCudaError {
        void* dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
        const void* src_ptr =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
        cudaStream_t cuda_stream =
            reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
        cudaError_t result = cudaMemcpyAsync(dst_ptr,
                                             src_ptr,
                                             count,
                                             static_cast<cudaMemcpyKind>(kind),
                                             cuda_stream);
        return PaddleCudaError(result);
      },
      "Copy memory asynchronously");

  cudart.def(
      "cudaStreamSynchronize",
      [](py::int_ stream) -> PaddleCudaError {
        cudaStream_t cuda_stream =
            reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
        cudaError_t result = cudaStreamSynchronize(cuda_stream);
        return PaddleCudaError(result);
      },
      "Synchronize stream");

  cudart.def(
      "cudaDeviceSynchronize",
      []() -> PaddleCudaError {
        cudaError_t result = cudaDeviceSynchronize();
        return PaddleCudaError(result);
      },
      "Synchronize device");

  cudart.def(
      "cudaGetLastError",
      []() -> PaddleCudaError {
        cudaError_t result = cudaGetLastError();
        return PaddleCudaError(result);
      },
      "Get last CUDA error");

  cudart.def(
      "cudaPeekAtLastError",
      []() -> PaddleCudaError {
        cudaError_t result = cudaPeekAtLastError();
        return PaddleCudaError(result);
      },
      "Peek at last CUDA error without clearing it");

  cudart.attr("cudaMemcpyHostToHost") = static_cast<int>(cudaMemcpyHostToHost);
  cudart.attr("cudaMemcpyHostToDevice") =
      static_cast<int>(cudaMemcpyHostToDevice);
  cudart.attr("cudaMemcpyDeviceToHost") =
      static_cast<int>(cudaMemcpyDeviceToHost);
  cudart.attr("cudaMemcpyDeviceToDevice") =
      static_cast<int>(cudaMemcpyDeviceToDevice);
  cudart.attr("cudaMemcpyDefault") = static_cast<int>(cudaMemcpyDefault);

  cudart.attr("cudaHostRegisterDefault") =
      static_cast<unsigned int>(cudaHostRegisterDefault);
  cudart.attr("cudaHostRegisterPortable") =
      static_cast<unsigned int>(cudaHostRegisterPortable);
  cudart.attr("cudaHostRegisterMapped") =
      static_cast<unsigned int>(cudaHostRegisterMapped);
  cudart.attr("cudaHostRegisterIoMemory") =
      static_cast<unsigned int>(cudaHostRegisterIoMemory);

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  struct PaddleCudaOutputMode {
    cudaOutputMode_t value;
    PaddleCudaOutputMode() : value(cudaKeyValuePair) {}
    explicit PaddleCudaOutputMode(cudaOutputMode_t v) : value(v) {}
    explicit PaddleCudaOutputMode(int v)
        : value(static_cast<cudaOutputMode_t>(v)) {}
    operator cudaOutputMode_t() const { return value; }
    operator int() const { return static_cast<int>(value); }
    bool operator==(const PaddleCudaOutputMode& other) const {
      return value == other.value;
    }
    bool operator!=(const PaddleCudaOutputMode& other) const {
      return value != other.value;
    }
    bool operator==(cudaOutputMode_t other) const { return value == other; }
    bool operator!=(cudaOutputMode_t other) const { return value != other; }
    bool operator==(int other) const {
      return static_cast<int>(value) == other;
    }
    bool operator!=(int other) const {
      return static_cast<int>(value) != other;
    }
    int to_int() const { return static_cast<int>(value); }
  };

  py::class_<PaddleCudaOutputMode>(cudart, "cudaOutputMode")
      .def(py::init<int>(), "Create from integer value")
      .def("__int__", &PaddleCudaOutputMode::to_int)
      .def("__eq__",
           [](const PaddleCudaOutputMode& a, const PaddleCudaOutputMode& b) {
             return a == b;
           })
      .def("__eq__",
           [](const PaddleCudaOutputMode& a, int b) { return a == b; })
      .def("__ne__",
           [](const PaddleCudaOutputMode& a, const PaddleCudaOutputMode& b) {
             return a != b;
           })
      .def("__ne__",
           [](const PaddleCudaOutputMode& a, int b) { return a != b; })
      .def("__repr__", [](const PaddleCudaOutputMode& mode) -> std::string {
        switch (mode.value) {
          case cudaKeyValuePair:
            return "cudaOutputMode.KeyValuePair";
          case cudaCSV:
            return "cudaOutputMode.CSV";
          default:
            return "cudaOutputMode(" +
                   std::to_string(static_cast<int>(mode.value)) + ")";
        }
      });

  cudart.attr("cudaOutputMode").attr("KeyValuePair") =
      PaddleCudaOutputMode(cudaKeyValuePair);
  cudart.attr("cudaOutputMode").attr("CSV") = PaddleCudaOutputMode(cudaCSV);
#endif

  cudart.def(
      "cudaGetErrorString",
      [](const PaddleCudaError& error) -> std::string {
        return std::string(cudaGetErrorString(error.value));
      },
      "Get error string for cuda error");

  cudart.def(
      "cudaGetErrorString",
      [](int error_code) -> std::string {
        return std::string(
            cudaGetErrorString(static_cast<cudaError_t>(error_code)));
      },
      "Get error string for cuda error code");

  cudart.def("cudaGetErrorString", cudaGetErrorString);

  cudart.def("cudaProfilerStart",
#ifdef USE_ROCM
             []() -> PaddleCudaError { return PaddleCudaError(hipSuccess); }
#else
      []() -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        return PaddleCudaError(cudaProfilerStart());
      }
#endif
  );

  cudart.def("cudaProfilerStop",
#ifdef USE_ROCM
             []() -> PaddleCudaError { return PaddleCudaError(hipSuccess); }
#else
      []() -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        return PaddleCudaError(cudaProfilerStop());
      }
#endif
  );

  cudart.def(
      "cudaHostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> PaddleCudaError {
        py::gil_scoped_release no_gil;
        cudaError_t result =
            cudaHostRegister(reinterpret_cast<void*>(ptr), size, flags);
        return PaddleCudaError(result);
      });

  cudart.def("cudaHostUnregister", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaHostUnregister(reinterpret_cast<void*>(ptr));
    return PaddleCudaError(result);
  });

  cudart.def("cudaStreamCreate", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(ptr));
    return PaddleCudaError(result);
  });

  cudart.def("cudaStreamDestroy", [](uintptr_t ptr) -> PaddleCudaError {
    py::gil_scoped_release no_gil;
    cudaError_t result = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ptr));
    return PaddleCudaError(result);
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
