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

#include "paddle/phi/kernels/legacy/gpu/tensor_debug.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "paddle/common/enforce.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// Maximum tensor rank supported.  Shapes are passed as plain int64_t values
// in registers to avoid any host/device memory transfers (which are forbidden
// inside a CUDA Graph capture region).
static constexpr int kMaxDims = 9;

// dtype tag passed as an integer so we need no device-side char* pointer.
enum class DebugDtype : int {
  FLOAT32 = 0,
  FLOAT64,
  FLOAT16,
  BFLOAT16,
  INT32,
  INT64,
  INT16,
  INT8,
  UINT8,
  BOOL,
};

// ---------------------------------------------------------------------------
// Helper: dtype name from enum tag (device-side string literal, no pointer)
// ---------------------------------------------------------------------------
__device__ static const char* DtypeName(DebugDtype tag) {
  switch (tag) {
    case DebugDtype::FLOAT32:
      return "FLOAT32";
    case DebugDtype::FLOAT64:
      return "FLOAT64";
    case DebugDtype::FLOAT16:
      return "FLOAT16";
    case DebugDtype::BFLOAT16:
      return "BFLOAT16";
    case DebugDtype::INT32:
      return "INT32";
    case DebugDtype::INT64:
      return "INT64";
    case DebugDtype::INT16:
      return "INT16";
    case DebugDtype::INT8:
      return "INT8";
    case DebugDtype::UINT8:
      return "UINT8";
    case DebugDtype::BOOL:
      return "BOOL";
    default:
      return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// Shape is passed as a fixed-size struct so all values live in registers.
// No device-memory pointer is needed.
// ---------------------------------------------------------------------------
struct ShapeArgs {
  int64_t dims[kMaxDims];
  int ndim;
};

// ---------------------------------------------------------------------------
// Device helpers: compute multi-dim indices, count leading/trailing bracket
// events for shape-aware nested printing.
//
// Algorithm (single thread, linear scan):
//   For linear index i:
//     * compute per-dimension indices idx[d] = (i / stride[d]) % dims[d]
//     * "open_dims"  = number of rightmost dims whose idx is 0
//                     -> print that many '[' before the value
//     * "close_dims" = number of rightmost dims at their last position
//                     -> print that many ']' after the value, then
//                     newline+indent
//   Special cases: scalar (ndim==0) and 1-D tensors.
// ---------------------------------------------------------------------------

// Compute strides for C-contiguous layout.
__device__ static void ComputeStrides(const ShapeArgs& shape,
                                      int64_t* strides) {
  strides[shape.ndim - 1] = 1;
  for (int d = shape.ndim - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * shape.dims[d + 1];
  }
}

// For element at linear index i, how many trailing dimensions have idx==0
// (i.e. we are at the start of a new sub-array in those dims).
__device__ static int CountOpenDims(int64_t i,
                                    const ShapeArgs& shape,
                                    const int64_t* strides) {
  int count = 0;
  for (int d = shape.ndim - 1; d >= 0; --d) {
    if ((i / strides[d]) % shape.dims[d] == 0) {
      ++count;
    } else {
      break;
    }
  }
  return count;
}

// For element at linear index i, how many trailing dimensions are at their
// last position (i.e. we just finished a sub-array in those dims).
__device__ static int CountCloseDims(int64_t i,
                                     const ShapeArgs& shape,
                                     const int64_t* strides) {
  int count = 0;
  for (int d = shape.ndim - 1; d >= 0; --d) {
    int64_t idx_d = (i / strides[d]) % shape.dims[d];
    if (idx_d == shape.dims[d] - 1) {
      ++count;
    } else {
      break;
    }
  }
  return count;
}

// Print the prefix (opening brackets / comma / newline+indent) before element
// i.
__device__ static void PrintPrefix(int64_t i,
                                   const ShapeArgs& shape,
                                   const int64_t* strides,
                                   int close_prev) {
  if (i == 0) {
    // Opening brackets for all dimensions.
    for (int d = 0; d < shape.ndim; ++d) printf("[");
    return;
  }

  if (close_prev > 0) {
    // After closing brackets we start a new row: comma, newline, indent.
    printf(",\n");
    // Indent = (ndim - close_prev) spaces so the opening '[' aligns.
    int indent = shape.ndim - close_prev;
    for (int d = 0; d < indent; ++d) printf(" ");
    for (int d = 0; d < close_prev; ++d) printf("[");
  } else {
    printf(", ");
  }
}

// ---------------------------------------------------------------------------
// Typed print helpers (device-side, called from the kernel body)
// ---------------------------------------------------------------------------
template <typename T>
__device__ static void PrintValue(T v) {
  printf("%.6g", static_cast<double>(v));
}

template <>
__device__ void PrintValue<__half>(__half v) {
  printf("%.6g", static_cast<double>(__half2float(v)));
}

template <>
__device__ void PrintValue<__nv_bfloat16>(__nv_bfloat16 v) {
  printf("%.6g", static_cast<double>(__bfloat162float(v)));
}

template <>
__device__ void PrintValue<bool>(bool v) {
  printf("%s", v ? "True" : "False");
}

// ---------------------------------------------------------------------------
// CUDA kernel: single thread, shape-aware nested printing.
// All state (shape, strides) lives in registers -- CUDA Graph safe.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void PrintTensorKernel(const T* data,
                                  int64_t numel,
                                  ShapeArgs shape,
                                  DebugDtype dtag) {
  // -- header ----------------------------------------------------------------
  printf("[TensorDebug] dtype : %s\n", DtypeName(dtag));

  printf("[TensorDebug] shape : [");
  for (int i = 0; i < shape.ndim; ++i) {
    if (i > 0) printf(", ");
    printf("%lld", static_cast<long long>(shape.dims[i]));  // NOLINT
  }
  printf("]\n");

  printf("[TensorDebug] numel : %lld\n",
         static_cast<long long>(numel));  // NOLINT

  // -- data: shape-aware nested printing -------------------------------------
  printf("[TensorDebug] data  :");

  if (numel == 0) {
    printf(" []\n");
    return;
  }

  // Scalar (0-D tensor)
  if (shape.ndim == 0) {
    printf(" ");
    PrintValue(data[0]);
    printf("\n");
    return;
  }

  printf("\n");

  // Compute strides in registers (no device malloc).
  int64_t strides[kMaxDims] = {};
  ComputeStrides(shape, strides);

  int close_prev = 0;  // closing brackets printed after the previous element

  for (int64_t i = 0; i < numel; ++i) {
    PrintPrefix(i, shape, strides, close_prev);
    PrintValue(data[i]);

    close_prev = CountCloseDims(i, shape, strides);
    // Print closing brackets after the value (before the next prefix).
    for (int d = 0; d < close_prev; ++d) printf("]");
  }

  printf("\n");
}

// ---------------------------------------------------------------------------
// Helper: build ShapeArgs on the host and launch the kernel.
// No dynamic allocation, no memcpy, no stream sync -> CUDA Graph safe.
// ---------------------------------------------------------------------------
template <typename T>
static void LaunchPrint(const T* d_data,
                        int64_t numel,
                        const int64_t* host_shape,
                        int ndim,
                        DebugDtype dtag,
                        cudaStream_t stream) {
  ShapeArgs shape{};
  shape.ndim = ndim;
  for (int i = 0; i < ndim && i < kMaxDims; ++i) {
    shape.dims[i] = host_shape[i];
  }

  // Single-thread kernel, all arguments in registers.
  // No malloc/free/memcpy -> safe inside a CUDA Graph capture region.
  PrintTensorKernel<T><<<1, 1, 0, stream>>>(d_data, numel, shape, dtag);
}

// ---------------------------------------------------------------------------
// Public entry: DebugPrintGPUTensor
// ---------------------------------------------------------------------------
void DebugPrintGPUTensor(const phi::DenseTensor& tensor, cudaStream_t stream) {
  PADDLE_ENFORCE_EQ(tensor.place().GetType() == phi::AllocationType::GPU,
                    true,
                    phi::errors::InvalidArgument(
                        "DebugPrintGPUTensor only supports GPU DenseTensors. "
                        "Please call tensor.cuda() first."));

  PADDLE_ENFORCE_LE(
      tensor.dims().size(),
      kMaxDims,
      phi::errors::InvalidArgument(
          "DebugPrintGPUTensor: tensor rank %d exceeds kMaxDims (%d).",
          tensor.dims().size(),
          kMaxDims));

  auto dtype = tensor.dtype();
  int64_t numel = tensor.numel();
  int ndim = tensor.dims().size();
  const int64_t* host_shape = tensor.dims().Get();

#define DISPATCH(cpp_type, enum_val)               \
  case phi::DataType::enum_val: {                  \
    LaunchPrint<cpp_type>(tensor.data<cpp_type>(), \
                          numel,                   \
                          host_shape,              \
                          ndim,                    \
                          DebugDtype::enum_val,    \
                          stream);                 \
    break;                                         \
  }

  switch (dtype) {
    DISPATCH(float, FLOAT32)
    DISPATCH(double, FLOAT64)
    DISPATCH(phi::dtype::float16, FLOAT16)
    DISPATCH(phi::dtype::bfloat16, BFLOAT16)
    DISPATCH(int32_t, INT32)
    DISPATCH(int64_t, INT64)
    DISPATCH(int16_t, INT16)
    DISPATCH(int8_t, INT8)
    DISPATCH(uint8_t, UINT8)
    DISPATCH(bool, BOOL)
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "DebugPrintGPUTensor: unsupported dtype %s",
          phi::DataTypeToString(dtype)));
  }

#undef DISPATCH
}

}  // namespace phi
