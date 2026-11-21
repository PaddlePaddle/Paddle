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

// paddle_tensor.cc: the parts that use paddle::Tensor and Paddle's current
// compute stream:
//   - send_window_tensor: wraps a range of fp32 in the group's send window as
//     a paddle::Tensor (each bucket's grad_storage is one). No copy: the
//     deleter passed to paddle::from_blob is nullptr, so destroying the tensor
//     does not free the memory, which belongs to the NCCL window.
//   - copy_2d: one pitched copy on the compute stream (driver API
//     cuMemcpy2DAsync through phi::dynload, see gin_dynload.h). Arguments are
//     tensors plus byte offsets; bounds are checked here.
//   - paddle_compute_stream: Paddle's current compute stream
//     (paddle::GetCurrentCUDAStream(GPUPlace)->raw_stream()), the same stream
//     as Python's paddle.device.current_stream(); it follows
//     paddle.device.stream_guard.
#include "paddle/fluid/distributed/collective/gin_overlap/paddle_tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/cuda_stream.h"

#include "paddle/fluid/distributed/collective/gin_overlap/gin_engine.h"

namespace gin_overlap {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("gin_overlap: " + what);
}

std::string describe(const paddle::Tensor& t) {
  if (!t.defined()) return "undefined tensor";
  if (!t.initialized()) return "uninitialized tensor";
  return "numel " + std::to_string(t.numel()) + ", " +
         std::to_string(phi::SizeOf(t.dtype())) + " B per element, place " +
         t.place().DebugString();
}

// Checks the source or destination of a 2D copy: it must be an initialized
// dense tensor on the GPU, and [offset, offset + (height-1)*pitch + width)
// must lie inside it (the last row counts only width). Returns the tensor's
// data pointer; throws otherwise
void* checked_block(const paddle::Tensor& t,
                    size_t offset,
                    size_t pitch,
                    size_t width,
                    size_t height,
                    const char* what) {
  if (!t.defined() || !t.initialized() || !t.is_dense_tensor() ||
      t.place().GetType() != phi::AllocationType::GPU) {
    fail(std::string("copy_2d: ") + what +
         " is not an initialized dense tensor on the GPU (" + describe(t) +
         ")");
  }
  if (width == 0 || height == 0 || width > pitch) {
    fail(std::string("copy_2d: ") + what + " width " + std::to_string(width) +
         " B / height " + std::to_string(height) + " / pitch " +
         std::to_string(pitch) + " B is invalid");
  }
  const size_t nbytes = (size_t)t.numel() * phi::SizeOf(t.dtype());
  const size_t need = offset + (height - 1) * pitch + width;
  if (offset > nbytes || need > nbytes) {
    fail(std::string("copy_2d: ") + what + " out of bounds (offset " +
         std::to_string(offset) + " B + " + std::to_string(height) +
         " rows x " + std::to_string(pitch) + " B pitch, last row " +
         std::to_string(width) + " B, needs " + std::to_string(need) +
         " B, the tensor has " + std::to_string(nbytes) + " B: " + describe(t) +
         ")");
  }
  return const_cast<void*>(t.data());
}

}  // namespace

// Paddle's current compute stream. On failure prints the reason and returns
// an error code; never falls back to the default stream
int paddle_compute_stream(cudaStream_t* out) {
  *out = nullptr;
  try {
    phi::CUDAStream* stream =
        paddle::GetCurrentCUDAStream(phi::GPUPlace(g_device_id));
    if (stream == nullptr) {
      fprintf(stderr,
              "[gin_overlap] paddle::GetCurrentCUDAStream(GPUPlace(%d)) "
              "returned null\n",
              g_device_id);
      return (int)cudaErrorInvalidValue;
    }
    *out = stream->raw_stream();
  } catch (const std::exception& ex) {
    fprintf(stderr,
            "[gin_overlap] paddle::GetCurrentCUDAStream(GPUPlace(%d)) "
            "threw: %s\n",
            g_device_id,
            ex.what());
    return (int)cudaErrorInvalidValue;
  }
  return 0;
}

// Wraps [send_offset, send_offset + 4*numel) of the group's send window as a
// 1-D fp32 paddle::Tensor (no copy, see the file header). send_offset must be
// 4-byte aligned and in bounds; after wrapping, the tensor's pointer, numel,
// dtype and place are checked, and a mismatch throws.
paddle::Tensor send_window_tensor(int group,
                                  size_t send_offset,
                                  long long numel) {
  if (!is_valid_group(group)) {
    fail("send_window_tensor: group id must be 0 / 1, got " +
         std::to_string(group));
  }
  CommGroup& comm_group = group_at(group);
  if (!comm_group.send) {
    fail("send_window_tensor: send window of the " +
         std::string(group_name(group)) +
         " group is not allocated yet (called before alloc_windows)");
  }
  if (numel <= 0 || (send_offset & 3) ||
      send_offset + (size_t)numel * 4 > comm_group.send_bytes) {
    fail("send_window_tensor: slice of the send window of the " +
         std::string(group_name(group)) +
         " group is out of bounds or not 4-byte aligned (offset " +
         std::to_string(send_offset) + " B + " + std::to_string(numel) +
         " fp32, window size " + std::to_string(comm_group.send_bytes) + " B)");
  }
  char* data = comm_group.send + send_offset;
  paddle::Tensor t = paddle::from_blob(data,
                                       {numel},
                                       phi::DataType::FLOAT32,
                                       phi::DataLayout::NCHW,
                                       phi::GPUPlace(g_device_id),
                                       /*deleter=*/nullptr);
  if (!t.defined() || !t.initialized() || !t.is_dense_tensor() ||
      t.data() != (const void*)data || t.numel() != numel ||
      t.dtype() != phi::DataType::FLOAT32 ||
      t.place() != phi::Place(phi::GPUPlace(g_device_id))) {
    fail(
        "send_window_tensor: the wrapped tensor does not match the send "
        "window slice (want data_ptr " +
        std::to_string((uintptr_t)data) + ", " + std::to_string(numel) +
        " fp32, GPUPlace(" + std::to_string(g_device_id) + "), got " +
        describe(t) + ")");
  }
  return t;
}

// One 2D copy on the compute stream (cuMemcpy2DAsync): height rows of width
// bytes, source pitch src_pitch and destination pitch dst_pitch, starting at
// byte offsets in the two tensors. Throws on error.
void copy_2d(const paddle::Tensor& dst,
             size_t dst_offset,
             size_t dst_pitch,
             const paddle::Tensor& src,
             size_t src_offset,
             size_t src_pitch,
             size_t width,
             size_t height) {
  void* dst_data =
      checked_block(dst, dst_offset, dst_pitch, width, height, "destination");
  void* src_data =
      checked_block(src, src_offset, src_pitch, width, height, "source");
  cudaStream_t stream = nullptr;
  if (paddle_compute_stream(&stream)) {
    fail("copy_2d: cannot get Paddle's current compute stream");
  }
  CUDA_MEMCPY2D params;
  memset(&params, 0, sizeof(params));
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = (CUdeviceptr)((char*)src_data + src_offset);
  params.srcPitch = src_pitch;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr)((char*)dst_data + dst_offset);
  params.dstPitch = dst_pitch;
  params.WidthInBytes = width;
  params.Height = height;
  const CUresult res = cuMemcpy2DAsync(&params, (CUstream)stream);
  if (res != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(res, &msg);
    fail("copy_2d: cuMemcpy2DAsync failed, CUresult=" +
         std::to_string((int)res) + " (" + (msg ? msg : "?") + "): width " +
         std::to_string(width) + " B, height " + std::to_string(height) +
         ", source pitch " + std::to_string(src_pitch) +
         " B, destination pitch " + std::to_string(dst_pitch) + " B");
  }
}

}  // namespace gin_overlap
