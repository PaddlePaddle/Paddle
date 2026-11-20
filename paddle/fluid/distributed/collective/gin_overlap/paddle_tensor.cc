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

// paddle_tensor.cc —— 用到 paddle::Tensor 和 Paddle 当前计算流的实现：
//   - send_window_tensor：把本组 send window 里一段 fp32 包成 paddle::Tensor
//     （各桶的 grad_storage 就是它）。零拷贝，paddle::from_blob 的 deleter 传
//     nullptr，张量析构时不释放内存；内存归 NCCL window 所有。
//   - copy_2d：在计算流上做一次 pitched 拷贝（驱动 API cuMemcpy2DAsync，经
//     phi::dynload，见 gin_dynload.h）。参数是张量 + 字节偏移，越界在这里检查。
//   - paddle_compute_stream：Paddle 的当前计算流
//     （paddle::GetCurrentCUDAStream(GPUPlace)->raw_stream()），和 Python 的
//     paddle.device.current_stream() 是同一条流，paddle.device.stream_guard
//     换流时也跟着变。
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
  if (!t.defined()) return "未定义的张量";
  if (!t.initialized()) return "没初始化的张量";
  return "numel " + std::to_string(t.numel()) + "、每元素 " +
         std::to_string(phi::SizeOf(t.dtype())) + " B、place " +
         t.place().DebugString();
}

// 检查 2D 拷贝的源或目的：必须是 GPU 上已初始化的 dense 张量，而且
// [offset, offset + (height-1)*pitch + width) 落在张量内（最后一行只算
// width）。返回张量的数据指针；不满足时抛异常
void* checked_block(const paddle::Tensor& t,
                    size_t offset,
                    size_t pitch,
                    size_t width,
                    size_t height,
                    const char* what) {
  if (!t.defined() || !t.initialized() || !t.is_dense_tensor() ||
      t.place().GetType() != phi::AllocationType::GPU) {
    fail(std::string("fused all_gather 的 2D 拷贝：") + what +
         "不是 GPU 上已初始化的 dense 张量（" + describe(t) + "）");
  }
  if (width == 0 || height == 0 || width > pitch) {
    fail(std::string("fused all_gather 的 2D 拷贝：") + what + "的 width " +
         std::to_string(width) + " B / height " + std::to_string(height) +
         " / pitch " + std::to_string(pitch) + " B 不合法");
  }
  const size_t nbytes = (size_t)t.numel() * phi::SizeOf(t.dtype());
  const size_t need = offset + (height - 1) * pitch + width;
  if (offset > nbytes || need > nbytes) {
    fail(std::string("fused all_gather 的 2D 拷贝：") + what + "越界（偏移 " +
         std::to_string(offset) + " B + " + std::to_string(height) + " 行 × " +
         std::to_string(pitch) + " B 行距，最后一行 " + std::to_string(width) +
         " B，共要 " + std::to_string(need) + " B，张量只有" +
         std::to_string(nbytes) + " B：" + describe(t) + "）");
  }
  return const_cast<void*>(t.data());
}

}  // namespace

// Paddle 的当前计算流。取不到时打印原因并返回错误码，不退回默认流
int paddle_compute_stream(cudaStream_t* out) {
  *out = nullptr;
  try {
    phi::CUDAStream* stream =
        paddle::GetCurrentCUDAStream(phi::GPUPlace(g_device_id));
    if (stream == nullptr) {
      fprintf(stderr,
              "[gin_overlap] paddle::GetCurrentCUDAStream(GPUPlace(%d)) "
              "返回空\n",
              g_device_id);
      return (int)cudaErrorInvalidValue;
    }
    *out = stream->raw_stream();
  } catch (const std::exception& ex) {
    fprintf(stderr,
            "[gin_overlap] paddle::GetCurrentCUDAStream(GPUPlace(%d)) "
            "抛异常：%s\n",
            g_device_id,
            ex.what());
    return (int)cudaErrorInvalidValue;
  }
  return 0;
}

// 把本组 send window 的 [send_offset, send_offset + 4*numel) 包成一个 1-D fp32
// paddle::Tensor（零拷贝，见文件头）。send_offset 必须 4 字节对齐且不越界；包完
// 检查张量的指针、numel、dtype 和 place，对不上就抛异常。
paddle::Tensor send_window_tensor(int group,
                                  size_t send_offset,
                                  long long numel) {
  if (!is_valid_group(group)) {
    fail("send_window_tensor: 通信组编号要是 0 / 1，给的是 " +
         std::to_string(group));
  }
  CommGroup& comm_group = group_at(group);
  if (!comm_group.send) {
    fail("send_window_tensor: " + std::string(group_name(group)) +
         " 组的 send window 还没分配（alloc_windows 之前就来切片了）");
  }
  if (numel <= 0 || (send_offset & 3) ||
      send_offset + (size_t)numel * 4 > comm_group.send_bytes) {
    fail("send_window_tensor: " + std::string(group_name(group)) +
         " 组的 send window 切片越界或没按 4 B 对齐（偏移 " +
         std::to_string(send_offset) + " B + " + std::to_string(numel) +
         " 个 fp32，window 一共 " + std::to_string(comm_group.send_bytes) +
         " B）");
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
        "send_window_tensor: 包出来的张量和 send window 的切片对不上（要 "
        "data_ptr " +
        std::to_string((uintptr_t)data) + "、" + std::to_string(numel) +
        " 个 fp32、GPUPlace(" + std::to_string(g_device_id) + ")，得到 " +
        describe(t) + "）");
  }
  return t;
}

// 在计算流上做一次 2D 拷贝（cuMemcpy2DAsync）：height 行，每行 width 字节，源
// 行距 src_pitch、目的行距 dst_pitch，起点分别是两个张量里的字节偏移。出错时
// 抛异常。
void copy_2d(const paddle::Tensor& dst,
             size_t dst_offset,
             size_t dst_pitch,
             const paddle::Tensor& src,
             size_t src_offset,
             size_t src_pitch,
             size_t width,
             size_t height) {
  void* dst_data =
      checked_block(dst, dst_offset, dst_pitch, width, height, "目的");
  void* src_data =
      checked_block(src, src_offset, src_pitch, width, height, "源");
  cudaStream_t stream = nullptr;
  if (paddle_compute_stream(&stream)) {
    fail("fused all_gather 的 2D 拷贝：取不到 Paddle 的当前计算流");
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
    fail("fused all_gather 拆回 cuMemcpy2DAsync 失败 CUresult=" +
         std::to_string((int)res) + "（" + (msg ? msg : "?") + "）：width " +
         std::to_string(width) + " B、height " + std::to_string(height) +
         "、源行距 " + std::to_string(src_pitch) + " B、目的行距 " +
         std::to_string(dst_pitch) + " B");
  }
}

}  // namespace gin_overlap
