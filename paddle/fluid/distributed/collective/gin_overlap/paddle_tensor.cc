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

// paddle_tensor.cc —— 所有要碰 paddle::Tensor 和 Paddle 当前计算流的代码都在这
// 一个 TU 里。
//
// 在 PaddleFleet 的扩展里这个文件后缀是 .cu：那边 Paddle 的头文件里有 thrust，
//   外部构建的 host 编译器解析不了，所以凡是包含它们的翻译单元都得交给 nvcc，
//   连收 / 还 paddle::Tensor 的 pybind 注册都只好挪到这边来。搬进 Paddle 之后
//   这两件事都不成立了：这些头文件本来就是 Paddle 自己的（.cc 直接包），pybind
//   的 type_caster<paddle::Tensor>（paddle/utils/pybind.h）在 pybind 目录里也
//   直接能用 ⇒ 这个文件变成普通 .cc，那两个方法的注册并回 pybind 那一份
//   （paddle/fluid/pybind/gin_overlap_api.cc），这边只留实现。
//
// 这里三件事：
//   send_window_tensor  本组 send window 里一段 fp32 包成
//   paddle::Tensor（各桶的 grad_storage 就是它）。
//     零拷贝、而且 Paddle 不接管释放：paddle::from_blob 的 deleter 传 nullptr ⇒
//     phi::Allocation 的 deleter_ 是空指针、张量析构时什么都不做（phi/api/lib/
//     tensor_utils.cc:137-143 的 FromBlobImpl 里 alloc_deleter 只在 deleter
//     非空时才建， phi/core/allocator.h:52-56 的 ~Allocation 只在 deleter_
//     非空时调它）；内存一直归 NCCL window。原来 Python 侧手搓 DLPack
//     胶囊（PaddleFleet 的 engine.py 里的 _wrap）干的就是这件事。 先例：
//     deep_ep_xpu/deep_ep.cpp:1063-1069 也是这样把一块不归 Paddle 释放的缓冲
//     （recv_count_ptr）包成 paddle::Tensor 交出去。
//   copy_2d  计算流上一次 pitched 拷贝（驱动 API
//   cuMemcpy2DAsync，和 Python 侧原来用 ctypes
//     调 libcuda.so.1 的 cuMemcpy2DAsync_v2
//     是同一个调用，拷贝引擎；Paddle 不链 libcuda，这一调用经 phi::dynload
//     的包装过去，见 gin_dynload.h），参数从裸指针改成 张量 +
//     字节偏移，边界在这里查。
//   paddle_compute_stream  Paddle 的当前计算流，等价于 Python 的
//     paddle.device.current_stream()._as_parameter_.value：
//     paddle::GetCurrentCUDAStream(GPUPlace)（phi/api/include/context_pool.h:114）
//     → phi::CUDAStream::raw_stream()。Python 那条路是 pybind 的
//     _get_current_stream → platform::get_current_stream →
//     DeviceContextPool::Get(GPUPlace) → GPUContext::cuda_stream() →
//     raw_stream()，取的是同一个流，所以 paddle.device.stream_guard
//     换流时这边也跟着换。
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

// 2D 拷贝的源 / 目的都要求：dense、已初始化、在 GPU 上，而且
// [offset, offset + (height-1)*pitch + width) 落在这块张量里（最后一行只算
// width，和 cuMemcpy2DAsync 真正会碰到的范围一致）
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

// Paddle 的当前计算流（Python 原来读 current_stream()._as_parameter_.value
// 的那一个）。
// 拿不到就打印原因并返回错误码：上层的方法一律按返回码报错，不静默往默认流上排工作。
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

// 本组 send window 的 [send_offset, send_offset + 4*numel) 包成一个 1-D fp32
// paddle::Tensor。 零拷贝（deleter = nullptr，见文件头），dtype / numel /
// data_ptr 和原来 Python 侧 DLPack 那版一致； 包完当场自检，对不上就抛（原来是
// _wrap 末尾的 assert）。
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

// 计算流上一次 2D 拷贝（驱动 API cuMemcpy2DAsync，拷贝引擎）：height
// 行，每行 width 字节， 源行距 src_pitch、目的行距
// dst_pitch，起点各自是张量里的字节偏移。 backend._fused_all_gather
// 把一次合批 all_gather 的结果按列拆回各桶的 param_storage 用这个。
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
