/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/runtime/rt_stream.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/runtime/rt_event.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_memory.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_resources.h"
#include "paddle/fluid/platform/enforce.h"

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/library.h"
#include "dtu/hlir/types.h"
#include "dtu/op_define/type.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/dynload/port.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {
namespace {
void ConvertLaunchParams(const std::vector<std::shared_ptr<Memory>> &pins,
                         const std::vector<std::shared_ptr<Memory>> &pouts,
                         Stream::GcuLaunchParams &params) {  // NOLINT
  for (size_t idx = 0; idx < pins.size(); ++idx) {
    auto gcu_mem = pins[idx];
    params.inputs.push_back(gcu_mem->GetDevAddr());
  }
  for (size_t idx = 0; idx < pouts.size(); ++idx) {
    auto gcu_mem = pouts[idx];
    params.outputs.push_back(gcu_mem->GetDevAddr());
  }
}
}  // namespace

Stream::Stream(Context *ctx, topsStream_t stream, bool owning_stream)
    : ctx(ctx), tops_stream(stream), owning_stream_(owning_stream) {
  ResourceMgr::GetInstance()->RTCounter("Stream", 1);
  PADDLE_ENFORCE_EQ(
      (ctx && tops_stream),
      true,
      platform::errors::InvalidArgument(
          "Invalid args, ctx is nullptr:%d, tops_stream is nullptr:%d",
          ctx == nullptr,
          tops_stream == nullptr));
}

Stream::~Stream() {
  Synchronize();
  if (owning_stream_) {
    RT_CHECK_NO_THROW(topsStreamDestroy(tops_stream));
  }
  ResourceMgr::GetInstance()->RTCounter("StreamRelease", 1);
}

std::shared_ptr<Stream> Stream::CreateStream(Context *ctx,
                                             topsStream_t stream_in) {
  if (stream_in != nullptr) {
    return std::make_shared<Stream>(ctx, stream_in, false);
  }
  GcuDeviceGuard guard(ctx->device);
  topsStream_t stream;
  RT_CHECK(topsStreamCreate(&stream));
  return std::make_shared<Stream>(ctx, stream);
}

void Stream::MemcpyH2DSync(std::shared_ptr<Memory> pmem,
                           const void *pdata,
                           uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(
      topsMemcpyHtoD(pmem->GetDevAddr(), const_cast<void *>(pdata), nbytes));
  ResourceMgr::GetInstance()->RTCounter("MemcpyH2DSync", 1);
}

void Stream::MemcpyD2HSync(void *pdata,
                           std::shared_ptr<Memory> pmem,
                           uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyDtoH(pdata, pmem->GetDevAddr(), nbytes));
  ResourceMgr::GetInstance()->RTCounter("MemcpyD2HSync", 1);
}

void Stream::MemcpyD2DSync(std::shared_ptr<Memory> pdst,
                           std::shared_ptr<Memory> psrc,
                           uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyDtoD(pdst->GetDevAddr(), psrc->GetDevAddr(), nbytes));
  ResourceMgr::GetInstance()->RTCounter("MemcpyD2DSync", 1);
}

void Stream::Memset32Sync(std::shared_ptr<Memory> pmem,
                          int value,
                          uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemset(pmem->GetDevAddr(), value, nbytes));
  ResourceMgr::GetInstance()->RTCounter("Memset32Sync", 1);
}

void Stream::MemcpyH2DAsync(std::shared_ptr<Memory> pmem,
                            const void *pdata,
                            uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyHtoDAsync(
      pmem->GetDevAddr(), const_cast<void *>(pdata), nbytes, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemcpyH2DAsync", 1);
}

void Stream::MemcpyD2HAsync(void *pdata,
                            std::shared_ptr<Memory> pmem,
                            uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyDtoHAsync(pdata, pmem->GetDevAddr(), nbytes, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemcpyD2HAsync", 1);
}

void Stream::MemcpyD2DAsync(std::shared_ptr<Memory> pdst,
                            std::shared_ptr<Memory> psrc,
                            uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyDtoDAsync(
      pdst->GetDevAddr(), psrc->GetDevAddr(), nbytes, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemcpyD2DAsync", 1);
}

void Stream::MemsetAsync(std::shared_ptr<Memory> pmem,
                         int value,
                         uint64_t nbytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemsetAsync(pmem->GetDevAddr(), value, nbytes, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemsetAsync", 1);
}

void Stream::MemcpySync(void *dst,
                        const void *src,
                        size_t size_bytes,
                        topsMemcpyKind kind) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpy(dst, src, size_bytes, kind));
  ResourceMgr::GetInstance()->RTCounter("MemcpySyncWithKind", 1);
}

void Stream::MemcpyAsync(void *dst,
                         const void *src,
                         size_t size_bytes,
                         topsMemcpyKind kind) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemcpyAsync(dst, src, size_bytes, kind, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemcpyAsyncWithKind", 1);
}

void Stream::MemsetAsync(void *dst, int value, size_t size_bytes) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsMemsetAsync(dst, value, size_bytes, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("MemsetAsync", 1);
}

void Stream::EventRecord(std::shared_ptr<Event> pevent) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsEventRecord(pevent->tops_event, tops_stream));
  ResourceMgr::GetInstance()->RTCounter("EventRecord", 1);
}

void Stream::WaitEvent(std::shared_ptr<Event> pevent) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsStreamWaitEvent(tops_stream, pevent->tops_event, 0));
  ResourceMgr::GetInstance()->RTCounter("WaitEvent", 1);
}

void Stream::AddHostCallback(const std::function<void()> &fn) {
  AddGcuCallback(ctx->device, tops_stream, std::move(fn));
  ResourceMgr::GetInstance()->RTCounter("Callback", 1);
}

void Stream::RunExecutableSync(
    const ExecutablePtr &exe,
    const std::vector<std::shared_ptr<Memory>> &ins,
    const std::vector<std::shared_ptr<Memory>> &outs) {
  GcuDeviceGuard guard(ctx->device);
  RunExecutableAsync(exe, ins, outs);
  Synchronize();
}

void Stream::RunExecutable(const ExecutablePtr &exe,
                           Stream::GcuLaunchParams &params) {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsLaunchExecutable(exe->GetTopsExecutable(),
                                nullptr,
                                params.inputs.data(),
                                params.inputs.size(),
                                nullptr,
                                nullptr,
                                params.outputs.data(),
                                params.outputs.size(),
                                nullptr,
                                nullptr,
                                tops_stream));
  ResourceMgr::GetInstance()->RTCounter("RunExecutable", 1);
}

static inline std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t> &sizes) {
  const int64_t ndims = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndims, 1);
  for (auto i = ndims - 2; i >= 0; --i) {
    // strides can't be 0 even if sizes are 0.
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], int64_t{1});
  }
  return strides;
}

static inline std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t> &sizes, const std::vector<uint8_t> &layouts) {
  auto rank = sizes.size();
  CHECK(rank == layouts.size()) << rank << " vs " << layouts.size();
  auto cstrides = contiguous_strides(sizes);
  auto strides = cstrides;
  for (size_t i = 0; i < rank; i++) {
    strides[layouts[i]] = cstrides[i];
  }
  return strides;
}

static inline std::vector<uint8_t> contiguous_layouts(const size_t ndims) {
  CHECK(ndims < UINT8_MAX) << " with ndims = " << ndims;
  std::vector<uint8_t> contiguous_layout(ndims, 0);
  std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
  return contiguous_layout;
}

static inline std::vector<int64_t> contiguous_layouts_ex(const size_t ndims) {
  std::vector<int64_t> contiguous_layout(ndims, 0);
  std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
  return contiguous_layout;
}

int GetGCUDataType(const phi::DataType &dtype) {
  switch (dtype) {
    case phi::DataType::UINT8:
      return hlir::U8;
    case phi::DataType::INT8:
      return hlir::S8;
    case phi::DataType::INT16:
      return hlir::S16;
    case phi::DataType::UINT16:
      return hlir::U16;
    case phi::DataType::INT32:
      return hlir::S32;
    case phi::DataType::UINT32:
      return hlir::U32;
    case phi::DataType::INT64:
      return hlir::S64;
    case phi::DataType::UINT64:
      return hlir::U64;
    case phi::DataType::FLOAT16:
      return hlir::F16;
    case phi::DataType::FLOAT32:
      return hlir::F32;
    case phi::DataType::FLOAT64:
      return hlir::F64;
    case phi::DataType::COMPLEX64:
      return hlir::C64;
    case phi::DataType::COMPLEX128:
      return hlir::C128;
    case phi::DataType::BOOL:
      return hlir::PRED;
    case phi::DataType::BFLOAT16:
      return hlir::BF16;
    default: {
      CHECK(false) << "Invalid scalar type " << DataTypeToString(dtype);
      return hlir::PRIMITIVE_TYPE_INVALID;
    }
  }
}

hlir::Tensor *GetHlirTensor(const void *mem_ptr, const phi::DataType &dtype) {
  auto *gcu_memory = static_cast<GcuMemory *>(const_cast<void *>(mem_ptr));
  auto *data_ptr = gcu_memory->mem_ptr;
  std::vector<int64_t> dims = {};
  size_t ndims = 16;
  std::vector<int64_t> tmp_dims(ndims, 0);
  RT_CHECK(topsMemoryGetDims(
      data_ptr, reinterpret_cast<int64_t *>(tmp_dims.data()), &ndims));
  dims = std::vector<int64_t>(tmp_dims.begin(), tmp_dims.begin() + ndims);

  auto strides = contiguous_strides(dims);
  auto layouts = contiguous_layouts_ex(dims.size());
  dims = dims.empty() ? std::vector<int64_t>{1} : dims;
  strides = strides.empty() ? std::vector<int64_t>{1} : strides;
  layouts = layouts.empty() ? std::vector<int64_t>{0} : layouts;
  return (new hlir::Tensor{data_ptr,
                           GetGCUDataType(dtype),
                           static_cast<int64_t>(gcu_memory->size),
                           dims,
                           strides,
                           layouts});
}

void BuildDispatchParam(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs,
                        const std::vector<phi::DataType> &in_types,
                        const std::vector<phi::DataType> &out_types,
                        hlir::DispatchParam &params) {  // NOLINT
  for (size_t i = 0; i < inputs.size(); i++) {
    auto data_ptr = inputs[i];
    VLOG(6) << "start get input hlir tensor: " << i;
    params.inputs.push_back(GetHlirTensor(data_ptr, in_types[i]));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto data_ptr = outputs[i];
    VLOG(6) << "start get output hlir tensor: " << i;
    params.outputs.push_back(GetHlirTensor(data_ptr, out_types[i]));
  }
}

void FreeDispatchParam(hlir::DispatchParam &params) {  // NOLINT
  for (auto input : params.inputs) delete input;
  for (auto output : params.outputs) delete output;
}

void Stream::RunExecutableAsync(const ExecutablePtr &exe,
                                const std::vector<void *> &ins,
                                const std::vector<void *> &outs,
                                const std::vector<phi::DataType> &in_types,
                                const std::vector<phi::DataType> &out_types) {
  PADDLE_GCU_TRACE_START(EXEC, exec);
  hlir::DispatchParam params;
  params.stream = tops_stream;
  BuildDispatchParam(ins, outs, in_types, out_types, params);
  AOTOPS_DEBUG("single_op", params);
  std::string compile_options =
      "hlir-training-pipeline{dispatch-jit=true tensor-split=true "
      "op-key=pavo dynamic-shape=false}";
  PADDLE_GCU_TRACE_START(DISPATCH, dispatch);
  exe->dispatch->dispatch(params, compile_options.c_str());
  PADDLE_GCU_TRACE_END(DISPATCH, dispatch);
  PADDLE_GCU_TRACE_END(EXEC, exec);
  FreeDispatchParam(params);
}

void Stream::RunExecutableAsync(
    const ExecutablePtr &exe,
    const std::vector<std::shared_ptr<Memory>> &ins,
    const std::vector<std::shared_ptr<Memory>> &outs) {
  Stream::GcuLaunchParams params;
  ConvertLaunchParams(ins, outs, params);
  RunExecutable(exe, params);
}

void Stream::RunExecutableAsync(const ExecutablePtr &exe,
                                const std::vector<void *> &ins,
                                const std::vector<void *> &outs) {
  GcuDeviceGuard guard(ctx->device);
  Stream::GcuLaunchParams params(ins, outs);
  RunExecutable(exe, params);
}

void Stream::Synchronize() {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsStreamSynchronize(tops_stream));
  ResourceMgr::GetInstance()->RTCounter("Synchronize", 1);
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
