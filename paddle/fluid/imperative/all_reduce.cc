// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_NCCL

#include "paddle/fluid/imperative/all_reduce.h"

#include <string>
#include <utility>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/nccl_helper.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace imperative {

static void AllReduce(const framework::Tensor &src, framework::Tensor *dst,
                      const ParallelStrategy &strategy, cudaStream_t stream) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));

  const void *src_ptr = src.data<void>();

  dst->Resize(src.dims());
  auto *dst_ptr = dst->mutable_data(src.place(), src.type());

  auto nccl_dtype = platform::ToNCCLDataType(src.type());
  auto comm = static_cast<platform::CUDADeviceContext *>(
                  platform::DeviceContextPool::Instance().Get(place))
                  ->nccl_comm();

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
      src_ptr, dst_ptr, src.numel(), nccl_dtype, ncclSum, comm, stream));
}

#if NCCL_VERSION_CODE >= 2212
static void AllReduce(const framework::SelectedRows &src,
                      framework::SelectedRows *dst,
                      const ParallelStrategy &strategy, cudaStream_t stream) {
  VLOG(0) << "SelectedRows AllReduce start";
  const auto &src_tensor = src.value();
  const auto &place = src_tensor.place();
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));

  auto dtype = src_tensor.type();
  auto nccl_dtype = platform::ToNCCLDataType(dtype);
  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  auto comm = dev_ctx->nccl_comm();

  // 1. Gather rows number from all workers. Here use ncclAllGather to do this,
  // but we can use other ways to implement is in the future
  const auto &src_rows = src.rows();
  framework::Vector<int64_t> rows_num_vector(strategy.nranks_);
  rows_num_vector[strategy.local_rank_] = static_cast<int64_t>(src_rows.size());
  auto *gpu_rows_num_ptr = rows_num_vector.CUDAMutableData(place);
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      gpu_rows_num_ptr + strategy.local_rank_, gpu_rows_num_ptr, 1, ncclInt64,
      comm, stream));

  if (stream != dev_ctx->stream()) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  }

  const auto *cpu_rows_num_ptr = rows_num_vector.data();
  auto rows_num =
      std::accumulate(cpu_rows_num_ptr, cpu_rows_num_ptr + strategy.nranks_,
                      static_cast<int64_t>(0));
  dst->set_height(src.height());

  VLOG(0) << "Gather rows: " << string::join_strings(rows_num_vector, ',')
          << ", total rows number: " << rows_num
          << ", height: " << src.height();

  PADDLE_ENFORCE_LE(
      rows_num, src.height(),
      platform::errors::Unimplemented(
          "The gathered SelectedRows's rows number should less than or equal "
          "to the SelectedRows's height, but the actual rows number is %d, the "
          "SelectedRows's height is %d.",
          rows_num, src.height()));

  auto *dst_rows = dst->mutable_rows();
  dst_rows->resize(rows_num);
  auto *dst_rows_ptr = dst_rows->CUDAMutableData(place);
  const auto *src_rows_ptr = src_rows.CUDAData(place);

  auto *dst_tensor = dst->mutable_value();
  auto dims = src_tensor.dims();
  dims[0] = rows_num;
  auto feature_size = framework::product(dims) / dims[0];
  dst_tensor->Resize(dims);
  auto *dst_tensor_ptr = dst_tensor->mutable_data(place, dtype);
  const auto *src_tensor_ptr = src_tensor.data<void>();

  auto sizeof_dtype = framework::SizeOfType(dtype);
  int64_t row_offset = 0;
  for (int i = 0; i < strategy.nranks_; ++i) {
    if (cpu_rows_num_ptr[i] > 0) {
      // 2. Broadcast the rows of SelectedRows
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBroadcast(
          src_rows_ptr, dst_rows_ptr + row_offset, cpu_rows_num_ptr[i],
          ncclInt64, i, comm, stream));
      // 3. Broadcast the tensor data of SelectedRows
      auto *dst_tensor_ptr_i = reinterpret_cast<uint8_t *>(dst_tensor_ptr) +
                               row_offset * feature_size * sizeof_dtype;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBroadcast(
          src_tensor_ptr, dst_tensor_ptr_i, cpu_rows_num_ptr[i] * feature_size,
          nccl_dtype, i, comm, stream));
      row_offset += cpu_rows_num_ptr[i];
    }
  }

  VLOG(0) << "Original SelectedRows rows: "
          << string::join_strings(src_rows, ',');
  VLOG(0) << "Result SelectedRows rows: "
          << string::join_strings(*dst_rows, ',');
}
#endif

void AllReduce(const framework::Variable &src, framework::Variable *dst,
               const ParallelStrategy &strategy, cudaStream_t stream) {
  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(), strategy, stream);
#if NCCL_VERSION_CODE >= 2212
  } else if (src.IsType<framework::SelectedRows>()) {
    if (&src != dst) {
      if (!dst->IsType<framework::SelectedRows>()) {
        dst->Clear();
      }
      AllReduce(src.Get<framework::SelectedRows>(),
                dst->GetMutable<framework::SelectedRows>(), strategy, stream);
    } else {
      // SelectedRows cannot be allreduce in-place
      framework::Variable tmp_dst;
      AllReduce(src.Get<framework::SelectedRows>(),
                tmp_dst.GetMutable<framework::SelectedRows>(), strategy,
                stream);
      *dst = std::move(tmp_dst);
    }
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "LoDTensor and SelectedRows are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

static const platform::Place &GetVarPlace(const framework::Variable &src) {
  if (src.IsType<framework::LoDTensor>()) {
    return src.Get<framework::LoDTensor>().place();
#if NCCL_VERSION_CODE >= 2212
  } else if (src.IsType<framework::SelectedRows>()) {
    return src.Get<framework::SelectedRows>().value().place();
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Cannot get unsupported variable type %s for imperative allreduce, "
        "only "
        "LoDTensor and SelectedRows are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void AllReduce(const framework::Variable &src, framework::Variable *dst,
               const ParallelStrategy &strategy) {
  const auto &place = GetVarPlace(src);
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));
  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();
  AllReduce(src, dst, strategy, stream);
}

}  // namespace imperative
}  // namespace paddle

#endif
