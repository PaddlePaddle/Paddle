// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// old op include, fluid should be removed
#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <vector>
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
// trace op include
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename Context>
void GetClassInterval(const gpuStream_t& stream,
                      const phi::Place& place,
                      const Context& dev_ctx,
                      const int rid,
                      const int rank,
                      const int nranks,
                      const int D,
                      DenseTensor* class_interval) {
  std::vector<int> shard_dim_vec(nranks + 1, 0);
  shard_dim_vec[rank + 1] = D;
  if (nranks <= 1) {
    paddle::framework::TensorFromVector(shard_dim_vec, dev_ctx, class_interval);
    return;
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  DenseTensor num_classes_per_device;
  paddle::framework::TensorFromVector(
      shard_dim_vec, dev_ctx, &num_classes_per_device);
  int* num_classes_per_device_ptr = num_classes_per_device.data<int>();

  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    // Use ProcessGroup
    paddle::distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(num_classes_per_device);
    out_tensor.push_back(num_classes_per_device);

    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = paddle::distributed::ReduceOp::SUM;
    auto task = pg->AllReduce(in_tensor, out_tensor, opts);
    task->Wait();
  } else {
    const auto& comm =
        paddle::platform::NCCLCommContext::Instance().Get(rid, place);
    // use global calculate stream
    const auto calcu_stream =
        static_cast<GPUContext*>(
            paddle::platform::DeviceContextPool::Instance().Get(place))
            ->stream();

    PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::ncclAllReduce(
        num_classes_per_device_ptr,
        num_classes_per_device_ptr,
        num_classes_per_device.numel(),
        paddle::platform::ToNCCLDataType(paddle::framework::TransToProtoVarType(
            num_classes_per_device.dtype())),
        ncclSum,
        comm->comm(),
        calcu_stream));
  }

  class_interval->Resize({nranks + 1});
  auto class_interval_ptr = dev_ctx.template Alloc<int>(class_interval);
  size_t cub_temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum<int*, int*>(
      nullptr, cub_temp_storage_bytes, nullptr, nullptr, nranks + 1, stream);
  auto cub_temp_storage = paddle::memory::Alloc(place, cub_temp_storage_bytes);
  cub::DeviceScan::InclusiveSum<int*, int*>(cub_temp_storage->ptr(),
                                            cub_temp_storage_bytes,
                                            num_classes_per_device_ptr,
                                            class_interval_ptr,
                                            nranks + 1,
                                            stream);
  return;
#endif
}

template <typename T, typename IndexT>
__global__ void AddMarginToPositiveLogitsKernel(T* logit,
                                                const IndexT* label,
                                                const float margin1,
                                                const float margin2,
                                                const float margin3,
                                                const int rank,
                                                const int nranks,
                                                const int64_t N,
                                                const int64_t D,
                                                const int* class_interval_ptr) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  int start_index = class_interval_ptr[rank];
  int end_index = class_interval_ptr[rank + 1];
  int num_classes = class_interval_ptr[nranks];
  CUDA_KERNEL_LOOP(i, N) {
    auto real_label = label[i];
    PADDLE_ENFORCE((real_label < num_classes) && (real_label >= 0),
                   "The index is out of bounds, "
                   "please check whether the value of label and "
                   "input meet the number of class. It should "
                   "be less than [%d], but received [%d]",
                   num_classes,
                   real_label);

    if (real_label >= start_index && real_label < end_index) {
      int64_t offset = i * D + real_label - start_index;
      if (fabs(margin1 - 1.0) > 1e-8 || fabs(margin2) > 1e-8) {
        MPType x = static_cast<MPType>(logit[offset]);
        MPType theta = acos(x);
        if (fabs(margin1 - 1.0) > 1e-8) {
          theta *= static_cast<MPType>(margin1);
        }
        if (fabs(margin2) > 1e-8) {
          theta += static_cast<MPType>(margin2);
        }
        logit[offset] = static_cast<T>(cos(theta));
      }
      if (fabs(margin3) > 1e-8) {
        MPType y = static_cast<MPType>(logit[offset]);
        y -= static_cast<MPType>(margin3);
        logit[offset] = static_cast<T>(y);
      }
    }
  }
}

template <typename T>
__global__ void ScaleLogitKernel(T* logits,
                                 const float scale,
                                 const int64_t N,
                                 const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) { logits[i] *= static_cast<T>(scale); }
}

template <typename T>
__global__ void LogitsMinusMaxKernel(T* logits,
                                     const T* logits_max_per_row,
                                     const int64_t N,
                                     const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    logits[i] -= logits_max_per_row[row];
  }
}

template <typename T>
__global__ void LogitsMinusLogSumKernel(T* logits,
                                        const T* logits_sum_per_row,
                                        const int64_t N,
                                        const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    logits[i] -= phi::kps::details::Log(logits_sum_per_row[row]);
  }
}

template <typename T, typename IndexT>
__global__ void HardLabelSoftmaxWithCrossEntropyKernel(
    T* loss,
    T* log_softmax,
    const IndexT* labels,
    const int rank,
    const int64_t N,
    const int64_t D,
    const int* class_interval_ptr) {
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      auto softmax = log_softmax[i];
      loss[row] = -softmax;
      log_softmax[i] = phi::kps::details::Exp(softmax);
    } else {
      log_softmax[i] = phi::kps::details::Exp(log_softmax[i]);
    }
  }
}

template <typename T, typename Context>
void MarginCrossEntropyKernel(const Context& dev_ctx,
                              const DenseTensor& logits,
                              const DenseTensor& labels,
                              bool return_softmax,
                              int ring_id,
                              int rank,
                              int nranks,
                              float margin1,
                              float margin2,
                              float margin3,
                              float scale,
                              DenseTensor* softmax,
                              DenseTensor* loss) {
  const auto& place = dev_ctx.GetPlace();  // old code

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  paddle::platform::NCCLComm* comm;
  paddle::distributed::ProcessGroup* pg = nullptr;
  gpuStream_t stream;
  if (nranks > 1) {
    auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(ring_id)) {
      // Use ProcessGroup
      pg = map->get(ring_id);
    } else {
      comm = paddle::platform::NCCLCommContext::Instance().Get(ring_id, place);

      // use global calculate stream
      stream = static_cast<GPUContext*>(
                   paddle::platform::DeviceContextPool::Instance().Get(place))
                   ->stream();
    }
  }
#endif

  // allocate memory on device.
  T* softmax_ptr = dev_ctx.template Alloc<T>(softmax);
  T* loss_ptr = dev_ctx.template Alloc<T>(loss);

  const auto& logits_dims = logits.dims();
  const auto& labels_dims = labels.dims();

  const int axis = logits_dims.size() - 1;
  const int N = phi::funcs::SizeToAxis(axis, logits_dims);
  const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

  int blocks = NumBlocks(N);
  int threads = kNumCUDAThreads;
  const auto& label_type =
      paddle::framework::TransToProtoVarType(labels.dtype());

  // copy logits to softmax variable since we can't modify logits,
  // and it also be used when calculate grad
  phi::Copy<Context>(dev_ctx, logits, dev_ctx.GetPlace(), true, softmax);

  DenseTensor softmax_2d;
  softmax_2d.ShareDataWith(*softmax).Resize({N, D});
  T* logits_ptr = softmax_2d.data<T>();

  DenseTensor class_interval;
  GetClassInterval<T, Context>(dev_ctx.stream(),
                               dev_ctx.GetPlace(),
                               dev_ctx,
                               ring_id,
                               rank,
                               nranks,
                               D,
                               &class_interval);

  // step 1, preprocess logits
  // add margin for positive elements
  // theta = acos(x_i)
  // (cos(m1 * theta + m2) - m3)
  // save match_logits, used for gradient computation.
  if (label_type == paddle::framework::proto::VarType::INT32) {
    typedef int32_t LabelT;
    AddMarginToPositiveLogitsKernel<T>
        <<<NumBlocks(N), threads, 0, dev_ctx.stream()>>>(
            logits_ptr,
            labels.data<LabelT>(),
            margin1,
            margin2,
            margin3,
            rank,
            nranks,
            N,
            D,
            class_interval.data<int>());
  } else if (label_type == paddle::framework::proto::VarType::INT64) {
    typedef int64_t LabelT;
    AddMarginToPositiveLogitsKernel<T>
        <<<NumBlocks(N), threads, 0, dev_ctx.stream()>>>(
            logits_ptr,
            labels.data<LabelT>(),
            margin1,
            margin2,
            margin3,
            rank,
            nranks,
            N,
            D,
            class_interval.data<int>());
  } else {
    PADDLE_THROW(errors::Unimplemented(
        "margin_cross_entropy label type noly support int32 and int64, "
        "but got %s",
        label_type));
  }

  // scale by s
  ScaleLogitKernel<T><<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
      logits_ptr, scale, N, D);

  // step 2, obtain logit_max
  DenseTensor logits_max;
  logits_max.Resize({N, 1});
  dev_ctx.template Alloc<T>(&logits_max);
  T* logits_max_buff = dev_ctx.template Alloc<T>(&logits_max);

  phi::funcs::
      ReduceKernel<T, T, phi::kps::MaxFunctor, phi::kps::IdentityFunctor<T>>(
          static_cast<const phi::GPUContext&>(dev_ctx),
          softmax_2d,
          &logits_max,
          phi::kps::IdentityFunctor<T>(),
          {1});

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (nranks > 1) {
    if (pg) {
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(logits_max);
      out_tensor.push_back(logits_max);

      paddle::distributed::AllreduceOptions opts;
      opts.reduce_op = paddle::distributed::ReduceOp::MAX;
      auto task = pg->AllReduce(in_tensor, out_tensor, opts);
      task->Wait();
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::ncclAllReduce(
          logits_max_buff,
          logits_max_buff,
          logits_max.numel(),
          paddle::platform::ToNCCLDataType(
              paddle::framework::TransToProtoVarType(logits_max.dtype())),
          ncclMax,
          comm->comm(),
          stream));
    }
  }
#endif

  // step 3, logit - logit_max
  LogitsMinusMaxKernel<T><<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
      logits_ptr, logits_max_buff, N, D);

  // step 4, sum(exp(logit - logit_max))
  DenseTensor sum_exp_logits;
  sum_exp_logits.Resize({N, 1});
  dev_ctx.template Alloc<T>(&sum_exp_logits);
  T* sum_exp_logits_buff = dev_ctx.template Alloc<T>(&sum_exp_logits);
  phi::funcs::ReduceKernel<T, T, phi::kps::AddFunctor, phi::kps::ExpFunctor<T>>(
      static_cast<const phi::GPUContext&>(dev_ctx),
      softmax_2d,
      &sum_exp_logits,
      phi::kps::ExpFunctor<T>(),
      {1});

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (nranks > 1) {
    if (pg) {
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(sum_exp_logits);
      out_tensor.push_back(sum_exp_logits);

      paddle::distributed::AllreduceOptions opts;
      opts.reduce_op = paddle::distributed::ReduceOp::SUM;
      auto task = pg->AllReduce(in_tensor, out_tensor, opts);
      task->Wait();
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::ncclAllReduce(
          sum_exp_logits_buff,
          sum_exp_logits_buff,
          sum_exp_logits.numel(),
          paddle::platform::ToNCCLDataType(
              paddle::framework::TransToProtoVarType(sum_exp_logits.dtype())),
          ncclSum,
          comm->comm(),
          stream));
    }
  }
#endif

  // step 5, (logit - logit_max) - log(sum(exp(logit - logit_max)))
  LogitsMinusLogSumKernel<T>
      <<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
          logits_ptr, sum_exp_logits_buff, N, D);

  // step 6, prob = exp((logit - logit_max) - log(sum(exp(logit -
  // logit_max))))
  // loss = -((logit_i - logit_max) - log(sum(exp(logit - logit_max))))

  phi::funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, loss, static_cast<T>(0.0));
  if (label_type == paddle::framework::proto::VarType::INT32) {
    typedef int32_t LabelT;
    HardLabelSoftmaxWithCrossEntropyKernel<T, LabelT>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_ptr,
                                                   logits_ptr,
                                                   labels.data<LabelT>(),
                                                   rank,
                                                   N,
                                                   D,
                                                   class_interval.data<int>());
  } else if (label_type == paddle::framework::proto::VarType::INT64) {
    typedef int64_t LabelT;
    HardLabelSoftmaxWithCrossEntropyKernel<T, LabelT>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_ptr,
                                                   logits_ptr,
                                                   labels.data<LabelT>(),
                                                   rank,
                                                   N,
                                                   D,
                                                   class_interval.data<int>());
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (nranks > 1) {
    if (pg) {
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*loss);
      out_tensor.push_back(*loss);

      paddle::distributed::AllreduceOptions opts;
      opts.reduce_op = paddle::distributed::ReduceOp::SUM;
      auto task = pg->AllReduce(in_tensor, out_tensor, opts);
      task->Wait();
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::ncclAllReduce(
          loss_ptr,
          loss_ptr,
          loss->numel(),
          paddle::platform::ToNCCLDataType(
              paddle::framework::TransToProtoVarType(loss->dtype())),
          ncclSum,
          comm->comm(),
          stream));
    }
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(margin_cross_entropy,
                   GPU,
                   ALL_LAYOUT,
                   phi::MarginCrossEntropyKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
