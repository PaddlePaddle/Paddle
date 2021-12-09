/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <vector>
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/margin_cross_entropy_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/string/string_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

void GetClassInterval(const gpuStream_t& stream, const platform::Place& place,
                      const platform::DeviceContext& ctx, const int rid,
                      const int rank, const int nranks, const int D,
                      Tensor* class_interval) {
  std::vector<int> shard_dim_vec(nranks + 1, 0);
  shard_dim_vec[rank + 1] = D;
  if (nranks <= 1) {
    framework::TensorFromVector(shard_dim_vec, ctx, class_interval);
    return;
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  Tensor num_classes_per_device;
  framework::TensorFromVector(shard_dim_vec, ctx, &num_classes_per_device);
  int* num_classes_per_device_ptr = num_classes_per_device.data<int>();

  const auto& comm = platform::NCCLCommContext::Instance().Get(rid, place);
  // use global calculate stream
  const auto calcu_stream =
      static_cast<platform::CUDADeviceContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      num_classes_per_device_ptr, num_classes_per_device_ptr,
      num_classes_per_device.numel(),
      platform::ToNCCLDataType(num_classes_per_device.type()), ncclSum,
      comm->comm(), calcu_stream));

  auto class_interval_ptr =
      class_interval->mutable_data<int>({nranks + 1}, place);
  size_t cub_temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum<int*, int*>(
      nullptr, cub_temp_storage_bytes, nullptr, nullptr, nranks + 1, stream);
  auto cub_temp_storage = memory::Alloc(place, cub_temp_storage_bytes);
  cub::DeviceScan::InclusiveSum<int*, int*>(
      cub_temp_storage->ptr(), cub_temp_storage_bytes,
      num_classes_per_device_ptr, class_interval_ptr, nranks + 1, stream);
  return;
#endif
}

template <typename T, typename IndexT>
__global__ void AddMarginToPositiveLogitsKernel(
    T* logit, const IndexT* label, const float margin1, const float margin2,
    const float margin3, const int rank, const int nranks, const int64_t N,
    const int64_t D, const int* class_interval_ptr) {
  using MPType = typename details::MPTypeTrait<T>::Type;
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
                   num_classes, real_label);

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

template <typename Tx, typename Ty = Tx>
struct ExpAndSum {
  using Transformer = kps::ExpFunctor<Tx>;

  inline Ty initial() { return static_cast<Ty>(0.0f); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b + a;
  }
};

template <typename T>
__global__ void ScaleLogitKernel(T* logits, const float scale, const int64_t N,
                                 const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) { logits[i] *= static_cast<T>(scale); }
}

template <typename T>
__global__ void LogitsMinusMaxKernel(T* logits, const T* logits_max_per_row,
                                     const int64_t N, const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    logits[i] -= logits_max_per_row[row];
  }
}

template <typename T>
__global__ void LogitsMinusLogSumKernel(T* logits, const T* logits_sum_per_row,
                                        const int64_t N, const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    logits[i] -= kps::details::Log(logits_sum_per_row[row]);
  }
}

template <typename T, typename IndexT>
__global__ void HardLabelSoftmaxWithCrossEntropyKernel(
    T* loss, T* log_softmax, const IndexT* labels, const int rank,
    const int64_t N, const int64_t D, const int* class_interval_ptr) {
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      auto softmax = log_softmax[i];
      loss[row] = -softmax;
      log_softmax[i] = kps::details::Exp(softmax);
    } else {
      log_softmax[i] = kps::details::Exp(log_softmax[i]);
    }
  }
}

template <typename T, typename IndexT>
__global__ void CalculateGrad(T* logits_grad, const T* loss_grad,
                              const T* logits, const IndexT* labels,
                              const float margin1, const float margin2,
                              const float scale, const int rank,
                              const int64_t N, const int64_t D,
                              const int* class_interval_ptr) {
  using MPType = typename details::MPTypeTrait<T>::Type;
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      logits_grad[i] = (logits_grad[i] - static_cast<T>(1.0)) * loss_grad[row];
      if (fabs(margin1 - 1.0) > 1e-8 || fabs(margin2) > 1e-8) {
        MPType dout = static_cast<MPType>(logits_grad[i]);
        MPType one = static_cast<MPType>(1.0f);
        MPType x = static_cast<MPType>(logits[i]);
        MPType m1 = static_cast<MPType>(margin1);
        MPType m2 = static_cast<MPType>(margin2);

        MPType d = m1 * sin(m1 * acos(x) + m2) / sqrt(one - x * x);
        logits_grad[i] = static_cast<T>(dout * d);
      }
    } else {
      logits_grad[i] *= loss_grad[row];
    }
    if (fabs(scale - 1.0) > 1e-8) {
      logits_grad[i] *= static_cast<T>(scale);
    }
  }
}

template <typename T>
class MarginCrossEntropyOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* logits = ctx.Input<Tensor>("Logits");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* softmax = ctx.Output<Tensor>("Softmax");
    Tensor* loss = ctx.Output<Tensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const float margin1 = ctx.Attr<float>("margin1");
    const float margin2 = ctx.Attr<float>("margin2");
    const float margin3 = ctx.Attr<float>("margin3");
    const float scale = ctx.Attr<float>("scale");

    const auto& place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    platform::NCCLComm* comm;
    gpuStream_t stream;
    if (nranks > 1) {
      comm = platform::NCCLCommContext::Instance().Get(rid, place);

      // use global calculate stream
      stream = static_cast<platform::CUDADeviceContext*>(
                   platform::DeviceContextPool::Instance().Get(place))
                   ->stream();
    }
#endif

    // allocate memory on device.
    T* softmax_ptr = softmax->mutable_data<T>(place);
    T* loss_ptr = loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = SizeToAxis(axis, logits_dims);
    const int D = SizeFromAxis(axis, logits_dims);

    int blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->type();

    // copy logits to softmax variable since we can't modify logits,
    // and it also be used when calculate grad
    framework::TensorCopy(*logits, ctx.GetPlace(), ctx.device_context(),
                          softmax);

    Tensor softmax_2d;
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    T* logits_ptr = softmax_2d.data<T>();

    Tensor class_interval;
    GetClassInterval(dev_ctx.stream(), place, ctx.cuda_device_context(), rid,
                     rank, nranks, D, &class_interval);

    // step 1, preprocess logits
    // add margin for positive elements
    // theta = acos(x_i)
    // (cos(m1 * theta + m2) - m3)
    // save match_logits, used for gradient computation.
    if (label_type == framework::proto::VarType::INT32) {
      typedef int32_t LabelT;
      AddMarginToPositiveLogitsKernel<
          T><<<NumBlocks(N), threads, 0, dev_ctx.stream()>>>(
          logits_ptr, labels->data<LabelT>(), margin1, margin2, margin3, rank,
          nranks, N, D, class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      typedef int64_t LabelT;
      AddMarginToPositiveLogitsKernel<
          T><<<NumBlocks(N), threads, 0, dev_ctx.stream()>>>(
          logits_ptr, labels->data<LabelT>(), margin1, margin2, margin3, rank,
          nranks, N, D, class_interval.data<int>());
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "margin_cross_entropy label type noly support int32 and int64, "
          "but got %s",
          label_type));
    }

    // scale by s
    ScaleLogitKernel<T><<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
        logits_ptr, scale, N, D);

    // step 2, obtain logit_max
    Tensor logits_max;
    logits_max =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    T* logits_max_buff = logits_max.mutable_data<T>(place);
    TensorReduceFunctorImpl<T, T, CustomMax>(softmax_2d, &logits_max, {1},
                                             dev_ctx.stream());

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          logits_max_buff, logits_max_buff, logits_max.numel(),
          platform::ToNCCLDataType(logits_max.type()), ncclMax, comm->comm(),
          stream));
    }
#endif

    // step 3, logit - logit_max
    LogitsMinusMaxKernel<T><<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
        logits_ptr, logits_max_buff, N, D);

    // step 4, sum(exp(logit - logit_max))
    Tensor sum_exp_logits;
    sum_exp_logits =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    T* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);
    TensorReduceFunctorImpl<T, T, ExpAndSum>(softmax_2d, &sum_exp_logits, {1},
                                             dev_ctx.stream());

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          sum_exp_logits_buff, sum_exp_logits_buff, sum_exp_logits.numel(),
          platform::ToNCCLDataType(sum_exp_logits.type()), ncclSum,
          comm->comm(), stream));
    }
#endif

    // step 5, (logit - logit_max) - log(sum(exp(logit - logit_max)))
    LogitsMinusLogSumKernel<
        T><<<NumBlocks(N * D), threads, 0, dev_ctx.stream()>>>(
        logits_ptr, sum_exp_logits_buff, N, D);

    // step 6, prob = exp((logit - logit_max) - log(sum(exp(logit -
    // logit_max))))
    // loss = -((logit_i - logit_max) - log(sum(exp(logit - logit_max))))
    math::SetConstant<platform::CUDADeviceContext, T>()(dev_ctx, loss,
                                                        static_cast<T>(0.0));
    if (label_type == framework::proto::VarType::INT32) {
      typedef int32_t LabelT;
      HardLabelSoftmaxWithCrossEntropyKernel<
          T, LabelT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          loss_ptr, logits_ptr, labels->data<LabelT>(), rank, N, D,
          class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      typedef int64_t LabelT;
      HardLabelSoftmaxWithCrossEntropyKernel<
          T, LabelT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          loss_ptr, logits_ptr, labels->data<LabelT>(), rank, N, D,
          class_interval.data<int>());
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          loss_ptr, loss_ptr, loss->numel(),
          platform::ToNCCLDataType(loss->type()), ncclSum, comm->comm(),
          stream));
    }
#endif
  }
};

template <typename T>
class MarginCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* labels = context.Input<Tensor>("Label");
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* softmax = context.Input<Tensor>("Softmax");

    const Tensor* loss_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));

    const bool return_softmax = context.Attr<bool>("return_softmax");

    const int rid = context.Attr<int>("ring_id");
    const int nranks = context.Attr<int>("nranks");
    const int rank = context.Attr<int>("rank");

    const float margin1 = context.Attr<float>("margin1");
    const float margin2 = context.Attr<float>("margin2");
    const float margin3 = context.Attr<float>("margin3");
    const float scale = context.Attr<float>("scale");

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int N = SizeToAxis(axis, sofrmax_dims);
    const int D = SizeFromAxis(axis, sofrmax_dims);

    if (return_softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    } else {
      logit_grad->ShareDataWith(*softmax);
    }

    int blocks = NumBlocks(N * D);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->type();

    Tensor class_interval;
    GetClassInterval(dev_ctx.stream(), context.GetPlace(),
                     context.cuda_device_context(), rid, rank, nranks, D,
                     &class_interval);

    if (label_type == framework::proto::VarType::INT32) {
      typedef int32_t LabelT;
      CalculateGrad<T, LabelT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad->data<T>(), loss_grad->data<T>(), logits->data<T>(),
          labels->data<LabelT>(), margin1, margin2, scale, rank, N, D,
          class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      typedef int64_t LabelT;
      CalculateGrad<T, LabelT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad->data<T>(), loss_grad->data<T>(), logits->data<T>(),
          labels->data<LabelT>(), margin1, margin2, scale, rank, N, D,
          class_interval.data<int>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(margin_cross_entropy,
                        ops::MarginCrossEntropyOpCUDAKernel<float>,
                        ops::MarginCrossEntropyOpCUDAKernel<double>,
                        ops::MarginCrossEntropyOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(margin_cross_entropy_grad,
                        ops::MarginCrossEntropyGradCUDAKernel<float>,
                        ops::MarginCrossEntropyGradCUDAKernel<double>,
                        ops::MarginCrossEntropyGradCUDAKernel<plat::float16>);
