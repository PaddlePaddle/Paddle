/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mean_iou_op.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void CountCUDAKernel(const int num_classes,
                                const int count,
                                const T* predictions,
                                const T* labels,
                                int* wrong,
                                int* correct) {
  extern __shared__ int blcok_cache[];
  int* wrong_c = blcok_cache;
  int* correct_c = blcok_cache + num_classes;
  // init cache
  for (int i = threadIdx.x; i < num_classes * 2; i += blockDim.x) {
    blcok_cache[i] = 0;
  }
  __syncthreads();

  T pred;
  T label;
  CUDA_KERNEL_LOOP(i, count) {
    pred = predictions[i];
    label = labels[i];
    if (pred == label) {
      atomicAdd(correct_c + pred, 1);
    } else {
      atomicAdd(wrong_c + pred, 1);
      atomicAdd(wrong_c + label, 1);
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    atomicAdd(wrong + i, wrong_c[i]);
    atomicAdd(correct + i, correct_c[i]);
  }
}

__global__ void ComputeIoUCUDAKernel(
    const int num_classes, int* wrong, int* correct, float* ious, float* iou) {
  __shared__ int valid_count_c;
  if (threadIdx.x == 0) {
    valid_count_c = 0;
  }
  __syncthreads();
  CUDA_KERNEL_LOOP(i, num_classes) {
    int wrong_n = wrong[i];
    int correct_n = correct[i];
    int denominator = wrong_n + correct_n;
    if (denominator > 0) {
      atomicAdd(&valid_count_c, 1);
      ious[i] = static_cast<float>(correct_n) / denominator;
    } else {
      ious[i] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float iou_sum = 0;
    for (int i = 0; i < num_classes; ++i) {
      iou_sum += ious[i];
    }
    iou[0] += iou_sum / valid_count_c;
  }
}

template <typename T>
class MeanIoUCUDAOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto& place = *dev_ctx.eigen_device();
    // get input and output tensor
    auto* predictions = ctx.Input<phi::DenseTensor>("Predictions");
    auto* labels = ctx.Input<phi::DenseTensor>("Labels");
    auto* out_mean_iou = ctx.Output<phi::DenseTensor>("OutMeanIou");
    auto* out_wrong = ctx.Output<phi::DenseTensor>("OutWrong");
    auto* out_correct = ctx.Output<phi::DenseTensor>("OutCorrect");
    int num_classes = static_cast<int>(ctx.Attr<int>("num_classes"));

    // Get data ptr
    const T* predictions_data = predictions->data<T>();
    const T* labels_data = labels->data<T>();
    int* out_wrong_data = out_wrong->mutable_data<int>(ctx.GetPlace());
    int* out_correct_data = out_correct->mutable_data<int>(ctx.GetPlace());
    float* out_mean_iou_data =
        out_mean_iou->mutable_data<float>(ctx.GetPlace());

    // Get Eigen tensor
    auto out_mean_iou_t = EigenTensor<float, 1>::From(*out_mean_iou);
    auto out_wrong_t = EigenTensor<int, 1>::From(*out_wrong);
    auto out_correct_t = EigenTensor<int, 1>::From(*out_correct);

    // Temporary memory
    auto tmp_ious_data = memory::Alloc(
        dev_ctx.GetPlace(),
        num_classes * sizeof(float),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    float* ious_data = static_cast<float*>(tmp_ious_data->ptr());

    // Init out_wrong, out_correct and out_mean_iou
    out_wrong_t.device(place) = out_wrong_t.constant(0);
    out_correct_t.device(place) = out_correct_t.constant(0);
    out_mean_iou_t.device(place) = out_mean_iou_t.constant(0.0f);

    // collect pre wrong, correct and mean_iou
    auto in_mean_ious = ctx.MultiInput<phi::DenseTensor>("InMeanIou");
    for (int i = 0; i < in_mean_ious.size(); ++i) {
      out_mean_iou_t.device(place) +=
          EigenTensor<float, 1>::From(*in_mean_ious[i]);
    }
    auto in_wrongs = ctx.MultiInput<phi::DenseTensor>("InWrongs");
    for (int i = 0; i < in_wrongs.size(); ++i) {
      out_wrong_t.device(place) += EigenTensor<int, 1>::From(*in_wrongs[i]);
    }
    auto in_corrects = ctx.MultiInput<phi::DenseTensor>("InCorrects");
    for (int i = 0; i < in_corrects.size(); ++i) {
      out_correct_t.device(place) += EigenTensor<int, 1>::From(*in_corrects[i]);
    }
    // compute
    auto stream = ctx.cuda_device_context().stream();
    int block = PADDLE_CUDA_NUM_THREADS;
    int grid = (predictions->numel() + block - 1) / block;
    int cache_size = (num_classes * 2 + 1) * sizeof(int);
    CountCUDAKernel<T>
        <<<grid, block, cache_size, stream>>>(num_classes,
                                              predictions->numel(),
                                              predictions_data,
                                              labels_data,
                                              out_wrong_data,
                                              out_correct_data);

    ComputeIoUCUDAKernel<<<1, block, 0, stream>>>(num_classes,
                                                  out_wrong_data,
                                                  out_correct_data,
                                                  ious_data,
                                                  out_mean_iou_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(mean_iou,
                        ops::MeanIoUCUDAOpKernel<int>,
                        ops::MeanIoUCUDAOpKernel<int64_t>,
                        ops::MeanIoUCUDAOpKernel<int32_t>);
