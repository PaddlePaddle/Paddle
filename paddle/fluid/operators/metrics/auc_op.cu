/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/metrics/auc_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

__global__ void ClearObsoleteDataKernel(int64_t *pos, int64_t *neg,
                                        const int bucket_length,
                                        const int slide_steps) {
  int cur_step_index =
      static_cast<int>(pos[(slide_steps + 1) * bucket_length]) % slide_steps;
  int cur_step_begin = cur_step_index * bucket_length;
  int sum_step_begin = slide_steps * bucket_length;
  CUDA_KERNEL_LOOP(i, bucket_length) {
    pos[sum_step_begin + i] -= pos[cur_step_begin + i];
    neg[sum_step_begin + i] -= neg[cur_step_begin + i];
    pos[cur_step_begin + i] = neg[cur_step_begin + i] = 0;
  }
}

__global__ void UpdateSumDataKernel(int64_t *pos, int64_t *neg,
                                    const int bucket_length,
                                    const int slide_steps) {
  int cur_step_index =
      static_cast<int>(pos[(slide_steps + 1) * bucket_length]) % slide_steps;
  int cur_step_begin = cur_step_index * bucket_length;
  int sum_step_begin = slide_steps * bucket_length;
  CUDA_KERNEL_LOOP(i, bucket_length) {
    pos[sum_step_begin + i] += pos[cur_step_begin + i];
    neg[sum_step_begin + i] += neg[cur_step_begin + i];
  }
}

template <typename T>
__global__ void AddDataKernel(const int64_t *label_data, const T *pred_data,
                              const int inference_width,
                              const int num_thresholds, int64_t *pos,
                              int64_t *neg, const int numel,
                              const int slide_steps) {
  int cur_step_begin = 0;
  if (slide_steps > 0) {
    int cur_step_index =
        static_cast<int>(pos[(slide_steps + 1) * (1 + num_thresholds)]) %
        slide_steps;
    cur_step_begin = cur_step_index * (1 + num_thresholds);
  }
  CUDA_KERNEL_LOOP(i, numel) {
    auto predict_data = pred_data[i * inference_width + (inference_width - 1)];
    PADDLE_ENFORCE(predict_data <= 1, "The predict data must less or equal 1.");
    PADDLE_ENFORCE(predict_data >= 0,
                   "The predict data must gather or equal 0.");
    uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
    if (label_data[i]) {
      paddle::platform::CudaAtomicAdd(pos + cur_step_begin + binIdx, 1);
    } else {
      paddle::platform::CudaAtomicAdd(neg + cur_step_begin + binIdx, 1);
    }
  }
}
__global__ void CalcAucKernel(int64_t *stat_pos, int64_t *stat_neg,
                              int num_thresholds, double *auc,
                              bool need_add_batch_num) {
  *auc = 0.0f;
  double totPos = 0.0;
  double totNeg = 0.0;
  double totPosPrev = 0.0;
  double totNegPrev = 0.0;

  int idx = num_thresholds;

  while (idx >= 0) {
    totPosPrev = totPos;
    totNegPrev = totNeg;
    totPos += stat_pos[idx];
    totNeg += stat_neg[idx];
    *auc += (totNeg - totNegPrev) * (totPos + totPosPrev) / 2.0;
    --idx;
  }

  if (totPos > 0.0 && totNeg > 0.0) {
    *auc = *auc / totPos / totNeg;
  }
  if (need_add_batch_num) {
    stat_pos[num_thresholds + 1] += 1;
    stat_neg[num_thresholds + 1] += 1;
  }
}

template <typename DeviceContext, typename T>
class AucCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *predict = ctx.Input<Tensor>("Predict");
    auto *label = ctx.Input<Tensor>("Label");

    int num_thresholds = ctx.Attr<int>("num_thresholds");
    int slide_steps = ctx.Attr<int>("slide_steps");

    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto *auc_tensor = ctx.Output<Tensor>("AUC");
    auto *stat_pos = ctx.Output<Tensor>("StatPosOut");
    auto *stat_neg = ctx.Output<Tensor>("StatNegOut");

    auto *origin_stat_pos = stat_pos->mutable_data<int64_t>(ctx.GetPlace());
    auto *origin_stat_neg = stat_neg->mutable_data<int64_t>(ctx.GetPlace());
    auto *auc_value = auc_tensor->mutable_data<double>(ctx.GetPlace());

    auto *stat_pos_in_tensor = ctx.Input<Tensor>("StatPos");
    auto *pos_in_data = stat_pos_in_tensor->data<int64_t>();
    auto *stat_neg_in_tensor = ctx.Input<Tensor>("StatNeg");
    auto *neg_in_data = stat_neg_in_tensor->data<int64_t>();
#ifdef PADDLE_WITH_CUDA
    if (stat_pos_in_tensor != stat_pos) {
      cudaMemcpy(origin_stat_pos, pos_in_data,
                 ((1 + slide_steps) * (num_thresholds + 1) +
                  (slide_steps > 0 ? 1 : 0)) *
                     sizeof(int64_t),
                 cudaMemcpyDeviceToDevice);
    }
    if (stat_neg_in_tensor != stat_neg) {
      cudaMemcpy(origin_stat_neg, neg_in_data,
                 ((1 + slide_steps) * (num_thresholds + 1) +
                  (slide_steps > 0 ? 1 : 0)) *
                     sizeof(int64_t),
                 cudaMemcpyDeviceToDevice);
    }
#else
    if (stat_pos_in_tensor != stat_pos) {
      hipMemcpy(origin_stat_pos, pos_in_data,
                ((1 + slide_steps) * (num_thresholds + 1) +
                 (slide_steps > 0 ? 1 : 0)) *
                    sizeof(int64_t),
                hipMemcpyDeviceToDevice);
    }
    if (stat_neg_in_tensor != stat_neg) {
      hipMemcpy(origin_stat_neg, neg_in_data,
                ((1 + slide_steps) * (num_thresholds + 1) +
                 (slide_steps > 0 ? 1 : 0)) *
                    sizeof(int64_t),
                hipMemcpyDeviceToDevice);
    }
#endif

    statAuc(ctx, label, predict, num_thresholds, slide_steps, origin_stat_pos,
            origin_stat_neg);
    int sum_offset = slide_steps * (num_thresholds + 1);
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    CalcAucKernel<<<1, 1, 0, stream>>>(
        origin_stat_pos + sum_offset, origin_stat_neg + sum_offset,
        num_thresholds, auc_value, slide_steps > 0);
  }

 private:
  inline static double trapezoidArea(double X1, double X2, double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }

  inline static void statAuc(const framework::ExecutionContext &ctx,
                             const framework::Tensor *label,
                             const framework::Tensor *predict,
                             const int num_thresholds, const int slide_steps,
                             int64_t *origin_stat_pos,
                             int64_t *origin_stat_neg) {
    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];
    const T *inference_data = predict->data<T>();
    const auto *label_data = label->data<int64_t>();
    const int bucket_length = num_thresholds + 1;
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    if (slide_steps == 0) {
      AddDataKernel<<<(batch_size + PADDLE_CUDA_NUM_THREADS - 1) /
                          PADDLE_CUDA_NUM_THREADS,
                      PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          label_data, inference_data, inference_width, num_thresholds,
          origin_stat_pos, origin_stat_neg, batch_size, slide_steps);
      return;
    }
    // the last number of origin_stat_pos store the index should be used in
    // current step
    int cur_step_index =
        static_cast<int>(origin_stat_pos[(slide_steps + 1) * bucket_length]) %
        slide_steps;
    int cur_step_begin = cur_step_index * bucket_length;
    int sum_step_begin = slide_steps * bucket_length;

    ClearObsoleteDataKernel<<<(bucket_length + PADDLE_CUDA_NUM_THREADS - 1) /
                                  PADDLE_CUDA_NUM_THREADS,
                              PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        origin_stat_pos, origin_stat_neg, bucket_length, slide_steps);

    AddDataKernel<<<(batch_size + PADDLE_CUDA_NUM_THREADS - 1) /
                        PADDLE_CUDA_NUM_THREADS,
                    PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        label_data, inference_data, inference_width, num_thresholds,
        origin_stat_pos, origin_stat_neg, batch_size, slide_steps);
    UpdateSumDataKernel<<<(bucket_length + PADDLE_CUDA_NUM_THREADS - 1) /
                              PADDLE_CUDA_NUM_THREADS,
                          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        origin_stat_pos, origin_stat_neg, bucket_length, slide_steps);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(auc,
                        ops::AucCUDAKernel<paddle::platform::CUDAPlace, float>);
