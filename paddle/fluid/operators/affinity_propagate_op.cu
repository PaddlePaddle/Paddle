/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <string>
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeCalcGateWbAndSumByGuidance2D(const T* guidance, T* gate_wb, T* gate_sum, const bool abs_flag, const int kernel_size, const int n, const int h, const int w) {
  int nthreads = n * h * w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / h / w;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int side_num = (kernel_size - 1) / 2;
    int channel_num = kernel_size * kernel_size - 1;
    int gh = h - (2 * side_num);
    int gw = w - (2 * side_num);

    T abs_sum = 0.;
    int rc = 0;
    for (int i = 0; i <= 2 * side_num; i++) {
      for (int j = 0; j <= 2 * side_num; j++) {
        if (i != side_num || j != side_num) {
          int rh = hi - i;
          int rw = wi - j;
          if (rh >= 0 && rh < gh && rw >= 0 && rw < gw) {
            int idx = rw + rh * gw + rc * gh * gw + ni * channel_num * gh * gw;
            abs_sum += std::fabs(guidance[idx]);
          }
          rc++;
        }
      }
    }

    rc = 0;
    T sum = 0.;
    for (int i = 0; i <= 2 * side_num; i++) {
      for (int j = 0; j <= 2 * side_num; j++) {
        if (i != side_num || j != side_num) {
          int rh = hi - i;
          int rw = wi - j;
          if (rh >= 0 && rh < gh && rw >= 0 && rw < gw) {
            int idx = rw + rh * gw + rc * gh * gw + ni * channel_num * gh * gw;
            T rwb = abs_flag ? std::fabs(guidance[idx]) : guidance[idx];
            gate_wb[tid] = rwb / abs_sum;
            printf("wb: %d, %d, %d, %d, %f\n", ni, rc, rh, rw, gate_wb[tid]);
            sum += gate_wb[tid];
          }
          rc++;
        }
      }
    }

    if (wi >= side_num && wi < w - side_num && hi >= side_num && hi < h - side_num) {
      int gate_sum_idx = wi - side_num + (hi - side_num) * w + ni * (h - 2 * side_num) * (w - 2 * side_num);
      gate_sum[gate_sum_idx] = sum;
      printf("sum: %d, %d, %d, %f\n", ni, hi-side_num, wi-side_num, sum);
    }
  }
}

template <typename T>
class AffinityPropagateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    auto* input = ctx.Input<Tensor>("X");
    auto* guidance = ctx.Input<Tensor>("Guidance");
    auto* mask = ctx.Input<Tensor>("Mask");
    auto* output = ctx.Output<Tensor>("Out");

    const int prop_iters = ctx.Attr<int>("prop_iters");
    const int kernel_size = ctx.Attr<int>("kernel_size");
    std::string norm_type = ctx.Attr<std::string>("norm_type");
    const bool abs_flag = norm_type == "abs_sum";

    const T* input_data = input->data<T>();
    const T* guidance_data = guidance->data<T>();
    const T* mask_data;
    if (mask) mask_data = mask->data<T>();
    else mask_data = NULL;

    auto input_dims = input->dims();
    if (input_dims.size() == 4) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int h = input_dims[2];
      const int w = input_dims[3];

      auto* output_data = output->mutable_data<T>({n, c, h, w}, ctx.GetPlace());

      int gate_c = kernel_size * kernel_size - 1;
      Tensor gate_wb = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({n, gate_c, h + 2, w + 2}, dev_ctx);
      auto* gate_wb_data = gate_wb.data<T>();
      Tensor gate_sum = ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({n, h, w}, dev_ctx);
      auto* gate_sum_data = gate_sum.data<T>();

      int pixelNum = n * h * w;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      int pad_size = kernel_size - 1;
      KeCalcGateWbAndSumByGuidance2D<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          guidance_data, gate_wb_data, gate_sum_data, abs_flag, kernel_size,
          n, h + pad_size, w + pad_size);

    }
  }
};

template <typename T>
class AffinityPropagateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(affinity_propagate, 
                        ops::AffinityPropagateOpCUDAKernel<float>,
                        ops::AffinityPropagateOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(affinity_propagate_grad,
                        ops::AffinityPropagateGradOpCUDAKernel<float>,
                        ops::AffinityPropagateGradOpCUDAKernel<double>);
