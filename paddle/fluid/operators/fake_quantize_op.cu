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

#include <string>
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void find_abs_max_kernel(const int n, const T* in, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ T shared_max_data[];
  if (gridDim.x > 1) {
    shared_max_data[tid] = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = fabs(in[i]);
      if (tmp > shared_max_data[tid]) {
        shared_max_data[tid] = tmp;
      }
    }
  } else {
    if (bid < n) {
      shared_max_data[tid] = fabs(in[bid]);
    } else {
      shared_max_data[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && shared_max_data[tid] < shared_max_data[tid + i]) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

float find_abs_max_gpu(const platform::CUDADeviceContext& ctx,
                       const float* array, int length) {
  float host_max;
  int NUM_THREADS = 1024;
  int gridDimx = (NUM_THREADS - 1 + length) / NUM_THREADS;
  gridDimx = (gridDimx > NUM_THREADS) ? NUM_THREADS : gridDimx;
  framework::Tensor t;
  float* device_max = t.mutable_data<float>(framework::make_ddim({gridDimx}),
                                            platform::CUDAPlace());
  find_abs_max_kernel<float><<<gridDimx, NUM_THREADS,
                               NUM_THREADS * sizeof(float), ctx.stream()>>>(
      length, array, device_max);
  find_abs_max_kernel<
      float><<<1, NUM_THREADS, NUM_THREADS * sizeof(float), ctx.stream()>>>(
      gridDimx, device_max, device_max);
  PADDLE_ENFORCE_EQ(
      cudaMemcpy(&host_max, device_max, sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess, "cudaMemcpy failed");
  return host_max;
}

template <typename T>
__global__ void apply_saturate_kernel(const int n, const T* in, T* out,
                                      int* num_saturate, const T min,
                                      const T max) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ int shared_count[];
  shared_count[tid] = 0;
  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    if (in[i] > max) {
      out[i] = max;
      shared_count[tid] += 1;
    } else if (in[i] < min) {
      out[i] = min;
      shared_count[tid] += 1;
    } else {
      out[i] = in[i];
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      shared_count[tid] += shared_count[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    num_saturate[blockIdx.x] = shared_count[0];
  }
}

template <typename T>
__global__ void reduce_kernel(const int n, const T* in, T* out) {
  int tid = threadIdx.x;
  extern __shared__ T shared_sum[];
  if (tid < n) {
    shared_sum[tid] = in[tid];
  } else {
    shared_sum[tid] = T(0);
  }
  __syncthreads();
  // blockDim.x must >= n
  for (int i = (n + 1) / 2; i > 0; i >>= 1) {
    if (tid < i) {
      shared_sum[tid] += shared_sum[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[0] = shared_sum[0];
  }
}

template <typename T>
int apply_saturate_gpu(const platform::CUDADeviceContext& ctx, const int n,
                       const T* in, T* out, const T min, const T max) {
  int host_num_saturate;
  int NUM_THREADS = 1024;
  int gridDimx = (n + NUM_THREADS - 1) / NUM_THREADS;
  gridDimx = (gridDimx > NUM_THREADS) ? NUM_THREADS : gridDimx;
  framework::Tensor t;
  int* device_num_saturate = t.mutable_data<int>(
      framework::make_ddim({gridDimx}), platform::CUDAPlace());
  apply_saturate_kernel<
      T><<<gridDimx, NUM_THREADS, NUM_THREADS * sizeof(T), ctx.stream()>>>(
      n, in, out, device_num_saturate, min, max);
  reduce_kernel<int><<<1, NUM_THREADS, NUM_THREADS * sizeof(T), ctx.stream()>>>(
      gridDimx, device_num_saturate, device_num_saturate);
  PADDLE_ENFORCE_EQ(cudaSuccess,
                    cudaMemcpy(&host_num_saturate, device_num_saturate,
                               sizeof(int), cudaMemcpyDeviceToHost),
                    "cudaMemcpy failed");
  return host_num_saturate;
}

template <typename DeviceContext, typename T>
class FakeQuantizeCUDAKernel : public framework::OpKernel<T> {
 public:
  T find_range_abs_max(const platform::CUDADeviceContext& ctx,
                       framework::Tensor* scale_list,
                       framework::Tensor* out_scale, const T& cur_scale,
                       int window_size, int current_iter) const {
    T* sl = scale_list->mutable_data<T>(platform::CPUPlace());
    T remove_tmp = sl[current_iter];
    sl[current_iter] = cur_scale;
    T& max_scale = out_scale->mutable_data<T>(platform::CPUPlace())[0];
    if (max_scale < cur_scale) {
      max_scale = cur_scale;
    } else if (fabs(remove_tmp - max_scale) < 1e-6) {
      int size = (current_iter > window_size) ? window_size : current_iter;
      max_scale = T(find_abs_max_gpu(ctx, scale_list->data<float>(), size));
    }
    return max_scale;
  }

  T find_abs_max(framework::Tensor* src, const int n) const {
    T* p = src->mutable_data<T>(platform::CPUPlace());
    T abs_max = T(0.00000001);
    for (int i = 0; i < n; i++) {
      T tmp = fabs(p[i]);
      if (tmp > abs_max) abs_max = tmp;
    }
    return abs_max;
  }

  T find_moving_average_abs_max(framework::Tensor* in_scale,
                                framework::Tensor* out_scale,
                                const T& cur_scale) const {
    T* ins = in_scale->mutable_data<T>(platform::CPUPlace());
    T* outs = out_scale->mutable_data<T>(platform::CPUPlace());
    outs[0] = 0.9 * cur_scale + 0.1 * ins[0];
    return T(outs[0]);
  }

  virtual void Compute(const framework::ExecutionContext& context) const {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto& device_ctx = context.cuda_device_context();
    auto* tensor = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    const bool is_test = context.Attr<bool>("is_test");
    tensor->mutable_data<T>(in->place());
    context.Output<framework::Tensor>("OutMovingScale")
        ->mutable_data<T>(
            context.Input<framework::Tensor>("InMovingScale")->place());
    auto quantize_type =
        static_cast<std::string>(context.Attr<std::string>("quantize_type"));
    if (quantize_type == std::string("range_abs_max")) {
      context.Output<framework::Tensor>("OutScales")
          ->mutable_data<T>(
              context.Input<framework::Tensor>("InScales")->place());
      context.Output<framework::Tensor>("OutCurrentIter")
          ->mutable_data<T>(
              context.Input<framework::Tensor>("InCurrentIter")->place());
    }

    T scale = T(1);
    int window_size = context.Attr<int>("window_size");
    T bin_cnt = (T)((1 << (context.Attr<int>("bit_length") - 1)) - 1);
    if (quantize_type == std::string("abs_max")) {
      auto* saving_scale = context.Output<framework::Tensor>("OutMovingScale");
      scale = (T)find_abs_max_gpu(device_ctx, in->data<float>(), in->numel());
      saving_scale->mutable_data<T>(platform::CPUPlace())[0] = scale;

      auto& device_ctx = context.template device_context<DeviceContext>();
      auto* scale_list = context.Output<framework::Tensor>("OutScales");
      math::SetConstant<DeviceContext, T> scalar;
      scale_list->mutable_data<T>(context.GetPlace());
      scalar(device_ctx, scale_list, static_cast<T>(0));
      auto* iter = context.Output<framework::Tensor>("OutCurrentIter");
      iter->mutable_data<T>(context.GetPlace());
      scalar(device_ctx, iter, static_cast<T>(0));
    } else if (quantize_type == std::string("range_abs_max")) {
      auto* moving_scale = const_cast<framework::Tensor*>(
          context.Input<framework::Tensor>("InMovingScale"));
      if (is_test) {
        scale = moving_scale->mutable_data<T>(platform::CPUPlace())[0];
      } else {
        auto* it = const_cast<framework::Tensor*>(
            context.Input<framework::Tensor>("InCurrentIter"));
        auto* iter = context.Output<framework::Tensor>("OutCurrentIter");
        int* last_iter = it->mutable_data<int>(platform::CPUPlace());
        int* current_iter = iter->mutable_data<int>(platform::CPUPlace());
        auto* scale_list = context.Output<framework::Tensor>("OutScales");
        auto* saving_scale =
            context.Output<framework::Tensor>("OutMovingScale");
        scale = (T)find_abs_max_gpu(device_ctx, in->data<float>(), in->numel());
        scale = find_range_abs_max(device_ctx, scale_list, saving_scale, scale,
                                   window_size, current_iter[0]);
        (*current_iter) = (*last_iter) + 1;
      }
    } else if (quantize_type == std::string("moving_average_abs_max")) {
      auto* moving_scale = const_cast<framework::Tensor*>(
          context.Input<framework::Tensor>("InMovingScale"));
      if (is_test) {
        scale = moving_scale->mutable_data<T>(platform::CPUPlace())[0];
      } else {
        scale = (T)find_abs_max_gpu(device_ctx, in->data<float>(), in->numel());
        auto* saving_scale =
            context.Output<framework::Tensor>("OutMovingScale");
        scale = find_moving_average_abs_max(
            const_cast<framework::Tensor*>(moving_scale), saving_scale, scale);
      }
    }

    apply_saturate_gpu<T>(device_ctx, in->numel(), in->data<T>(),
                          tensor->mutable_data<T>(in->place()), -scale, scale);
    scale = bin_cnt / scale;

    auto& dev =
        *context.template device_context<DeviceContext>().eigen_device();
    auto eigen_out = framework::EigenVector<T>::Flatten(*tensor);
    auto eigen_in = framework::EigenVector<T>::Flatten(*tensor);
    eigen_out.device(dev) = (scale * eigen_in).round();
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(fake_quantize,
                        paddle::operators::FakeQuantizeCUDAKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        paddle::operators::FakeQuantizeCUDAKernel<
                            paddle::platform::CUDADeviceContext, double>);
