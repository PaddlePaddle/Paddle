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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/sgd_group_op.h"
#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {

namespace {

template <typename T>
__global__ void SGDGroupKernel(const T* g, const T* p, const T* learning_rate,
                               const int num, T* p_out) {
  T lr = learning_rate[0];
  int grid_size = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += grid_size) {
    T g_data = g[i];
    T p_data = p[i];
    p_out[i] = p_data - lr * g_data;
  }
}

template <typename T>
__device__ T upper_bound(const T* first, T count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  T step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first - orig;
}

template <typename T>
__global__ void SGDGroupKernel(T** grads, T** params, T** learning_rate,
                               const int* p_numbers, int para_num, int ele_num,
                               T** params_out) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(p_numbers, para_num, tid_x) - 1;
  int curr_offset = p_numbers[segment];
  int curr_segment = segment;

  for (; tid_x < ele_num; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset;
    while ((curr_col_offset = p_numbers[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    T* grad_ptr = grads[curr_segment];
    T* params_ptr = params[curr_segment];
    T* learning_rate_ptr = learning_rate[curr_segment];
    T* params_out_ptr = params_out[curr_segment];

    T g_data = grad_ptr[local_col];
    T p_data = params_ptr[local_col];
    T lr = *learning_rate_ptr;
    params_out_ptr[local_col] = p_data - lr * g_data;
  }
}

}  // namespace

template <typename T>
class SGDGroupOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto params = ctx.MultiInput<framework::Tensor>("Params");
    auto learning_rates = ctx.MultiInput<framework::Tensor>("LearningRates");
    auto grads = ctx.MultiInput<framework::Tensor>("Grads");

    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOuts");

    auto grad_var = ctx.MultiInputVar("Grads");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (grad_var[0]->IsType<framework::LoDTensor>()) {
      int p_num = params.size();
      int p_ele_num = 0;
      framework::Vector<int16_t> params_data(p_num * sizeof(T*) / 2);
      framework::Vector<int16_t> grads_data(p_num * sizeof(T*) / 2);
      framework::Vector<int16_t> param_outs_data(p_num * sizeof(T*) / 2);
      framework::Vector<int16_t> lrs_data(p_num * sizeof(T*) / 2);

      T** params_ptr = reinterpret_cast<T**>(params_data.data());
      T** grads_ptr = reinterpret_cast<T**>(grads_data.data());
      T** param_out_ptr = reinterpret_cast<T**>(param_outs_data.data());
      T** lrs_ptr = reinterpret_cast<T**>(lrs_data.data());

      framework::Vector<int> param_num(p_num + 1);
      param_num[0] = 0;

      for (int i = 0; i < p_num; ++i) {
        p_ele_num += params[i]->numel();
        param_num[i + 1] = p_ele_num;
        params_ptr[i] = const_cast<T*>(params[i]->data<T>());
        grads_ptr[i] = const_cast<T*>(grads[i]->data<T>());
        lrs_ptr[i] = const_cast<T*>(learning_rates[i]->data<T>());
        param_out_ptr[i] = param_outs[i]->mutable_data<T>(ctx.GetPlace());
      }

      T** params_gpu =
          reinterpret_cast<T**>(params_data.CUDAMutableData(ctx.GetPlace()));
      T** grads_gpu =
          reinterpret_cast<T**>(grads_data.CUDAMutableData(ctx.GetPlace()));
      T** param_out_gpu = reinterpret_cast<T**>(
          param_outs_data.CUDAMutableData(ctx.GetPlace()));
      T** lrs_data_gpu =
          reinterpret_cast<T**>(lrs_data.CUDAMutableData(ctx.GetPlace()));
      const int* param_num_gpu = param_num.CUDAData(ctx.GetPlace());

      // computation
      // set the thread block and grid according to CurrentDeviceId
      const int kThreadsPerBlock = 1024;
      int block = kThreadsPerBlock;
      if (p_ele_num < kThreadsPerBlock) {  // block is aligned by 32.
        block = ((p_ele_num + 31) >> 5) << 5;
      }
      int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
      int max_blocks = std::max(max_threads / block, 1);

      int grid = std::min((p_ele_num + block - 1) / block, max_blocks);

      SGDGroupKernel<T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          grads_gpu, params_gpu, lrs_data_gpu, param_num_gpu, p_num, p_ele_num,
          param_out_gpu);
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sgd_group, ops::SGDGroupOpCUDAKernel<float>,
                        ops::SGDGroupOpCUDAKernel<double>);
