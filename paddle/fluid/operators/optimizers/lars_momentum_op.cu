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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"
#include "paddle/fluid/platform/fast_divmod.h"

#if defined(__NVCC__) && CUDA_VERSION >= 11000
/* Once CUDA_VERSION is beyond 11.0, cooperative_groups can be involved in
   without adding --rdc=true compile flag, then L2_norm cuda kernel can be
   set as a __device__ kernel rather than global kernel. On the contrary,
   the compile flag shall be set in old version, which may affect the cuda
   kernel performance in paddle, consequently, L2_norm kernel shall be set
   as a __global__ kernel.
*/
#include <cooperative_groups.h>
#define LARS_FUNCTION_FLAG __device__
#else
#define LARS_FUNCTION_FLAG __global__
#endif

#ifdef __HIPCC__
#define LARS_BLOCK_SIZE 256
#else
#define LARS_BLOCK_SIZE 512
#endif

#define MAX_MERGED_OPS 200

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

template <typename T, typename MT>
struct MergedParameter {
 public:
  int64_t numel_arr[MAX_MERGED_OPS];
  int repeat_arr[MAX_MERGED_OPS];
  int thresh_arr[MAX_MERGED_OPS];
  const T* __restrict__ p_arr[MAX_MERGED_OPS];
  const T* __restrict__ g_arr[MAX_MERGED_OPS];
  const MT* __restrict__ v_arr[MAX_MERGED_OPS];
  const MT* __restrict__ lr_arr[MAX_MERGED_OPS];
  const MT* __restrict__ master_p_arr[MAX_MERGED_OPS];
  T* __restrict__ p_out_arr[MAX_MERGED_OPS];
  MT* __restrict__ v_out_arr[MAX_MERGED_OPS];
  MT* __restrict__ master_p_out_arr[MAX_MERGED_OPS];
};

__device__ __forceinline__ float Sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ double Sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ float Fma(float x, float y, float z) {
  return fmaf(x, y, z);
}
__device__ __forceinline__ double Fma(double x, double y, double z) {
  return fma(x, y, z);
}

template <typename T, typename MT, int VecSize, bool IsAmp = false>
__device__ inline void VectorizeLarsUpdate(
    const T* __restrict__ grad, const MT* __restrict__ param,
    const MT* __restrict__ velocity, T* __restrict__ param_out,
    MT* __restrict__ velocity_out, const MT mu, MT local_lr,
    const MT lars_weight_decay, const MT rescale_grad, const int tid,
    const int grid_stride, const int numel,
    MT* __restrict__ master_param_out = nullptr) {
  using VecType = paddle::platform::AlignedVector<T, VecSize>;
  using VecMType = paddle::platform::AlignedVector<MT, VecSize>;
  int main = numel >> (VecSize >> 1);
  int tail_offset = main * VecSize;

  const VecType* __restrict__ grad_vec = reinterpret_cast<const VecType*>(grad);
  const VecMType* __restrict__ param_vec =
      reinterpret_cast<const VecMType*>(param);
  const VecMType* __restrict__ velocity_vec =
      reinterpret_cast<const VecMType*>(velocity);
  VecType* param_out_vec = reinterpret_cast<VecType*>(param_out);
  VecMType* velocity_out_vec = reinterpret_cast<VecMType*>(velocity_out);

  VecMType* master_param_out_vec;
  if (IsAmp) {
    master_param_out_vec = reinterpret_cast<VecMType*>(master_param_out);
  }

  for (int i = tid; i < main; i += grid_stride) {
    VecType param_out_tmp;
    VecMType velocity_tmp, param_tmp;
    VecType grad_data = grad_vec[i];
    VecMType param_data = param_vec[i];
    VecMType velocity_data = velocity_vec[i];

#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      MT grad_val = static_cast<MT>(grad_data[j]) * rescale_grad;
      velocity_tmp[j] =
          Fma(velocity_data[j], mu,
              local_lr * Fma(lars_weight_decay, param_data[j], grad_val));
      param_tmp[j] = param_data[j] - velocity_tmp[j];
      param_out_tmp[j] = static_cast<T>(param_tmp[j]);
    }
    param_out_vec[i] = param_out_tmp;
    velocity_out_vec[i] = velocity_tmp;
    if (IsAmp) {
      master_param_out_vec[i] = param_tmp;
    }
  }

  for (int i = tid + tail_offset; i < numel; i += grid_stride) {
    MT grad_val = static_cast<MT>(grad[i]) * rescale_grad;
    MT param_val = param[i];
    MT velocity_tmp = Fma(velocity[i], mu, local_lr * Fma(lars_weight_decay,
                                                          param_val, grad_val));
    MT param_tmp = param_val - velocity_tmp;
    param_out[i] = static_cast<T>(param_tmp);
    velocity_out[i] = velocity_tmp;
    if (IsAmp) {
      master_param_out[i] = param_tmp;
    }
  }
}

template <typename T, typename MT>
LARS_FUNCTION_FLAG void L2NormKernel(
    const T* __restrict__ p_data, const T* __restrict__ g_data,
    MT* __restrict__ p_buffer, MT* __restrict__ g_buffer, MT s_buffer[],
    const int64_t numel, const int repeat_times, const int thresh,
    const MT rescale_grad, MT* __restrict__ p_n = nullptr,
    MT* __restrict__ g_n = nullptr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_stride = LARS_BLOCK_SIZE * gridDim.x;
  const MT rescale_grad_pow = rescale_grad * rescale_grad;
  MT p_arr_val = static_cast<MT>(0);
  MT g_arr_val = static_cast<MT>(0);

  if (repeat_times == 0) {
    if (tid < numel) {
      p_arr_val = static_cast<MT>(p_data[tid]);
      g_arr_val = static_cast<MT>(g_data[tid]);
    }
    s_buffer[0] += math::blockReduceSum<MT>(p_arr_val * p_arr_val, FINAL_MASK);
    s_buffer[1] += math::blockReduceSum<MT>(g_arr_val * g_arr_val, FINAL_MASK);
  } else {
    /* To avoid occupy too much temp buffer. Hence, slice the whole data into 2
    parts, the front of them whose quantity is excatly multiple of grid-thread
    number, and this part of data is delt in for loop, the rest of data is delt
    with another step to avoid visiting data address beyond bound. */
    for (int i = 0; i < repeat_times; ++i) {
      p_arr_val = static_cast<MT>(p_data[tid]);
      g_arr_val = static_cast<MT>(g_data[tid]);
      tid += grid_stride;
      s_buffer[0] +=
          math::blockReduceSum<MT>(p_arr_val * p_arr_val, FINAL_MASK);
      s_buffer[1] +=
          math::blockReduceSum<MT>(g_arr_val * g_arr_val, FINAL_MASK);
      __syncthreads();
    }
    MT p_val = 0;
    MT g_val = 0;
    if (tid < numel) {
      p_val = static_cast<MT>(p_data[tid]);
      g_val = static_cast<MT>(g_data[tid]);
    }
    s_buffer[0] += math::blockReduceSum<MT>(p_val * p_val, FINAL_MASK);
    s_buffer[1] += math::blockReduceSum<MT>(g_val * g_val, FINAL_MASK);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    p_buffer[blockIdx.x] = s_buffer[0];
    g_buffer[blockIdx.x] = s_buffer[1];
  }
  // Grid sync for completely writring partial result back to gloabl memory
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  cg.sync();
  MT p_partial_sum = threadIdx.x < thresh ? p_buffer[threadIdx.x] : 0;
  MT g_partial_sum = threadIdx.x < thresh ? g_buffer[threadIdx.x] : 0;
  *p_n = sqrt(math::blockReduceSum<MT>(p_partial_sum, FINAL_MASK));
  *g_n = sqrt(rescale_grad_pow *
              math::blockReduceSum<MT>(g_partial_sum, FINAL_MASK));
}

template <typename T, typename MT>
__global__ void MergedMomentumLarsKernel(
    MergedParameter<T, MT>* merged_param, MT* __restrict__ p_buffer,
    MT* __restrict__ g_buffer, const int op_num, const MT mu,
    const MT lars_coeff, const MT lars_weight_decay, const MT epsilon,
    const MT rescale_grad) {
  __shared__ MT s_buffer[2];
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < op_num; ++i) {
    int numel = merged_param->numel_arr[i];
    MT p_n = static_cast<MT>(0);
    MT g_n = static_cast<MT>(0);
    s_buffer[0] = static_cast<MT>(0);
    s_buffer[1] = static_cast<MT>(0);
    L2NormKernel<T, MT>(merged_param->p_arr[i], merged_param->g_arr[i],
                        p_buffer, g_buffer, s_buffer, numel,
                        merged_param->repeat_arr[i],
                        merged_param->thresh_arr[i], rescale_grad, &p_n, &g_n);
    const MT lr = *(merged_param->lr_arr[i]);
    MT local_lr = lr;
    if (lars_weight_decay > static_cast<MT>(0)) {
      local_lr =
          lr * lars_coeff * p_n / (fma(lars_weight_decay, p_n, g_n) + epsilon);
    }

    if (std::is_same<T, paddle::platform::float16>::value) {
      VectorizeLarsUpdate<T, MT, 4, true>(
          merged_param->g_arr[i], merged_param->master_p_arr[i],
          merged_param->v_arr[i], merged_param->p_out_arr[i],
          merged_param->v_out_arr[i], mu, local_lr, lars_weight_decay,
          rescale_grad, tid, grid_stride, numel,
          merged_param->master_p_out_arr[i]);
    } else if (std::is_same<T, float>::value) {
      VectorizeLarsUpdate<T, MT, 4, false>(
          merged_param->g_arr[i],
          reinterpret_cast<const MT*>(merged_param->p_arr[i]),
          merged_param->v_arr[i], merged_param->p_out_arr[i],
          merged_param->v_out_arr[i], mu, local_lr, lars_weight_decay,
          rescale_grad, tid, grid_stride, numel);
    } else {
      VectorizeLarsUpdate<T, MT, 2, false>(
          merged_param->g_arr[i],
          reinterpret_cast<const MT*>(merged_param->p_arr[i]),
          merged_param->v_arr[i], merged_param->p_out_arr[i],
          merged_param->v_out_arr[i], mu, local_lr, lars_weight_decay,
          rescale_grad, tid, grid_stride, numel);
    }
  }
}

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* __restrict__ param, const T* __restrict__ grad,
    const MT* __restrict__ velocity, T* param_out, MT* velocity_out,
    const MT* __restrict__ master_param, MT* __restrict__ master_param_out,
    const MT* __restrict__ learning_rate, MT* __restrict__ p_buffer,
    MT* __restrict__ g_buffer, const MT mu, const MT lars_coeff,
    const MT lars_weight_decay, const MT epsilon, const MT rescale_grad,
    const int repeat_times, const int thresh, const int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
  MT param_norm = static_cast<MT>(0);
  MT grad_norm = static_cast<MT>(0);
  __shared__ MT s_buffer[2];
  s_buffer[0] = static_cast<MT>(0);
  s_buffer[1] = static_cast<MT>(0);
  L2NormKernel<T, MT>(param, grad, p_buffer, g_buffer, s_buffer, numel,
                      repeat_times, gridDim.x, rescale_grad, &param_norm,
                      &grad_norm);

  const MT lr = learning_rate[0];
  MT local_lr = lr;
  if (lars_weight_decay > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * param_norm /
               (fma(lars_weight_decay, param_norm, grad_norm) + epsilon);
  }

  if (master_param_out) {
    VectorizeLarsUpdate<T, MT, 4, true>(grad, master_param, velocity, param_out,
                                        velocity_out, mu, local_lr,
                                        lars_weight_decay, rescale_grad, tid,
                                        grid_stride, numel, master_param_out);
  } else {
    if (std::is_same<T, float>::value ||
        std::is_same<T, paddle::platform::float16>::value) {
      // As for multiple-precision, type T and MT cannot be more than fp16 or
      // fp32, Then, the maximum data IO size could be set to 4.
      VectorizeLarsUpdate<T, MT, 4, false>(
          grad, reinterpret_cast<const MT*>(param), velocity, param_out,
          velocity_out, mu, local_lr, lars_weight_decay, rescale_grad, tid,
          grid_stride, numel);
    } else {
      VectorizeLarsUpdate<T, MT, 2, false>(
          grad, reinterpret_cast<const MT*>(param), velocity, param_out,
          velocity_out, mu, local_lr, lars_weight_decay, rescale_grad, tid,
          grid_stride, numel);
    }
  }
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MT = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool multi_precision = ctx.Attr<bool>("multi_precision");
    const bool merge_operation = ctx.Attr<bool>("merge_operation");
    bool has_master = false;
    int num_blocks_per_sm = 0;
    auto& cuda_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int sm_num = cuda_ctx.GetSMCount();
    framework::Tensor tmp_buffer_t =
        ctx.AllocateTmpTensor<MT, platform::CUDADeviceContext>(
            {LARS_BLOCK_SIZE << 1}, cuda_ctx);
    auto* p_buffer = tmp_buffer_t.mutable_data<MT>(ctx.GetPlace());
    auto* g_buffer = p_buffer + LARS_BLOCK_SIZE;

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT lars_coeff = static_cast<MT>(ctx.Attr<float>("lars_coeff"));
    MT lars_weight_decay =
        static_cast<MT>(ctx.Attr<float>("lars_weight_decay"));
    MT epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));

    if (merge_operation) {
      auto param = ctx.MultiInput<framework::LoDTensor>("Param");
      auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
      auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
      auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
      auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
      auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, MergedMomentumLarsKernel<T, MT>, LARS_BLOCK_SIZE,
          sizeof(MT) << 1);

      MergedParameter<T, MT> merged_params;
      int total_numel = 0;
      size_t op_num = grad.size();
      PADDLE_ENFORCE_LT(
          op_num, MAX_MERGED_OPS,
          platform::errors::InvalidArgument(
              "Currently, the maximum quantity of merged-op supported is (%d), "
              "but lars op required for trainning this model is (%d)\n",
              MAX_MERGED_OPS, op_num));

      for (int i = 0; i < op_num; ++i) {
        int temp_numel = param[i]->numel();
        // max_numel = max_numel < temp_numel ? temp_numel : max_numel;
        total_numel += temp_numel;
        merged_params.numel_arr[i] = temp_numel;
        merged_params.p_arr[i] = param[i]->data<T>();
        merged_params.g_arr[i] = grad[i]->data<T>();
        merged_params.v_arr[i] = velocity[i]->data<MT>();
        merged_params.lr_arr[i] = learning_rate[i]->data<MT>();
        merged_params.p_out_arr[i] =
            param_out[i]->mutable_data<T>(ctx.GetPlace());
        merged_params.v_out_arr[i] =
            velocity_out[i]->mutable_data<MT>(ctx.GetPlace());
      }
      int avg_numel = total_numel / op_num;
      int grid_num = (avg_numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
      int thread_num = std::min(sm_num * num_blocks_per_sm, LARS_BLOCK_SIZE);
      int grid_real = std::min(thread_num, grid_num);
      int grid_stride = LARS_BLOCK_SIZE * grid_real;

      for (int i = 0; i < op_num; ++i) {
        grid_num = (merged_params.numel_arr[i] + LARS_BLOCK_SIZE - 1) /
                   LARS_BLOCK_SIZE;
        merged_params.repeat_arr[i] =
            (merged_params.numel_arr[i] + grid_stride - 1) / grid_stride - 1;
        merged_params.thresh_arr[i] = std::min(thread_num, grid_num);
      }
      if (multi_precision) {
        auto master_param = ctx.MultiInput<framework::LoDTensor>("MasterParam");
        auto master_param_out =
            ctx.MultiOutput<framework::LoDTensor>("MasterParamOut");
        for (int i = 0; i < op_num; ++i) {
          merged_params.master_p_arr[i] = master_param[i]->data<MT>();
          merged_params.master_p_out_arr[i] =
              master_param_out[i]->mutable_data<MT>(ctx.GetPlace());
        }
      }
      auto merged_buf = memory::Alloc(cuda_ctx, sizeof(merged_params));
      auto* merged_ptr =
          reinterpret_cast<MergedParameter<T, MT>*>(merged_buf->ptr());
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, cuda_ctx.GetPlace()),
                   reinterpret_cast<void*>(merged_ptr), platform::CPUPlace(),
                   reinterpret_cast<void*>(&merged_params),
                   sizeof(merged_params), cuda_ctx.stream());
      void* cuda_param[] = {reinterpret_cast<void*>(&merged_ptr),
                            reinterpret_cast<void*>(&p_buffer),
                            reinterpret_cast<void*>(&g_buffer),
                            reinterpret_cast<void*>(&op_num),
                            reinterpret_cast<void*>(&mu),
                            reinterpret_cast<void*>(&lars_coeff),
                            reinterpret_cast<void*>(&lars_weight_decay),
                            reinterpret_cast<void*>(&epsilon),
                            reinterpret_cast<void*>(&rescale_grad)};
      // Lanuch all sm theads.
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(MergedMomentumLarsKernel<T, MT>), grid_real,
          LARS_BLOCK_SIZE, cuda_param, 0, cuda_ctx.stream());
    } else {
      auto param = ctx.MultiInput<framework::LoDTensor>("Param");
      auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
      auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
      auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
      auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
      auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");

      auto* param_data = param[0]->data<T>();
      auto* grad_data = grad[0]->data<T>();
      auto* velocity_data = velocity[0]->data<MT>();
      auto* lr = learning_rate[0]->data<MT>();
      auto* param_out_data = param_out[0]->mutable_data<T>(ctx.GetPlace());
      auto* velocity_out_data =
          velocity_out[0]->mutable_data<MT>(ctx.GetPlace());
      const MT* master_param_data = nullptr;
      MT* master_param_out_data = nullptr;
      if (multi_precision) {
        auto master_param = ctx.MultiInput<framework::Tensor>("MasterParam");
        auto master_param_out =
            ctx.MultiOutput<framework::Tensor>("MasterParamOut");
        master_param_data = master_param[0]->data<MT>();
        master_param_out_data =
            master_param_out[0]->mutable_data<MT>(ctx.GetPlace());
      }
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, MomentumLarsKernel<T, MT>, LARS_BLOCK_SIZE,
          sizeof(MT) << 1);
      int64_t numel = param[0]->numel();
      int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
      int grid_real =
          std::min(std::min(sm_num * num_blocks_per_sm, grid), LARS_BLOCK_SIZE);
      int grid_stride = LARS_BLOCK_SIZE * grid_real;
      int repeat_times = (numel + grid_stride - 1) / grid_stride - 1;
      int thresh = 0;

      // Uniform kernel parameter for cudaLaunchCooperativeKernel
      void* cuda_param[] = {
          reinterpret_cast<void*>(&param_data),
          reinterpret_cast<void*>(&grad_data),
          reinterpret_cast<void*>(&velocity_data),
          reinterpret_cast<void*>(&param_out_data),
          reinterpret_cast<void*>(&velocity_out_data),
          reinterpret_cast<void*>(&master_param_data),
          reinterpret_cast<void*>(&master_param_out_data),
          reinterpret_cast<void*>(&lr),
          reinterpret_cast<void*>(&p_buffer),
          reinterpret_cast<void*>(&g_buffer),
          reinterpret_cast<void*>(&mu),
          reinterpret_cast<void*>(&lars_coeff),
          reinterpret_cast<void*>(&lars_weight_decay),
          reinterpret_cast<void*>(&epsilon),
          reinterpret_cast<void*>(&rescale_grad),
          reinterpret_cast<void*>(&repeat_times),
          reinterpret_cast<void*>(&thresh),  // Just a placeholder
          reinterpret_cast<void*>(&numel)};
      // Lanuch all sm theads.
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(MomentumLarsKernel<T, MT>), grid_real,
          LARS_BLOCK_SIZE, cuda_param, 0, cuda_ctx.stream());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    lars_momentum,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>);
