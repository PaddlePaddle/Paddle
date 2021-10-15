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

#if CUDA_VERSION >= 11000
#include <cooperative_groups.h>
#endif

#ifdef __HIPCC__
#define LARS_BLOCK_SIZE 256
#else
#define LARS_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

__device__ __forceinline__ float Sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ double Sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ float Fma(float x, float y, float z) {
  return fmaf(x, y, z);
}
__device__ __forceinline__ double Fma(double x, double y, double z) {
  return fma(x, y, z);
}

template <typename T>
class LarsThreadConfig {
 public:
  int grid_for_norm;
  int grid_for_lars;
#if CUDA_VERSION >= 11000

 private:
  int grid_stride;

 public:
  explicit LarsThreadConfig(int64_t numel, int sm_num, int num_blocks_per_sm) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_lars =
        std::min(std::min(sm_num * num_blocks_per_sm, grid), LARS_BLOCK_SIZE);
    grid_stride = LARS_BLOCK_SIZE * grid_for_lars;
  }

  int GetRepeatTimes(int64_t numel) {
    return (numel + grid_stride - 1) / grid_stride - 1;
  }
#else
  int repeat_times;
  explicit LarsThreadConfig(const int64_t numel) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_norm = std::min(grid, LARS_BLOCK_SIZE);
    const int grid_stride = grid_for_norm * LARS_BLOCK_SIZE;
    repeat_times = (numel + grid_stride - 1) / grid_stride - 1;
    // Determine to read 4 fp16 or float data once, but 2 double data once.
    grid_for_lars =
        std::is_same<double, T>::value
            ? (numel + (LARS_BLOCK_SIZE << 1) - 1) / (LARS_BLOCK_SIZE << 1)
            : (numel + (LARS_BLOCK_SIZE << 2) - 1) / (LARS_BLOCK_SIZE << 2);
  }
#endif
};

template <typename T, typename MT, int VecSize, bool IsAmp = false>
__device__ inline void VectorizeLarsUpdate(
    const T* __restrict__ grad, MT* param, MT* velocity, const MT mu,
    MT local_lr, const MT lars_weight_decay, const MT rescale_grad,
    const int tid, const int grid_stride, const int numel) {
  using VecType = paddle::platform::AlignedVector<T, VecSize>;
  using VecMType = paddle::platform::AlignedVector<MT, VecSize>;
  int main = numel >> (VecSize >> 1);
  int tail_offset = main * VecSize;

  const VecType* __restrict__ grad_vec = reinterpret_cast<const VecType*>(grad);
  VecMType* velocity_vec = reinterpret_cast<VecMType*>(velocity);
  VecMType* param_vec = reinterpret_cast<VecMType*>(param);

  for (int i = tid; i < main; i += grid_stride) {
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
    }
    param_vec[i] = param_tmp;
    velocity_vec[i] = velocity_tmp;
  }

  for (int i = tid + tail_offset; i < numel; i += grid_stride) {
    MT grad_val = static_cast<MT>(grad[i]) * rescale_grad;
    MT param_val = param[i];
    MT velocity_tmp = Fma(velocity[i], mu, local_lr * Fma(lars_weight_decay,
                                                          param_val, grad_val));
    MT param_tmp = param_val - velocity_tmp;
    param[i] = param_tmp;
    velocity[i] = velocity_tmp;
  }
}

#if CUDA_VERSION >= 11000
/* Once CUDA_VERSION is beyond 11, cooperative_groups can be involved in without
  --rdc=true compile flag, then L2_norm kernel can be set with __device__ and
  cooperative_groups::grid_group also can be involved. Otherwise, adding this
  flag may affect much, L2_norm kernel shall be set with __global__.*/
// TODO(limingshu): declaration of cooperative_groups wapper is invalid in host.
template <typename T, typename MT>
__forceinline__ __device__ void L2NormKernel(
    const cooperative_groups::grid_group* cg,
#else
template <typename T, typename MT>
__global__ void L2NormKernel(
#endif
    T* p_data, const T* __restrict__ g_data, MT* p_buffer, MT* g_buffer,
    const int64_t numel, const int repeat_times, const MT rescale_grad,
    const int thresh = 0, MT* p_n = nullptr, MT* g_n = nullptr) {
  __shared__ MT s_buffer[2];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_stride = LARS_BLOCK_SIZE * gridDim.x;
  const MT rescale_pow = rescale_grad * rescale_grad;
  if (threadIdx.x == 0) {
    s_buffer[0] = static_cast<MT>(0);
    s_buffer[1] = static_cast<MT>(0);
  }
  MT p_tmp = static_cast<MT>(0);
  MT g_tmp = static_cast<MT>(0);

  if (repeat_times == 0) {
    if (tid < numel) {
      p_tmp = static_cast<MT>(p_data[tid]);
      g_tmp = static_cast<MT>(g_data[tid]);
    }
    MT tmp0 = math::blockReduceSum<MT>(p_tmp * p_tmp, FINAL_MASK);
    MT tmp1 = math::blockReduceSum<MT>(g_tmp * g_tmp, FINAL_MASK);
    if (threadIdx.x == 0) {
      s_buffer[0] += tmp0;
      s_buffer[1] += tmp1;
    }
  } else {
    /* Avoid occupy too much temp buffer. Slice the whole data into 2 parts,
    the front of data whose quantity is excatly multiple of grid-thread
    number, and delt in for loop, the rest is delt with another step. */
    for (int i = 0; i < repeat_times; ++i) {
      p_tmp = static_cast<MT>(p_data[tid]);
      g_tmp = static_cast<MT>(g_data[tid]);
      tid += grid_stride;
      MT tmp0 = math::blockReduceSum<MT>(p_tmp * p_tmp, FINAL_MASK);
      MT tmp1 = math::blockReduceSum<MT>(g_tmp * g_tmp, FINAL_MASK);
      if (threadIdx.x == 0) {
        s_buffer[0] += tmp0;
        s_buffer[1] += tmp1;
      }
      __syncthreads();
    }
    MT p_val = 0;
    MT g_val = 0;
    if (tid < numel) {
      p_val = static_cast<MT>(p_data[tid]);
      g_val = static_cast<MT>(g_data[tid]);
    }
    MT tmp0 = math::blockReduceSum<MT>(p_val * p_val, FINAL_MASK);
    MT tmp1 = math::blockReduceSum<MT>(g_val * g_val, FINAL_MASK);
    if (threadIdx.x == 0) {
      s_buffer[0] += tmp0;
      s_buffer[1] += tmp1;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    p_buffer[blockIdx.x] = s_buffer[0];
    g_buffer[blockIdx.x] = s_buffer[1];
  }
#if CUDA_VERSION >= 11000
  cg->sync();  // Grid sync for writring partial result to gloabl memory
  MT p_part_sum = threadIdx.x < gridDim.x ? p_buffer[threadIdx.x] : 0;
  MT g_part_sum = threadIdx.x < gridDim.x ? g_buffer[threadIdx.x] : 0;
  MT tmp0 = math::blockReduceSum<MT>(p_part_sum, FINAL_MASK);
  MT tmp1 = math::blockReduceSum<MT>(g_part_sum, FINAL_MASK);
  if (threadIdx.x == 0) {
    s_buffer[0] = tmp0;
    s_buffer[1] = tmp1;
  }
  __syncthreads();
  *p_n = Sqrt(s_buffer[0]);
  *g_n = Sqrt(rescale_pow * s_buffer[1]);
#endif
}

template <typename T, typename MT>
__forceinline__ __device__ void MomentumUpdate(
    T* param, const T* __restrict__ grad, MT* velocity, MT* master_param,
    const MT* __restrict__ learn_rate, const MT mu, const MT lars_weight_decay,
    const MT lars_coeff, const MT epsilon, const MT rescale_grad,
    const MT param_norm, const MT grad_norm, const int tid,
    const int grid_stride, const int64_t numel, const bool is_amp) {
  const MT lr = learn_rate[0];
  MT local_lr = lr;
  if (lars_weight_decay > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * param_norm /
               (fma(lars_weight_decay, param_norm, grad_norm) + epsilon);
  }
  if (is_amp) {
    VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/true>(
        grad, master_param, velocity, mu, local_lr, lars_weight_decay,
        rescale_grad, tid, grid_stride, numel);
  } else {
    if (std::is_same<T, float>::value ||
        std::is_same<T, paddle::platform::float16>::value) {
      /* TODO(limingshu): pointer cast may damage memory accessing for fp16 */
      VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/false>(
          grad, reinterpret_cast<MT*>(param), velocity, mu, local_lr,
          lars_weight_decay, rescale_grad, tid, grid_stride, numel);
    } else {
      VectorizeLarsUpdate<T, MT, /*VecSize=*/2, /*IsAmp=*/false>(
          grad, reinterpret_cast<MT*>(param), velocity, mu, local_lr,
          lars_weight_decay, rescale_grad, tid, grid_stride, numel);
    }
  }
}

#if CUDA_VERSION >= 11000
template <typename MT, int kOpNum, typename T>
struct MergedLarsMasterParam {
  DEVICE inline MT* GetMasterParam(size_t) const { return nullptr; }
  constexpr void SetMasterParam(size_t, MT*) {}
};

template <typename MT, int kOpNum>
struct MergedLarsMasterParam<MT, kOpNum, paddle::platform::float16> {
  MT* master_params[kOpNum];

  DEVICE inline MT* GetMasterParam(size_t idx) const {
    return master_params[idx];
  }
  void SetMasterParam(size_t idx, MT* p) { master_params[idx] = p; }
};

template <typename T, typename MT,
          int kOpNum =
              std::is_same<T, paddle::platform::float16>::value ? 80 : 90>
struct LarsParamWarpper : public MergedLarsMasterParam<MT, kOpNum, T> {
  static constexpr int kNum = kOpNum;

  int numel_arr[kOpNum];
  int repeat_arr[kOpNum];
  const T* __restrict__ g_arr[kOpNum];
  T* p_arr[kOpNum];
  MT* v_arr[kOpNum];
  MT weight_decay_arr[kOpNum];
};

template <typename T, typename MT, typename LarsWarpperType>
__global__ void MergedMomentumLarsKernel(
    LarsWarpperType lars_warpper, MT* p_buffer, MT* g_buffer, const MT* lr,
    const int op_num, const MT mu, const MT lars_coeff, const MT epsilon,
    const MT rescale_grad, const bool is_amp) {
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();

  for (int i = 0; i < lars_warpper.kNum; ++i) {
    if (i > op_num) break;
    int numel = lars_warpper.numel_arr[i];
    MT param_norm = static_cast<MT>(0);
    MT grad_norm = static_cast<MT>(0);
    L2NormKernel<T, MT>(&cg, lars_warpper.p_arr[i], lars_warpper.g_arr[i],
                        p_buffer, g_buffer, numel, lars_warpper.repeat_arr[i],
                        rescale_grad, 0, &param_norm, &grad_norm);
    MomentumUpdate<T, MT>(lars_warpper.p_arr[i], lars_warpper.g_arr[i],
                          lars_warpper.v_arr[i], lars_warpper.GetMasterParam(i),
                          lr, mu, lars_warpper.weight_decay_arr[i], lars_coeff,
                          epsilon, rescale_grad, param_norm, grad_norm, tid,
                          grid_stride, numel, is_amp);
  }
}
#endif

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    T* param, const T* __restrict__ grad, MT* velocity, MT* master_param,
    const MT* __restrict__ learn_rate, MT* p_buffer, MT* g_buffer, const MT mu,
    const MT lars_coeff, const MT lars_weight_decay, const MT epsilon,
    const MT rescale_grad, const int repeat_times, int thresh,
    const int64_t numel, const bool is_amp) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
#if CUDA_VERSION >= 11000
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  MT param_norm = static_cast<MT>(0);
  MT grad_norm = static_cast<MT>(0);
  L2NormKernel<T, MT>(&cg, param, grad, p_buffer, g_buffer, numel, repeat_times,
                      rescale_grad, gridDim.x, &param_norm, &grad_norm);
#else
  __shared__ MT s_buffer[2];
  const MT rescale_pow = rescale_grad * rescale_grad;
  MT param_part_norm = threadIdx.x < thresh ? p_buffer[threadIdx.x] : 0;
  MT grad_part_norm = threadIdx.x < thresh ? g_buffer[threadIdx.x] : 0;
  MT tmp0 = math::blockReduceSum<MT>(param_part_norm, FINAL_MASK);
  MT tmp1 = math::blockReduceSum<MT>(param_part_norm, FINAL_MASK);
  if (threadIdx.x == 0) {
    s_buffer[0] = tmp0;
    s_buffer[1] = tmp1;
  }
  __syncthreads();
  MT param_norm = Sqrt(s_buffer[0]);
  MT grad_norm = Sqrt(rescale_pow * s_buffer[1]);
#endif
  MomentumUpdate<T, MT>(param, grad, velocity, master_param, learn_rate, mu,
                        lars_weight_decay, lars_coeff, epsilon, rescale_grad,
                        param_norm, grad_norm, tid, grid_stride, numel, is_amp);
}

template <typename T, typename MT>
inline void SeparatedLarsMomentumOpCUDAKernel(
    const platform::CUDADeviceContext& cuda_ctx, T* param_data,
    MT* velocity_data, const T* grad_data, MT* master_param_data, const MT* lr,
    MT* p_buffer, MT* g_buffer, const MT mu, const MT lars_coeff,
    const MT weight_decay, const MT epsilon, const MT rescale_grad,
    const int64_t numel, const bool is_amp) {
  LarsThreadConfig<T> lars_thread_config(numel);
  L2NormKernel<T, MT><<<lars_thread_config.grid_for_norm, LARS_BLOCK_SIZE, 0,
                        cuda_ctx.stream()>>>(
      param_data, grad_data, p_buffer, g_buffer, numel,
      lars_thread_config.repeat_times, rescale_grad);

  MomentumLarsKernel<T, MT><<<lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE,
                              0, cuda_ctx.stream()>>>(
      param_data, grad_data, velocity_data, master_param_data, lr, p_buffer,
      g_buffer, mu, lars_coeff, weight_decay, epsilon, rescale_grad, 0,
      lars_thread_config.grid_for_norm, numel, is_amp);
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MT = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int num_blocks_per_sm = 0;
    auto& cuda_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int sm_num = cuda_ctx.GetSMCount();
    framework::Tensor tmp_buffer_t =
        ctx.AllocateTmpTensor<MT, platform::CUDADeviceContext>(
            {LARS_BLOCK_SIZE << 1}, cuda_ctx);
    auto* p_buffer = tmp_buffer_t.mutable_data<MT>(ctx.GetPlace());
    auto* g_buffer = p_buffer + LARS_BLOCK_SIZE;
    bool multi_precision = ctx.Attr<bool>("multi_precision");
    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT lars_coeff = static_cast<MT>(ctx.Attr<float>("lars_coeff"));
    MT epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    bool merge_option = ctx.Attr<bool>("merge_option");

    auto weight_decay_arr = ctx.Attr<std::vector<float>>("lars_weight_decay");
    auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
    auto param = ctx.MultiInput<framework::LoDTensor>("Param");
    auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
    auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
    auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
    auto master_param = ctx.MultiInput<framework::LoDTensor>("MasterParam");
    auto master_param_out =
        ctx.MultiOutput<framework::LoDTensor>("MasterParamOut");
    auto* lr = learning_rate[0]->data<MT>();

    int op_num = grad.size();
    for (size_t i = 0; i < op_num; ++i) {
      PADDLE_ENFORCE_EQ(
          param[i], param_out[i],
          platform::errors::InvalidArgument(
              "Input(Param) and Output(ParamOut) must be the same Tensors."));
      PADDLE_ENFORCE_EQ(velocity[i], velocity_out[i],
                        platform::errors::InvalidArgument(
                            "Input(Velocity) and Output(VelocityOut) must be "
                            "the same Tensors."));
      if (multi_precision) {
        PADDLE_ENFORCE_EQ(master_param[i], master_param_out[i],
                          platform::errors::InvalidArgument(
                              "Input(MasterParam) and Output(MasterParamOut) "
                              "must be the same Tensors."));
      }
    }
#if CUDA_VERSION >= 11000
    // if (op_num > 1) {
    if (merge_option) {
      LarsParamWarpper<T, MT> lars_warpper;
      /* Implementation of lars optimizer consists of following two steps:
        1. Figure out the L2 norm statistic result of grad data and param data.
        2. Update param and velocity with usage of L2 norm statistic result.
      Step1 and step2 can be merged with api provided by nvida
        cudaLaunchCooperativeKernel:
        - The thread quantity shall less than pyhsical SM limited threads
        - Launche as thread-block can synchronizlly execute. */
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm,
          MergedMomentumLarsKernel<T, MT, decltype(lars_warpper)>,
          LARS_BLOCK_SIZE, sizeof(MT) << 1);

      VLOG(10) << "Num of ops merged in lars_warpper is " << lars_warpper.kNum;

      int merge_times = (op_num + lars_warpper.kNum - 1) / lars_warpper.kNum;
      for (int j = 0; j < merge_times; ++j) {
        size_t total_numel = 0;
        int start_idx = j * lars_warpper.kNum;
        int loop_num = std::min(lars_warpper.kNum, op_num - start_idx);

        for (int i = 0; i < loop_num; ++i) {
          size_t temp_numel = param[start_idx + i]->numel();
          total_numel += temp_numel;
          lars_warpper.numel_arr[i] = temp_numel;
          lars_warpper.g_arr[i] = grad[start_idx + i]->data<T>();
          lars_warpper.p_arr[i] = param_out[start_idx + i]->data<T>();
          lars_warpper.v_arr[i] = velocity_out[start_idx + i]->data<MT>();
          lars_warpper.weight_decay_arr[i] =
              static_cast<MT>(weight_decay_arr[start_idx + i]);
        }
        int64_t avg_numel = total_numel / loop_num;
        LarsThreadConfig<float> lars_thread_config(avg_numel, sm_num,
                                                   num_blocks_per_sm);
        for (int i = 0; i < loop_num; ++i) {
          lars_warpper.repeat_arr[i] =
              lars_thread_config.GetRepeatTimes(lars_warpper.numel_arr[i]);
          if (multi_precision) {
            lars_warpper.SetMasterParam(
                i, master_param_out[i]->mutable_data<MT>(ctx.GetPlace()));
          }
        }
        void* cuda_param[] = {reinterpret_cast<void*>(&lars_warpper),
                              reinterpret_cast<void*>(&p_buffer),
                              reinterpret_cast<void*>(&g_buffer),
                              reinterpret_cast<void*>(&lr),
                              reinterpret_cast<void*>(&loop_num),
                              reinterpret_cast<void*>(&mu),
                              reinterpret_cast<void*>(&lars_coeff),
                              reinterpret_cast<void*>(&epsilon),
                              reinterpret_cast<void*>(&rescale_grad),
                              reinterpret_cast<void*>(&multi_precision)};
        cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(
                MergedMomentumLarsKernel<T, MT, decltype(lars_warpper)>),
            lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE, cuda_param, 0,
            cuda_ctx.stream());

        VLOG(10) << "Lanuched ops number is " << loop_num;
      }
    } else {
      auto* grad_data = grad[0]->data<T>();
      auto* param_data = param_out[0]->data<T>();
      auto* velocity_data = velocity_out[0]->data<MT>();
      auto* lr = learning_rate[0]->data<MT>();
      const MT* master_param_data =
          multi_precision ? master_param_out[0]->data<MT>() : nullptr;
      int64_t numel = param[0]->numel();
      MT lars_weight_decay = weight_decay_arr[0];

      // Figure out how many blocks can be active in each sm.
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, MomentumLarsKernel<T, MT>, LARS_BLOCK_SIZE,
          sizeof(MT) << 1);
      LarsThreadConfig<float> lars_thread_config(numel, sm_num,
                                                 num_blocks_per_sm);
      int repeat_times = lars_thread_config.GetRepeatTimes(numel);
      int thresh = 0;

      void* cuda_param[] = {
          reinterpret_cast<void*>(&param_data),
          reinterpret_cast<void*>(&grad_data),
          reinterpret_cast<void*>(&velocity_data),
          reinterpret_cast<void*>(&master_param_data),
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
          reinterpret_cast<void*>(&numel),
          reinterpret_cast<void*>(&multi_precision)};
      // Lanuch all sm theads.
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(MomentumLarsKernel<T, MT>),
          lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE, cuda_param, 0,
          cuda_ctx.stream());
    }
#else
    for (int i = 0; i < op_num; ++i) {
      MT* master_param_data =
          multi_precision ? master_param_out[i]->data<MT>() : nullptr;
      SeparatedLarsMomentumOpCUDAKernel<T, MT>(
          cuda_ctx, param_out[i]->data<T>(), velocity_out[i]->data<MT>(),
          grad[i]->data<T>(), master_param_data, lr, p_buffer, g_buffer, mu,
          lars_coeff, weight_decay_arr[i], epsilon, rescale_grad,
          param[i]->numel(), multi_precision);
    }
#endif
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
