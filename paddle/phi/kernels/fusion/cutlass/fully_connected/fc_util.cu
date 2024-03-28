#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_util.h"
#include <iostream>
#include <cmath>

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename T>
float diff(const T *C_cutlass, const T *C_naive, int n) {
  float max_diff = -1.;
  for (int i = 0; i < n; i++) {
    float cutlass_value = static_cast<float>(C_cutlass[i]);
    float naive_value = static_cast<float>(C_naive[i]);
    // std::cout << "cutlass-naive: " << cutlass_value << "-" << naive_value << "    ";
    if (std::abs(naive_value - cutlass_value) > max_diff) {
      max_diff = std::abs(naive_value - cutlass_value);
    }
  }
  return max_diff;
}

__device__ inline float tanh_kai(float x){
  if(x > 0)
      return (1-exp(-2*x))/(1+exp(-2*x));
  else
      return (exp(2*x)-1)/(1+exp(2*x));
}

// 暂时假设输入都是行主序的。
template <typename T = half>
__global__ void naive_fc_kernel(
    const T *input,
    const T *weight,
    const T *bias,
    T *output,
    int M, int N, int K,
    int lda, int ldb, int ldd,
    float leaky_alpha,
    bool vecBias,
    OpType op_type
){
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0.;
    for (int k = 0; k < K; ++k) {
      float input_ele = static_cast<float>(input[i * lda + k]);
      float weight_ele = static_cast<float>(weight[k * ldb + j]);
      accumulator += input_ele * weight_ele;
    }
    if(vecBias){
      accumulator += static_cast<float>(bias[j]);
    }
    else{
      accumulator += static_cast<float>(bias[i * ldd + j]);
    }

    switch (op_type) {
        case FC_BIAS:
            break;
        case FC_BIAS_RELU:
            accumulator = accumulator > 0 ? accumulator : 0;
            break;
        case FC_BIAS_SILU:
            accumulator = accumulator * (1.f / (1 + exp(-accumulator)));
            break;
        case FC_BIAS_LEAKY_RELU:
          accumulator = accumulator > 0 ? accumulator : (accumulator * leaky_alpha);
          break;
        case FC_BIAS_SIGMOID:
          accumulator = 1.f / (1.f + std::exp(-accumulator));
          break;
        case FC_BIAS_GELU:
          accumulator = 0.5*accumulator*(1+tanh_kai(std::sqrt(2/M_PI)*(accumulator+0.044715*std::pow(accumulator,3))));
          break;
        default:
            break;
    }
    output[i*ldd+j] = (T)accumulator;
  }
}

template <typename T>
float fc_diff_gpu(const FcAllParams& params, OpType op_type){
    const T *input = (const T*)params.input;
    const T *weight = (const T*)params.weight;
    const T *bias = (const T*)params.bias;
    T *output_cutlass_D = (T*)params.output;
    int M = params.m, N = params.n, K = params.k;
    int lda = params.lda, ldb = params.ldb, ldd= params.ldd;
    float leaky_alpha = params.leaky_alpha;
    bool vecBias = params.vecBias;
    
    size_t outSize = sizeof(T) * M * N;
    T *output_naive_D;
    CUDA_CHECK(cudaMalloc((void**)&output_naive_D, outSize));
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_fc_kernel<T><<<grid, block>>>(input, weight, bias, output_naive_D,
                                        M, N, K,lda, ldb, ldd, leaky_alpha, vecBias, op_type);
    cudaGetLastError();
    CUDA_CHECK(cudaDeviceSynchronize());

    T *output_cutlass_H = reinterpret_cast<T*>(malloc(outSize));
    CUDA_CHECK(cudaMemcpy(output_cutlass_H, output_cutlass_D, outSize, cudaMemcpyDeviceToHost));
    T *output_naive_H = reinterpret_cast<T*>(malloc(outSize));
    CUDA_CHECK(cudaMemcpy(output_naive_H, output_naive_D, outSize, cudaMemcpyDeviceToHost));

    float max_diff = diff(output_cutlass_H, output_naive_H, M*N);

    free(output_cutlass_H);
    free(output_naive_H);
    cudaFree(output_naive_D);
    return max_diff;
}

std::string OpType2String(OpType op_type) {
  switch (op_type) {
    case FC_BIAS:
      return "fc_bias";
      break;
    case FC_BIAS_RELU:
      return "fc_bias_relu";
      break;
    case FC_BIAS_SILU:
      return "fc_bias_silu";
      break;
    case FC_BIAS_SIGMOID:
      return "fc_bias_sigmoid";
      break;
    case FC_BIAS_LEAKY_RELU:
      return "fc_bias_leaky_relu";
    case FC_BIAS_GELU:
      return "fc_bias_gelu";
    default:
      break;
  }
  return "unnamed_op";
}


int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(FcAllParams)>> &all_func,
    const FcAllParams &params,
    OpType op_type) {
    constexpr int WARMUP = 10;
    constexpr int REPEAT = 10;
    float min_time = 100000.f;
    int min_time_index = -1;
    for (int i = 0; i < all_func.size(); i++) {
      cutlass::Status status;
      auto func = all_func[i];
      // When func has large diff, we will make it nullptr.
      if (!func) continue;
      // 这里写死了sizeof(half) 注意！！
      CUDA_CHECK(cudaMemset(params.output,
                0,
                sizeof(half) * params.m * params.n));
      status = func(params);
      if (status != cutlass::Status::kSuccess) continue;

      for (int ii = 0; ii < WARMUP; ii++) {
        status = func(params);
      }

      cudaEvent_t beg, end;
      CUDA_CHECK(cudaEventCreate(&beg));
      CUDA_CHECK(cudaEventCreate(&end));
      CUDA_CHECK(cudaEventRecord(beg));
      for (int ii = 0; ii < REPEAT; ii++) {
        status = func(params);
      }

      CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaEventSynchronize(end));
      float elapsed_time;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, beg, end));
      if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
        min_time = elapsed_time;
        min_time_index = i;

        if (params.data_type == FcDataType::fp16) {
          // debug code
          std::cout << "fp16_"
                    << OpType2String(op_type) << ": tactic " << i
                    << " has max diff "
                    << fc_diff_gpu<half>(params, op_type)
                    << " compared with baseline,"
                    << "cost_time: " << elapsed_time << "ms." << std::endl;
        } 
        else if (params.data_type == FcDataType::bf16) {
          // debug code
          std::cout << "bf16_"
                    << OpType2String(op_type) << ": tactic " << i
                    << " has max diff "
                    << fc_diff_gpu<__nv_bfloat16>(params, op_type)
                    << " compared with baseline,"
                    << "cost_time: " << elapsed_time << "ms." << std::endl;
        }
        else if(params.data_type == FcDataType::fp32){
          // debug code
          std::cout << "fp32_"
                    << OpType2String(op_type) << ": tactic " << i
                    << " has max diff "
                    << fc_diff_gpu<float>(params, op_type)
                    << " compared with baseline,"
                    << "cost_time: " << elapsed_time << "ms." << std::endl;
        }
      }
    }

    if (min_time_index < 0) {
      std::cout << "Can't find any cutlass config for " << OpType2String(op_type)
                << std::endl;
    }
    return min_time_index;
}

template float fc_diff_gpu<float>(const FcAllParams& params, OpType op_type);
template float fc_diff_gpu<half>(const FcAllParams& params, OpType op_type);
template float fc_diff_gpu<__nv_bfloat16>(const FcAllParams& params, OpType op_type);

}
}
}