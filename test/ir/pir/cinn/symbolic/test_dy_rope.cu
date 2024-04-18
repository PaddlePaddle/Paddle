#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <cstdint>

#define CINN_WITH_CUDA

#include "/work/Paddle/paddle/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh"

extern "C" {

__global__
void __launch_bounds__(256) fn_reshape_reshape_gather_nd_reshape_broadcast_to_elementwise_mul__0__COND__FPA__FPA__FPA_16384llGE1ll_BPA_AND_FPA_16384llLE2147483647ll_BPA__BPA_AND_FPA__FPA_1GE1ll_BPA_AND_FPA_1LE1ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const int64_t* __restrict__ var_1, const float* __restrict__ var_6, float* __restrict__ var_7)
{
  if (((int)blockIdx.x < 16)) {
    if (((int)threadIdx.x < 256)) {
      var_7[((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047))))] = (var_6[((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047))))] * var[(var_1[((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)] & 2047)]);
      var_7[(1ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] = (var_6[(1ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] * var[(var_1[((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)] & 2047)]);
      var_7[(2ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] = (var_6[(2ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] * var[(var_1[((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)] & 2047)]);
      var_7[(3ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] = (var_6[(3ll + ((((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) / 2048ll) * 16384ll) + ((4ll * ((int)threadIdx.x & 1)) + (8ll * ((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)))))] * var[(var_1[((((int)threadIdx.x / 2) + (128ll * (int)blockIdx.x)) & 2047)] & 2047)]);
    };
  };
}


__global__
void __launch_bounds__(1024) fn_reshape_reshape_gather_nd_reshape_generate_shape_broadcast_to_broadcast_to_elementwise_mul__3__COND__FPA__FPA__FPA__FPA_8llMULS0_BPA_GE1024ll_BPA_AND_FPA__FPA_8llMULS0_BPA_LE1048575ll_BPA__BPA_AND_FPA__FPA_1GE1ll_BPA_AND_FPA_1LE1ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const int64_t* __restrict__ var_1, const float* __restrict__ var_5, float* __restrict__ var_8, int64_t S1, int64_t S0)
{
  if (((int)blockIdx.x < (((1ll + (2ll * S0)) / 1024) + 1ll))) {
    if (((int)threadIdx.x < 1024)) {
      if ((((1024 * (int)blockIdx.x) + (int)threadIdx.x) < (1ll + (2ll * S0)))) {
        if ((((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x)) < (8ll * S0))) {
          var_8[((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0))))] = (var_5[((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0))))] * var[(var_1[(((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0] % S1)]);
        };
        if (((1 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_8[(1ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] = (var_5[(1ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] * var[(var_1[(((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0] % S1)]);
        };
        if (((2 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_8[(2ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] = (var_5[(2ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] * var[(var_1[(((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0] % S1)]);
        };
        if (((3 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_8[(3ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] = (var_5[(3ll + ((4ll * ((int)threadIdx.x & 1)) + ((((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) / S0) * (8ll * S0)) + (8ll * ((((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0)))))] * var[(var_1[(((int)threadIdx.x / 2) + (512ll * (int)blockIdx.x)) % S0] % S1)]);
        };
      };
    };
  };
}
__global__
void __launch_bounds__(1024) fn_reshape_gather_nd_generate_shape_broadcast_to_broadcast_to_elementwise_mul__1__COND__FPA__FPA__FPA__FPA_8llMULS0_BPA_GE1024ll_BPA_AND_FPA__FPA_8llMULS0_BPA_LE1048575ll_BPA__BPA_AND_FPA__FPA_1GE1ll_BPA_AND_FPA_1LE1ll_BPA__BPA__BPA___kernel(const int64_t* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_3, float* __restrict__ var_6, int64_t S0, int64_t S1)
{
  if (((int)blockIdx.x < (((1ll + (2ll * S0)) / 1024) + 1ll))) {
    if (((int)threadIdx.x < 1024)) {
      if ((((1024 * (int)blockIdx.x) + (int)threadIdx.x) < (1ll + (2ll * S0)))) {
        if ((((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x)) < (8ll * S0))) {
          var_6[((8ll * ((int)threadIdx.x / 2)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x)))] = (var_3[((8ll * ((int)threadIdx.x / 2ll)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x)))] * var_1[var[((((int)threadIdx.x / 2ll) + (512ll * (int)blockIdx.x)) % S0)]]);
        };
        if (((1 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_6[(1ll + ((8ll * ((int)threadIdx.x / 2)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] = (var_3[(1ll + ((8ll * ((int)threadIdx.x / 2ll)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] * var_1[var[((((int)threadIdx.x / 2ll) + (512ll * (int)blockIdx.x)) % S0)]]);
        };
        if (((2 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_6[(2ll + ((8ll * ((int)threadIdx.x / 2)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] = (var_3[(2ll + ((8ll * ((int)threadIdx.x / 2ll)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] * var_1[var[((((int)threadIdx.x / 2ll) + (512ll * (int)blockIdx.x)) % S0)]]);
        };
        if (((3 + ((4096 * (int)blockIdx.x) + (4 * (int)threadIdx.x))) < (8ll * S0))) {
          var_6[(3ll + ((8ll * ((int)threadIdx.x / 2)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] = (var_3[(3ll + ((8ll * ((int)threadIdx.x / 2ll)) + ((4ll * ((int)threadIdx.x & 1)) + (4096ll * (int)blockIdx.x))))] * var_1[var[((((int)threadIdx.x / 2ll) + (512ll * (int)blockIdx.x)) % S0)]]);
        };
      };
    };
  };
}
}

void CheckAccuracy(float *out, float *manual_out, int n) {
  int error_count = 0;
	for(int i = 0; i < n; ++i){
		if(std::abs(float(out[i]) - float(manual_out[i])) > 1e-5) {
			std::cout << i << "\t" << float(out[i]) << "\t" << float(manual_out[i]) << std::endl;
      error_count++;
		}
    if (error_count > 10) {
      printf("the ans is wrong\n");
      return;
    }
	}
	printf("the ans is right\n");
}

int main(){
	const int S = 2048;
	const int N = 1 * S * 8 * 1;//
  const int M = 1 * S * 1 * 1;//
	float *q=(float *)malloc(N*sizeof(float));
	float *d_q;
	cudaMalloc((void **)&d_q,N*sizeof(float));

	float *cos=(float *)malloc(M*sizeof(float));
	float *d_cos;
	cudaMalloc((void **)&d_cos,M*sizeof(float));

	int64_t *position_id = (int64_t *)malloc(S*sizeof(int64_t));
	int64_t *d_pi;
	cudaMalloc((void **)&d_pi,M*sizeof(int64_t));

	float *out=(float *)malloc(N *sizeof(float));
	float *d_out;
	cudaMalloc((void **)&d_out, N *sizeof(float));

	float *stout=(float *)malloc(N *sizeof(float));
	float *d_stout;
	cudaMalloc((void **)&d_stout, N *sizeof(float));
	
	srand(0);
	for(int i=0;i<N;i++){
		q[i]= rand() % 100 / 100.0f;
		
	}
  for(int i=0; i<M; i++)
  {
    cos[i]= rand() % 100 / 100.0f;
  }
	for(int i=0;i<S;i++){
		position_id[i]=i;
	}
	std::cerr << "before copy" << std::endl;
	cudaMemcpy(d_q,q,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_cos,cos,M*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi,position_id,S*sizeof(int64_t),cudaMemcpyHostToDevice);

	fn_reshape_reshape_gather_nd_reshape_broadcast_to_elementwise_mul__0__COND__FPA__FPA__FPA_16384llGE1ll_BPA_AND_FPA_16384llLE2147483647ll_BPA__BPA_AND_FPA__FPA_1GE1ll_BPA_AND_FPA_1LE1ll_BPA__BPA__BPA___kernel<<<16,256>>>(d_cos, d_pi, d_q, d_stout);
	
	int iter = 1;
	for(int i = 0; i < iter; i++) {
    // Kernel by CINN
  fn_reshape_gather_nd_generate_shape_broadcast_to_broadcast_to_elementwise_mul__1__COND__FPA__FPA__FPA__FPA_8llMULS0_BPA_GE1024ll_BPA_AND_FPA__FPA_8llMULS0_BPA_LE1048575ll_BPA__BPA_AND_FPA__FPA_1GE1ll_BPA_AND_FPA_1LE1ll_BPA__BPA__BPA___kernel<<<4,1024>>>(d_pi,d_cos,  d_q, d_out, S, S);
	}

	std::cerr << "after copy" << std::endl;
	cudaMemcpy(out, d_out, N *sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(stout, d_stout, N *sizeof(float), cudaMemcpyDeviceToHost);



	CheckAccuracy(out, stout, N);

	cudaFree(d_q);
	cudaFree(d_cos);
	cudaFree(d_pi);
	cudaFree(d_out);
	cudaFree(d_stout);
}