// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_fp16.h>
#include <vector>
#include "generated/w4a8/w4a8.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_triton_w4a8;

std::vector<paddle::Tensor> TritonW4a8(const paddle::Tensor& x,
                                        const paddle::Tensor& qweight,
                                        bool bool_trans_w) {
  int m = x.shape()[0];
  int k = x.shape()[1];
  int n = qweight.shape()[1];
  if (bool_trans_w) {
    n = qweight.shape()[0];
  }
  
  std::cout << "TritonW4a8: m=" << m << ", k=" << k << ", n=" << n << std::endl;
  auto c_out = paddle::full({m, n}, 0, paddle::DataType::INT32, x.place());

  auto dev_x = x.data<int8_t>();
  auto dev_weight = qweight.data<int32_t>();
  auto dev_c = c_out.data<int32_t>();

  int stride_bk = n;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = k / 8;
  }

  std::vector<int> problem_size = {m, k, n};

  if (map_problem_triton_w4a8.count(problem_size)) {
    int algo_id = map_problem_triton_w4a8[problem_size];
    printf("TritonW4a8: %d\n", algo_id);
    auto status = w4a8_kernel(c_out.stream(),
                               (CUdeviceptr)(dev_x),
                               (CUdeviceptr)(dev_weight),
                               (CUdeviceptr)(dev_c),
                               m,
                               n,
                               k,
                               k,
                               1,
                               stride_bk,
                               stride_bn,
                               n,
                               1,
                               algo_id);
    assert(status == CUDA_SUCCESS);
    return {c_out};
  }

  float min_time = 10000.f;
  int select_id = -1;
  constexpr int WARMUP = 10;
  constexpr int REPEAT = 10;

  for (int algo_id = 0; algo_id < w4a8_kernel_get_num_algos(); ++algo_id) {
    cudaEvent_t beg, end;
    auto status = CUDA_SUCCESS;
    printf("TritonW4a8: %d\n", algo_id);

    for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
      if (ii == WARMUP) {
        (cudaEventCreate(&beg));
        (cudaEventCreate(&end));
        (cudaEventRecord(beg));
      }
      cudaMemset(dev_c, 0, sizeof(int32_t) * m * n);
      status = w4a8_kernel(c_out.stream(),
                                 (CUdeviceptr)(dev_x),
                                 (CUdeviceptr)(dev_weight),
                                 (CUdeviceptr)(dev_c),
                                 m,
                                 n,
                                 k,
                                 k,
                                 1,
                                 stride_bk,
                                 stride_bn,
                                 n,
                                 1,
                                 algo_id);
      // assert(status == CUDA_SUCCESS);
    }

    (cudaEventRecord(end));
    (cudaEventSynchronize(end));
    float elapsed_time;
    (cudaEventElapsedTime(&elapsed_time, beg, end));
    if (elapsed_time < min_time && status == CUDA_SUCCESS) {
      min_time = elapsed_time;
      select_id = algo_id;
    }
  }

  map_problem_triton_w4a8[problem_size] = select_id;
  std::cout << "select algo id: " << select_id << std::endl;

  return {c_out};
}

PD_BUILD_OP(triton_w4a8)
    .Inputs({"x", "qweight"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TritonW4a8))
    .Attrs({"bool_trans_w: bool"});

