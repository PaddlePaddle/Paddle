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
  //auto dev_weight = qweight.data<int8_t>();
  auto dev_weight = qweight.data<int32_t>();
  auto dev_c = c_out.data<int32_t>();

  int stride_bk = n;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    //stride_bn = k / 8;
    stride_bn = k / 4; 
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
  constexpr int WARMUP = 5;
  constexpr int REPEAT = 10;

  for (int algo_id = 0; algo_id < w4a8_kernel_get_num_algos(); ++algo_id) {
    cudaEvent_t beg[REPEAT];
    cudaEvent_t end[REPEAT];
    float elapsed_times[REPEAT];

    auto status = CUDA_SUCCESS;

    for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
      int repeat_id = ii - WARMUP;

      if (repeat_id >= 0) {
        (cudaEventCreate(beg + repeat_id));
        (cudaEventCreate(end + repeat_id));
        (cudaEventRecord(beg[repeat_id]));
      }

      auto flush_l2_cache = paddle::full(
          {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
      // std::cout << &flush_l2_cache  << std::endl;

      cudaMemset(dev_c, 0, sizeof(phi::dtype::float16) * m * n);
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

      if (repeat_id >= 0) {
        (cudaEventRecord(end[repeat_id]));
        (cudaEventSynchronize(end[repeat_id]));
        (cudaEventElapsedTime(
            elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
      }
    }

    float avg_elapsed_time = 0.f;
    for (int ii = 0; ii < REPEAT; ++ii) {
      avg_elapsed_time += elapsed_times[ii];
    }

    if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
      min_time = avg_elapsed_time;
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

