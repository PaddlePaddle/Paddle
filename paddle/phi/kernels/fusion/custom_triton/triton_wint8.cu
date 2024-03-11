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
#include "generated/wint8/wint8.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_triton_wint8;

std::vector<paddle::Tensor> TritonWint8(const paddle::Tensor& x,
                                        const paddle::Tensor& qweight,
                                        const paddle::Tensor& scales,
                                        paddle::optional<paddle::Tensor>& bias,
                                        bool bool_trans_w,
                                        bool with_bias) {
  int m = x.shape()[0];
  int k = x.shape()[1];
  int n = scales.shape()[0];

  auto c_out = paddle::full({m, n}, 0, x.dtype(), x.place());

  auto dev_x = x.data<phi::dtype::float16>();
  auto dev_weight = qweight.data<int8_t>();
  auto dev_c = c_out.data<phi::dtype::float16>();
  auto dev_scales = scales.data<phi::dtype::float16>();
  phi::dtype::float16* dev_bias = nullptr;
  if (with_bias) {
    dev_bias = bias->data<phi::dtype::float16>();
  }

  int stride_bk = n;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = k;
  }

  std::vector<int> problem_size = {m, k, n};

  if (map_problem_triton_wint8.count(problem_size)) {
    int algo_id = map_problem_triton_wint8[problem_size];
    auto status = wint8_kernel(c_out.stream(),
                               (CUdeviceptr)(dev_x),
                               (CUdeviceptr)(dev_weight),
                               (CUdeviceptr)(dev_c),
                               (CUdeviceptr)(dev_scales),
                               (CUdeviceptr)(dev_bias),
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

  for (int algo_id = 0; algo_id < wint8_kernel_get_num_algos(); ++algo_id) {
    cudaEvent_t beg, end;
    auto status = CUDA_SUCCESS;

    for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
      if (ii == WARMUP) {
        (cudaEventCreate(&beg));
        (cudaEventCreate(&end));
        (cudaEventRecord(beg));
      }
      cudaMemset(dev_c, 0, sizeof(phi::dtype::float16) * m * n);
      status = wint8_kernel(c_out.stream(),
                                 (CUdeviceptr)(dev_x),
                                 (CUdeviceptr)(dev_weight),
                                 (CUdeviceptr)(dev_c),
                                 (CUdeviceptr)(dev_scales),
                                 (CUdeviceptr)(dev_bias),
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

  map_problem_triton_wint8[problem_size] = select_id;
  std::cout << "select algo id: " << select_id << std::endl;

  return {c_out};
}

// std::vector<std::vector<int64_t>> TritonWint8InferShape(const
// std::vector<int64_t>& a_shape,
//                                                               const
//                                                               std::vector<int64_t>&
//                                                               b_shape) {
//     return {{a_shape[0], b_shape[1]}};
// }

// std::vector<paddle::DataType> TritonWint8InferDtype(const paddle::DataType&
// A_dtype,
//                                                         const
//                                                         paddle::DataType&
//                                                         B_dtype) {
//     return {A_dtype};
// }

PD_BUILD_OP(triton_wint8)
    .Inputs({"x", "qweight", "scales", paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TritonWint8))
    .Attrs({"bool_trans_w: bool", "with_bias: bool"});
// .SetInferShapeFn(PD_INFER_SHAPE(TritonWint8InferShape))
// .SetInferDtypeFn(PD_INFER_DTYPE(TritonWint8InferDtype));
