// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {

/**
 * This file is not a common test, it is used as a util for developers to
 * write source CUDA code to debug whether it runs correctly during runtime
 */
using runtime::cuda::CUDAModule;

/**
 * Utility function to create cuda memory of non-empty shape.
 *
 * @param shape: a non-empty shape for the created cuda memory
 * @param data: the data to initialize the cuda memory. Function doesn't
 *     initialize if it is nullptr
 * @return the CUdeviceptr pointing to the created memory
 */
template <typename T>
CUdeviceptr CreateCudaMemory(const std::vector<int>& shape, const T* data) {
  PADDLE_ENFORCE_EQ(!shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Couldn't create CUDA memory for empty shape. Please "
                        "ensure the shape is not empty."));
  CUDA_CALL(cudaDeviceSynchronize());

  int numel = 1;
  for (int s : shape) {
    numel = numel * s;
  }

  CUdeviceptr cuda_ptr = cuMemAlloc(&cuda_ptr, numel * sizeof(T));
  if (data != nullptr) {
    CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(cuda_ptr),
                         data,
                         numel * sizeof(T),
                         cudaMemcpyHostToDevice));
  }
  return cuda_ptr;
}

TEST(CodeGenDebug, RunCudaSourceCode) {
  cinn::common::Context::Global().ResetNameId();

  std::string source_code = R"ROC(
extern "C" {

__global__
void __launch_bounds__(512) fn_relu_1_kernel(const float* __restrict__ var_1, float* __restrict__ Relu_output)
{
  for (int32_t j_0 = 0; j_0 < 8; j_0 += 1) {
    for (int32_t j_1 = 0; j_1 < 1; j_1 += 1) {
      for (int32_t j_2 = 0; j_2 < 1; j_2 += 1) {
        for (int32_t j_3 = 0; j_3 < 8; j_3 += 1) {
          for (int32_t j_4 = 0; j_4 < 1; j_4 += 1) {
            for (int32_t k_0 = 0; k_0 < 1; k_0 += 1) {
              for (int32_t k_1 = 0; k_1 < 7; k_1 += 1) {
                for (int32_t k_2 = 0; k_2 < 4; k_2 += 1) {
                  for (int32_t k_3 = 0; k_3 < 4; k_3 += 1) {
                    for (int32_t k_4 = 0; k_4 < 1; k_4 += 1) {
                      for (int32_t a_0 = 0; a_0 < 16; a_0 += 1) {
                        for (int32_t a_1 = 0; a_1 < 1; a_1 += 1) {
                          for (int32_t a_2 = 0; a_2 < 1; a_2 += 1) {
                            for (int32_t a_3 = 0; a_3 < 1; a_3 += 1) {
                              for (int32_t a_4 = 0; a_4 < 7; a_4 += 1) {
                                Relu_output[((7 * a_0) + ((7 * a_1) + ((7 * a_2) + ((7 * a_3) + ((100352 * j_0) + ((100352 * j_1) + ((100352 * j_2) + ((12544 * j_3) + ((12544 * j_4) + ((12544 * k_0) + ((1792 * k_1) + ((448 * k_2) + ((112 * k_3) + ((112 * k_4) + a_4))))))))))))))] = max(var_1[((7 * a_0) + ((7 * a_1) + ((7 * a_2) + ((7 * a_3) + ((100352 * j_0) + ((100352 * j_1) + ((100352 * j_2) + ((12544 * j_3) + ((12544 * j_4) + ((12544 * k_0) + ((1792 * k_1) + ((448 * k_2) + ((112 * k_3) + ((112 * k_4) + a_4))))))))))))))], 0.00000000f);
                              };
                            };
                          };
                        };
                      };
                    };
                  };
                };
              };
            };
          };
        };
      };
    };
  };
}

}
)ROC";

  backends::nvrtc::Compiler compiler;

  std::string ptx = compiler(CodeGenCudaDev::GetSourceHeader() + source_code);
  ASSERT_FALSE(ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);
  CUdeviceptr var =
      CreateCudaMemory<float>(/* shape */ {64 * 112 * 112}, /* data */ nullptr);
  CUdeviceptr out =
      CreateCudaMemory<float>(/* shape */ {64 * 112 * 112}, /* data */ nullptr);

  void* args[] = {&var, &out};
  dim3 grid(512, 1, 1);
  dim3 block(512, 1, 1);
  cuda_module.LaunchKernel(
      /*device_id*/ 0, "fn_relu_1_kernel", grid, block, args);
}

}  // namespace backends
}  // namespace cinn
