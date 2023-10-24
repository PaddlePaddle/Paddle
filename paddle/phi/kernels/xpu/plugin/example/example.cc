// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#include <assert.h>
#include "xpu/plugin.h"
#include "xpu/xdnn.h"

namespace xdnn = baidu::xpu::api;

int main() {
  int num = 5;
  int errcode = 0;
  auto ctx = xdnn::create_context();
  float* A = nullptr;
  errcode = xpu_malloc(reinterpret_cast<void**>(&A), num * sizeof(float));
  assert(errcode == 0);
  float* B = nullptr;
  errcode = xpu_malloc(reinterpret_cast<void**>(&B), num * sizeof(float));
  assert(errcode == 0);

  std::vector<float> A_cpu = {1, 2, 3, 4, 5};
  std::vector<float> B_cpu(num, 0.0f);
  std::vector<float> B_ref = {3, 4, 5, 6, 7};
  xpu_memcpy(reinterpret_cast<void*>(A),
             reinterpret_cast<void*>(&(A_cpu[0])),
             num * sizeof(float),
             XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  errcode = xdnn::plugin::add2(ctx, A, B, num);
  assert(errcode == 0);
  xpu_memcpy(reinterpret_cast<void*>(&(B_cpu[0])),
             reinterpret_cast<void*>(B),
             num * sizeof(float),
             XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  printf("A(%p):\n", A);
  for (size_t i = 0; i < num; i++) {
    printf("%f ", A_cpu[i]);
  }
  printf("\nB(%p):\n", B);
  for (size_t i = 0; i < num; i++) {
    printf("%f ", B_cpu[i]);
  }
  bool pass = true;
  for (size_t i = 0; i < num; i++) {
    if (fabs(B_cpu[i] - B_ref[i]) > 1e-5f) {
      pass = false;
      break;
    }
  }
  printf("\nCheck %s! \n", pass ? "pass" : "fail");

  destroy_context(ctx);
  errcode = xpu_free(A);
  assert(errcode == 0);
  errcode = xpu_free(B);
  assert(errcode == 0);
  return 0;
}
