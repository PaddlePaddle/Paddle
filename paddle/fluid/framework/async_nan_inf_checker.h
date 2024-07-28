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

#pragma once

#ifdef PADDLE_WITH_CUDA
#include <ThreadPool.h>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace framework {

class AsyncNaNInfChecker {
  DISABLE_COPY_AND_ASSIGN(AsyncNaNInfChecker);

 public:
  explicit AsyncNaNInfChecker(phi::GPUPlace place);

  ~AsyncNaNInfChecker();

  bool Check(const phi::DenseTensor &x);

  bool Check(const paddle::Tensor &x);

  void Wait();

 private:
  template <typename T>
  bool CheckImpl(const phi::DenseTensor &x);

 private:
  std::unique_ptr<phi::GPUContext> ctx_;
  cudaEvent_t event_{nullptr};
  phi::GPUPlace place_;
};

}  // namespace framework
}  // namespace paddle

#endif
