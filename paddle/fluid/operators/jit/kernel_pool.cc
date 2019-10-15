/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/kernel_pool.h"
#include <memory>  // for shared_ptr
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {
namespace jit {

JitCodeCreatorPool& JitCodeCreatorPool::Instance() {
  static JitCodeCreatorPool g_creator_pool;
  return g_creator_pool;
}

KernelPool& KernelPool::Instance() {
  static KernelPool g_kernel_pool;
  return g_kernel_pool;
}

ReferKernelPool& ReferKernelPool::Instance() {
  static ReferKernelPool g_refer_kernel_pool;
  return g_refer_kernel_pool;
}

}  // namespace jit
}  // namespace operators
}  // namespace paddle
