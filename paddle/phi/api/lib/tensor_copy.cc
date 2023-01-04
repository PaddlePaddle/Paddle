/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/lib/tensor_copy.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace experimental {

void copy(const Tensor& src, const Place& place, bool blocking, Tensor* dst) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(src);
  kernel_key_set.backend_set =
      kernel_key_set.backend_set | BackendSet(phi::TransToPhiBackend(place));
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  VLOG(6) << "start copy. ";

  auto target_place = phi::TransToPhiPlace(kernel_key.backend());
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetMutable(
      target_place.GetType() == place.GetType() ? place : target_place);

  auto dense_x = TensorToDenseTensor(src);

  auto kernel_out = SetKernelOutput(dst);
  phi::MetaTensor meta_out(kernel_out);
  phi::UnchangedInferMeta(*dense_x, &meta_out);

  phi::Copy(*dev_ctx, *dense_x, place, blocking, kernel_out);

  VLOG(6) << "copy finished. ";
}

}  // namespace experimental
}  // namespace paddle
