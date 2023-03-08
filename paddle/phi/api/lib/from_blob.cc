/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/include/from_blob.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

char touch_extension_api() { return '\0'; }

using AllocationDeleter = void (*)(phi::Allocation*);

PADDLE_API Tensor from_blob(void* data,
                            const phi::DDim& shape,
                            DataType dtype,
                            const Place& place,
                            phi::DataLayout layout,
                            size_t storage_offset,
                            const Deleter& deleter) {
  PADDLE_ENFORCE_NOT_NULL(
      data, phi::errors::InvalidArgument("data can not be nullptr"));

  auto meta = phi::DenseTensorMeta(dtype, shape, layout, storage_offset);

  size_t size =
      SizeOf(dtype) * (meta.is_scalar ? 1 : product(shape)) + storage_offset;

  AllocationDeleter alloc_deleter = nullptr;
  if (deleter) {
    static thread_local Deleter g_deleter = deleter;
    alloc_deleter = [](phi::Allocation* p) { g_deleter(p->ptr()); };
  }

  auto alloc =
      std::make_shared<phi::Allocation>(data, size, alloc_deleter, place);

  return Tensor(std::make_shared<phi::DenseTensor>(alloc, meta));
}

}  // namespace experimental
}  // namespace paddle
