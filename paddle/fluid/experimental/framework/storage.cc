/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/experimental/framework/storage.h"

namespace paddle {
namespace experimental {
namespace framework {

void Storage::Realloc(size_t size) {
  alloc_->Deallocate(data(), size_);
  size_ = size;
  data_ = alloc_->Allocate(size_);
}

ExternalStorage::ExternalStorage(void* ptr, size_t size,
                                 const platform::Place& place)
    : StorageInterface(ptr), size_(size), place_(place) {}

ExternalStorage::ExternalStorage(const intrusive_ptr<Storage>& root,
                                 size_t delta, size_t size)
    : StorageInterface(static_cast<uint8_t*>(root->data()) + delta),
      size_(size),
      place_(root->place()) {
  PADDLE_ENFORCE_LE(
      static_cast<size_t>(delta + size), root->size(),
      platform::errors::InvalidArgument("The size of the external storage does "
                                        "not meet the metadata requirements."));
}

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
