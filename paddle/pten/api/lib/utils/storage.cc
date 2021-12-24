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

#include "paddle/pten/api/lib/utils/storage.h"

namespace paddle {
namespace experimental {

ExternalStorage::ExternalStorage(void* ptr,
                                 size_t size,
                                 const paddle::platform::Place& place)
    : pten::Storage(
          std::make_shared<paddle::memory::Allocation>(ptr, size, place)),
      size_(size) {}

ExternalStorage::ExternalStorage(const pten::intrusive_ptr<pten::Storage>& root,
                                 size_t delta,
                                 size_t size)
    : Storage(std::make_shared<paddle::memory::Allocation>(
          static_cast<uint8_t*>(root->data()) + delta, size, root->place())),
      size_(size) {
  PADDLE_ENFORCE_LE(static_cast<size_t>(delta + size),
                    root->size(),
                    paddle::platform::errors::InvalidArgument(
                        "The size of the external storage does "
                        "not meet the metadata requirements."));
}

}  // namespace experimental
}  // namespace paddle
