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

#include "paddle/phi/api/lib/utils/tensor_utils.h"

#include <utility>
#include <vector>

namespace paddle {
namespace experimental {

template <typename DstLoD, typename SrcLoD>
void SetLoD(DstLoD* dst, const SrcLoD& src) {
  dst->reserve(src.size());
  dst->clear();
  for (auto&& v : src) {
    dst->emplace_back(v);
  }
}

std::unique_ptr<phi::DenseTensor> MakePhiDenseTensor(
    const phi::DenseTensor& src) {
  return std::make_unique<phi::DenseTensor>(src);
}

phi::IntArray MakePhiIntArray(const phi::DenseTensor& src) { return {src}; }

}  // namespace experimental
}  // namespace paddle
