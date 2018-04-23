//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/reader.h"

namespace paddle {
namespace framework {
ReaderBase::~ReaderBase() {}

FileReader::FileReader(const std::vector<DDim> &dims) : dims_(dims) {}

std::unique_ptr<std::vector<LoDTensor>> FileReader::ReadNext() {
  auto out = ReadNextImpl();
  if (out == nullptr) {
    return out;
  }
  for (size_t i = 0; i < dims_.size(); ++i) {
    auto &actual = out->at(i).dims();
    auto &expect = dims_[i];

    PADDLE_ENFORCE_EQ(actual.size(), expect.size());
  }

  return out;
}
}  // namespace framework
}  // namespace paddle
