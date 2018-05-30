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

#include <algorithm>
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
ReaderBase::~ReaderBase() {}

FileReader::FileReader(const std::vector<DDim> &dims) : dims_(dims) {}

void FileReader::ReadNext(std::vector<LoDTensor> *out) {
  ReadNextImpl(out);
  if (out->empty()) {
    return;
  }
  
  /*
  * Comment from sneaxiy
  *   Is it necessary to use std::vector<LoDTensor>::at() to access the data in 'out'?
  *   If std::out_of_range is expected while accessing the data in 'out', try to add PADDLE_ENFORCE
  *   @code_start
  *   PADDLE_ENFORCE_EQ(out->size(), dims_.size(), "Expected out->size() == %d but read %d", dims_.size(), out->size());
  *   @code_end
  *
  *   If std::out_of_range is not expected, try to change to upper bound of the loop to be std::min(out->size(), dims_.size())
  */
  for (size_t i = 0; i < std::min(out->size(), dims_.size()); ++i) {
    auto &actual = (*out)[i].dims();
    //auto &actual = out->at(i).dims();
    auto &expect = dims_[i];

    PADDLE_ENFORCE_EQ(actual.size(), expect.size());
    for (int j = 0; j < actual.size(); ++j) {
      //      PADDLE_ENFORCE(actual[i] == expect[i] || expect[i] == -1);
    }
  }
}
}  // namespace framework
}  // namespace paddle
