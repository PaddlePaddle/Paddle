// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/py_reader.h"
#include <memory>

namespace paddle {
namespace operators {
namespace reader {

PyReader::PyReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue)
    : framework::FileReader() {
  PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
  queue_ = queue;
}

void PyReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  bool success;
  *out = queue_->Pop(&success);
  if (!success) out->clear();
}

PyReader::~PyReader() { queue_->Close(); }

void PyReader::Shutdown() { queue_->Close(); }

void PyReader::Start() { queue_->ReOpen(); }

}  // namespace reader
}  // namespace operators
}  // namespace paddle
