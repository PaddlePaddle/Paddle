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

#include "paddle/phi/core/operators/reader/py_reader.h"

namespace paddle::operators::reader {

PyReader::PyReader(
    const std::shared_ptr<DenseTensorBlockingQueue>& queue,
    const std::vector<phi::DDim>& dims,
    const std::vector<framework::proto::VarType::Type>& var_types,
    const std::vector<bool>& need_check_feed)
    : framework::FileReader(dims, var_types, need_check_feed) {
  PADDLE_ENFORCE_NOT_NULL(queue,
                          common::errors::PreconditionNotMet(
                              "DenseTensorBlockingQueue must not be null."));
  queue_ = queue;
}

void PyReader::ReadNext(phi::TensorArray* out) {
  bool success = false;
  *out = queue_->Pop(&success);
  if (!success) out->clear();
}

PyReader::~PyReader() {  // NOLINT
  queue_->Close();
}

void PyReader::Shutdown() { queue_->Close(); }

void PyReader::Start() { queue_->ReOpen(); }

}  // namespace paddle::operators::reader
