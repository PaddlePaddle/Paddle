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

static void TraceDecorations(
    std::weak_ptr<ReaderBase> reader,
    std::vector<std::function<void()>> *restart_methods) {
  auto reader_ptr = reader.lock();
  PADDLE_ENFORCE_NOT_NULL(reader_ptr);
  for (auto &d_reader : reader_ptr->GetDecorations()) {
    TraceDecorations(d_reader, restart_methods);
  }
  restart_methods->emplace_back(reader_ptr->CloseAndGetRestartMethod(false));
}

void ReaderBase::ReInitAllReaders() {
  std::vector<std::function<void()>> restart_methods;
  for (auto &d_reader : decorations_) {
    TraceDecorations(d_reader, &restart_methods);
  }
  restart_methods.emplace_back(CloseAndGetRestartMethod(true));

  for (auto it = restart_methods.rbegin(); it != restart_methods.rend(); ++it) {
    (*it)();
  }
}

DecoratedReader::DecoratedReader(const std::shared_ptr<ReaderBase> &reader)
    : ReaderBase(), reader_(reader) {
  PADDLE_ENFORCE_NOT_NULL(reader_);
}

void DecoratedReader::ReadNext(std::vector<LoDTensor> *out) {
  PADDLE_ENFORCE(!is_closed_, "Can not read data from a closed reader.");
  ReadNextImpl(out);
}

std::function<void()> DecoratedReader::CloseAndGetRestartMethod(
    bool recursively) {
  Close();
  std::function<void()> restart_method;
  if (recursively) {
    auto underlying_restart_method = reader_->CloseAndGetRestartMethod(true);
    restart_method = [=] {
      underlying_restart_method();
      ReStart();
    };
  } else {
    restart_method = [=] { ReStart(); };
  }
  return restart_method;
}

void RootReader::ReadNext(std::vector<LoDTensor> *out) {
  PADDLE_ENFORCE(!is_closed_, "Can not read data from a closed reader.");
  ReadNextImpl(out);
}

std::function<void()> RootReader::CloseAndGetRestartMethod(bool recursively) {
  Close();
  return [=] { ReStart(); };
}

FileReader::FileReader(const std::vector<DDim> &dims)
    : RootReader(), dims_(dims) {}

void FileReader::ReadNext(std::vector<LoDTensor> *out) {
  PADDLE_ENFORCE(!is_closed_, "Can not read data from a closed reader.");
  ReadNextImpl(out);
  if (out->empty()) {
    return;
  }

  PADDLE_ENFORCE_EQ(out->size(), dims_.size());
  for (size_t i = 0; i < dims_.size(); ++i) {
    auto &actual = (*out)[i].dims();
    auto &expect = dims_[i];

    PADDLE_ENFORCE_EQ(actual.size(), expect.size());
    for (int j = 0; j < actual.size(); ++j) {
      //      PADDLE_ENFORCE(actual[i] == expect[i] || expect[i] == -1);
    }
  }
}

}  // namespace framework
}  // namespace paddle
