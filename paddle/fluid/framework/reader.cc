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
#include <deque>

namespace paddle {
namespace framework {
ReaderBase::~ReaderBase() {}

void ReaderBase::InsertDecoratedReader(ReaderBase *decorated_reader) {
  decorated_readers_.emplace(decorated_reader);
}
void ReaderBase::EraseDecoratedReader(ReaderBase *decorated_reader) {
  auto it = decorated_readers_.find(decorated_reader);
  PADDLE_ENFORCE(it != decorated_readers_.end(),
                 "Cannot find the decorated reader to erase");
  decorated_readers_.erase(it);
}
std::unordered_set<ReaderBase *> ReaderBase::GetEndPoints() {
  std::unordered_set<ReaderBase *> result;
  std::deque<ReaderBase *> queue;
  queue.emplace_back(this);
  while (!queue.empty()) {  // BFS search
    auto *front = queue.front();
    queue.pop_front();
    if (front->decorated_readers_.empty()) {
      result.emplace(front);
    } else {
      for (ReaderBase *reader : front->decorated_readers_) {
        queue.emplace_back(reader);
      }
    }
  }

  return result;
}

FileReader::FileReader(const std::vector<DDim> &dims) : dims_(dims) {}

void FileReader::ReadNext(std::vector<LoDTensor> *out) {
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
DecoratedReader::~DecoratedReader() { reader_->EraseDecoratedReader(this); }
}  // namespace framework
}  // namespace paddle
