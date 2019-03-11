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

#include "paddle/fluid/recordio/scanner.h"

#include <string>
#include <utility>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace recordio {

Scanner::Scanner(std::unique_ptr<std::istream> &&stream)
    : stream_(std::move(stream)), parser_(*stream_) {
  Reset();
}

Scanner::Scanner(const std::string &filename)
    : stream_(new std::ifstream(filename, std::ios::in | std::ios::binary)),
      parser_(*stream_) {
  PADDLE_ENFORCE(static_cast<bool>(*stream_), "Cannot open file %s", filename);
  Reset();
}

void Scanner::Reset() {
  stream_->clear();
  stream_->seekg(0, std::ios::beg);
  parser_.Init();
}

std::string Scanner::Next() {
  if (stream_->eof()) {
    return "";
  }

  auto res = parser_.Next();
  if (!parser_.HasNext() && HasNext()) {
    parser_.Init();
  }
  return res;
}

bool Scanner::HasNext() const { return !stream_->eof(); }
}  // namespace recordio
}  // namespace paddle
