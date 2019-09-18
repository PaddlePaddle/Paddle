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

#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "paddle/fluid/recordio/chunk.h"

namespace paddle {
namespace recordio {

class Scanner {
 public:
  explicit Scanner(std::unique_ptr<std::istream>&& stream);

  explicit Scanner(const std::string& filename);

  void Reset();

  std::string Next();

  bool HasNext() const;

 private:
  std::unique_ptr<std::istream> stream_;
  ChunkParser parser_;
};
}  // namespace recordio
}  // namespace paddle
