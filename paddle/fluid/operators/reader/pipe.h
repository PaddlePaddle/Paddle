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

#pragma once
#include <array>
#include <cstddef>

namespace paddle {
namespace operators {
namespace reader {

class Pipe {
 public:
  explicit Pipe(const int file_descriptor);
  ~Pipe();
  static std::array<int, 2> Create();

 protected:
  const int file_descriptor_;
};

class ReadPipe : public Pipe {
 public:
  explicit ReadPipe(const int file_descriptor) : Pipe(file_descriptor) {}
  void read(uint8_t* buffer, std::size_t size);
};

class WritePipe : public Pipe {
 public:
  explicit WritePipe(const int file_descriptor) : Pipe(file_descriptor) {}
  void write(const uint8_t* buffer, std::size_t size);
};
}  // namespace reader
}  // namespace operators
}  // namespace paddle
