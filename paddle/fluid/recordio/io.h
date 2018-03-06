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

#include <stdio.h>
#include <string>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"  // DISABLE_COPY_ASSIGN

namespace paddle {
namespace recordio {

// Seekable Stream Interface for read and write
class Stream {
public:
  virtual ~Stream() {}
  virtual size_t Read(void* ptr, size_t size) = 0;
  virtual void Write(const void* ptr, size_t size) = 0;
  virtual size_t Tell() = 0;
  virtual void Seek(size_t p) = 0;
  // Create Stream Instance
  static Stream* Open(const char* filename, const char* mode);
};

// FileStream
class FileStream : public Stream {
public:
  explicit FileStream(FILE* fp) : fp_(fp) {}
  ~FileStream() { this->Close(); }
  size_t Read(void* ptr, size_t size);
  void Write(const void* ptr, size_t size);
  size_t Tell();
  void Seek(size_t p);
  bool Eof();
  void Close();

private:
  FILE* fp_;
  DISABLE_COPY_AND_ASSIGN(FileStream);
};

}  // namespace recordio
}  // namespace paddle
