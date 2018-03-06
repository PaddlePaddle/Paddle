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

#include "paddle/fluid/recordio/io.h"
#include "paddle/fluid/string/piece.h"

#include <iostream>

namespace paddle {
namespace recordio {
Stream* Stream::Open(const char* filename, const char* mode) {
  // Create IOStream for different filesystems
  // HDFS:   hdfs://tmp/file.txt
  // Default: /tmp/file.txt
  FILE* fp = nullptr;
  if (string::HasPrefix(string::Piece(filename), string::Piece("/"))) {
    fp = fopen(filename, mode);
  }
  return new FileStream(fp);
}

size_t FileStream::Read(void* ptr, size_t size) {
  return fread(ptr, 1, size, fp_);
}

void FileStream::Write(const void* ptr, size_t size) {
  size_t real = fwrite(ptr, 1, size, fp_);
  PADDLE_ENFORCE(real == size, "FileStream write incomplete.");
}

size_t FileStream::Tell() { return ftell(fp_); }
void FileStream::Seek(size_t p) { fseek(fp_, p, SEEK_SET); }

bool FileStream::Eof() { return feof(fp_); }

void FileStream::Close() {
  if (fp_ != nullptr) {
    fclose(fp_);
    fp_ = nullptr;
  }
}

}  // namespace recordio
}  // namespace paddle
