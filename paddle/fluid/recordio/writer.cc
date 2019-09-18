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
#include "paddle/fluid/recordio/writer.h"

#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace recordio {

void Writer::Write(const std::string& record) {
  cur_chunk_.Add(record);
  if (cur_chunk_.NumRecords() >= max_num_records_in_chunk_) {
    Flush();
  }
}

void Writer::Flush() {
  cur_chunk_.Write(stream_, compressor_);
  cur_chunk_.Clear();
}

Writer::~Writer() {
  PADDLE_ENFORCE(cur_chunk_.Empty(), "Writer must be flushed when destroy.");
}

}  // namespace recordio
}  // namespace paddle
