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

#include <string>

#include "paddle/fluid/recordio/chunk.h"
namespace paddle {
namespace recordio {

class Writer {
 public:
  Writer(std::ostream* sout, Compressor compressor,
         size_t max_num_records_in_chunk = 1000)
      : stream_(*sout),
        max_num_records_in_chunk_(max_num_records_in_chunk),
        compressor_(compressor) {}

  void Write(const std::string& record);

  void Flush();

  ~Writer();

 private:
  std::ostream& stream_;
  size_t max_num_records_in_chunk_;
  Chunk cur_chunk_;
  Compressor compressor_;
};

}  // namespace recordio
}  // namespace paddle
