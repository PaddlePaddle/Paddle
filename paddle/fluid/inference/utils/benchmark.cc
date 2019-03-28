// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/utils/benchmark.h"
#include <sstream>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {

std::string Benchmark::SerializeToString() const {
  std::stringstream ss;
  ss << "-----------------------------------------------------\n";
  ss << "name\t";
  ss << "batch_size\t";
  ss << "num_threads\t";
  ss << "latency\t";
  ss << "qps";
  ss << '\n';

  ss << name_ << "\t";
  ss << batch_size_ << "\t\t";
  ss << num_threads_ << "\t";
  ss << latency_ << "\t";
  ss << 1000.0 / latency_;
  ss << '\n';
  return ss.str();
}
void Benchmark::PersistToFile(const std::string &path) const {
  std::ofstream file(path, std::ios::app);
  PADDLE_ENFORCE(file.is_open(), "Can not open %s to add benchmark", path);
  file << SerializeToString();
  file.flush();
  file.close();
}

}  // namespace inference
}  // namespace paddle
