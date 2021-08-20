// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <string>

#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

paddle::framework::ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();

  paddle::framework::ProgramDesc program_desc(buffer);
  return program_desc;
}

int main() {
  paddle::framework::InitDevices();
  auto place = paddle::platform::CUDAPlace(0);
  auto test_prog = load_from_file("lm_startup_program");

  auto main_prog = load_from_file("lm_main_program");

  paddle::framework::Scope scope;
  paddle::framework::StandaloneExecutor exec(place, test_prog, main_prog,
                                             &scope);

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < 2320; ++i) {
    if (i % 200 == 0) {
      std::cout << i << std::endl;
    }

    std::vector<paddle::framework::Tensor> vec_out;
    exec.Run({}, {}, {}, &vec_out);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "time cost " << diff.count() << std::endl;

  return 1;
}
