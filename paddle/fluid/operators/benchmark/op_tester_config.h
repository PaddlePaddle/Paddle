/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {
namespace benchmark {

struct OpInputConfig {
  OpInputConfig() {}
  explicit OpInputConfig(std::istream& is);

  void ParseDims(std::istream& is);
  void ParseLoD(std::istream& is);

  std::string name;
  std::vector<int64_t> dims;
  std::vector<std::vector<size_t>> lod;
};

struct OpTesterConfig {
  OpTesterConfig() {}
  explicit OpTesterConfig(const std::string& filename);

  bool Init(std::istream& is);

  bool ParseAttrs(std::istream& is);

  const OpInputConfig* GetInput(const std::string& name);

  std::string op_type;
  std::vector<OpInputConfig> inputs;
  std::unordered_map<std::string, std::string> attrs;
  int device_id{-1};  // CPU: -1
  int repeat{1};
  int profile{0};
  int print_debug_string{0};
  double runtime{0.0};
};

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
