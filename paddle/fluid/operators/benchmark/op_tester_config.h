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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {
namespace benchmark {

struct OpInputConfig {
  OpInputConfig() {}
  explicit OpInputConfig(std::istream& is);

  void ParseDType(std::istream& is);
  void ParseInitializer(std::istream& is);
  void ParseDims(std::istream& is);
  void ParseLoD(std::istream& is);

  std::string name;
  std::string dtype{"fp32"};  // int32/int, int64/long, fp32/float, fp64/double
  std::string initializer{"random"};  // random, natural
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

static bool Has(const std::vector<std::string>& vec, const std::string& item) {
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == item) {
      return true;
    }
  }
  return false;
}

template <typename T>
T StringTo(const std::string& str) {
  std::istringstream is(str);
  T value;
  is >> value;
  return value;
}

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
