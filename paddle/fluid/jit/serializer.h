// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace jit {
class Layer;
class IValue;

// Export Layer into local disk
class Serializer {
 public:
  void operator()(const Layer& layer, const std::string& file_dir);

 private:
  void WriteTensorData(const Layer& layer, const std::string& file_name) const;
  void WriteExtraInfo(const Layer& layer, const std::string& file_name) const;
  void WriteByteCode(const Layer& layer, const std::string& file_name) const;
};

class Deserializer {
 public:
  Layer operator()(const std::string& file_dir);

 private:
  IValue ReadTensorData(const std::string& file_name) const;
  void ReadExtraInfo(const std::string& file_name) const;
  void ReadByteCode(const std::string& file_name) const;
};

void Export(const Layer& layer, const std::string& file_path);

Layer Load(const std::string& file_path);

}  // namespace jit
}  // namespace paddle
