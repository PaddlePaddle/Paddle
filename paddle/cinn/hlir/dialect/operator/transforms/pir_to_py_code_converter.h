// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace pir {

class Program;

}

namespace cinn::dialect::ir {

class PirToPyCodeConverter {
 public:
  explicit PirToPyCodeConverter(const pir::Program* program)
      : program_(program), file_name_(), dump_symbolic_shape_(true) {}
  PirToPyCodeConverter(const PirToPyCodeConverter&) = delete;
  PirToPyCodeConverter(PirToPyCodeConverter&&) = delete;

  PirToPyCodeConverter& file_name(const std::string& file_name) {
    file_name_ = file_name;
    return *this;
  }

  PirToPyCodeConverter& dump_symbolic_shape(bool val) {
    dump_symbolic_shape_ = val;
    return *this;
  }

  void SaveIfFlagEnabled() const;

 private:
  const pir::Program* program_;
  std::string file_name_;
  bool dump_symbolic_shape_;
};

}  // namespace cinn::dialect::ir
