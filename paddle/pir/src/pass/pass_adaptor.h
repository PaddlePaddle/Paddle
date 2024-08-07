// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/pass/pass.h"

namespace pir {

class Operation;
class PassManager;

namespace detail {
// Used to run operation passes over nested operations.
class PassAdaptor final : public Pass {
 public:
  explicit PassAdaptor(PassManager* pm) : Pass("pass_adaptor", 0), pm_(pm) {}

  void Run(Operation*) override {}

  void Run(Operation*, uint8_t opt_level, bool verify);

 private:
  void RunImpl(Operation* op, uint8_t opt_level, bool verify);

  static bool RunPass(Pass* pass,
                      Operation* op,
                      AnalysisManager am,
                      uint8_t opt_level,
                      bool verify);

  static bool RunPipeline(const PassManager& pm,
                          Operation* op,
                          AnalysisManager am,
                          uint8_t opt_level,
                          bool verify);

 private:
  PassManager* pm_;

  // For accessing RunPipeline.
  friend class pir::PassManager;
};
}  // namespace detail

}  // namespace pir
