// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/utils/registry.h"

namespace cinn {
namespace frontend {

class ProgramPass {
 public:
  explicit ProgramPass(const std::string& name) : name_(name) {}

  /**
   * \brief Apply a sequence of passes on a program.
   * @param prog The input program to apply passes on.
   * @param passes The sequence of pass.
   * @return The program after being modified by the passes.
   */
  static void Apply(Program* prog,
                    const std::unordered_set<std::string>& fetch_ids,
                    const common::Target& target,
                    const std::vector<std::string>& passes);

  const std::string& name() const { return name_; }

 protected:
  virtual void ApplyImpl(Program* prog,
                         const std::unordered_set<std::string>& fetch_ids,
                         const common::Target& target) {}
  virtual void ApplyImpl(Program* prog,
                         const std::unordered_set<std::string>& fetch_ids,
                         const common::Target& target) const {
    return const_cast<ProgramPass*>(this)->ApplyImpl(prog, fetch_ids, target);
  }

  virtual void Clear() = 0;

 private:
  std::string name_;
};

class ProgramPassRegistry : public Registry<ProgramPass> {
 public:
  static ProgramPassRegistry* Global() {
    static ProgramPassRegistry x;
    return &x;
  }

  inline const ProgramPass* Get(const std::string& name) {
    const ProgramPass* pass = Registry<ProgramPass>::Find(name);
    CHECK(pass) << "Pass [" << name << "] is not registered";
    return pass;
  }

  inline ProgramPass* __REGISTER__(const std::string& name, ProgramPass* pass) {
    std::lock_guard<std::mutex> guard(registering_mutex);
    if (fmap_.count(name)) {
      return fmap_[name];
    }

    fmap_[name] = pass;
    const_list_.push_back(pass);
    entry_list_.push_back(pass);
    return pass;
  }

  inline ProgramPass* __REGISTER_OR_GET__(const std::string& name,
                                          ProgramPass* pass) {
    if (!fmap_.count(name)) {
      return __REGISTER__(name, pass);
    } else {
      return fmap_.at(name);
    }
  }

 private:
  ProgramPassRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(ProgramPassRegistry);
};

/**
 * @def CINN_REGISTER_PROGRAM_PASS
 * \brief Register a new program pass
 *
 * @param PassType The type of pass
 * @param PassClass The pass inherited from ProgramPass
 *
 * \code
 *  CINN_REGISTER_PROGRAM_PASS(decompose, DecomposerPass());
 * \endcode
 */
#define CINN_REGISTER_PROGRAM_PASS(PassType, PassClass)                     \
  static ::cinn::frontend::ProgramPass* __make_##PassType##__ =             \
      ::cinn::frontend::ProgramPassRegistry::Global()->__REGISTER_OR_GET__( \
          #PassType, new PassClass{#PassType})

}  // namespace frontend
}  // namespace cinn
