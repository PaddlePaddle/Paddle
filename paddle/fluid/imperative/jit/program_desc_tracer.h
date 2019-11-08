// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/imperative/jit/op_desc_meta.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {
namespace jit {

class ProgramDescTracer {
  DISABLE_COPY_AND_ASSIGN(ProgramDescTracer);

 public:
  ProgramDescTracer() = default;

  void SetNamePrefix(const std::string &name_prefix);

  void SetFeedVars(const std::vector<std::shared_ptr<VarBase>> &feed_vars,
                   std::vector<std::string> feed_names);

  void SetFetchVars(const std::vector<std::shared_ptr<VarBase>> &fetch_vars,
                    std::vector<std::string> fetch_names);

  void InsertOp(const std::string &type, const NameVarBaseMap &inputs,
                const NameVarBaseMap &outputs,
                const framework::AttributeMap &attrs);

  std::unique_ptr<framework::ProgramDesc> CreateProgramDesc() const;

  void Reset();

 private:
  void InsertVarIfNotExist(const std::shared_ptr<VarBase> &new_var);

  std::vector<std::unique_ptr<OpDescMeta>> ops_;

  std::map<std::weak_ptr<VarBase>,
           std::pair<size_t, std::unique_ptr<framework::VarDesc>>,
           std::owner_less<std::weak_ptr<VarBase>>>
      vars_;

  // The following fields are used to polish the converted ProgramDesc
  std::map<std::weak_ptr<VarBase>, std::string,
           std::owner_less<std::weak_ptr<VarBase>>>
      feed_vars_;

  std::map<std::weak_ptr<VarBase>, std::string,
           std::owner_less<std::weak_ptr<VarBase>>>
      fetch_vars_;

  std::string name_prefix_;
};

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
