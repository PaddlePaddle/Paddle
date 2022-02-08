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
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/imperative/jit/op_desc_meta.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {
class VarBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace imperative {
namespace jit {

using VarDescMetaMap =
    std::map<std::weak_ptr<VarBase>, std::unique_ptr<framework::VarDesc>,
             std::owner_less<std::weak_ptr<VarBase>>>;

using VarBaseSet = std::set<std::shared_ptr<VarBase>,
                            std::owner_less<std::shared_ptr<VarBase>>>;

using TracedProgramTuple =
    std::tuple<std::unique_ptr<framework::ProgramDesc> /*program*/,
               std::vector<std::string> /*feed_var_names*/,
               std::vector<std::string> /*fetch_var_names*/,
               std::vector<std::shared_ptr<VarBase>> /*persistable_vars*/>;

class ProgramDescTracer {
  DISABLE_COPY_AND_ASSIGN(ProgramDescTracer);

 public:
  ProgramDescTracer() = default;

  void InsertOp(const std::string &type, const NameVarBaseMap &inputs,
                const NameVarBaseMap &outputs,
                const framework::AttributeMap &attrs);

  void InsertOp(const std::string &type, const NameTensorMap &inputs,
                const NameTensorMap &outputs,
                const framework::AttributeMap &attrs);

  TracedProgramTuple CreateProgramDesc(
      const std::vector<std::shared_ptr<VarBase>> &feed_vars,
      const std::string &feed_prefix,
      const std::vector<std::shared_ptr<VarBase>> &fetch_vars,
      const std::string &fetch_prefix, const std::string &tmp_prefix) const;
  bool ContainVar(const std::weak_ptr<VarBase> &var) const;
  void Reset();

 private:
  void InsertVarIfNotExist(const std::shared_ptr<VarBase> &new_var,
                           bool is_input);

 private:
  std::vector<std::unique_ptr<OpDescMeta>> ops_;
  VarDescMetaMap vars_;
  VarBaseSet non_exist_input_vars_;
};

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
