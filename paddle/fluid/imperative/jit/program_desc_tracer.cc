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

#include "paddle/fluid/imperative/jit/program_desc_tracer.h"
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace paddle {
namespace imperative {
namespace jit {

void ProgramDescTracer::SetNamePrefix(const std::string &name_prefix) {
  name_prefix_ = name_prefix;
}

void ProgramDescTracer::SetFeedVars(
    const std::vector<std::shared_ptr<VarBase>> &feed_vars,
    std::vector<std::string> feed_names) {
  feed_vars_.clear();

  if (feed_names.empty()) {
    feed_names.reserve(feed_vars.size());
    for (auto &var : feed_vars) {
      feed_names.emplace_back(var->Name());
    }
  }

  PADDLE_ENFORCE_EQ(feed_names.size(), feed_vars.size(),
                    "The feeded variable names number must be equal to the "
                    "feeded variable number");

  for (size_t i = 0; i < feed_names.size(); ++i) {
    feed_vars_[feed_vars[i]] = feed_names[i];
  }
}

void ProgramDescTracer::SetFetchVars(
    const std::vector<std::shared_ptr<VarBase>> &fetch_vars,
    std::vector<std::string> fetch_names) {
  fetch_vars_.clear();

  if (fetch_names.empty()) {
    fetch_names.reserve(fetch_vars.size());
    for (auto &var : fetch_vars) {
      fetch_names.emplace_back(var->Name());
    }
  }

  PADDLE_ENFORCE_EQ(fetch_names.size(), fetch_vars.size(),
                    "The fetched variable names number must be equal to the "
                    "fetched variable number");
  for (size_t i = 0; i < fetch_names.size(); ++i) {
    fetch_vars_[fetch_vars[i]] = fetch_names[i];
  }
}

void ProgramDescTracer::InsertOp(const std::string &type,
                                 const NameVarBaseMap &inputs,
                                 const NameVarBaseMap &outputs,
                                 const framework::AttributeMap &attrs) {
  ops_.emplace_back(new OpDescMeta(type, inputs, outputs, attrs));
  auto &new_op = ops_.back();
  for (auto &pair : new_op->Inputs()) {
    for (auto &var : pair.second) {
      InsertVarIfNotExist(var.lock());
    }
  }

  for (auto &pair : new_op->Outputs()) {
    for (auto &var : pair.second) {
      InsertVarIfNotExist(var.lock());
    }
  }
}

std::unique_ptr<framework::ProgramDesc> ProgramDescTracer::CreateProgramDesc()
    const {
  std::unique_ptr<framework::ProgramDesc> prog(new framework::ProgramDesc());
  auto *block = prog->MutableBlock(0);

  size_t var_num = vars_.size();
  std::vector<framework::VarDesc *> var_descs(var_num, nullptr);
  std::unordered_map<framework::VarDesc *, std::weak_ptr<VarBase>>
      var_desc_to_var_base;

  for (auto &pair : vars_) {
    size_t var_id = pair.second.first;
    PADDLE_ENFORCE_LT(var_id, var_num);
    var_descs[var_id] = pair.second.second.get();
    PADDLE_ENFORCE_NOT_NULL(var_descs[var_id]);
    var_desc_to_var_base[var_descs[var_id]] = pair.first;
  }

  std::unordered_set<std::string> existing_var_names;
  for (auto *var_desc : var_descs) {
    if (var_desc->Persistable()) {
      existing_var_names.insert(var_desc->Name());
    }
  }

  for (auto &pair : feed_vars_) {
    existing_var_names.insert(pair.second);
  }

  for (auto &pair : fetch_vars_) {
    existing_var_names.insert(pair.second);
  }

  size_t counter = 0;
  auto generate_unique_name = [&]() -> std::string {
    do {
      auto name = name_prefix_ + std::to_string(counter++);
      if (existing_var_names.count(name) == 0) {
        existing_var_names.insert(name);
        return name;
      }
    } while (counter > 0);
    PADDLE_THROW("Too many vars in the program");
  };

  std::map<std::weak_ptr<VarBase>, std::string,
           std::owner_less<std::weak_ptr<VarBase>>>
      var_to_name;
  for (auto *var_desc : var_descs) {
    auto var_name = var_desc->Name();
    PADDLE_ENFORCE_EQ(var_desc_to_var_base.count(var_desc), 1);
    std::weak_ptr<VarBase> var_base = var_desc_to_var_base.at(var_desc);
    if (feed_vars_.count(var_base) > 0) {
      var_name = feed_vars_.at(var_base);
    } else if (fetch_vars_.count(var_base) > 0) {
      var_name = fetch_vars_.at(var_base);
    } else if (!var_desc->Persistable()) {
      var_name = generate_unique_name();
    }

    auto *new_var_desc = block->Var(var_name);
    *new_var_desc = *var_desc;
    new_var_desc->SetName(std::move(var_name));
    var_to_name[var_base] = new_var_desc->Name();
  }

  for (auto &op : ops_) {
    auto *op_desc = block->AppendOp();
    op_desc->SetType(op->Type());
    op_desc->SetAttrMap(op->Attrs());

    for (auto &pair : op->Inputs()) {
      std::vector<std::string> names;
      names.reserve(pair.second.size());
      for (auto &var : pair.second) {
        auto iter = var_to_name.find(var);
        PADDLE_ENFORCE_EQ(iter != var_to_name.end(), true,
                          "Cannot find input variable");
        names.emplace_back(iter->second);
      }

      op_desc->SetInput(pair.first, std::move(names));
    }

    for (auto &pair : op->Outputs()) {
      std::vector<std::string> names;
      names.reserve(pair.second.size());
      for (auto &var : pair.second) {
        auto iter = var_to_name.find(var);
        PADDLE_ENFORCE_EQ(iter != var_to_name.end(), true,
                          "Cannot find output variable");
        names.emplace_back(iter->second);
      }

      op_desc->SetOutput(pair.first, std::move(names));
    }
  }

  prog->Flush();
  return prog;
}

void ProgramDescTracer::InsertVarIfNotExist(
    const std::shared_ptr<VarBase> &new_var) {
  PADDLE_ENFORCE_NOT_NULL(new_var);
  if (vars_.count(new_var) != 0) return;

  size_t var_id = vars_.size();
  auto new_var_desc = new framework::VarDesc("");
  vars_[new_var] =
      std::make_pair(var_id, std::unique_ptr<framework::VarDesc>(new_var_desc));

  if (new_var->Persistable()) {
    new_var_desc->SetName(new_var->Name());
    new_var_desc->SetPersistable(true);
  } else {
    new_var_desc->SetPersistable(false);
  }

  const auto &inner_var = new_var->Var();
  PADDLE_ENFORCE_EQ(inner_var.IsInitialized(), true);
  if (inner_var.IsType<framework::LoDTensor>()) {
    const auto &tensor = inner_var.Get<framework::LoDTensor>();
    new_var_desc->SetType(framework::proto::VarType::LOD_TENSOR);
    new_var_desc->SetShape(framework::vectorize<int64_t>(tensor.dims()));
    new_var_desc->SetLoDLevel(tensor.lod().size());
    if (tensor.IsInitialized()) {
      new_var_desc->SetDataType(tensor.type());
    } else {
      new_var_desc->SetDataType(framework::proto::VarType::FP32);
    }
  } else {
    PADDLE_THROW("Not support variable type %s",
                 framework::ToTypeName(inner_var.Type()));
  }
}

void ProgramDescTracer::Reset() {
  ops_.clear();
  vars_.clear();
  feed_vars_.clear();
  fetch_vars_.clear();
  name_prefix_.clear();
}

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
