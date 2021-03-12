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

namespace paddle {
namespace imperative {
class VarBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace imperative {
namespace jit {

// A helper class to generate unique name for each non-persistable var
class UniqueBlockVarGenerator {
 public:
  UniqueBlockVarGenerator(const VarDescMetaMap &all_vars,
                          const VarBaseSet &non_exist_input_vars,
                          framework::BlockDesc *block);

  std::string NameOf(const std::weak_ptr<VarBase> &var,
                     const std::string &prefix);

 private:
  void InsertNewVarInBlock(const std::weak_ptr<VarBase> &var,
                           const framework::VarDesc &ref_desc,
                           const std::string &name,
                           bool force_persistable = false);

 private:
  const VarDescMetaMap &all_vars_;
  framework::BlockDesc *block_;
  std::unordered_map<std::string, size_t> counter_;

  std::map<std::weak_ptr<VarBase>, std::string,
           std::owner_less<std::weak_ptr<VarBase>>>
      var_to_name_;
  std::unordered_set<std::string> existing_names_;
};

UniqueBlockVarGenerator::UniqueBlockVarGenerator(
    const VarDescMetaMap &all_vars, const VarBaseSet &non_exist_input_vars,
    framework::BlockDesc *block)
    : all_vars_(all_vars), block_(block) {
  for (auto &var_pair : all_vars_) {
    auto *var_desc = var_pair.second.get();
    if (var_desc->Persistable()) {
      InsertNewVarInBlock(var_pair.first, *var_desc, var_desc->Name());
    } else if (non_exist_input_vars.count(var_pair.first.lock()) > 0) {
      VLOG(10) << "Mark " << var_desc->Name() << " as persistable";
      InsertNewVarInBlock(var_pair.first, *var_desc, var_desc->Name(),
                          /*force_persistable=*/true);
    }
  }
}

std::string UniqueBlockVarGenerator::NameOf(const std::weak_ptr<VarBase> &var,
                                            const std::string &prefix) {
  VLOG(3) << "Finding: " << var.lock()->Name();
  auto all_vars_iter = all_vars_.find(var);
  PADDLE_ENFORCE_EQ(all_vars_iter != all_vars_.end(), true,
                    platform::errors::NotFound(
                        "Variable is not found in UniqueBlockVarGenerator"));

  auto iter = var_to_name_.find(var);
  if (iter != var_to_name_.end()) {
    VLOG(5) << "Return existing var name " << iter->second;
    return iter->second;
  } else {
    auto generate_unique_name = [this, &prefix] {
      auto &cnt = counter_[prefix];
      do {
        auto name = prefix + std::to_string(cnt++);
        if (existing_names_.count(name) == 0) {
          return name;
        }
      } while (cnt > 0);
      PADDLE_THROW(
          platform::errors::OutOfRange("Too many vars in the program"));
    };

    auto unique_name = generate_unique_name();
    VLOG(5) << "Generate new var name " << unique_name;
    InsertNewVarInBlock(var, *(all_vars_iter->second), unique_name);
    return unique_name;
  }
}

void UniqueBlockVarGenerator::InsertNewVarInBlock(
    const std::weak_ptr<VarBase> &var, const framework::VarDesc &var_desc,
    const std::string &name, bool force_persistable) {
  var_to_name_[var] = name;
  existing_names_.insert(name);
  auto *new_var_desc = block_->Var(name);
  *new_var_desc = var_desc;
  new_var_desc->SetName(name);
  if (force_persistable) {
    new_var_desc->SetPersistable(true);
  }
}

bool ProgramDescTracer::ContainVar(const std::weak_ptr<VarBase> &var) const {
  auto vars_iter = vars_.find(var);
  bool ret = (vars_iter != vars_.end());
  if (!ret) {
    VLOG(5) << "Can't found variable: " << var.lock()->Name();
  }
  return ret;
}

void ProgramDescTracer::InsertOp(const std::string &type,
                                 const NameVarBaseMap &inputs,
                                 const NameVarBaseMap &outputs,
                                 const framework::AttributeMap &attrs) {
  ops_.emplace_back(new OpDescMeta(type, inputs, outputs, attrs));
  auto &new_op = ops_.back();
  for (auto &pair : new_op->Inputs()) {
    for (auto &var : pair.second) {
      InsertVarIfNotExist(var.lock(), true);
    }
  }

  for (auto &pair : new_op->Outputs()) {
    for (auto &var : pair.second) {
      InsertVarIfNotExist(var.lock(), false);
    }
  }
}

TracedProgramTuple ProgramDescTracer::CreateProgramDesc(
    const std::vector<std::shared_ptr<VarBase>> &feed_vars,
    const std::string &feed_prefix,
    const std::vector<std::shared_ptr<VarBase>> &fetch_vars,
    const std::string &fetch_prefix, const std::string &tmp_prefix) const {
  std::unique_ptr<framework::ProgramDesc> prog(new framework::ProgramDesc());
  auto *block = prog->MutableBlock(0);

  auto non_exist_vars_copy = non_exist_input_vars_;
  for (auto &feed_var : feed_vars) {
    non_exist_vars_copy.erase(feed_var);
  }

  UniqueBlockVarGenerator generator(vars_, non_exist_vars_copy, block);

  std::vector<std::string> feed_var_names;
  for (auto &feed_var : feed_vars) {
    if (ContainVar(feed_var)) {
      feed_var_names.emplace_back(generator.NameOf(feed_var, feed_prefix));
    }
  }

  std::vector<std::string> fetch_var_names;
  for (auto &fetch_var : fetch_vars) {
    if (ContainVar(fetch_var)) {
      fetch_var_names.emplace_back(generator.NameOf(fetch_var, fetch_prefix));
    }
  }

  for (auto &op : ops_) {
    auto *op_desc = block->AppendOp();
    op_desc->SetType(op->Type());
    op_desc->SetAttrMap(op->Attrs());

    for (auto &pair : op->Inputs()) {
      std::vector<std::string> names;
      names.reserve(pair.second.size());
      for (auto &var : pair.second) {
        if (ContainVar(var)) {
          names.emplace_back(generator.NameOf(var, tmp_prefix));
        }
      }

      op_desc->SetInput(pair.first, std::move(names));
    }

    for (auto &pair : op->Outputs()) {
      std::vector<std::string> names;
      names.reserve(pair.second.size());
      for (auto &var : pair.second) {
        if (ContainVar(var)) {
          names.emplace_back(generator.NameOf(var, tmp_prefix));
        }
      }

      op_desc->SetOutput(pair.first, std::move(names));
    }
  }

  prog->Flush();

  std::vector<std::shared_ptr<VarBase>> persistable_vars(
      non_exist_vars_copy.begin(), non_exist_vars_copy.end());
  for (auto &pair : vars_) {
    if (pair.second->Persistable()) {
      auto var = pair.first.lock();
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound("Persistable var %s does not exist",
                                          pair.second->Name()));
      persistable_vars.emplace_back(var);
    }
  }
  return std::make_tuple(std::move(prog), std::move(feed_var_names),
                         std::move(fetch_var_names),
                         std::move(persistable_vars));
}

void ProgramDescTracer::InsertVarIfNotExist(
    const std::shared_ptr<VarBase> &new_var, bool is_input) {
  PADDLE_ENFORCE_NOT_NULL(new_var, platform::errors::InvalidArgument(
                                       "The variable to insert is NULL."));
  if (vars_.count(new_var) != 0) return;

  auto new_var_desc = new framework::VarDesc("");
  vars_[new_var].reset(new_var_desc);

  if (new_var->Persistable() || is_input) {
    new_var_desc->SetName(new_var->Name());
    new_var_desc->SetPersistable(new_var->Persistable());
    if (!new_var->Persistable()) {
      non_exist_input_vars_.insert(new_var);
    }
  } else {
    new_var_desc->SetPersistable(false);
  }

  const auto &inner_var = new_var->Var();
  PADDLE_ENFORCE_EQ(inner_var.IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "The variable to insert is not initialized."));
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
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Not support variable type %s.",
        framework::ToTypeName(inner_var.Type())));
  }
}

void ProgramDescTracer::Reset() {
  ops_.clear();
  vars_.clear();
  non_exist_input_vars_.clear();
}

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
