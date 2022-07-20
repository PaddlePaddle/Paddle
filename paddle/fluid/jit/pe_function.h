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

#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {
namespace jit {

class PEFunction : public BaseFunction {
 public:
  PEFunction(const std::shared_ptr<FunctionInfo> &info,
             const Name2VariableMap &params_dict,
             const phi::Place &place)
      : info_(info), place_(place) {
    utils::ShareParamsIntoScope(info_->ParamNames(), params_dict, &scope_);
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    info_->RemoveDescFeedFetch();
  }

  ~PEFunction() noexcept {}

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs) {
    auto dense_tensors = utils::ToDenseTensors(inputs);
    return utils::ToTensors(this->operator()(dense_tensors));
  }

  std::vector<DenseTensor> operator()(const std::vector<DenseTensor> &inputs) {
    std::string prog_string;
    std::hash<std::string> string_hash;

    auto &program_desc = info_->ProgramDesc();
    // TODO(dev): Serialize is very slow.
    const_cast<framework::ProgramDesc *>(&program_desc)
        ->Proto()
        ->SerializePartialToString(&prog_string);
    int64_t program_id = static_cast<int64_t>(string_hash(prog_string));

    const framework::BlockDesc &global_block = program_desc.Block(0);
    int64_t start_op_index = 0;
    int64_t end_op_index = static_cast<int64_t>(global_block.OpSize());

    utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);
    std::vector<std::string> input_var_names = info_->InputArgNames();
    std::vector<std::string> output_var_names = info_->OutputArgNames();

    if (end_op_index > start_op_index) {
      auto cache_info = framework::GetExecutorInfoFromCache(program_desc,
                                                            place_,
                                                            start_op_index,
                                                            end_op_index,
                                                            /*is_grad=*/false,
                                                            program_id,
                                                            &scope_);
      auto &parallel_executor = cache_info.first;
      auto &skip_eager_delete_vars =
          framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, false);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_var_names);
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      output_var_names.begin(),
                                      output_var_names.end());

        framework::details::ParseSafeEagerDeletionSkipVars(
            program_desc,
            end_op_index,
            output_var_names,
            &skip_eager_delete_vars);
      }
      parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
    }
    std::vector<DenseTensor> res;
    utils::FetchOuts(info_->OutputArgNames(), scope_, &res);
    return res;
  }

  const std::shared_ptr<FunctionInfo> &Info() const { return info_; }

 private:
  std::shared_ptr<FunctionInfo> info_;
  framework::Scope scope_;
  phi::Place place_;
};

}  // namespace jit
}  // namespace paddle
