// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/table/tensor_table.h"
#include "paddle/fluid/distributed/common/utils.h"

namespace paddle {
namespace distributed {

int32_t DenseTensorTable::initialize() {
  _shards_task_pool.resize(10);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }
  return 0;
}

int32_t DenseTensorTable::initialize_tensor(framework::Scope *scope,
                                            framework::ProgramDesc *program,
                                            framework::Executor *executor) {
  scope_ = scope;
  program_ = program;
  executor_ = executor;

  auto tensor_config = _config.tensor();
  if (tensor_config.has_common_block_map()) {
    auto block_maps =
        paddle::string::split_string(tensor_config.common_block_map(), "#");
    for (auto &block_map : block_maps) {
      auto block = paddle::string::split_string(block_map, ":");
      auto block_id = std::stoi(block[0]);
      std::vector<int> block_ids{block_id};
      auto block_cmd = block[1];
      auto prepared = executor_->Prepare(*program_, block_ids);
      (*prepared_ctx_)[block_cmd] = prepared[0];
    }
  }
}

int32_t DenseTensorTable::pull_dense(float *values, size_t numel) {
  PADDLE_ENFORCE_EQ(numel, _data.numel(),
                    paddle::platform::errors::PreconditionNotMet(
                        "pull dense error, excepted numel %d, but actually %d.",
                        _data.numel(), numel));

  GetBlas<float>().VCOPY(numel, _data.data<float>(), values);
  return 0;
}

int32_t DenseTensorTable::push_dense(const float *values, size_t numel) {
  auto varname = _config.tensor().grad();
  auto local_scope = scope_->NewTmpScope();
  auto *var = local_scope->Var(varname);
  auto *t = var->GetMutable<framework::LoDTensor>();
  auto dims = paddle::framework::make_ddim({});

  auto ctx = paddle::platform::CPUDeviceContext();
  t->mutable_data<float>(_data.dims(), ctx.GetPlace());

  GetBlas<float>().VCOPY(numel, values, t->data<float>());
  executor_->RunPreparedContext((*prepared_ctx_)["push"].get(),
                                local_scope.get());
}

int32_t DenseTensorTable::push_dense_param(const float *values, size_t numel) {
  auto ctx = paddle::platform::CPUDeviceContext();
  if (_data.IsInitialized()) {
    PADDLE_ENFORCE_EQ(
        numel, _data.numel(),
        paddle::platform::errors::PreconditionNotMet(
            "pull dense error, excepted numel %d, but actually %d.",
            _data.numel(), numel));
  } else {
    _data.mutable_data<float>(
        framework::make_ddim({static_cast<int64_t>(numel), 1}), ctx.GetPlace());
  }

  GetBlas<float>().VCOPY(numel, values, _data.data<float>());
  return 0;
}
}  // namespace distributed
}  // namespace paddle
