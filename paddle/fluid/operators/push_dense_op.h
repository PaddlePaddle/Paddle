//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
void PushDenseFunctor(const framework::ExecutionContext& ctx) {
#ifdef PADDLE_WITH_PSLIB
  const auto& input_names = ctx.Attr<std::vector<std::string>>("InputNames");
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  PADDLE_ENFORCE_GT(table_id,
                    0,
                    common::errors::InvalidArgument(
                        "table id should > 0, but value is ", table_id));
  float scale_datanorm = ctx.Attr<float>("ScaleDataNorm");
  const auto& ids = ctx.MultiInput<phi::DenseTensor>("Ids");
  int batch_size =
      ids[0]->lod().size() ? ids[0]->lod()[0].size() - 1 : ids[0]->dims()[0];
  PADDLE_ENFORCE_GT(batch_size,
                    0,
                    common::errors::InvalidArgument(
                        "batch size should > 0, but value is ", batch_size));

  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  fleet_ptr->PushDenseVarsAsync(
      ctx.scope(), table_id, input_names, nullptr, scale_datanorm, batch_size);

  // note: GetInstance() is not thread-safe
  // we assume PullDenseWorker has been already initialized in DistMultiTrainer
  auto pull_dense_worker = framework::PullDenseWorker::GetInstance();
  PADDLE_ENFORCE_NE(pull_dense_worker,
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "pull_dense_worker should not be null"));
  int thread_id = pull_dense_worker->GetThreadIdByScope(&ctx.scope());
  pull_dense_worker->IncreaseThreadVersion(thread_id, table_id);
#endif
}

template <typename T, typename DeviceContext>
class PushDenseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PushDenseFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
