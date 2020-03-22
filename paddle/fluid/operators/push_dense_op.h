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
#include <vector>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
void PushDenseFunctor(const framework::ExecutionContext& ctx) {
#ifdef PADDLE_WITH_PSLIB
  const auto& inputs = ctx.MultiInput<framework::LoDTensor>("DenseGrads");
  const auto& input_names = ctx.Attr<std::vector<std::string>>("InputNames");
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  float scale_datanorm = ctx.Attr<float>("ScaleDataNorm");
  std::vector<paddle::ps::Region> regions;
  CHECK(inputs.size() == input_names.size());

  const auto& ids = ctx.MultiInput<framework::LoDTensor>("Ids");
  int batch_size =
      ids[0]->lod().size() ? ids[0]->lod()[0].size() - 1 : ids[0]->dims()[0];
  CHECK_GT(batch_size, 0);  
  
  for (size_t index = 0; index < inputs.size(); ++index) {
    framework::LoDTensor* tensor = const_cast<framework::LoDTensor*>(inputs[index]);
    T* g = tensor->data<T>();
    size_t count = tensor->numel();
    const std::string& name = input_names[index];
    if (scale_datanorm >= 0.0f) {
      if (name.find(".batch_size@GRAD") != std::string::npos ||
          name.find(".batch_sum@GRAD") != std::string::npos) {
        Eigen::Map<Eigen::MatrixXf> mat(g, 1, count);
        float scale = 1.0 / batch_size;
        mat *= scale;
      }
    } else if (name.find(".batch_square_sum@GRAD") != std::string::npos) {
      for (int i = 0; i < count; ++i) {
        g[i] = (g[i] - batch_size * scale_datanorm) / batch_size +
               batch_size * scale_datanorm;
      }
    }
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }

  CHECK(framework::FleetWrapper::pslib_ptr_ != nullptr);
  auto status = framework::FleetWrapper::pslib_ptr_->_worker_ptr->push_dense(
      regions.data(), regions.size(), table_id);

  // not use GetInstance() here, because it's not thread-safe,
  // s_instance_ should be already initialized in DistMultiTrainer
  auto pull_dense_worker = framework::PullDenseWorker::s_instance_;
  CHECK(pull_dense_worker != nullptr);
  int thread_id = pull_dense_worker->GetThreadIdByScope(&ctx.scope());
  pull_dense_worker->IncreaseThreadVersion(thread_id, table_id);
#endif
}

template <typename T>
class PushDenseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushDenseFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
