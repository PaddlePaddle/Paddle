/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/common.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/hccl_backend.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/hierarchical_backend.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

class HierarchicalHcclFactory {
 public:
  static paddle::operators::HierarchicalHccl *create(
      HierarchicalHcclInitConfig _init_config,
      std::shared_ptr<paddle::operators::rendezvous::BRPCStore> _brpc_store) {
    std::string backend = _init_config.backend;

    if (backend == "hccl-adapter") {
      return new paddle::operators::HcclBackend();
    }
    if (backend == "hierarchical") {
      return new paddle::operators::HierarchicalBackend(_init_config,
                                                        _brpc_store);
    }
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "unknown hierarchical hccl implementation backend"));
  }

  static paddle::operators::HierarchicalHccl *create(
      HierarchicalHcclLayerConfig _init_config,
      std::shared_ptr<paddle::operators::rendezvous::BRPCStore> _brpc_store) {
    std::string backend = _init_config.backend;
    if (backend == "hccl-adapter") {
      return new paddle::operators::HcclBackend();
    }
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "unknown hierarchical hccl implementation backend"));
  }

 private:
  HierarchicalHcclFactory() {}
  DISALLOW_COPY_AND_ASSIGN(HierarchicalHcclFactory);
};

}  // namespace operators
}  // namespace paddle
