// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/nccl_config.h"

#include "glog/logging.h"

namespace phi::distributed {

std::shared_ptr<NCCLConfig> NCCLConfig::CreateNCCLConfig(
#ifdef NCCL_HAS_CONFIG
    const int blocking,
    const int cga_cluster_size,
    const int min_ctas,
    const int max_ctas
#endif
) {
  return std::make_shared<NCCLConfig>(
#ifdef NCCL_HAS_CONFIG
      blocking, cga_cluster_size, min_ctas, max_ctas
#endif
  );
}

NCCLConfig::NCCLConfig(
#ifdef NCCL_HAS_CONFIG
    const int blocking,
    const int cga_cluster_size,
    const int min_ctas,
    const int max_ctas
#endif
    )
    : nccl_config_ptr(nullptr) {

#ifdef NCCL_HAS_CONFIG
  nccl_config_ptr = new ncclConfig_t;
  *nccl_config_ptr = NCCL_CONFIG_INITIALIZER;
  nccl_config_ptr->blocking = blocking;
  nccl_config_ptr->cgaClusterSize = cga_cluster_size;
  nccl_config_ptr->minCTAs = min_ctas;
  nccl_config_ptr->maxCTAs = max_ctas;
#endif
}

ncclConfig_t* NCCLConfig::GetOrigin() { return nccl_config_ptr; }

NCCLConfig::~NCCLConfig() {
  if (nccl_config_ptr) {
    delete nccl_config_ptr;
  }
}

}  // namespace phi::distributed
