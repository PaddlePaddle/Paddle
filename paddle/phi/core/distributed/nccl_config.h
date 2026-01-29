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
#pragma once

#include <memory>
#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
namespace distributed {

class NCCLConfig {
 public:
  static std::shared_ptr<NCCLConfig> CreateNCCLConfig(
#ifdef NCCL_HAS_CONFIG
      const int blocking,
      const int cga_cluster_size,
      const int min_ctas,
      const int max_ctas
#endif
  );

  NCCLConfig(
#ifdef NCCL_HAS_CONFIG
      const int blocking,
      const int cga_cluster_size,
      const int min_ctas,
      const int max_ctas
#endif
  );
  ~NCCLConfig();

 public:
  ncclConfig_t* GetOrigin();

 private:
  ncclConfig_t* nccl_config_ptr{nullptr};
};

}  // namespace distributed
}  // namespace phi
