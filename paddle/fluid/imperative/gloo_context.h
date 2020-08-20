//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int rank{0};
  int rank_num{0};
  std::string iface;
  std::string prefix;
  int init_seconds{9999999};
  int run_seconds{9999999};
  std::string path;
  std::string fs_name;
  std::string fs_ugi;
};

class ParallelContext {
 public:
  explicit ParallelContext(const ParallelStrategy& strategy)
      : strategy_(strategy) {}

  virtual ~ParallelContext() {}

  virtual void Init() = 0;

 protected:
  ParallelStrategy strategy_;
};

#if defined(PADDLE_WITH_GLOO)
class GLOOParallelContext : ParallelContext {
 public:
  explicit GLOOParallelContext(const ParallelStrategy& strategy)
      : ParallelContext(strategy) {}

  ~GLOOParallelContext() {}

  void Init() override;
};
#endif

}  //  namespace imperative
}  //  namespace paddle
