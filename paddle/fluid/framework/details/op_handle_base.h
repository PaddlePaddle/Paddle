//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/platform/device_context.h"
namespace paddle {
namespace framework {
namespace details {

struct OpHandleBase {
  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::unordered_map<platform::Place, platform::DeviceContext *,
                     platform::PlaceHash>
      dev_ctx_;

#ifdef PADDLE_WITH_CUDA
  std::unordered_map<int, cudaEvent_t> events_;
#endif

  std::string DebugString() const;

  virtual std::string Name() const = 0;

  virtual ~OpHandleBase();

  void Run(bool use_event);

  virtual void Wait(platform::DeviceContext *waited_dev);

  void AddInput(VarHandleBase *in);

  void AddOutput(VarHandleBase *out);

 protected:
  virtual void RunImpl() = 0;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
