/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/ps/service/communicator/communicator_common.h"
#include "paddle/fluid/distributed/ps/service/ps_service/service.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {
class Scope;
class SelectedRows;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

class PSCore;

using framework::LoDTensor;
using framework::Scope;
using phi::SelectedRows;
using framework::Variable;

using RpcCtxMap = std::unordered_map<std::string, CommContext>;

struct WrapperContext {
  uint32_t table_id;
  const std::string path;
  const int mode;
  const std::string meta;
};

struct InitContext {
  const std::vector<int> dev_ids;  // for gpu
};

class PSWrapper {
 public:
  virtual ~PSWrapper() {}
  PSWrapper() {}
  // init server

  virtual int32_t Initialize(InitContext& context) = 0;

  virtual void Stop() = 0;

  virtual void Load(WrapperContext& context) = 0;

  virtual void Save(WrapperContext& context) = 0;
};

}  // end namespace distributed
}  // end namespace paddle
