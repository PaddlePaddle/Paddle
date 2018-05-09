/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/detail/checkpoint.h"

#include <string>

namespace paddle {
namespace framework {
namespace details {
Checkpoint::Save(const framework::Scope& scope, const platform::Place& place,
                 const std::string& save_dir, const std::string& var_name,
                 const bool overwrite) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s for save_op",
                 var_name);
  PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                 "Checkpoint only supports LoDTensor, %s has wrong type",
                 var_name);

  bool is_present = FileExists(save_dir);
  if (is_present && !overwrite) {
    PADDLE_THROW("%s exists!, checkpoint cannot write  it when overwrite=false",
                 save_dir, overwrite);
  }

  MkDirRecursively(DirName(save_dir).c_str());
  std::ofstream fout(save_dir);
  PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write", save_dir);

  // get device context from pool
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(place);

  auto& tensor = var->Get<framework::LoDTensor>();
  // Serialize tensor
  framework::SerializeToStream(fout, tensor, dev_ctx);
  fout.close();
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
