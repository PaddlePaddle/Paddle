/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <memory>
#include <vector>
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/proto_desc.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

class ProgramDesc;

class InferenceDesc {
 public:
  InferenceDesc();

  explicit InferenceDesc(const ProgramDesc &prog,
                         const std::vector<std::string> &feed_var_names,
                         const std::vector<std::string> &fetch_var_names);

  proto::InferenceDesc *Proto();

 private:
  proto::InferenceDesc desc_;
  const ProgramDesc *prog_;  // not_own
  std::vector<std::string> _feed_var_names;
  std::vector<std::string> _fetch_var_names;
};
}  // namespace framework
}  // namespace paddle
