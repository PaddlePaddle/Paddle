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

#include "paddle/framework/inference_desc.h"
#include "paddle/framework/program_desc.h"

namespace paddle {
namespace framework {

InferenceDesc::InferenceDesc(const ProgramDesc &prog,
                             const std::vector<std::string> &feed_var_names,
                             const std::vector<std::string> &fetch_var_names)
    : prog_(&prog),
      _feed_var_names(feed_var_names),
      _fetch_var_names(fetch_var_names) {
  desc_.set_allocated_program(const_cast<ProgramDesc *>(prog_)->Proto());
  for (auto name : _feed_var_names) {
    desc_.add_feed_var_names(name);
  }
  for (auto name : _fetch_var_names) {
    desc_.add_fetch_var_names(name);
  }
}

proto::InferenceDesc *InferenceDesc::Proto() { return &desc_; }

}  // namespace framework
}  // namespace paddle
