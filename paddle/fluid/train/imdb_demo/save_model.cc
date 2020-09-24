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

#include "include/save_model.h"
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"

using std::unique_ptr;

namespace paddle {
namespace framework {
void save_model(const unique_ptr<ProgramDesc>& main_program, Scope* scope,
                const std::vector<std::string>& param_names,
                const std::string& model_name, bool save_combine) {
  auto place = platform::CPUPlace();
  const BlockDesc& global_block = main_program->Block(0);
  std::vector<std::string> paralist;
  for (auto* var : global_block.AllVars()) {
    bool is_model_param = false;
    for (auto param_name : param_names) {
      if (var->Name() == param_name) {
        is_model_param = true;
        break;
      }
    }

    if (!is_model_param) continue;

    if (!save_combine) {
      VLOG(3) << "model var name: %s" << var->Name().c_str();

      paddle::framework::AttributeMap attrs;
      attrs.insert({"file_path", model_name + "/" + var->Name()});
      auto save_op = paddle::framework::OpRegistry::CreateOp(
          "save", {{"X", {var->Name()}}}, {}, attrs);

      save_op->Run(*scope, place);
    } else {
      paralist.push_back(var->Name());
    }
  }
  if (save_combine) {
    std::sort(paralist.begin(), paralist.end());
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_name});
    auto save_op = paddle::framework::OpRegistry::CreateOp(
        "save_combine", {{"X", paralist}}, {}, attrs);
    save_op->Run(*scope, place);
  }
}
}  // namespace framework
}  // namespace paddle
