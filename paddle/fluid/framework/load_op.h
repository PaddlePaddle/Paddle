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
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace framework {

void LoadOpLib(const std::string &dso_name) {
  VLOG(1) << "Start to load op...";
  void* op_dso_handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpInfoMap* get_op_info_t();

  VLOG(1) << "Start to dlsym...";
  get_op_info_t* get_op_info = (get_op_info_t*)dlsym(op_dso_handle, "PD_GetAllOpProtos");
  auto* op_info = get_op_info();
  auto* other_info_map = op_info->mutable_map();

  auto* cur_info = OpInfoMap::Instance();
  for( const auto& n : *(other_info_map)) {
    VLOG(1) << "Load OP INFO " << n.first;
    if (!cur_info->Has(n.first)) {
      cur_info->Insert(n.first, n.second);
    }
  }

  typedef void init_device(bool);
  init_device* init_dev = (init_device*)dlsym(op_dso_handle, "PD_InitDevices");
  init_dev(true);
  VLOG(1) << "Stop to load op.";
}

}  // namespace framework
}  // namespace paddle
