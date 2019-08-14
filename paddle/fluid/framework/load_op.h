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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace framework {

void LoadOpLib(const std::string &dso_name) {
  void *op_dso_handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpInfoMap *get_op_info_t();
  get_op_info_t *get_op_info = reinterpret_cast<get_op_info_t *>(
      dlsym(op_dso_handle, "PD_GetAllOpProtos"));
  auto *op_info = get_op_info();
  auto *other_info_map = op_info->mutable_map();

  typedef std::vector<std::string> grad_op_desc_maker_t(
      const OpDesc &, const std::unordered_set<std::string> &,
      std::unordered_map<std::string, std::string> *,
      const std::vector<BlockDesc *> &);

  grad_op_desc_maker_t *grad_op_desc_maker =
      reinterpret_cast<grad_op_desc_maker_t *>(
          dlsym(op_dso_handle, "PD_GetGradOpDescStrs"));

  auto *cur_info = OpInfoMap::Instance();
  for (const auto &n : *(other_info_map)) {
    VLOG(1) << "Load OP " << n.first;
    auto type = n.first;
    if (type == "recurrent" || type == "recurrent_grad" ||
        type == "conditional_block" || type == "conditional_block_grad") {
      continue;
    }
    if (cur_info->Has(n.first)) {
      PADDLE_THROW("Op %s has been registered.");
    }
    VLOG(1) << "Inser OP " << n.first;
    OpInfo info;
    info.creator_ = n.second.creator_;
    info.grad_op_maker_ = [grad_op_desc_maker](
        const OpDesc &op_desc,
        const std::unordered_set<std::string> &no_grad_set,
        std::unordered_map<std::string, std::string> *grad_to_var,
        const std::vector<BlockDesc *> &grad_block) {
      std::vector<std::string> strs =
          grad_op_desc_maker(op_desc, no_grad_set, grad_to_var, grad_block);
      std::vector<std::unique_ptr<OpDesc>> ret;
      for (auto &str : strs) {
        proto::OpDesc proto_desc;
        PADDLE_ENFORCE(proto_desc.ParseFromString(str),
                       "Failed to parse OpDesc from string");
        ret.emplace_back(new OpDesc(proto_desc, nullptr));
      }
      return ret;
    };
    info.proto_ = n.second.proto_;
    info.checker_ = n.second.checker_;
    info.infer_var_type_ = n.second.infer_var_type_;
    info.infer_shape_ = n.second.infer_shape_;
    info.infer_inplace_ = n.second.infer_inplace_;
    info.infer_no_need_buffer_vars_ = n.second.infer_no_need_buffer_vars_;
    info.use_default_grad_op_desc_maker_ =
        n.second.use_default_grad_op_desc_maker_;

    cur_info->Insert(type, info);
  }

  typedef void init_device(platform::DeviceContextPool *);
  init_device *init_dev = reinterpret_cast<init_device *>(
      dlsym(op_dso_handle, "PD_InitDevicesPool"));
  init_dev(&(platform::DeviceContextPool::Instance()));
}

}  // namespace framework
}  // namespace paddle
