/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

template <typename T>
T *DynLoad(void *handle, std::string name) {
  T *func = reinterpret_cast<T *>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func,
      platform::errors::NotFound(
          "Failed to load dynamic operator library, error code(%s).", errorno));
  return func;
}

void LoadOpLib(const std::string &dso_name) {
  void *handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpInfoMap &get_op_info_t();
  get_op_info_t *get_op_info =
      DynLoad<get_op_info_t>(handle, "PD_GetOpInfoMap");
  auto &op_info = get_op_info();
  auto *dyn_info_map = op_info.mutable_map();

  typedef std::vector<std::string> grad_op_desc_maker_t(
      const OpDesc &, const std::unordered_set<std::string> &,
      std::unordered_map<std::string, std::string> *,
      const std::vector<BlockDesc *> &);

  grad_op_desc_maker_t *grad_op_desc_maker =
      DynLoad<grad_op_desc_maker_t>(handle, "PD_GetGradOpDescStrs");

  auto &info_map = OpInfoMap::Instance();
  for (const auto &n : *(dyn_info_map)) {
    auto type = n.first;
    if (type == "recurrent" || type == "recurrent_grad" ||
        type == "conditional_block" || type == "conditional_block_grad") {
      continue;
    }
    PADDLE_ENFORCE_NE(info_map.Has(n.first), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", type));
    OpInfo info;
    info.creator_ = n.second.creator_;

    // If get the protocol buffer from dynamic library directly, there
    // will be deconstruction error
    // ** Error in `python`: free(): invalid pointer:
    //  ...  paddle::framework::proto::OpDesc::SharedDtor()
    // It seems a bug in protobuf, see
    // https://github.com/protocolbuffers/protobuf/issues/435
    // So, get the serialized binary string from dynamic library,
    // then deserialize to protocol buffer.
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
        PADDLE_ENFORCE_EQ(proto_desc.ParseFromString(str), true,
                          platform::errors::InvalidArgument(
                              "Failed to parse OpDesc from string."));
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
    info.use_empty_grad_op_desc_maker_ = n.second.use_empty_grad_op_desc_maker_;

    info_map.Insert(type, info);
  }

  typedef void init_device_t(platform::DeviceContextPool *);
  init_device_t *init_dev =
      DynLoad<init_device_t>(handle, "PD_InitDevicesPool");
  init_dev(&(platform::DeviceContextPool::Instance()));
}

}  // namespace framework
}  // namespace paddle
