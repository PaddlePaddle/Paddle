// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/commit.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {

int PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    case PaddleDType::INT32:
      return sizeof(int32_t);
    case PaddleDType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

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

void LoadCustomOpLibrary(std::string lib_path) {
#if defined(__APPLE__) || defined(__OSX__) || defined(_WIN32)
  PADDLE_THROW(platform::errors::Unimplemented(
      "Loading custom cpp op in inference does not support Apple or Windows."));
#else
  void *handle = paddle::platform::dynload::GetOpDsoHandle(lib_path);

  typedef framework::OpInfoMap &get_op_info_t();
  get_op_info_t *get_op_info =
      DynLoad<get_op_info_t>(handle, "PD_GetOpInfoMap");
  auto &op_info = get_op_info();
  auto *dyn_info_map = op_info.mutable_map();

  typedef std::vector<std::string> grad_op_desc_maker_t(
      const framework::OpDesc &, const std::unordered_set<std::string> &,
      std::unordered_map<std::string, std::string> *,
      const std::vector<framework::BlockDesc *> &);

  grad_op_desc_maker_t *grad_op_desc_maker =
      DynLoad<grad_op_desc_maker_t>(handle, "PD_GetGradOpDescStrs");

  auto &info_map = framework::OpInfoMap::Instance();
  for (const auto &n : *(dyn_info_map)) {
    auto type = n.first;
    if (type == "recurrent" || type == "recurrent_grad" ||
        type == "conditional_block" || type == "conditional_block_grad") {
      continue;
    }
    PADDLE_ENFORCE_NE(info_map.Has(n.first), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", type));
    framework::OpInfo info;
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
        const framework::OpDesc &op_desc,
        const std::unordered_set<std::string> &no_grad_set,
        std::unordered_map<std::string, std::string> *grad_to_var,
        const std::vector<framework::BlockDesc *> &grad_block) {
      std::vector<std::string> strs =
          grad_op_desc_maker(op_desc, no_grad_set, grad_to_var, grad_block);
      std::vector<std::unique_ptr<framework::OpDesc>> ret;
      for (auto &str : strs) {
        framework::proto::OpDesc proto_desc;
        PADDLE_ENFORCE_EQ(proto_desc.ParseFromString(str), true,
                          platform::errors::InvalidArgument(
                              "Failed to parse OpDesc from string."));
        ret.emplace_back(new framework::OpDesc(proto_desc, nullptr));
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
#endif
}

PaddleBuf::PaddleBuf(PaddleBuf &&other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

PaddleBuf::PaddleBuf(const PaddleBuf &other) { *this = other; }

PaddleBuf &PaddleBuf::operator=(const PaddleBuf &other) {
  if (!other.memory_owned_) {
    data_ = other.data_;
    length_ = other.length_;
    memory_owned_ = other.memory_owned_;
  } else {
    Resize(other.length());
    // if other.length() == 0 or other.data() == nullptr, then the memcpy
    // behavior is undefined
    if (other.length() && other.data())
      memcpy(data_, other.data(), other.length());
    else if (other.length())
      PADDLE_THROW(
          "Invalid argument, null pointer data with length %u is passed",
          other.length());

    length_ = other.length();
    memory_owned_ = true;
  }
  return *this;
}

PaddleBuf &PaddleBuf::operator=(PaddleBuf &&other) {
  // only the buffer with external memory can be copied
  data_ = other.data_;
  length_ = other.length_;
  memory_owned_ = other.memory_owned_;
  other.data_ = nullptr;
  other.length_ = 0;
  other.memory_owned_ = false;
  return *this;
}

void PaddleBuf::Resize(size_t length) {
  // Only the owned memory can be reset, the external memory can't be changed.
  if (length_ >= length) return;
  if (memory_owned_) {
    Free();
    data_ = new char[length];
    length_ = length;
    memory_owned_ = true;
  } else {
    PADDLE_THROW("The memory is allocated externally, can not Resized");
  }
}

void PaddleBuf::Reset(void *data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void PaddleBuf::Free() {
  if (memory_owned_ && data_) {
    PADDLE_ENFORCE_GT(length_, 0UL);
    delete[] static_cast<char *>(data_);
    data_ = nullptr;
    length_ = 0;
  }
}

std::string get_version() {
  std::stringstream ss;
  ss << "version: " << framework::paddle_version() << "\n";
  ss << "commit: " << framework::paddle_commit() << "\n";
  ss << "branch: " << framework::paddle_compile_branch() << "\n";
  return ss.str();
}

std::string UpdateDllFlag(const char *name, const char *value) {
  std::string ret;
  LOG(WARNING)
      << "The function \"UpdateDllFlag\" is only used to update the flag "
         "on the Windows shared library";
  ret = google::SetCommandLineOption(name, value);

  PADDLE_ENFORCE_EQ(
      ret.empty(), false,
      platform::errors::InvalidArgument(
          "Fail to update flag: %s, please make sure the flag exists.", name));
  LOG(INFO) << ret;
  return ret;
}

}  // namespace paddle
