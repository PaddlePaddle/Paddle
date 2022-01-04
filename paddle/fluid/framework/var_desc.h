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

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

// convert between std::vector and protobuf repeated.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(repeated_field.begin(), repeated_field.end(),
            std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

// Specialize vector<bool>.
template <typename RepeatedField>
inline void VectorToRepeated(const std::vector<bool> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (auto elem : vec) {
    *repeated_field->Add() = elem;
  }
}

class VarDesc {
 public:
  explicit VarDesc(const std::string &name) {
    desc_.set_name(name);
    // TODO(paddle-dev): Why default to lodtensor.
    desc_.mutable_type()->set_type(proto::VarType::LOD_TENSOR);
  }

  explicit VarDesc(const proto::VarDesc &desc) : desc_(desc) {}

  // Explicitly implement the copy constructor for auto parallel
  VarDesc(const VarDesc &other)
      : desc_(other.desc_),
        attrs_(other.attrs_),
        original_id_(other.original_id_) {}

  proto::VarDesc *Proto() { return &desc_; }

  const proto::VarDesc *Proto() const { return &desc_; }

  std::string Name() const { return desc_.name(); }

  void SetName(std::string name) { desc_.set_name(name); }

  void SetTensorDescNum(size_t num);

  size_t GetTensorDescNum() const;

  void SetShape(const std::vector<int64_t> &dims);

  void SetShapes(const std::vector<std::vector<int64_t>> &multiple_dims);

  std::vector<int64_t> GetShape() const;

  std::vector<std::vector<int64_t>> GetShapes() const;

  void SetDataType(proto::VarType::Type data_type);

  void SetDataTypes(
      const std::vector<proto::VarType::Type> &multiple_data_type);

  proto::VarType::Type GetDataType() const;

  size_t ElementSize() const;

  std::vector<proto::VarType::Type> GetDataTypes() const;

  void SetLoDLevel(int32_t lod_level);

  void SetLoDLevels(const std::vector<int32_t> &multiple_lod_level);

  int32_t GetLoDLevel() const;

  std::vector<int32_t> GetLoDLevels() const;

  proto::VarType::Type GetType() const;

  void SetType(proto::VarType::Type type);

  bool Persistable() const { return desc_.persistable(); }

  void SetPersistable(bool persistable) { desc_.set_persistable(persistable); }

  bool IsParameter() const { return desc_.is_parameter(); }

  void SetIsParameter(bool is_parameter) {
    desc_.set_is_parameter(is_parameter);
  }

  void ClearIsParameter() { desc_.clear_is_parameter(); }

  bool HasIsParameter() const { return desc_.has_is_parameter(); }

  bool StopGradient() const { return desc_.stop_gradient(); }

  void SetStopGradient(bool stop_gradient) {
    desc_.set_stop_gradient(stop_gradient);
  }

  void ClearStopGradient() { desc_.clear_stop_gradient(); }

  bool HasStopGradient() const { return desc_.has_stop_gradient(); }

  bool NeedCheckFeed() const { return desc_.need_check_feed(); }

  void SetNeedCheckFeed(bool need_check_feed) {
    desc_.set_need_check_feed(need_check_feed);
  }

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);
  void RemoveAttr(const std::string &name);

  Attribute GetAttr(const std::string &name) const;

  // The Id() and OriginalId() are only used for auto parallel.
  uint64_t Id() const { return id_; }
  uint64_t OriginalId() const { return original_id_; }
  void SetOriginalId(uint64_t original_id) { original_id_ = original_id; }

 private:
  const proto::VarType::TensorDesc &tensor_desc() const;
  std::vector<proto::VarType::TensorDesc> tensor_descs() const;
  proto::VarType::TensorDesc *mutable_tensor_desc();
  std::vector<proto::VarType::TensorDesc *> mutable_tensor_descs();

  // This thread-safe implementation seems to be redudent since the neural
  // networks are usually constructed in a single thread.
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    return ++uid;
  }

  proto::VarDesc desc_;
  AttributeMap attrs_;

  // Note: the id_ is unique for all VarDesc (only for auto parallel).
  uint64_t id_ = GenerateId();
  // Note: the orignal_id_ is used for referring to the original VarDesc
  // that the current VarDesc is built from (only for auto parallel).
  // The default original_id_ is same as the id_, which means the
  // current VarDesc is not built from the other one.
  uint64_t original_id_ = id_;
};

bool operator==(const VarDesc &left, const VarDesc &right);
}  // namespace framework
}  // namespace paddle
