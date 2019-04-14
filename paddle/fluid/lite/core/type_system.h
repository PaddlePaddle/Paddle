// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
// This file contains the file system of the lite system. Every data type in
// Variable should be registered here, and the analysis phase will check the
// data type correction.
// This mechanism is made for keeping our system simpler and more stable, for
// the dubious typed Variables in the Operators' inputs and outputs are disaster
// for analysis and runtime.

#include <glog/logging.h>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

// Type is the definition of all the types that supported by the Variable that
// represents as the input and output of an operator or kernel.
// The DNN system is simple, and the architecture can not process that many data
// types as a compiler, or that will turn out to a chaos.
//
// We should make sure that supported data types should be registered here, and
// keep the quantity small. And avoid using some special data types as op's IO,
// such as some runtime cache, that need to be avoided.
//
// TODO(Superjomn) Add operator/kernel-wise static checking to avoid unsupported
// type mixed in the system.
class DataTypeBase {
 public:
  // The Void type can cast to any other type.
  // The Unsupported is the data type that developed include in the system, for
  // example, some `std::set` is used as input of some operator. It wan't be
  // analyzed or optimized by the system, that way results in many bugs in
  // previous system, so it should be avoided.
  enum class ID : int {
    Void = 0,     // unknown type that can be cast to any data type.
    Unsupported,  // Unsupported data type that will not be analyzed.
    Tensor_Fp32_NCHW,
    Tensor_Int8_NCHW,
    Tensor_Int64_NCHW,
    NumTypes,  // Must remains as last defined ID.
  };

  ID id() const { return id_; }

  // type check.
  bool IsTensor() const { return is_tensor_; }
  bool IsVoid() const { return id_ == ID::Void; }
  bool IsUnsupported() const { return id_ == ID::Unsupported; }
  bool IsTensorFp32NCHW() const { return id_ == ID::Tensor_Fp32_NCHW; }
  bool IsTensorInt8NCHW() const { return id_ == ID::Tensor_Int8_NCHW; }
  bool IsTensorInt64NCHW() const { return id_ == ID::Tensor_Int64_NCHW; }

  int num_types() const { return static_cast<int>(ID::NumTypes); }

 protected:
  // Can only extended by subclass.
  DataTypeBase(ID id, bool is_tensor) : id_(id), is_tensor_(is_tensor) {}

  ID id_{ID::Unsupported};
  bool is_tensor_{false};
};

/*
 * Datatype with device info considered.
 * NOTE A Type with different device is treated as different DeviceDataType.
 */
class DeviceDataType : public DataTypeBase {
 public:
  TargetType target() const { return place_.target; }
  PrecisionType precision() const { return place_.precision; }
  DataLayoutType layout() const { return place_.layout; }
  const Place& place() const { return place_; }
  const std::string& name() const { return name_; }

  bool operator==(const DeviceDataType& other) {
    return id_ == other.id() && place_ == other.place();
  }

  // Can cast to another type. This is heavily used in MIR, by determine whether
  // is is possible to add a instruction to transform a type to another.
  virtual bool TypeCastable(const DeviceDataType& type) const {
    return id_ == type.id();
  }

  virtual ~DeviceDataType() = default;

 protected:
  DeviceDataType(ID id, const std::string& name, bool is_tensor,
                 TargetType target = TargetType::kHost,
                 PrecisionType precision = PrecisionType::kFloat,
                 DataLayoutType layout = DataLayoutType::kNCHW)
      : DataTypeBase(id, is_tensor),
        place_{target, precision, layout},
        name_(name) {}

 protected:
  Place place_;
  const std::string name_;
};

// -------------------------------- predefined types ---------------------------
class Void : public DeviceDataType {
 public:
  Void() : DeviceDataType(ID::Void, "Void", false /*is_tensor*/) {}
};
class TensorFp32NCHW : public DeviceDataType {
 public:
  TensorFp32NCHW(TargetType target)
      : DeviceDataType(ID::Tensor_Fp32_NCHW, "TensorFp32NCHW",
                       true /*is_tensor*/, target, PrecisionType::kFloat,
                       DataLayoutType::kNCHW) {}
};
class TensorInt8NCHW : public DeviceDataType {
 public:
  TensorInt8NCHW(TargetType target)
      : DeviceDataType(ID::Tensor_Int8_NCHW, "TensorInt8NCHW",
                       true /*is_tensor*/, target, PrecisionType::kInt8,
                       DataLayoutType::kNCHW) {}
};
class TensorInt64NCHW : public DeviceDataType {
 public:
  TensorInt64NCHW(TargetType target)
      : DeviceDataType(ID::Tensor_Int64_NCHW, "TensorInt64NCHW",
                       true /*is_tensor*/, target, PrecisionType::kInt8,
                       DataLayoutType::kNCHW) {}
};
// ------------------------- end predefined types ---------------------------

// NOTE TypeSystem has some overhead, and better to be used in analysis phase.
class TypeSystem {
 private:
  // Put all valid types for Variables here!
  TypeSystem() {
    // Tensor is a valid data type for Variable.
    Register<Tensor>("tensor");
  }

 public:
  static TypeSystem& Global() {
    static TypeSystem x;
    return x;
  }

  template <typename T>
  void Register(const std::string& type) {
    size_t hash = typeid(T).hash_code();
    CHECK(!types_.count(hash)) << "duplicate register type " << type
                               << " found!";
    types_[hash] = type;
    names_.insert(type);
  }

  template <typename T>
  bool Contains() const {
    return types_.count(typeid(T).hash_code());
  }

  bool Contains(size_t hash) const { return types_.count(hash); }

  bool Contains(const std::string& type) { return names_.count(type); }

  std::string DebugInfo() const {
    std::stringstream ss;
    for (const auto& it : types_) {
      ss << it.second << "\n";
    }
    return ss.str();
  }

 private:
  std::unordered_map<size_t /*hash*/, std::string /*name*/> types_;
  TypeSystem(const TypeSystem&) = delete;
  std::unordered_set<std::string> names_;
};

}  // namespace lite
}  // namespace paddle
