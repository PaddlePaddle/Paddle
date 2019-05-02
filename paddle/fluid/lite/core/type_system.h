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
#include <map>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/lite/core/tensor.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// Type is the definition of all the types that supported by the Variable that
// represents as the input and output of an operator or kernel.
// The DNN system is simple, and the architecture can not process that many data
// types as a compiler, or that will turn out to a chaos.
//
// We should make sure that the supported data types be registered here, and
// keep the quantity small and avoid using some special data types as op's
// inputs or outputs, such as some runtime cache, those types can't be processed
// by the MIR.
//
// A tensor with different places(target, precision, data layout or device)
// should be treated as different types. Different types might be compatible
// with each other, for example, the `VoidTy` means any type, so any other types
// can be treated as a `VoidTy`.
//
// The Different Types can transform to others by adding some special
// transforming operators, for example, a DataLayoutTransformOp can convert a
// `TensorFp32NCHWTy` to a `TensorFp32NHWCTy`; a IoCopyOp can convert a
// `TensorFp32NCHWTy(kHost)` to `TensorFp32NCHWTy(kCUDA)`. There are many other
// convertions between different Types, but there are some unsupportted type
// convertions, for example, there is noway to convert a `UnsupportedTy` to a
// `TensorAnyTy`.
//
// We use Types to declare the definition of a kernel, each inputs' and outputs'
// arguments have a specific Types.
//
// REGISTER_LITE_KERNEL(mul, kHost, kFloat,
//     paddle::lite::kernels::host::MulCompute, def)
//   .BindInput("X", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
//       TARGET(kHost))})
//   .BindInput("Y", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
//       TARGET(kHost))})
//   .BindOutput("Out",
//   {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(TARGET(kHost))})
//   .Finalize();
//
// The above definition will be used in MIR by Type inference and uncompatible
// types check.
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
    // Tensor_Any represents a Tensor with any place, data, layout. It is used
    // in some IO kernels those doesn't care the data.
    Tensor_Any,
    // Used by feed or fetch op.
    TensorList_Any,
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
class Type : public DataTypeBase {
 public:
  TargetType target() const { return place_.target; }
  PrecisionType precision() const { return place_.precision; }
  DataLayoutType layout() const { return place_.layout; }
  short device() const { return place().device; }
  const Place& place() const { return place_; }
  const std::string& name() const { return name_; }

  bool operator==(const Type& other) {
    return id_ == other.id() && place_ == other.place();
  }
  friend std::ostream& operator<<(std::ostream& os, const Type& other) {
    if (other.IsUnsupported()) {
      os << "<Unsupported>";
      return os;
    }
    if (other.IsVoid()) {
      os << "<Void>";
      return os;
    }
    if (other.is_tensor_) {
      os << "<Tensor:";
    } else {
      os << "<";
    }
    os << TargetToStr(other.target()) << "/"
       << PrecisionToStr(other.precision()) << "/"
       << DataLayoutToStr(other.layout()) << ">";
    return os;
  }

  // Can cast to another type. This is heavily used in MIR, by determine whether
  // is is possible to add a statement to transform a type to another.
  virtual bool TypeCastable(const Type& type) const { return id_ == type.id(); }

  template <bool is_unknown, bool is_tensor = true,
            TargetType target = TargetType::kHost,
            PrecisionType precision = PrecisionType::kFloat,
            DataLayoutType layout = DataLayoutType::kNCHW>
  // Get a type.
  static const Type* Get();

  template <typename TypeTy>
  static const Type* Get(TargetType target = TargetType::kHost);

  virtual ~Type() = default;

 protected:
  Type(ID id, const std::string& name, bool is_tensor,
       TargetType target = TargetType::kHost,
       PrecisionType precision = PrecisionType::kFloat,
       DataLayoutType layout = DataLayoutType::kNCHW, short device = 0)
      : DataTypeBase(id, is_tensor),
        place_{target, precision, layout, device},
        name_(name) {}

 protected:
  Place place_;
  const std::string name_;
};

// -------------------------------- compatible check ---------------------------
static bool TargetCompatibleTo(const Type& a, const Type& b) {
  auto is_host = [](TargetType x) {
    return x == TARGET(kHost) || x == TARGET(kX86);
  };
  if (a.IsVoid() || b.IsVoid()) return true;
  if (a.IsTensor() || b.IsTensor()) {
    if (a.IsTensor() && b.IsTensor()) {
      return is_host(a.target()) ? is_host(b.target())
                                 : a.target() == b.target();
    }
    return false;
  }
  return true;
}

static bool DataLayoutCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||                                                  //
         (a.IsTensor() && b.IsTensor() && (a.layout() == b.layout() ||  //
                                           b.layout() == DATALAYOUT(kAny)));
}

static bool PrecisionCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||                                                        //
         (a.IsTensor() && b.IsTensor() && (a.precision() == b.precision() ||  //
                                           b.precision() == PRECISION(kAny)));
}

static bool DeviceCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||  //
         (a.IsTensor() && b.IsTensor() && (a.device() == b.device()));
}

// Can type 'a' be passed to 'b' directly.
static bool TypeCompatibleTo(const Type& a, const Type& b) {
  return TargetCompatibleTo(a, b) && DataLayoutCompatibleTo(a, b) &&
         PrecisionCompatibleTo(a, b) && DeviceCompatibleTo(a, b);
}

// -------------------------------- predefined types ---------------------------
// TODO(Superjomn) make all the Types' constructs protected to make sure there
// is only one instance across the system.
class VoidTy : public Type {
 public:
  VoidTy() : Type(ID::Void, "Void", false /*is_tensor*/) {}
};
class UnsupportedTy : public Type {
 public:
  UnsupportedTy() : Type(ID::Unsupported, "Unsupported", false /*is_tensor*/) {}
};
class TensorAnyTy : public Type {
 public:
  explicit TensorAnyTy(TargetType target)
      : Type(ID::Tensor_Any, "TensorAny", true, target, PRECISION(kAny),
             DATALAYOUT(kAny)) {}
};
// A list of tensor, and no assumption on the data layout or data type.
class TensorListAnyTy : public Type {
 public:
  explicit TensorListAnyTy(TargetType target)
      : Type(ID::TensorList_Any, "TensorList_Any", false, target,
             PRECISION(kAny), DATALAYOUT(kAny)) {}
};
class TensorFp32NCHWTy : public Type {
 public:
  explicit TensorFp32NCHWTy(TargetType target)
      : Type(ID::Tensor_Fp32_NCHW, "TensorFp32NCHW", true /*is_tensor*/, target,
             PrecisionType::kFloat, DataLayoutType::kNCHW) {}
};
class TensorInt8NCHWTy : public Type {
 public:
  explicit TensorInt8NCHWTy(TargetType target)
      : Type(ID::Tensor_Int8_NCHW, "TensorInt8NCHW", true /*is_tensor*/, target,
             PrecisionType::kInt8, DataLayoutType::kNCHW) {}
};
class TensorInt64NCHWTy : public Type {
 public:
  explicit TensorInt64NCHWTy(TargetType target)
      : Type(ID::Tensor_Int64_NCHW, "TensorInt64NCHW", true /*is_tensor*/,
             target, PrecisionType::kInt8, DataLayoutType::kNCHW) {}
};

const Type* LookupType(DataTypeBase::ID type_id, bool is_unknown,
                       bool is_tensor, Place place);
// ------------------------- end predefined types ---------------------------

/*
 * ParamType is used to represent a data type of a parameter for the kernel. It
 * can represent any Variable data type.
 * The element_type_hash is the hash code of the element, it should be
 * registered in the `TypeSystem`.
 */
struct ParamType {
  Place tensor_place{};
  const Type* type;

  ParamType() = default;
  ParamType(const Type* type) : type(type) { tensor_place = type->place(); }

  std::string DebugString() const { return tensor_place.DebugString(); }
};

/*
 * The data types of kernel parameters. It is used to track the type of kernel's
 * inputs and outputs.
 */
struct ParamTypeRecorder {
  std::map<std::string, ParamType> inputs;
  std::map<std::string, ParamType> outputs;

  void RegisterInputType(const std::string& arg_name, const ParamType& type) {
    Register(&inputs, arg_name, type);
  }

  void RegisterOutputType(const std::string& arg_name, const ParamType& type) {
    Register(&outputs, arg_name, type);
  }

 private:
  void Register(std::map<std::string, ParamType>* ts,
                const std::string& arg_name, ParamType type) {
    (*ts)[arg_name] = type;
  }
};

/*
 * The ParamTypeRegistry help register the input and output data types for all
 * the kernels. It is made singleton so that all the objects of the same kernel
 * can share the same information.
 *
 * Usage:
 * for register a kernel for FC operator.
 * ParamTypeRegistry::Global().Register(
 *        "fc", {TARGET(kCUDA), PRECISION(kFloat)}, 0,
 *        {typeid(Tensor), {TARGET(kCUDA)}});
 */
class ParamTypeRegistry {
 public:
  enum class IO : int { kInput = 0, kOutput };

  template <TargetType target, PrecisionType precision,
            DataLayoutType layout = DataLayoutType::kNCHW>
  /*
   * Helper class for registering a ParamType for a Kernel.
   * Usage:
   *
   * NewInstance<TARGET(kHost), PRECISION(kFloat)>("fc")
   *   .BindInput(0, {typeid(Tensor).hash_code(), {TARGET(kHost)})
   *   .BindInput(1, {typeid(Tensor).hash_code(), {TARGET(kHost),
   *                                               PRECISION(kFloat)});
   */
  struct NewInstance {
    explicit NewInstance(const std::string& kernel_type)
        : kernel_type_(kernel_type) {}

    NewInstance& BindInput(const std::string& arg_name,
                           const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kInput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }
    NewInstance& BindOutput(const std::string& arg_name,
                            const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kOutput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }

    bool Finalize() { return true; }

   private:
    std::string kernel_type_;
  };

  template <IO io>
  void Register(const std::string& kernel_type, const Place& place,
                const std::string& arg_name, ParamType data_type) {
    KernelIdTy key{kernel_type, place, io, arg_name};
    types_[key] = data_type;
    CHECK(types_.count(key));
  }

  const ParamType* RetrieveInArgument(const Place& place,
                                      const std::string& op_type,
                                      const std::string& arg_name) {
    return Retrieve<IO::kInput>(place, op_type, arg_name);
  }
  const ParamType* RetrieveOutArgument(const Place& place,
                                       const std::string& op_type,
                                       const std::string& arg_name) {
    return Retrieve<IO::kOutput>(place, op_type, arg_name);
  }

  static ParamTypeRegistry& Global() {
    static ParamTypeRegistry x;
    return x;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ParamTypeRegistry& other) {
    for (auto& item : other.types_) {
      os << item.first << " " << item.second.DebugString() << "\n";
    }
    return os;
  }

 protected:
  template <IO io>
  const ParamType* Retrieve(const Place& place, const std::string& op_type,
                            const std::string& arg_name) {
    KernelIdTy key{op_type, place, io, arg_name};
    auto it = types_.find(key);
    if (it == types_.end()) return nullptr;
    return &it->second;
  }

 private:
  ParamTypeRegistry() = default;

 public:
  // Identification for a Kernel.
  struct KernelIdTy {
    std::string kernel_type;
    Place place;
    IO io;
    std::string arg_name;

    size_t hash() const;
    friend std::ostream& operator<<(std::ostream& os, const KernelIdTy& other);
  };

  using key_t = KernelIdTy;
  struct KeyCmp {
    bool operator()(const key_t& a, const key_t& b) const;
  };

 private:
  std::map<key_t, ParamType, ParamTypeRegistry::KeyCmp> types_;
};

}  // namespace lite
}  // namespace paddle
