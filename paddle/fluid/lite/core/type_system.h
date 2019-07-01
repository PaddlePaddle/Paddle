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

#include <map>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// Type is the definition of all the types that supported by the Variable that
// represents as the input and output of an operator or kernel.
// The DNN system is simple, just a list of operators, and the architecture
// can not process that many data types as a compiler, or that will turn out to
// a chaos.
//
// We should make sure that the supported data types be registered here, and
// keep the set small and avoid using some special data types as op's
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
// convertions between different Types, but there are some unsupported type
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
class DataType {
 public:
  // The Void type can cast to any other type.
  // The Unsupported is the data type that developed include in the system, for
  // example, some `std::set` is used as input of some operator. It wan't be
  // analyzed or optimized by the system, that way results in many bugs in
  // previous system, so it should be avoided.
  enum class ID : int {
    Void = 0,     // unknown type that can be cast to any data type.
    Unsupported,  // Unsupported data type that will not be analyzed.
    // Tensor_Any represents a Tensor with any place, data, layout. It is used
    // in some IO kernels those doesn't care the data.
    Tensor,
    // A tensor list, but all the elements should have the same type.
    TensorList,
    // ---------
    NumTypes,  // Must remains as last defined ID.
  };

  ID id() const { return id_; }

  // type check.
  bool IsVoid() const { return id_ == ID::Void; }
  bool IsUnsupported() const { return id_ == ID::Unsupported; }
  bool IsTensor() const { return id_ == ID::Tensor; }
  bool IsTensorList() const { return id_ == ID::TensorList; }
  // Get number of types.
  int num_types() const { return static_cast<int>(ID::NumTypes); }

 protected:
  // Can only extended by subclass.
  explicit DataType(ID id) : id_(id) {}

  ID id_{ID::Unsupported};
};

/*
 * Datatype with device info considered.
 * NOTE A Type with different device is treated as different DeviceDataType.
 */
class Type : public DataType {
 public:
  // Can cast to another type. This is heavily used in MIR, by determine whether
  // is is possible to add a statement to transform a type to another.
  virtual bool TypeCastable(const Type& type) const { return id_ == type.id(); }

  /// Get a Tensor type.
  static const Type* GetTensorTy(TargetType target,
                                 PrecisionType precision = PRECISION(kFloat),
                                 DataLayoutType layout = DATALAYOUT(kNCHW),
                                 int device = 0);
  /// Get a TensorList type.
  static const Type* GetTensorListTy(
      TargetType target, PrecisionType precision = PRECISION(kFloat),
      DataLayoutType layout = DATALAYOUT(kNCHW), int device = 0);
  /// Get an Unsupported type.
  static const Type* GetUnsupportedTy();
  /// Get an Void type.
  static const Type* GetVoidTy();

  static const Type* Get(DataType::ID type_id, TargetType target = TARGET(kUnk),
                         PrecisionType precision = PRECISION(kUnk),
                         DataLayoutType layout = DATALAYOUT(kUnk),
                         int device = 0);

  TargetType target() const { return place_.target; }
  PrecisionType precision() const { return place_.precision; }
  DataLayoutType layout() const { return place_.layout; }
  int16_t device() const { return place().device; }
  const Place& place() const { return place_; }
  const std::string& name() const { return name_; }

  bool operator==(const Type& other) {
    return id_ == other.id() && place_ == other.place();
  }
  friend std::ostream& operator<<(std::ostream& os, const Type& other);

  virtual ~Type() = default;

 protected:
  /// One should avoid using this construct.
  Type(ID id, const std::string& name, TargetType target = TargetType::kHost,
       PrecisionType precision = PrecisionType::kFloat,
       DataLayoutType layout = DataLayoutType::kNCHW, int16_t device = 0)
      : DataType(id), place_{target, precision, layout, device}, name_(name) {}

  Place place_;
  const std::string name_;
};

// -------------------------------- compatible check ---------------------------
static bool TargetCompatibleTo(const Type& a, const Type& b) {
  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
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

/*
 * ParamType is used to represent a data type of a parameter for the kernel. It
 * can represent any Variable data type.
 * The element_type_hash is the hash code of the element, it should be
 * registered in the `TypeSystem`.
 */
struct ParamType {
  const Type* type;

  ParamType() = default;
  ParamType(const Type* type) : type(type) {}  // NOLINT

  std::string DebugString() const { return type->name(); }
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
