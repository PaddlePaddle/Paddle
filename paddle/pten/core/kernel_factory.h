//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_def.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace pten {

using DataType = paddle::experimental::DataType;
using DataLayout = paddle::experimental::DataLayout;

/**
 * [ Naming considerations ]
 *
 * The tensor operation library contains many kernels, and the computation
 * in each specific scenario is represented by an kernel.
 *
 * We directly named it `Kernel` instead of `Kernel`, the tensor operation
 * library here and fluid are independent, avoiding developers from
 * misunderstanding the relationship between the two concepts.
 */

class KernelContext;

using KernelFn = void (*)(KernelContext* ctx);

class KernelName final {
 public:
  KernelName(std::string name, std::string overload_name)
      : name_(std::move(name)), overload_name_(std::move(overload_name)) {}

  KernelName(const std::string& kernel_name) {
    ParseNameAndOverloadNameFromString(kernel_name);
  }

  KernelName(const char* kernel_name) {
    std::string kernel_name_str(kernel_name);
    ParseNameAndOverloadNameFromString(kernel_name_str);
  }

  const std::string& name() const { return name_; }
  const std::string& overload_name() const { return overload_name_; }

  struct Hash {
    size_t operator()(const KernelName& kernel_name) const {
      return std::hash<std::string>()(kernel_name.name()) ^
             (std::hash<std::string>()(kernel_name.overload_name()) << 1);
    }
  };

  size_t hash_value() const { return Hash()(*this); }

  bool operator<(const KernelName& kernel_name) const {
    return hash_value() < kernel_name.hash_value();
  }

  bool operator==(const KernelName& kernel_name) const {
    return hash_value() == kernel_name.hash_value();
  }

  bool operator!=(const KernelName& kernel_name) const {
    return hash_value() != kernel_name.hash_value();
  }

 private:
  void ParseNameAndOverloadNameFromString(const std::string& kernel_name) {
    size_t pos = kernel_name.find_first_of('.');
    if (pos == std::string::npos) {
      name_ = kernel_name;
      overload_name_ = "";
    } else {
      name_ = kernel_name.substr(0, pos);
      overload_name_ = kernel_name.substr(pos + 1, kernel_name.size());
    }
  }

  // TODO(chenweihang): use string_view to improve performance later
  std::string name_;
  std::string overload_name_;
};

class KernelKey {
 public:
  KernelKey() = default;

  KernelKey(Backend backend, DataLayout layout, DataType dtype)
      : backend_(backend), layout_(layout), dtype_(dtype) {}

  Backend backend() const { return backend_; }
  DataLayout layout() const { return layout_; }
  DataType dtype() const { return dtype_; }

  struct Hash {
    // Note: Now the number of bits we need does not exceed 32 bits, so there is
    // no need to use 64 bits. If needed in the future, it can be expanded,
    // but now we don’t over-design.
    uint32_t operator()(const KernelKey& key) const;
  };

  uint32_t hash_value() const { return Hash()(*this); }

  bool operator<(const KernelKey& key) const {
    return hash_value() < key.hash_value();
  }

  bool operator==(const KernelKey& key) const {
    return hash_value() == key.hash_value();
  }

  bool operator!=(const KernelKey& key) const {
    return hash_value() != key.hash_value();
  }

 private:
  // In total should be smaller than 32.
  constexpr static int kBackendBitLength = 8;
  constexpr static int kDataLayoutBitLength = 4;
  constexpr static int kDataTypeBitLength = 8;

  Backend backend_{Backend::UNDEFINED};
  DataLayout layout_{DataLayout::UNDEFINED};
  DataType dtype_{DataType::UNDEFINED};
};

// TODO(chenweihang): how deal with vector<Param>?
struct TensorArgDef {
  Backend backend;
  DataLayout layout;
  DataType dtype;

  TensorArgDef(Backend in_backend, DataLayout in_layout, DataType in_dtype)
      : backend(in_backend), layout(in_layout), dtype(in_dtype) {}

  TensorArgDef& SetBackend(Backend in_backend) {
    backend = in_backend;
    return *this;
  }

  TensorArgDef& SetDataLayout(DataLayout in_layout) {
    layout = in_layout;
    return *this;
  }

  TensorArgDef& SetDataType(DataType in_dtype) {
    dtype = in_dtype;
    return *this;
  }
};

struct AttributeArgDef {
  std::type_index type_index;

  explicit AttributeArgDef(std::type_index type_index)
      : type_index(type_index) {}
};

class KernelArgsDef {
 public:
  KernelArgsDef() = default;

  void AppendInput(Backend backend, DataLayout layout, DataType dtype) {
    input_defs_.emplace_back(TensorArgDef(backend, layout, dtype));
  }

  void AppendOutput(Backend backend, DataLayout layout, DataType dtype) {
    output_defs_.emplace_back(TensorArgDef(backend, layout, dtype));
  }

  void AppendAttribute(std::type_index type_index) {
    attribute_defs_.emplace_back(AttributeArgDef(type_index));
  }

  const paddle::SmallVector<TensorArgDef>& input_defs() const {
    return input_defs_;
  }

  const paddle::SmallVector<TensorArgDef>& output_defs() const {
    return output_defs_;
  }

  const paddle::SmallVector<AttributeArgDef>& attribute_defs() const {
    return attribute_defs_;
  }

  paddle::SmallVector<TensorArgDef>& input_defs() { return input_defs_; }

  paddle::SmallVector<TensorArgDef>& output_defs() { return output_defs_; }

  paddle::SmallVector<AttributeArgDef>& attribute_defs() {
    return attribute_defs_;
  }

 private:
  paddle::SmallVector<TensorArgDef> input_defs_{{}};
  paddle::SmallVector<TensorArgDef> output_defs_{{}};
  paddle::SmallVector<AttributeArgDef> attribute_defs_{{}};
};

class Kernel {
 public:
  // for map element contruct
  Kernel() = default;

  explicit Kernel(KernelFn fn) : fn_(fn) {}

  void operator()(KernelContext* ctx) const { fn_(ctx); }

  KernelArgsDef* mutable_args_def() { return &args_def_; }

  const KernelArgsDef& args_def() const { return args_def_; }

  TensorArgDef& InputAt(size_t idx) { return args_def_.input_defs().at(idx); }

  TensorArgDef& OutputAt(size_t idx) { return args_def_.output_defs().at(idx); }

  bool IsValid() { return fn_ != nullptr; }

 private:
  KernelFn fn_{nullptr};
  KernelArgsDef args_def_;
};

/**
 * Note: Each Computation need a basic kernel map that named by kernel_name.
 *       Such as for scale op, KernelMap contains a `scale` kernel map,
 *       if it still need other overload kernel, the op name can be
 *       `scale.***`.
 */
class KernelFactory {
 public:
  // replaced by paddle::flat_hash_map later
  using KernelMap = paddle::flat_hash_map<
      KernelName,
      paddle::flat_hash_map<KernelKey, Kernel, KernelKey::Hash>,
      KernelName::Hash>;

  static KernelFactory& Instance();

  KernelMap& kernels() { return kernels_; }

  void InsertCompatibleOpType(const std::string& op_type) {
    compatible_op_types_.insert(op_type);
  }

  bool HasCompatiblePtenKernel(const std::string& op_type) const {
    return compatible_op_types_.count(TransToPtenKernelName(op_type)) > 0;
  }

  const Kernel& SelectKernelOrThrowError(const KernelName& kernel_name,
                                         const KernelKey& kernel_key) const;

  const Kernel& SelectKernelOrThrowError(const KernelName& kernel_name,
                                         Backend backend,
                                         DataLayout layout,
                                         DataType dtype) const;

  Kernel SelectKernel(const KernelName& kernel_name,
                      const KernelKey& kernel_key) const;

 private:
  KernelFactory() = default;

  KernelMap kernels_;
  // Used to be compatible with the original execution system and
  // quickly confirm whether the new kernel can be called
  std::unordered_set<std::string> compatible_op_types_;
};

/** operator << overload **/

inline std::ostream& operator<<(std::ostream& os,
                                const KernelName& kernel_name) {
  if (kernel_name.overload_name().empty()) {
    os << kernel_name.name();
  } else {
    os << kernel_name.name() << "." << kernel_name.overload_name();
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const KernelKey& kernel_key) {
  os << "(" << kernel_key.backend() << ", " << kernel_key.layout() << ", "
     << kernel_key.dtype() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

std::ostream& operator<<(std::ostream& os, KernelFactory& kernel_factory);

}  // namespace pten
