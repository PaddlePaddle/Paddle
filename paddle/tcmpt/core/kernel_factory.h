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
#include <utility>

#include "paddle/tcmpt/core/backend.h"
#include "paddle/tcmpt/core/dtype.h"
#include "paddle/tcmpt/core/kernel_def.h"
#include "paddle/tcmpt/core/layout.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"

namespace pt {

/**
 * [ Naming considerations ]
 *
 * The tensor Compute library contains many kernels, and the computation
 * in each specific scenario is represented by an kernel.
 *
 * We directly named it `Kernel` instead of `Kernel`, the tensor Compute
 * library here and fluid are independent, avoiding developers from
 * misunderstanding the relationship between the two concepts.
 */

class KernelContext;

using KernelFn = void (*)(KernelContext* ctx);

struct KernelName final {
  // TODO(chenweihang): use string_view later?
  std::string name;
  std::string overload_name;
  // Avoid calculating Hash value at runtime
  size_t hash_value;

  KernelName(std::string name, std::string overload_name)
      : name(std::move(name)), overload_name(std::move(overload_name)) {
    hash_value = std::hash<std::string>()(name) ^
                 (std::hash<std::string>()(overload_name) << 1);
  }

  KernelName(const char* kernel_name) {
    std::string kernel_name_str(kernel_name);
    size_t pos = kernel_name_str.find_first_of('.');
    if (pos == std::string::npos) {
      name = kernel_name_str;
      overload_name = "";
    } else {
      name = kernel_name_str.substr(0, pos);
      overload_name = kernel_name_str.substr(pos + 1, kernel_name_str.size());
    }
    hash_value = std::hash<std::string>()(name) ^
                 (std::hash<std::string>()(overload_name) << 1);
  }

  struct Hash {
    size_t operator()(const KernelName& kernel_name) const {
      return kernel_name.hash_value;
    }
  };

  bool operator<(const KernelName& kernel_name) const {
    return hash_value < kernel_name.hash_value;
  }

  bool operator==(const KernelName& kernel_name) const {
    return hash_value == kernel_name.hash_value;
  }

  bool operator!=(const KernelName& kernel_name) const {
    return hash_value != kernel_name.hash_value;
  }
};

class KernelKey {
 public:
  KernelKey() = default;

  KernelKey(Backend backend, DataLayout layout, DataType dtype)
      : backend_(backend), layout_(layout), dtype_(dtype) {
    // |----31-20------|---19-12---|---11-8----|---7-0---|
    // | For extension | DataType | DataLayout | Backend |

    hash_value_ = 0;
    hash_value_ |= static_cast<uint8_t>(backend_);
    hash_value_ |= (static_cast<uint8_t>(layout_) << kBackendBitLength);
    hash_value_ |= (static_cast<uint16_t>(dtype_)
                    << (kBackendBitLength + kDataTypeBitLength));
  }

  Backend backend() const { return backend_; }
  DataLayout layout() const { return layout_; }
  DataType dtype() const { return dtype_; }

  uint32_t hash_value() const { return hash_value_; }

  bool operator<(const KernelKey& key) const {
    return hash_value_ < key.hash_value();
  }

  bool operator==(const KernelKey& key) const {
    return hash_value_ == key.hash_value();
  }

  bool operator!=(const KernelKey& key) const {
    return hash_value_ != key.hash_value();
  }

  struct Hash {
    uint32_t operator()(const KernelKey& key) const { return key.hash_value(); }
  };

 private:
  // In total should be smaller than 32.
  constexpr static int kBackendBitLength = 8;
  constexpr static int kDataLayoutBitLength = 4;
  constexpr static int kDataTypeBitLength = 8;

  Backend backend_{Backend::kUndef};
  DataLayout layout_{DataLayout::kUndef};
  DataType dtype_{DataType::kUndef};

  // Avoid calculating Hash value at runtime.
  // Note: Now the number of bits we need does not exceed 32 bits, so there is
  // no need to use 64 bits. If needed in the future, it can be expanded,
  // but now we don’t over-design.
  uint32_t hash_value_;
};

// TODO(chenweihang): how deal with vector<Param>?
struct TensorArgDef {
  Backend backend;
  DataLayout layout;
  DataType dtype;

  TensorArgDef(Backend backend, DataLayout layout, DataType dtype)
      : backend(backend), layout(layout), dtype(dtype) {}

  TensorArgDef& SetBackend(Backend backend) {
    backend = backend;
    return *this;
  }

  TensorArgDef& SetDataLayout(DataLayout layout) {
    layout = layout;
    return *this;
  }

  TensorArgDef& SetDataType(DataType dtype) {
    dtype = dtype;
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

  const std::vector<TensorArgDef>& input_defs() const { return input_defs_; }

  const std::vector<TensorArgDef>& output_defs() const { return output_defs_; }

  const std::vector<AttributeArgDef>& attribute_defs() const {
    return attribute_defs_;
  }

  std::vector<TensorArgDef>& input_defs() { return input_defs_; }

  std::vector<TensorArgDef>& output_defs() { return output_defs_; }

  std::vector<AttributeArgDef>& attribute_defs() { return attribute_defs_; }

 private:
  // TODO(chenweihang): replaced by paddle::small_vector
  std::vector<TensorArgDef> input_defs_{{}};
  std::vector<TensorArgDef> output_defs_{{}};
  std::vector<AttributeArgDef> attribute_defs_{{}};
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
  using KernelMap =
      std::unordered_map<KernelName,
                         std::unordered_map<KernelKey, Kernel, KernelKey::Hash>,
                         KernelName::Hash>;

  static KernelFactory& Instance();

  KernelMap& kernels() { return kernels_; }

  bool ContainsKernel(const char* name) const;

  const Kernel& SelectKernel(const KernelName& kernel_name,
                             const KernelKey& kernel_key) const;

  const Kernel& SelectKernel(const KernelName& kernel_name,
                             Backend backend,
                             DataLayout layout,
                             DataType dtype) const;

 private:
  KernelFactory() = default;

  KernelMap kernels_;
};

/** operator << overload **/

inline std::ostream& operator<<(std::ostream& os,
                                const KernelName& kernel_name) {
  if (kernel_name.overload_name.empty()) {
    os << kernel_name.name;
  } else {
    os << kernel_name.name << "." << kernel_name.overload_name;
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

}  // namespace pt
