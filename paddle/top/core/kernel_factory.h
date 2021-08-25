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

#include "paddle/top/core/backend.h"
#include "paddle/top/core/dtype.h"
#include "paddle/top/core/kernel_def.h"
#include "paddle/top/core/layout.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"

namespace pt {

class OpKernelContext;

using OpKernelFn = void (*)(OpKernelContext* ctx);

struct OperationName final {
  // TODO(chenweihang): use string_view later?
  std::string op_type;
  std::string overload_type;
  // Avoid calculating Hash value at runtime
  size_t hash_value;

  OperationName(std::string op_type, std::string overload_type)
      : op_type(std::move(op_type)), overload_type(std::move(overload_type)) {
    hash_value = std::hash<std::string>()(op_type) ^
                 (std::hash<std::string>()(overload_type) << 1);
  }

  OperationName(const char* op_name) {
    std::string op_name_str(op_name);
    size_t pos = op_name_str.find_first_of('.');
    if (pos == std::string::npos) {
      op_type = op_name_str;
      overload_type = "";
    } else {
      op_type = op_name_str.substr(0, pos);
      PADDLE_ENFORCE_EQ(op_name_str.find('.', pos + 1),
                        std::string::npos,
                        paddle::platform::errors::InvalidArgument(
                            "OperationName only can contains one '.'."));
      overload_type = op_name_str.substr(pos + 1, op_name_str.size());
    }
    hash_value = std::hash<std::string>()(op_type) ^
                 (std::hash<std::string>()(overload_type) << 1);
  }

  struct Hash {
    size_t operator()(const OperationName& op_name) const {
      return op_name.hash_value;
    }
  };

  bool operator<(const OperationName& op_name) const {
    return hash_value < op_name.hash_value;
  }

  bool operator==(const OperationName& op_name) const {
    return hash_value == op_name.hash_value;
  }

  bool operator!=(const OperationName& op_name) const {
    return hash_value != op_name.hash_value;
  }
};

class OpKernelKey {
 public:
  OpKernelKey() = default;

  OpKernelKey(Backend backend, DataLayout layout, DataType dtype)
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

  bool operator<(const OpKernelKey& key) const {
    return hash_value_ < key.hash_value();
  }

  bool operator==(const OpKernelKey& key) const {
    return hash_value_ == key.hash_value();
  }

  bool operator!=(const OpKernelKey& key) const {
    return hash_value_ != key.hash_value();
  }

  struct Hash {
    uint32_t operator()(const OpKernelKey& key) const {
      return key.hash_value();
    }
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
struct ParamDef {
  Backend backend;
  DataLayout layout;
  DataType dtype;

  ParamDef(Backend backend, DataLayout layout, DataType dtype)
      : backend(backend), layout(layout), dtype(dtype) {}
};

class OpKernelParamDef {
 public:
  OpKernelParamDef() = default;

  void AppendInput(Backend backend, DataLayout layout, DataType dtype) {
    input_defs_.emplace_back(ParamDef(backend, layout, dtype));
  }

  void AppendOutput(Backend backend, DataLayout layout, DataType dtype) {
    output_defs_.emplace_back(ParamDef(backend, layout, dtype));
  }

  const std::vector<ParamDef>& input_defs() const { return input_defs_; }

  const std::vector<ParamDef>& output_defs() const { return output_defs_; }

 private:
  // TODO(chenweihang): replaced by paddle::small_vector
  std::vector<ParamDef> input_defs_{{}};
  std::vector<ParamDef> output_defs_{{}};
};

class OpKernel {
 public:
  // for map element contruct
  OpKernel() = default;

  explicit OpKernel(OpKernelFn fn) : fn_(fn) {}

  void operator()(OpKernelContext* ctx) const { fn_(ctx); }

  OpKernelParamDef* mutable_param_def() { return &param_def_; }

  const OpKernelParamDef& param_def() const { return param_def_; }

 private:
  OpKernelFn fn_{nullptr};
  OpKernelParamDef param_def_;
};

/**
 * Note: Each Operation need a basic kernel map that named by op_type.
 *       Such as for scale op, OpKernelMap contains a `scale` kernel map,
 *       if it still need other overload kernel, the op name can be
 *       `scale.***`.
 */
class OpKernelFactory {
 public:
  // replaced by paddle::flat_hash_map later
  using OpKernelMap = std::unordered_map<
      OperationName,
      std::unordered_map<OpKernelKey, OpKernel, OpKernelKey::Hash>,
      OperationName::Hash>;

  static OpKernelFactory& Instance();

  OpKernelMap& kernels() { return kernels_; }

  bool ContainsOperation(const char* op_type) const;

  const OpKernel& SelectKernel(const OperationName& op_name,
                               const OpKernelKey& kernel_key) const;

  const OpKernel& SelectKernel(const OperationName& op_name,
                               Backend backend,
                               DataLayout layout,
                               DataType dtype) const;

 private:
  OpKernelFactory() = default;

  OpKernelMap kernels_;
};

/** operator << overload **/

inline std::ostream& operator<<(std::ostream& os,
                                const OperationName& op_name) {
  if (op_name.overload_type.empty()) {
    os << op_name.op_type;
  } else {
    os << op_name.op_type << "." << op_name.overload_type;
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const OpKernelKey& kernel_key) {
  os << "(" << kernel_key.backend() << ", " << kernel_key.layout() << ", "
     << kernel_key.dtype() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, OpKernelFactory& kernel_factory);

}  // namespace pt
