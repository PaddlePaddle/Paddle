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
#include "paddle/top/core/layout.h"

namespace pt {

class OpKernelContext;

using OpKernelFn = void (*)(OpKernelContext* ctx);

struct OperationName final {
  std::string op_type;
  std::string overload_type;
  // Avoid calculating Hash value at runtime
  size_t hash_value;

  OperationName(std::string op_type, std::string overload_type)
      : op_type(std::move(op_type)), overload_type(std::move(overload_type)) {
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
  OpKernelKey(Backend backend, DataType dtype, DataLayout layout)
      : backend_(backend), dtype_(dtype), layout_(layout) {
    // |----31-20------|---19-16----|---15-8---|---7-0---|
    // | For extension | DataLayout | DataType | Backend |

    hash_value_ = 0;
    hash_value_ |= static_cast<uint8_t>(backend_);
    hash_value_ |= (static_cast<uint16_t>(dtype_) << kBackendBitLength);
    hash_value_ |= (static_cast<uint32_t>(layout_)
                    << (kBackendBitLength + kDataTypeBitLength));
  }

  Backend backend() const { return backend_; }
  DataType dtype() const { return dtype_; }
  DataLayout layout() const { return layout_; }

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
  constexpr static int kDataTypeBitLength = 8;
  constexpr static int kDataLayoutBitLength = 4;

  Backend backend_;
  DataType dtype_;
  DataLayout layout_;

  // Avoid calculating Hash value at runtime.
  // Note: Now the number of bits we need does not exceed 32 bits, so there is
  // no need to use 64 bits. If needed in the future, it can be expanded,
  // but now we don’t over-design.
  uint32_t hash_value_;
};

class OpKernelFactory {
 public:
  static OpKernelFactory& Instance();

  const OpKernelFn& FindOpKernel(const OperationName& op_name,
                                 const OpKernelKey& kernel_key) const;

 private:
  OpKernelFactory();

  // replaced by paddle::flat_hash_map later
  std::unordered_map<
      OperationName,
      std::unordered_map<OpKernelKey, OpKernelFn, OpKernelKey::Hash>,
      OperationName::Hash>
      kernels_;
};

/** operator << overload **/

inline std::ostream& operator<<(std::ostream& os,
                                const OperationName& op_name) {
  os << op_name.op_type << "." << op_name.overload_type;
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const OpKernelKey& kernel_key) {
  os << "(" << kernel_key.backend() << ", " << kernel_key.dtype() << ", "
     << kernel_key.layout() << ")";
  return os;
}

}  // namespace pt
