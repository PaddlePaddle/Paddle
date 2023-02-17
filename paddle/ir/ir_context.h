// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <memory>
#include <unordered_map>

namespace ir {
class IrContextImpl;
class StorageManager;
class AbstractType;
class TypeId;

///
/// \brief IrContext is a global parameterless class used to store and manage
/// Type and its related data structures.
///
class IrContext {
 public:
  ///
  /// \brief Initializes a new instance of IrContext.
  ///
  /// \return Global parameterless for IrContext.
  ///
  static IrContext *Instance() {
    VLOG(4) << "=> Instance called with context (ir_context_ == " << ir_context_
            << ").";
    if (ir_context_ == nullptr) {
      ir_context_ = new IrContext();
    }
    VLOG(4) << "    |-> (ir_context_ == " << ir_context_ << ").";
    return ir_context_;
  }

  ~IrContext();

  ///
  /// \brief Get an instance of IrContextImpl, a private member of IrContext.
  /// For the specific definition of IrContextImpl, see ir_context.cc.
  ///
  /// \return The instance of IrContextImpl.
  ///
  IrContextImpl &impl() { return *impl_; }

  ///
  /// \brief Returns the storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  /// \return The storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  StorageManager &storage_manager();

  ///
  /// \brief Returns the storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  /// \return The storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  std::unordered_map<TypeId, AbstractType *> &registed_abstracted_type();

 private:
  IrContext();

  IrContext(const IrContext &) = delete;

  void operator=(const IrContext &) = delete;

  static IrContext *ir_context_;

  const std::unique_ptr<IrContextImpl> impl_;
};

}  // namespace ir
