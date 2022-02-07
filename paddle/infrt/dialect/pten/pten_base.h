// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <string>

#include "paddle/infrt/dialect/pten/infrt_pten_base.h.inc"
#include "paddle/infrt/dialect/pten/infrt_pten_baseDialect.h.inc"
#include "paddle/infrt/dialect/pten/infrt_pten_baseTypes.h.inc"

namespace infrt {
namespace pten {

// struct AllocatorTypeStorage : public mlir::TypeStorage {
//   AllocatorTypeStorage(const std::string& kind) : kind_(kind) {}

//   bool operator==(const std::string& key) const { return key == kind_; }

//   static llvm::hash_code hashKey(const std::string& key) {
//     return llvm::hash_value(key);
//   }

//   static AllocatorTypeStorage* construct(mlir::TypeStorageAllocator&
//   allocator,
//                                          const std::string& key) {
//     return new (allocator.allocate<AllocatorTypeStorage>())
//         AllocatorTypeStorage(key);
//   }

//  private:
//   std::string kind_;
// };

// class AllocatorType : public mlir::Type::TypeBase<AllocatorType,
//                                                   mlir::Type,
//                                                   AllocatorTypeStorage> {
//  public:
//   using Base::Base;
//   static AllocatorType get(const std::string& kind);

//   static std::string& kind();
// };

}  // namespace pten
}  // namespace infrt
