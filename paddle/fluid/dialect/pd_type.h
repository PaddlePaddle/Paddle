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

#ifdef GET_TYPE_LIST
#undef GET_TYPE_LIST
paddle::dialect::DenseTensorType
#else

#include "paddle/fluid/dialect/pd_type_storage.h"
#include "paddle/ir/type.h"

namespace paddle {
namespace dialect {
///
/// \brief Define built-in parametric types.
///
class DenseTensorType : public ir::Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(DenseTensorType, DenseTensorTypeStorage);

  const ir::Type &dtype() const;

  const paddle::dialect::DenseTensorTypeStorage::Dim &dim() const;

  const paddle::dialect::DenseTensorTypeStorage::DataLayout &data_layout()
      const;

  const paddle::dialect::DenseTensorTypeStorage::LoD &lod() const;

  const size_t &offset() const;
};

}  // namespace dialect
}  // namespace paddle
#endif
