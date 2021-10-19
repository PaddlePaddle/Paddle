/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/tcmpt/api/include/core.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace paddle {
namespace framework {

/* tensor translate */

template <typename PtTensorImplT, typename VariableT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const VariableT& tensor,
                                              pt::Backend backend,
                                              pt::DataType dtype,
                                              pt::DataLayout layout);

template <typename PtTensorImplT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const LoDTensor& tensor,
                                              const platform::Place& place,
                                              proto::VarType::Type type);

template <typename PtTensorImplT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const Tensor& tensor,
                                              const platform::Place& place,
                                              proto::VarType::Type type);

std::shared_ptr<pt::TensorInterface> InputVariableToPtTensor(
    const framework::Variable& variable, const pt::TensorArgDef& arg_def);
std::shared_ptr<pt::TensorInterface> OutputVariableToPtTensor(
    framework::Variable* variable, const pt::TensorArgDef& arg_def);

/* Kernel Key translate */

OpKernelType TransPtKernelKeyToOpKernelType(const pt::KernelKey& kernel_key);
pt::KernelKey TransOpKernelTypeToPtKernelKey(const OpKernelType& kernel_type);

/* Kernel Args parse */

// TODO(chenweihang): we can generate this map by proto info in compile time
class KernelSignatureMap {
 public:
  static KernelSignatureMap& Instance();

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& op_type, const KernelSignature& signature) {
    if (!Has(op_type)) {
      map_.insert({op_type, signature});
    }
  }

  const KernelSignature* GetNullable(const std::string& op_type) const {
    auto it = map_.find(op_type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

 private:
  KernelSignatureMap() = default;
  paddle::flat_hash_map<std::string, KernelSignature> map_;

  DISABLE_COPY_AND_ASSIGN(KernelSignatureMap);
};

class KernelArgsNameMaker {
 public:
  virtual ~KernelArgsNameMaker() {}
  virtual const paddle::SmallVector<std::string>& GetInputArgsNames() = 0;
  virtual const paddle::SmallVector<std::string>& GetOutputArgsNames() = 0;
  virtual const paddle::SmallVector<std::string>& GetAttrsArgsNames() = 0;
};

class KernelArgsNameMakerByOpProto : public KernelArgsNameMaker {
 public:
  explicit KernelArgsNameMakerByOpProto(framework::proto::OpProto* op_proto)
      : op_proto_(op_proto) {}

  ~KernelArgsNameMakerByOpProto() {}

  const paddle::SmallVector<std::string>& GetInputArgsNames() override;
  const paddle::SmallVector<std::string>& GetOutputArgsNames() override;
  const paddle::SmallVector<std::string>& GetAttrsArgsNames() override;

  KernelSignature GetKernelSignature();

 private:
  framework::proto::OpProto* op_proto_;

  paddle::SmallVector<std::string> input_names_;
  paddle::SmallVector<std::string> output_names_;
  paddle::SmallVector<std::string> attr_names_;
};

std::string KernelSignatureToString(const KernelSignature& signature);

}  // namespace framework
}  // namespace paddle
