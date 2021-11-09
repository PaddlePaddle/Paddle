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

#include <sstream>

#include "paddle/fluid/framework/pten_utils.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

OpKernelType TransPtenKernelKeyToOpKernelType(
    const pten::KernelKey& kernel_key) {
  proto::VarType::Type data_type =
      pten::TransToProtoVarType(kernel_key.dtype());
  platform::Place place = pten::TransToFluidPlace(kernel_key.backend());
  DataLayout data_layout = pten::TransToFluidDataLayout(kernel_key.layout());
  LibraryType library_type = LibraryType::kPlain;
  if (kernel_key.backend() == pten::Backend::MKLDNN) {
    library_type = LibraryType::kMKLDNN;
  } else if (kernel_key.backend() == pten::Backend::CUDNN) {
    library_type = LibraryType::kCUDNN;
  } else {
    // do nothing
  }
  // TODO(chenweihang): the customized_type_value is lost
  return OpKernelType(data_type, place, data_layout, library_type);
}

pten::KernelKey TransOpKernelTypeToPtenKernelKey(
    const OpKernelType& kernel_type) {
  pten::Backend backend = pten::TransToPtenBackend(kernel_type.place_);
  if (kernel_type.library_type_ == LibraryType::kMKLDNN) {
    backend = pten::Backend::MKLDNN;
  } else if (kernel_type.library_type_ == LibraryType::kCUDNN) {
    backend = pten::Backend::CUDNN;
  } else {
    // do
  }
  paddle::experimental::DataLayout layout =
      pten::TransToPtenDataLayout(kernel_type.data_layout_);
  paddle::experimental::DataType dtype =
      pten::TransToPtenDataType(kernel_type.data_type_);
  return pten::KernelKey(backend, layout, dtype);
}

KernelSignatureMap* KernelSignatureMap::kernel_signature_map_ = nullptr;
std::mutex KernelSignatureMap::mutex_;

KernelSignatureMap& KernelSignatureMap::Instance() {
  if (kernel_signature_map_ == nullptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (kernel_signature_map_ == nullptr) {
      kernel_signature_map_ = new KernelSignatureMap;
    }
  }
  return *kernel_signature_map_;
}

bool KernelSignatureMap::Has(const std::string& op_type) const {
  return map_.find(op_type) != map_.end();
}

void KernelSignatureMap::Emplace(const std::string& op_type,
                                 KernelSignature&& signature) {
  if (!Has(op_type)) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!Has(op_type)) {
      map_.emplace(op_type, signature);
    }
  }
}

const KernelSignature& KernelSignatureMap::Get(
    const std::string& op_type) const {
  auto it = map_.find(op_type);
  PADDLE_ENFORCE_NE(
      it, map_.end(),
      platform::errors::NotFound(
          "Operator `%s`'s kernel signature is not registered.", op_type));
  return it->second;
}

const paddle::SmallVector<std::string>&
KernelArgsNameMakerByOpProto::GetInputArgsNames() {
  for (int i = 0; i < op_proto_->inputs_size(); ++i) {
    auto& in = op_proto_->inputs()[i];
    auto& in_name = in.name();
    if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
      VLOG(1) << "Parse PtenKernel input: skip extra & quant input - "
              << in_name;
      continue;
    }
    // If contains dispensable input, we should override the
    // GetExpectedPtenKernelArgs method self
    if (in.has_dispensable() && in.dispensable()) {
      VLOG(1) << "Parse PtenKernel input: skip dispensable input - " << in_name;
      continue;
    }
    VLOG(1) << "Parse PtenKernel input: " << in_name;
    input_names_.emplace_back(in_name);
  }
  return input_names_;
}

const paddle::SmallVector<std::string>&
KernelArgsNameMakerByOpProto::GetOutputArgsNames() {
  for (int i = 0; i < op_proto_->outputs_size(); ++i) {
    auto& out = op_proto_->outputs()[i];
    auto& out_name = out.name();
    // TODO(chenweihang): outputs also need skip some cases
    VLOG(1) << "Parse PtenKernel output: " << out_name;
    output_names_.emplace_back(out_name);
  }
  return output_names_;
}

const paddle::SmallVector<std::string>&
KernelArgsNameMakerByOpProto::GetAttrsArgsNames() {
  for (int i = 0; i < op_proto_->attrs_size(); ++i) {
    auto& attr = op_proto_->attrs()[i];
    auto& attr_name = attr.name();
    if (attr_name == "use_mkldnn" || attr_name == "op_role" ||
        attr_name == "op_role_var" || attr_name == "op_namescope" ||
        attr_name == "op_callstack" || attr_name == "op_device") {
      VLOG(1) << "Parse PtenKernel attribute: skip needless attr - "
              << attr_name;
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      VLOG(1) << "Parse PtenKernel attribute: skip extra & quant attr - "
              << attr_name;
      continue;
    }
    VLOG(1) << "Parse PtenKernel attribute: " << attr_name;
    attr_names_.emplace_back(attr_name);
  }

  return attr_names_;
}

KernelSignature KernelArgsNameMakerByOpProto::GetKernelSignature() {
  return KernelSignature(op_proto_->type(), GetInputArgsNames(),
                         GetAttrsArgsNames(), GetOutputArgsNames());
}

std::string KernelSignatureToString(const KernelSignature& signature) {
  std::stringstream os;
  os << "Kernel Signature - name: " << signature.name
     << "; inputs: " << string::join_strings(std::get<0>(signature.args), ", ")
     << "; attributes: "
     << string::join_strings(std::get<1>(signature.args), ", ") << "; outputs: "
     << string::join_strings(std::get<2>(signature.args), ", ");
  return os.str();
}

}  // namespace framework
}  // namespace paddle
