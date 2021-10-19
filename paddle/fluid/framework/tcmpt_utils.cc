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

#include "paddle/fluid/framework/tcmpt_utils.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

// TODO(chenweihang, shixiaowei): adapt SelectedRows
template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor, LoDTensor>(
    const LoDTensor& tensor, pt::Backend backend, pt::DataType dtype,
    pt::DataLayout layout) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(tensor.dims(), backend, dtype, layout, tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    VLOG(1) << "Old LoDTensor holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor, Tensor>(
    const Tensor& tensor, pt::Backend backend, pt::DataType dtype,
    pt::DataLayout layout) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(tensor.dims(), backend, dtype, layout, tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    VLOG(1) << "Old Tensor holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const LoDTensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  return MakeTensorImpl<pt::DenseTensor, LoDTensor>(
      tensor, pt::TransToPtBackend(place), pt::TransToPtDataType(type),
      pt::TransToPtDataLayout(tensor.layout()));
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const Tensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  return MakeTensorImpl<pt::DenseTensor, Tensor>(
      tensor, pt::TransToPtBackend(place), pt::TransToPtDataType(type),
      pt::TransToPtDataLayout(tensor.layout()));
}

std::shared_ptr<pt::TensorInterface> InputVariableToPtTensor(
    const framework::Variable& variable, const pt::TensorArgDef& arg_def) {
  auto expected_place = pt::TransToFluidPlace(arg_def.backend);

  if (variable.template IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.template Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
              tmp_tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    } else {
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
              tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    }
  } else if (variable.template IsType<framework::SelectedRows>()) {
    // TODO(chenweihang): now we don't deal with row and height
    // by xiaowei's advice
    const auto& tensor = variable.template Get<framework::SelectedRows>();
    if (!platform::is_same_place(tensor.value().place(), expected_place)) {
      framework::Tensor tmp_tensor;
      TensorCopySync(tensor.value(), expected_place, &tmp_tensor);
      // TODO(chenweihang): adapt SelectedRows by xiaowei's design
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
              tmp_tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    } else {
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
              tensor.value(), arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable.Type())));
  }
  return nullptr;
}

std::shared_ptr<pt::TensorInterface> OutputVariableToPtTensor(
    framework::Variable* variable, const pt::TensorArgDef& arg_def) {
  // mutable_data before run kernel, to avoid share output form
  // KernelContext to original tensor
  if (variable->template IsType<framework::LoDTensor>()) {
    auto* tensor = variable->template GetMutable<framework::LoDTensor>();
    tensor->mutable_data(pt::TransToFluidPlace(arg_def.backend),
                         pt::TransToProtoVarType(arg_def.dtype));
    auto pt_out =
        framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
            *tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
    return pt_out;
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    tensor->mutable_value()->mutable_data(
        pt::TransToFluidPlace(arg_def.backend),
        pt::TransToProtoVarType(arg_def.dtype));
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    auto pt_out = framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
        tensor->value(), arg_def.backend, arg_def.dtype, arg_def.layout);
    return pt_out;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared output `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }

  return nullptr;
}

OpKernelType TransPtKernelKeyToOpKernelType(const pt::KernelKey& kernel_key) {
  proto::VarType::Type data_type = pt::TransToProtoVarType(kernel_key.dtype());
  platform::Place place = pt::TransToFluidPlace(kernel_key.backend());
  DataLayout data_layout = pt::TransToFluidDataLayout(kernel_key.layout());
  LibraryType library_type = LibraryType::kPlain;
  if (kernel_key.backend() == pt::Backend::kMKLDNN) {
    library_type = LibraryType::kMKLDNN;
  } else if (kernel_key.backend() == pt::Backend::kCUDNN) {
    library_type = LibraryType::kCUDNN;
  } else {
    // do nothing
  }
  // TODO(chenweihang): the customized_type_value is lost
  return OpKernelType(data_type, place, data_layout, library_type);
}

pt::KernelKey TransOpKernelTypeToPtKernelKey(const OpKernelType& kernel_type) {
  pt::Backend backend = pt::TransToPtBackend(kernel_type.place_);
  if (kernel_type.library_type_ == LibraryType::kMKLDNN) {
    backend = pt::Backend::kMKLDNN;
  } else if (kernel_type.library_type_ == LibraryType::kCUDNN) {
    backend = pt::Backend::kCUDNN;
  } else {
    // do
  }
  pt::DataLayout layout = pt::TransToPtDataLayout(kernel_type.data_layout_);
  pt::DataType dtype = pt::TransToPtDataType(kernel_type.data_type_);
  return pt::KernelKey(backend, layout, dtype);
}

KernelSignatureMap& KernelSignatureMap::Instance() {
  static KernelSignatureMap g_kernel_signature_map;
  return g_kernel_signature_map;
}

const paddle::SmallVector<std::string>&
KernelArgsNameMakerByOpProto::GetInputArgsNames() {
  for (int i = 0; i < op_proto_->inputs_size(); ++i) {
    auto& in = op_proto_->inputs()[i];
    auto& in_name = in.name();
    if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
      VLOG(1) << "Parse PtKernel input: skip extra & quant input - " << in_name;
      continue;
    }
    // If contains dispensable input, we should override the
    // GetExpectedPtKernelArgs method self
    if (in.has_dispensable() && in.dispensable()) {
      VLOG(1) << "Parse PtKernel input: skip dispensable input - " << in_name;
      continue;
    }
    VLOG(1) << "Parse PtKernel input: " << in_name;
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
    VLOG(1) << "Parse PtKernel output: " << out_name;
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
      VLOG(1) << "Parse PtKernel attribute: skip needless attr - " << attr_name;
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      VLOG(1) << "Parse PtKernel attribute: skip extra & quant attr - "
              << attr_name;
      continue;
    }
    VLOG(1) << "Parse PtKernel attribute: " << attr_name;
    attr_names_.emplace_back(attr_name);
  }

  return attr_names_;
}

KernelSignature KernelArgsNameMakerByOpProto::GetKernelSignature() {
  return std::make_pair(
      op_proto_->type(),
      std::make_tuple(GetInputArgsNames(), GetAttrsArgsNames(),
                      GetOutputArgsNames()));
}

std::string KernelSignatureToString(const KernelSignature& signature) {
  std::stringstream os;
  os << "Kernel Signature - name: " << signature.first << "; inputs: "
     << string::join_strings(std::get<0>(signature.second), ", ")
     << "; attributes: "
     << string::join_strings(std::get<1>(signature.second), ", ")
     << "; outputs: "
     << string::join_strings(std::get<2>(signature.second), ", ");
  return os.str();
}

}  // namespace framework
}  // namespace paddle
