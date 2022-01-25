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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_factory.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

class KernelArgsNameMakerByOpProto : public KernelArgsNameMaker {
 public:
  explicit KernelArgsNameMakerByOpProto(
      const framework::proto::OpProto* op_proto)
      : op_proto_(op_proto) {
    PADDLE_ENFORCE_NOT_NULL(op_proto_, platform::errors::InvalidArgument(
                                           "Op proto cannot be nullptr."));
  }

  ~KernelArgsNameMakerByOpProto() {}

  const paddle::SmallVector<std::string>& GetInputArgsNames() override;
  const paddle::SmallVector<std::string>& GetOutputArgsNames() override;
  const paddle::SmallVector<std::string>& GetAttrsArgsNames() override;

  KernelSignature GetKernelSignature();

 private:
  DISABLE_COPY_AND_ASSIGN(KernelArgsNameMakerByOpProto);

 private:
  const framework::proto::OpProto* op_proto_;

  paddle::SmallVector<std::string> input_names_;
  paddle::SmallVector<std::string> output_names_;
  paddle::SmallVector<std::string> attr_names_;
};

OpKernelType TransPtenKernelKeyToOpKernelType(
    const pten::KernelKey& kernel_key) {
  proto::VarType::Type data_type =
      pten::TransToProtoVarType(kernel_key.dtype());
  platform::Place place = pten::TransToFluidPlace(kernel_key.backend());
  DataLayout data_layout = kernel_key.layout();
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
  paddle::experimental::DataLayout layout = kernel_type.data_layout_;
  paddle::experimental::DataType dtype =
      pten::TransToPtenDataType(kernel_type.data_type_);
  return pten::KernelKey(backend, layout, dtype);
}

KernelSignatureMap* KernelSignatureMap::kernel_signature_map_ = nullptr;
std::once_flag KernelSignatureMap::init_flag_;

KernelSignatureMap& KernelSignatureMap::Instance() {
  std::call_once(init_flag_, [] {
    kernel_signature_map_ = new KernelSignatureMap();
    for (const auto& pair : OpInfoMap::Instance().map()) {
      const auto& op_type = pair.first;
      const auto* op_proto = pair.second.proto_;
      if (pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type) &&
          op_proto) {
        KernelArgsNameMakerByOpProto maker(op_proto);
        VLOG(10) << "Register kernel signature for " << op_type;
        auto success = kernel_signature_map_->map_
                           .emplace(pten::TransToPtenKernelName(op_type),
                                    std::move(maker.GetKernelSignature()))
                           .second;
        PADDLE_ENFORCE_EQ(
            success, true,
            platform::errors::PermissionDenied(
                "Kernel signature of the operator %s has been registered.",
                op_type));
      }
    }
  });
  return *kernel_signature_map_;
}

bool KernelSignatureMap::Has(const std::string& op_type) const {
  return map_.find(op_type) != map_.end();
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
      VLOG(6) << "Parse PtenKernel input: skip extra & quant input - "
              << in_name;
      continue;
    }
    // If contains dispensable input, we should override the
    // GetExpectedPtenKernelArgs method self
    if (in.has_dispensable() && in.dispensable()) {
      VLOG(6) << "Parse PtenKernel input: skip dispensable input - " << in_name;
      continue;
    }
    VLOG(6) << "Parse PtenKernel input: " << in_name;
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
    VLOG(6) << "Parse PtenKernel output: " << out_name;
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
      VLOG(6) << "Parse PtenKernel attribute: skip needless attr - "
              << attr_name;
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      VLOG(6) << "Parse PtenKernel attribute: skip extra & quant attr - "
              << attr_name;
      continue;
    }
    VLOG(6) << "Parse PtenKernel attribute: " << attr_name;
    attr_names_.emplace_back(attr_name);
  }

  return attr_names_;
}

KernelSignature KernelArgsNameMakerByOpProto::GetKernelSignature() {
  return KernelSignature(pten::TransToPtenKernelName(op_proto_->type()),
                         GetInputArgsNames(), GetAttrsArgsNames(),
                         GetOutputArgsNames());
}

void SetAllocationForOutputTenosr(pten::DenseTensor* tensor,
                                  const platform::Place& place) {
  if (!tensor->IsInitialized() || !(tensor->place() == place)) {
    int dtype_size = tensor->dtype() == DataType::UNDEFINED
                         ? 0
                         : experimental::SizeOf(tensor->dtype());
    int64_t numels = product(tensor->dims());
    numels = numels < 0 ? 0 : numels;
    auto tmp_allocation_ptr = memory::Alloc(place, numels * dtype_size);
    auto& deleter = tmp_allocation_ptr.get_deleter();
    auto* allocation_ptr = tmp_allocation_ptr.release();
    auto shared_allocation =
        std::shared_ptr<pten::Allocation>(allocation_ptr, deleter);

    tensor->ResetHolder(shared_allocation);
  }
}

}  // namespace framework
}  // namespace paddle
