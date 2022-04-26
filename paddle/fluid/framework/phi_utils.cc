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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/phi_utils.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/type_defs.h"

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

  const paddle::SmallVector<const char*>& GetInputArgsNames() override;
  const paddle::SmallVector<const char*>& GetOutputArgsNames() override;
  const paddle::SmallVector<const char*>& GetAttrsArgsNames() override;

  phi::KernelSignature GetKernelSignature();

 private:
  DISABLE_COPY_AND_ASSIGN(KernelArgsNameMakerByOpProto);

 private:
  const framework::proto::OpProto* op_proto_;

  paddle::SmallVector<const char*> input_names_;
  paddle::SmallVector<const char*> output_names_;
  paddle::SmallVector<const char*> attr_names_;
};

OpKernelType TransPhiKernelKeyToOpKernelType(const phi::KernelKey& kernel_key) {
  proto::VarType::Type data_type =
      paddle::framework::TransToProtoVarType(kernel_key.dtype());
  // no need to set current device id here
  platform::Place place = phi::TransToPhiPlace(kernel_key.backend(), false);
  DataLayout data_layout = kernel_key.layout();
  LibraryType library_type = LibraryType::kPlain;
  if (kernel_key.backend() == phi::Backend::MKLDNN) {
    library_type = LibraryType::kMKLDNN;
  } else if (kernel_key.backend() == phi::Backend::GPUDNN) {
    library_type = LibraryType::kCUDNN;
  } else if (kernel_key.backend() == phi::Backend::KPS) {
    library_type = LibraryType::kKP;
  } else {
    // do nothing
  }
  // TODO(chenweihang): the customized_type_value is lost
  return OpKernelType(data_type, place, data_layout, library_type);
}

phi::KernelKey TransOpKernelTypeToPhiKernelKey(
    const OpKernelType& kernel_type) {
  phi::Backend backend = phi::TransToPhiBackend(kernel_type.place_);
  if (kernel_type.library_type_ == LibraryType::kMKLDNN) {
    backend = phi::Backend::MKLDNN;
  } else if (kernel_type.library_type_ == LibraryType::kCUDNN) {
    backend = phi::Backend::GPUDNN;
  } else if (kernel_type.library_type_ == LibraryType::kKP) {
    backend = phi::Backend::KPS;
  } else {
    // do nothing
  }
  paddle::experimental::DataLayout layout = kernel_type.data_layout_;
  paddle::experimental::DataType dtype =
      paddle::framework::TransToPhiDataType(kernel_type.data_type_);
  return phi::KernelKey(backend, layout, dtype);
}

phi::KernelKey FallBackToCpu(const OpKernelType& expected_kernel_key,
                             const phi::KernelKey& kernel_key,
                             const framework::OperatorBase& op) {
#ifdef PADDLE_WITH_XPU
  if (platform::is_xpu_place(expected_kernel_key.place_) ||
      paddle::platform::is_in_xpu_black_list(op.Type())) {
    VLOG(3) << "phi missing XPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    return phi::KernelKey(phi::Backend::CPU, kernel_key.layout(),
                          kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(expected_kernel_key.place_)) {
    VLOG(3) << "phi missing NPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    return phi::KernelKey(phi::Backend::CPU, kernel_key.layout(),
                          kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_MLU
  if (platform::is_mlu_place(expected_kernel_key.place_)) {
    VLOG(3) << "phi missing MLU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    return phi::KernelKey(phi::Backend::CPU, kernel_key.layout(),
                          kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_IPU
  if (platform::is_ipu_place(expected_kernel_key.place_)) {
    VLOG(3) << "phi missing IPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    return phi::KernelKey(phi::Backend::CPU, kernel_key.layout(),
                          kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (platform::is_custom_place(expected_kernel_key.place_)) {
    VLOG(3) << "phi missing " << expected_kernel_key.place_.GetDeviceType()
            << " kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    return phi::KernelKey(phi::Backend::CPU, kernel_key.layout(),
                          kernel_key.dtype());
  }
#endif
  return phi::KernelKey();
}

const paddle::SmallVector<const char*>&
KernelArgsNameMakerByOpProto::GetInputArgsNames() {
  for (int i = 0; i < op_proto_->inputs_size(); ++i) {
    auto& in = op_proto_->inputs()[i];
    auto& in_name = in.name();
    if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
      continue;
    }
    // If contains dispensable input, we should override the
    // OpArgumentMapping method self in phi/ops/compat dir
    if (in.has_dispensable() && in.dispensable()) {
      continue;
    }
    input_names_.emplace_back(in_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel inputs: ";
    std::copy(input_names_.begin(), input_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return input_names_;
}

const paddle::SmallVector<const char*>&
KernelArgsNameMakerByOpProto::GetOutputArgsNames() {
  for (int i = 0; i < op_proto_->outputs_size(); ++i) {
    auto& out = op_proto_->outputs()[i];
    auto& out_name = out.name();
    if ((out.has_extra() && out.extra()) || (out.has_quant() && out.quant())) {
      continue;
    }
    output_names_.emplace_back(out_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel outputs: ";
    std::copy(output_names_.begin(), output_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return output_names_;
}

const paddle::SmallVector<const char*>&
KernelArgsNameMakerByOpProto::GetAttrsArgsNames() {
  for (int i = 0; i < op_proto_->attrs_size(); ++i) {
    auto& attr = op_proto_->attrs()[i];
    auto& attr_name = attr.name();
    if (attr_name == "use_mkldnn" || attr_name == "use_cudnn" ||
        attr_name == "op_role" || attr_name == "op_role_var" ||
        attr_name == "op_namescope" || attr_name == "op_callstack" ||
        attr_name == "op_device") {
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      continue;
    }
    attr_names_.emplace_back(attr_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel attributes: ";
    std::copy(attr_names_.begin(), attr_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return attr_names_;
}

phi::KernelSignature KernelArgsNameMakerByOpProto::GetKernelSignature() {
  return phi::KernelSignature(
      phi::TransToPhiKernelName(op_proto_->type()).c_str(), GetInputArgsNames(),
      GetAttrsArgsNames(), GetOutputArgsNames());
}

std::once_flag kernel_sig_map_init_flag;

void InitDefaultKernelSignatureMap() {
  std::call_once(kernel_sig_map_init_flag, [] {
    for (const auto& pair : paddle::framework::OpInfoMap::Instance().map()) {
      const auto& op_type = pair.first;
      const auto* op_proto = pair.second.proto_;
      if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type) &&
          op_proto) {
        paddle::framework::KernelArgsNameMakerByOpProto maker(op_proto);
        VLOG(10) << "Register `" << op_type << "` kernel signature:";
        phi::DefaultKernelSignatureMap::Instance().Insert(
            op_type, std::move(maker.GetKernelSignature()));
      }
    }
  });
}

static void SetAllocationForUninitializedDenseTensor(
    phi::DenseTensor* dense_tensor, const platform::Place& place) {
  int dtype_size = dense_tensor->dtype() == DataType::UNDEFINED
                       ? 0
                       : experimental::SizeOf(dense_tensor->dtype());
  int64_t numels = product(dense_tensor->dims());
  numels = numels < 0 ? 0 : numels;
  auto tmp_allocation_ptr = memory::Alloc(place, numels * dtype_size);
  auto& deleter = tmp_allocation_ptr.get_deleter();
  auto* allocation_ptr = tmp_allocation_ptr.release();
  auto shared_allocation =
      std::shared_ptr<phi::Allocation>(allocation_ptr, deleter);

  dense_tensor->ResetHolder(shared_allocation);
}

}  // namespace framework
}  // namespace paddle
