// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/tcmpt_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/pten/core/layout.h"

/* ---- Tensor -> VarBase ---- */
static std::shared_ptr<paddle::imperative::VarBase> TensorToVarBase(
    const paddle::experimental::Tensor& tensor) {
  // Create imperative::VarBase with underlying type of framework::Tensor
  auto var_base = std::make_shared<paddle::imperative::VarBase>(
      false /*has_grad*/, "whatever" /*name*/);
  paddle::framework::Variable* var = var_base->MutableVar();
  paddle::framework::Tensor* framework_tensor =
      var->GetMutable<paddle::framework::LoDTensor>();

  framework_tensor->Resize(tensor.shape());
  framework_tensor->set_layout(ptenTransToFluidDataLayout(tensor.layout()));

  std::shared_ptr<ptenTensorInterface> tensor_interface = tensor.impl();
  // Contruct framework::Tensor from paddle::experimental::Tensor
  if (auto tensor_dense =
          std::dynamic_pointer_cast<ptenDenseTensor>(tensor_interface)) {
    paddle::framework::ShareTensorImpl<ptenDenseTensor>(tensor_dense.get(),
                                                        framework_tensor);

  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Unrecognized paddle::experimental::Tensor type, only "
        "DenseTensor is supported for now."));
  }
  return var_base;
}

std::vector<std::shared_ptr<paddle::imperative::VarBase>> TensorsToVarBases(
    const paddle::experimental::Tensor& tensor) {
  return {TensorToVarBase(tensor)};
}

std::vector<std::shared_ptr<paddle::imperative::VarBase>> TensorsToVarBases(
    const std::vector<paddle::experimental::Tensor>& tensors) {
  std::vector<std::shared_ptr<paddle::imperative::VarBase>> var_bases;

  for (const paddle::experimental::Tensor& tensor : tensors) {
    var_bases.emplace_back(std::move(TensorToVarBase(tensor)));
  }

  return var_bases;
}

/* ---- VarBase -> Tensor ---- */
paddle::experimental::Tensor VarBaseToTensor(
    const std::shared_ptr<paddle::imperative::VarBase>& var_base) {
  // Get Underlying Tensor from VarBase
  paddle::framework::Variable* var = var_base->MutableVar();

  paddle::framework::DDim ddim;
  ptenBackend backend;
  ptenDataType dtype;
  ptenDataLayout layout;
  std::shared_ptr<paddle::memory::allocation::Allocation> allocation;

  if (var->IsType<paddle::framework::LoDTensor>()) {
    const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();
    ddim = framework_tensor.dims();
    backend = ptenTransToPtBackend(framework_tensor.place());
    dtype = ptenTransToPtDataType(framework_tensor.type());
    layout = ptenTransToPtLayout(framework_tensor.layout());
    allocation = framework_tensor.Holder();

  } else if (var->IsType<paddle::framework::Tensor>()) {
    const auto& framework_tensor = var->Get<paddle::framework::Tensor>();
    ddim = framework_tensor.dims();
    backend = ptenTransToPtBackend(framework_tensor.place());
    dtype = ptenTransToPtDataType(framework_tensor.type());
    layout = ptenTransToPtLayout(framework_tensor.layout());
    allocation = framework_tensor.Holder();

  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Unable to fetch underlying tensor from VarBase, only LoDTensor and "
        "Tensor are supported for now"));
  }
  auto tensor_meta = ptenTensorMeta(ddim, backend, dtype, layout);
  auto tensor_dense = std::make_shared<ptenDenseTensor>(std::move(tensor_meta),
                                                        ptenTensorStatus());
  tensor_dense->ShareAllocation(allocation);

  return paddle::experimental::Tensor(tensor_dense);
}

std::vector<paddle::experimental::Tensor> VarBasesToTensors(
    const std::shared_ptr<paddle::imperative::VarBase>& var_base) {
  return {VarBaseToTensor(var_base)};
}

std::vector<paddle::experimental::Tensor> VarBasesToTensors(
    const std::vector<std::shared_ptr<paddle::imperative::VarBase>>&
        var_bases) {
  std::vector<paddle::experimental::Tensor> tensors;

  for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
       var_bases) {
    tensors.emplace_back(std::move(VarBaseToTensor(var_base)));
  }

  return tensors;
}

std::vector<std::shared_ptr<paddle::imperative::VarBase>>
ConstructDuplicableOutput(const size_t num) {
  auto tracer = paddle::imperative::GetCurrentTracer();
  std::vector<std::shared_ptr<paddle::imperative::VarBase>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto var_base_name = tracer->GenerateUniqueName();
    res.emplace_back(new paddle::imperative::VarBase(var_base_name));
  }
  return res;
}
