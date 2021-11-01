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

// Eager Dygraph

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/framework/variable.h"

TEST(Utils, SyncToVarsSingle) {
  egr::InitEnv(paddle::platform::CPUPlace());

  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
  egr::EagerTensor tensor = egr::EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0, true);

  std::vector<std::shared_ptr<egr::EagerTensor>> var_bases = SyncToVars(tensor);

  paddle::framework::Variable* var = var_bases[0]->MutableVar();
  const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

  const float* ptr = framework_tensor.data<float>();
  if (framework_tensor.numel() != tensor.numel())
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "VarBase does not match egr::EagerTensor in size"));

  for (int i = 0; i < framework_tensor.numel(); i++) {
    if (ptr[i] != 5.0)
      PADDLE_THROW(
          paddle::platform::errors::Fatal("%d does not match 5.0", ptr[i]));
  }
}

TEST(Utils, SyncToVarsMultiple) {
  egr::InitEnv(paddle::platform::CPUPlace());

  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
  std::vector<egr::EagerTensor> tensors = {
      egr::EagerUtils::CreateTensorWithValue(ddim, pten::Backend::CPU,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW, 1.0, true),
      egr::EagerUtils::CreateTensorWithValue(
          ddim, pten::Backend::CPU, pten::DataType::FLOAT32,
          pten::DataLayout::NCHW, 2.0, true)};

  std::vector<std::shared_ptr<egr::EagerTensor>> var_bases =
      SyncToVars(tensors);

  {
    paddle::framework::Variable* var = var_bases[0]->MutableVar();
    const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

    const float* ptr = framework_tensor.data<float>();
    if (framework_tensor.numel() != tensors[0].numel())
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "VarBase does not match egr::EagerTensor in size"));

    for (int i = 0; i < framework_tensor.numel(); i++) {
      if (ptr[i] != 1.0)
        PADDLE_THROW(
            paddle::platform::errors::Fatal("%d does not match 1.0", ptr[i]));
    }
  }

  {
    paddle::framework::Variable* var = var_bases[1]->MutableVar();
    const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

    const float* ptr = framework_tensor.data<float>();
    if (framework_tensor.numel() != tensors[0].numel())
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "VarBase does not match egr::EagerTensor in size"));

    for (int i = 0; i < framework_tensor.numel(); i++) {
      if (ptr[i] != 2.0)
        PADDLE_THROW(
            paddle::platform::errors::Fatal("%d does not match 2.0", ptr[i]));
    }
  }
}

TEST(Utils, SyncToTensorSingle) {
  egr::InitEnv(paddle::platform::CPUPlace());

  std::shared_ptr<egr::EagerTensor> X(new egr::EagerTensor());
  std::vector<float> src_data(128, 5.0);
  std::vector<int64_t> dims = {2, 4, 4, 4};
  paddle::platform::CPUPlace place;

  auto* x_tensor = X->MutableVar()->GetMutable<paddle::framework::LoDTensor>();
  x_tensor->Resize(paddle::framework::make_ddim(dims));
  auto* mutable_x = x_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  auto X_ = SyncToTensors(*(X.get()));
  egr::EagerTensor tensor = GetOutput(X_[0]);

  PADDLE_ENFORCE(
      egr::CompareTensorWithValue<float>(tensor, 5.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 5.0));
}

TEST(Utils, SyncToTensorMultiple) {
  egr::InitEnv(paddle::platform::CPUPlace());
  std::vector<int64_t> dims = {2, 4, 4, 4};
  paddle::platform::CPUPlace place;

  std::vector<egr::EagerTensor> egr_tensors;
  {
    auto egr_tensor = egr::EagerTensor();
    std::vector<float> src_data(128, 1.0);
    auto* x_tensor =
        egr_tensor.MutableVar()->GetMutable<paddle::framework::LoDTensor>();
    x_tensor->Resize(paddle::framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());
    egr_tensors.emplace_back(egr_tensor);
  }
  {
    auto egr_tensor = egr::EagerTensor();
    std::vector<float> src_data(128, 2.0);
    auto* x_tensor =
        egr_tensor.MutableVar()->GetMutable<paddle::framework::LoDTensor>();
    x_tensor->Resize(paddle::framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());
    egr_tensors.emplace_back(std::move(egr_tensor));
  }
  std::vector<egr::EagerTensor> tensors =
      GetOutputs(SyncToTensors(egr_tensors));

  PADDLE_ENFORCE(
      egr::CompareTensorWithValue<float>(tensors[0], 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));
  PADDLE_ENFORCE(
      egr::CompareTensorWithValue<float>(tensors[1], 2.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 2.0));
}
