// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <memory>
#include <set>
#include "paddle/common/macros.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace egr {

class ActivationOffloaderWithPlace;

class ReloadFunctor {
 public:
  explicit ReloadFunctor(std::weak_ptr<phi::DenseTensor> tensor,
                         ActivationOffloaderWithPlace *offloader);

  void Reload();

 private:
  std::weak_ptr<phi::DenseTensor> tensor_;
  ActivationOffloaderWithPlace *offloader_;
};

class ActivationOffloaderWithPlace {
 public:
  explicit ActivationOffloaderWithPlace(phi::GPUPlace place);

  void SetSkipTensors(const std::vector<paddle::Tensor> &tensors);

  paddle::optional<ReloadFunctor> Add(const paddle::Tensor &activation);

  size_t Offload(size_t size);

  void Remove(const std::weak_ptr<phi::DenseTensor> &tensor);

  phi::GPUPlace Place() const { return place_; }

  size_t CachedSize() const;

 private:
  void Shrink();

  DISABLE_COPY_AND_ASSIGN(ActivationOffloaderWithPlace);

 private:
  using WeakTensorSet =
      std::set<std::weak_ptr<phi::DenseTensor>,
               std::owner_less<std::weak_ptr<phi::DenseTensor>>>;
  using WeakTensorMap =
      std::map<std::weak_ptr<phi::DenseTensor>,
               size_t,
               std::owner_less<std::weak_ptr<phi::DenseTensor>>>;
  phi::GPUPlace place_;
  WeakTensorMap activations_;
  WeakTensorSet skip_tensors_;
};

class ActivationOffloader {
 private:
  ActivationOffloader() = default;

 public:
  void SetSkipTensors(const std::vector<paddle::Tensor> &tensors);

  paddle::optional<ReloadFunctor> Add(const paddle::Tensor &activation);

  size_t Offload(phi::Place place, size_t size);

  size_t CachedSize() const;

  static ActivationOffloader *Instance();

 private:
  ActivationOffloaderWithPlace *GetOrCreateOffloader(phi::Place place);

  DISABLE_COPY_AND_ASSIGN(ActivationOffloader);

 private:
  std::map<phi::Place, std::unique_ptr<ActivationOffloaderWithPlace>>
      offloaders_;
};

}  // namespace egr
