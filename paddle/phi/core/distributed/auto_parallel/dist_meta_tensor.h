/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {
namespace distributed {

class DistMetaTensor : public MetaTensor {
 public:
  DistMetaTensor() : MetaTensor() {}

  // supporting implicit construction is easier to use
  DistMetaTensor(TensorBase* tensor)  // NOLINT
      : MetaTensor(tensor) {}
  DistMetaTensor(const TensorBase& tensor)  // NOLINT
      : MetaTensor(tensor) {}
  DistMetaTensor(const TensorBase* tensor)  // NOLINT
      : MetaTensor(tensor) {}
  DistMetaTensor(TensorBase& tensor)  // NOLINT
      : MetaTensor(tensor) {}
  // For static mode only
  DistMetaTensor(const phi::DDim& dims, const TensorDistAttr& dist_attr)
      : dims_(dims), dist_attr_(dist_attr) {}

  DistMetaTensor(DistMetaTensor&&) = default;
  DistMetaTensor& operator=(DistMetaTensor&&) = default;
  DistMetaTensor(const DistMetaTensor&) = default;
  DistMetaTensor& operator=(const DistMetaTensor&) = default;

  virtual ~DistMetaTensor() = default;

  DDim dims() const override;

  const distributed::TensorDistAttr& dist_attr() const;

  bool initialized() const override;

 private:
  /**
   * Note: When using the semi-automatic parallel segmentation derivation rules
   * of the static graph, in order to facilitate the packaging of the input
   * parameters of the construction, the DistMetaTensor is inherited and
   * encapsulated, and the class members dims_ and dist_attr_ are added to it.
   *
   * The information contained in these two members is also in the tensor of the
   * meta_tensor of the base class, and there is redundancy.
   *
   * We need to pay attention when using it to ensure the consistency.
   * These two members are read-only, and their values cannot be changed
   * after construction. To change their values, they need to be set
   * directly in tensor_*/
  phi::DDim dims_;
  TensorDistAttr dist_attr_;
};

}  // namespace distributed
}  //  namespace phi
