/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <memory>
#include <vector>

#include "paddle/math/Matrix.h"
#include "paddle/math/SparseRowMatrix.h"
#include "paddle/parameter/Parameter.h"

namespace paddle {

class Weight {
private:
  MatrixPtr weight_;
  MatrixPtr weightGrad_;
  ParameterPtr parameter_;

public:
  Weight(size_t height, size_t width, ParameterPtr parameter);
  Weight(size_t height, size_t width, ParameterPtr parameter, size_t offset);

  const MatrixPtr& getW() { return weight_; }
  const MatrixPtr& getWGrad() { return weightGrad_; }
  const ParameterPtr& getParameterPtr();

  void incUpdate(const UpdateCallback& callback) {
    getParameterPtr()->incUpdate(callback);
  }

  void setParameterPtr(ParameterPtr param);
};

typedef std::vector<std::unique_ptr<Weight>> WeightList;

}  // namespace paddle
