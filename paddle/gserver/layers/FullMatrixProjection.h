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
#include "paddle/utils/Stat.h"

#include "Projection.h"

namespace paddle {

/**
 * FullMatrixProjection performs full matrix multiplication:
 * \f[
 *    out.row[i] += in.row[i] * weight
 * \f]
 *
 * The config file api is full_matrix_projection.
 */
class FullMatrixProjection : public Projection {
public:
  FullMatrixProjection(const ProjectionConfig& config,
                       const ParameterPtr& parameter,
                       bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

protected:
  std::unique_ptr<Weight> weight_;
};

}  // namespace paddle
