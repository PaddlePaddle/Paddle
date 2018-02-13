/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "TableProjection.h"

namespace paddle {

REGISTER_PROJECTION(table, TableProjection);

TableProjection::TableProjection(const ProjectionConfig& config,
                                 const ParameterPtr& parameter,
                                 bool useGpu)
    : Projection(config, parameter, useGpu) {
  table_.reset(
      new Weight(config.input_size(), config.output_size(), parameter));
}

void TableProjection::prefetch(const Argument* in) {
  CHECK(in->ids);
  auto* sparseParam =
      dynamic_cast<SparsePrefetchRowCpuMatrix*>(table_->getW().get());
  if (sparseParam) {
    sparseParam->addRows(in->ids);
  }
}

void TableProjection::forward() {
  CHECK(in_->ids);
  out_->value->selectRows(*table_->getW(), *in_->ids);
}

void TableProjection::backward(const UpdateCallback& callback) {
  if (table_->getWGrad()) {
    CHECK(in_->ids);
    out_->grad->addToRows(*table_->getWGrad(), *in_->ids);
    parameter_->incUpdate(callback);
  }
}

}  // namespace paddle
