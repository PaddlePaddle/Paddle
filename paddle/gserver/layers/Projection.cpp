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

#include "Projection.h"

#include "ContextProjection.h"
#include "FullMatrixProjection.h"
#include "TableProjection.h"

namespace paddle {

ClassRegistrar<Projection, ProjectionConfig, ParameterPtr, bool>
    Projection::registrar_;

Projection* Projection::create(const ProjectionConfig& config,
                               ParameterPtr parameter,
                               bool useGpu) {
  return registrar_.createByType(config.type(), config, parameter, useGpu);
}

}  // namespace paddle
