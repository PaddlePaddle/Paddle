// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/controlflow/op_variant.h"

namespace paddle {
namespace operators {

const framework::VariableNameMap& OpVariant::Inputs() const {
  return *boost::apply_visitor(InputsVisitor(), op_);
}

const framework::VariableNameMap& OpVariant::Outputs() const {
  return *boost::apply_visitor(OutputsVisitor(), op_);
}

const framework::AttributeMap& OpVariant::Attrs() const {
  return *boost::apply_visitor(AttributeMapVisitor(), op_);
}

}  // namespace operators
}  // namespace paddle
