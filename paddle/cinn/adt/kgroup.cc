// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/schedule_dim.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn::adt {

using AnchorTensor = Variable;

namespace {

std::size_t GetTensorNumel(const Tensor& tensor) {
  CHECK(tensor.Has<adapter::Tensor>());
  return tensor.Get<adapter::Tensor>().GetNumel();
}

const std::vector<int32_t>& GetTensorShape(const Tensor& tensor) {
  CHECK(tensor.Has<adapter::Tensor>());
  return tensor.Get<adapter::Tensor>().GetShape();
}

}  // namespace

List<LoopSize> KGroup::GetDefaultScheduleSizes(
    const std::shared_ptr<IGroup>& igroup) const {
  List<LoopSize> ret{};

  const Tensor& tensor = igroup->anchor_tensor();
  const auto tensor_shape = GetTensorShape(tensor);
  for (int32_t dim : tensor_shape) {
    ret->emplace_back(LoopSize{dim});
  }
  return ret;
}

}  // namespace cinn::adt
