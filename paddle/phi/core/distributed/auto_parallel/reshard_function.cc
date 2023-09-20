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

#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi {
namespace distributed {

std::shared_ptr<DistTensor> ReshardFunction::Eval(
    DeviceContext* dev_ctx,
    const DistTensor& in,
    const TensorDistAttr& out_dist_attr) {
  std::shared_ptr<DistTensor> out = std::make_shared<DistTensor>();
  Eval(dev_ctx, in, out_dist_attr, out.get());
  return out;
}

void ReshardFunction::SetValue(DistTensor* tensor, const DenseTensor& value) {
  tensor->value_ = value;
}

void ReshardFunction::SetDistProps(DistTensor* tensor,
                                   const DDim& dims,
                                   const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(dist_attr.verify(vectorize(dims)),
                    true,
                    phi::errors::InvalidArgument(
                        "The input dist_attr and dims are improper."));

  tensor->dims_ = dims;
  tensor->dist_attr_ = dist_attr;
}

void ReshardFunction::SetDistProps(DistTensor* tensor,
                                   const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(dist_attr.verify(vectorize(tensor->dims())),
                    true,
                    phi::errors::InvalidArgument(
                        "The input dist_attr and dims are improper."));

  tensor->dist_attr_ = dist_attr;
}

DenseTensor* ReshardFunction::GetMutableTensor(DistTensor* tensor) {
  return &tensor->value_;
}

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  for (const auto& func : GetReshardFunctionList()) {
    if (func->IsSuitable(in, out_dist_attr)) {
      return func.get();
    }
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "Can not reshard from in_dist_attr=%s to out_dist_attr=%s.",
      in.dist_attr().to_string(),
      out_dist_attr.to_string()));
}

std::vector<std::unique_ptr<ReshardFunction>>& GetReshardFunctionList() {
  static std::vector<std::unique_ptr<ReshardFunction>> func_list;
  return func_list;
}

}  // namespace distributed
}  // namespace phi
