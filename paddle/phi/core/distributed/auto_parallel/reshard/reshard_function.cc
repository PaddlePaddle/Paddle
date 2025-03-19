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

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

std::shared_ptr<DistTensor> ReshardFunction::Eval(
    DeviceContext* dev_ctx,
    const DistTensor& in,
    const TensorDistAttr& out_dist_attr) {
  phi::RecordEvent reshard_record_event(
      Name(), phi::TracerEventType::OperatorInner, 1);
  std::shared_ptr<DistTensor> out = std::make_shared<DistTensor>();
  Eval(dev_ctx, in, out_dist_attr, out.get());
  return out;
}

void ReshardFunction::SetValue(DistTensor* tensor, const DenseTensor& value) {
  tensor->value_ = std::make_shared<DenseTensor>(value);
}

void ReshardFunction::SetDistProps(DistTensor* tensor,
                                   const DDim& dims,
                                   const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(dist_attr.verify_dynamic(common::vectorize(dims)),
                    true,
                    common::errors::InvalidArgument(
                        "The input dist_attr [%s] and dims [%s] are improper.",
                        dist_attr.to_string(),
                        str_join(vectorize(dims))));

  tensor->global_dims_ = dims;
  tensor->dist_attr_ = dist_attr;
  tensor->process_mesh_ = dist_attr.process_mesh();
  tensor->placements_ = ToPlacements(dist_attr);
}

void ReshardFunction::SetDistProps(DistTensor* tensor,
                                   const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(dist_attr.verify_dynamic(common::vectorize(tensor->dims())),
                    true,
                    common::errors::InvalidArgument(
                        "The input dist_attr [%s] and dims [%s] are improper.",
                        dist_attr.to_string(),
                        str_join(vectorize(tensor->dims()))));

  tensor->dist_attr_ = dist_attr;
  tensor->process_mesh_ = dist_attr.process_mesh();
  tensor->placements_ = ToPlacements(dist_attr);
}

DenseTensor* ReshardFunction::GetMutableTensor(DistTensor* tensor) {
  return tensor->value_.get();
}

}  // namespace phi::distributed
