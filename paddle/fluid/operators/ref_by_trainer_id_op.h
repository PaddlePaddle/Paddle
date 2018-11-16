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

#pragma once

#include <stdio.h>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class RefByTrainerIdKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto in_list = context.MultiInput<framework::Tensor>("X");
    auto* trainer_id_t = context.Input<framework::Tensor>("TrainerId");
    int64_t trainer_id = 0;
    auto* trainer_id_data = trainer_id_t->data<int64_t>();
    if (platform::is_gpu_place(context.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      auto stream = context.cuda_device_context().stream();
      memory::Copy<>(platform::CPUPlace(), &trainer_id,
                     boost::get<platform::CUDAPlace>(context.GetPlace()),
                     trainer_id_data, sizeof(int64_t), stream);
#endif
    } else {
      trainer_id = *trainer_id_data;
    }
    PADDLE_ENFORCE_LT((size_t)trainer_id, in_list.size());
    out->mutable_data<T>(context.GetPlace());
    out->ShareDataWith(*(in_list[trainer_id]));
  }
};

}  // namespace operators
}  // namespace paddle
