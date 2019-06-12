/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle{
namespace operators{
template<typename DeviceContext,typename T>
class CReduceScatterOpKernel:public framework::OpKernel<T>{
public:
 void Compute(const framework::ExecutionContext &ctx) const override{
  auto place = ctx.GetPlace();
  PADDLE_ENFORCE(is_gpu_place(place),
                   "CAllReduce op can run on gpu place only for now.");
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  auto in = ctx.Input<framework::Tensor>("X");
  auto out = ctx.Output<framework::Tensor>("Out");
  int rid = ctx.Attr<int>("ring_id");
  auto comm = platform::NCCLCommContext::Instance().Get(rid);
  int nccl_nranks = comm->nranks();
  auto out_dims = in->dims();
  out_dims[0]=out_dims[0]/static_cast<int64_t>(nccl_nranks);
  out->mutable_data<T>(out_dims,place);
  int64_t recv_numel = in->numel()/nccl_nranks;
  const T *send_buff = in->data<T>();
  T *recv_buff = out->data<T>();
  int dtype = platform::ToNCCLDataType(in->type());
  PADDLE_ENFORCE(platform::dynload::ncclReduceScatter(
      send_buff, recv_buff, recv_numel, static_cast<ncclDataType_t>(dtype),
      ncclSum,comm->comm(), comm->stream()));
#else
  PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
};
}// namespace operators
}// namespace paddle


