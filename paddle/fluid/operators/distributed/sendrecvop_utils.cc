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

#ifdef PADDLE_WITH_CUDA
#include <nccl.h>
#endif
#include <sys/time.h>
#include <thread>  // NOLINT

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/variable_response.h"

namespace paddle {
namespace operators {
namespace distributed {

using VarMsg = sendrecv::VariableMessage;

#ifdef PADDLE_WITH_CUDA
void* GetVarPayLoad(const std::string varname, int64_t size) {
  platform::CUDAPinnedPlace cuda_pinned;
  return memory::Alloc(cuda_pinned, size);
}
#endif

void GetTensorPayload(framework::Variable* var,
                      const platform::DeviceContext& ctx, VarMsg* request,
                      void** payload, size_t* payload_size) {
  auto tensor = var->Get<framework::LoDTensor>();
  // FIXME(wuyi): data types in send_recv.proto is copied from
  // framework.proto
  request->set_data_type(
      static_cast<VarMsg::Type>(framework::ToDataType(tensor.type())));
  for (auto& dim : framework::vectorize(tensor.dims())) {
    request->add_dims(dim);
  }
  const framework::LoD lod = tensor.lod();
  if (lod.size() > 0) {
    request->set_lod_level(lod.size());
    for (auto& each : lod) {
      VarMsg::LodData* lod_inner = request->add_lod();
      for (auto& d : each) {
        lod_inner->add_lod_data(d);
      }
    }
  }
  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(platform::is_gpu_place(tensor.place()));
    // platform::CUDAPinnedPlace cuda_pinned;
    auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
    auto copy_size = tensor.numel() * framework::SizeOfType(tensor.type());
    *payload = GetVarPayLoad(request->varname(), copy_size);

    platform::CUDAPinnedPlace cuda_pinned;
    memory::Copy(cuda_pinned, *payload,
                 boost::get<platform::CUDAPlace>(tensor.place()),
                 reinterpret_cast<const void*>(tensor.data<void>()), copy_size,
                 gpu_dev_ctx.stream());

    ctx.Wait();
#endif
  } else {
    *payload = tensor.data<void>();
  }
  *payload_size = tensor.numel() * framework::SizeOfType(tensor.type());
}

void GetSelectedRowsPayload(framework::Variable* var,
                            const platform::DeviceContext& ctx, VarMsg* request,
                            void** payload, size_t* payload_size) {
  auto* slr = var->GetMutable<framework::SelectedRows>();
  request->set_data_type(
      static_cast<VarMsg::Type>(framework::ToDataType(slr->value().type())));
  request->set_lod_level(0);
  request->set_slr_height(slr->height());

  for (auto& dim : framework::vectorize(slr->value().dims())) {
    request->add_dims(dim);
  }

  auto* tensor = slr->mutable_value();
  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
    auto copy_size = tensor->numel() * framework::SizeOfType(tensor->type());
    *payload = GetVarPayLoad(request->varname(), copy_size);

    platform::CUDAPinnedPlace cuda_pinned;
    memory::Copy(cuda_pinned, *payload,
                 boost::get<platform::CUDAPlace>(tensor->place()),
                 reinterpret_cast<const void*>(tensor->data<void>()), copy_size,
                 gpu_dev_ctx.stream());
    ctx.Wait();
#endif
  } else {
    *payload = slr->mutable_value()->data<void>();
  }
  *payload_size = tensor->numel() * framework::SizeOfType(tensor->type());
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
