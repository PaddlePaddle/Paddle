/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class NCCLBroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()));

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    int dev_id = boost::get<platform::CUDAPlace>(ctx.GetPlace()).device;
    int root_dev_id = ctx.Attr<int>("root");

    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());

    const int in_dev_id = boost::get<platform::CUDAPlace>(in->place()).device;
    PADDLE_ENFORCE_EQ(dev_id, in_dev_id);

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto comm = dev_ctx.nccl_comm();
    auto stream = dev_ctx.stream();
    PADDLE_ENFORCE_NOT_NULL(stream, "Should initialize NCCL firstly.");

    void* data_buffer = const_cast<void*>(in->data<void>());
    if (root_dev_id != in_dev_id) {
      data_buffer = out->mutable_data<T>(ctx.GetPlace());
    }

    VLOG(3) << "Bcast " << ctx.Inputs("X")[0] << " From " << root_dev_id
            << " to " << in_dev_id;

    auto dtype = platform::ToNCCLDataType(in->type());

    PADDLE_ENFORCE(platform::dynload::ncclBcast(
        data_buffer, static_cast<size_t>(in->numel()), dtype, root_dev_id, comm,
        stream));

    if (ctx.Attr<bool>("sync_mode")) {
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(broadcast, ops::NCCLBroadcastOpKernel<float>,
                        ops::NCCLBroadcastOpKernel<double>,
                        ops::NCCLBroadcastOpKernel<int>,
                        ops::NCCLBroadcastOpKernel<int64_t>,
                        ops::NCCLBroadcastOpKernel<plat::float16>);
