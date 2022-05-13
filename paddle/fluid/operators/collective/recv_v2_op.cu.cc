/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/recv_v2_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
class RecvOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    int rid = ctx.Attr<int>("ring_id");
    bool dynamic_shape = ctx.Attr<bool>("dynamic_shape");
    PADDLE_ENFORCE_GE(
        rid, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for recv_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for recv_v2 op must be non-negative.", peer));

    gpuStream_t stream = nullptr;
    auto place = ctx.GetPlace();
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup *pg = map->get(rid);
      std::vector<phi::DenseTensor> out_tensor;
      auto out_shape = ctx.Attr<std::vector<int>>("out_shape");
      auto out = ctx.Output<framework::LoDTensor>("Out");
      auto out_dims = out->dims();
      out->mutable_data<T>(out_dims, place);

      out_tensor.emplace_back(*out);
      auto task = pg->Recv(out_tensor, peer);
      return;
    }
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    PADDLE_ENFORCE_LT(
        peer, comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer, comm->nranks()));

    int data_type = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);
    ncclDataType_t dtype = platform::ToNCCLDataType(type);

    auto *out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensorArray>()) {
      PADDLE_ENFORCE_EQ(
          dynamic_shape, false,
          platform::errors::InvalidArgument("Dynamic shape for send/recv not "
                                            "support LoDTensorArray for now."));
      auto out_array = out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t idx = 0; idx < out_array->size(); ++idx) {
        VLOG(3) << "LodTensorArray: idx(" << idx << ")";
        auto out = &out_array->at(idx);
        auto out_dims = out->dims();
        out->mutable_data<T>(out_dims, place, 0);
        auto numel = out->numel();
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
            out->data<T>(), numel, dtype, peer, comm->comm(), stream));
        VLOG(3) << "rank " << comm->rank() << " recv " << phi::product(out_dims)
                << " from " << peer;
      }
      return;
    }

    auto out_shape = ctx.Attr<std::vector<int>>("out_shape");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    auto out_dims = out->dims();
    auto numel = out->numel();

    if (dynamic_shape) {
      VLOG(3) << "recv_v2 will use dynamic shape with send_v2";
      paddle::experimental::DataType shape_dytpe =
          paddle::experimental::DataType::INT64;
      ncclDataType_t nccl_dtype =
          platform::ToNCCLDataType(framework::TransToProtoVarType(shape_dytpe));

      // step1: recv the shape size

      // recv the shape size tensor on gpu
      framework::Tensor gpu_shape_size_tensor(shape_dytpe);
      gpu_shape_size_tensor.Resize({1});
      gpu_shape_size_tensor.mutable_data(place, shape_dytpe);
      auto *gpu_data = gpu_shape_size_tensor.data<int64_t>();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          gpu_data, 1, nccl_dtype, peer, comm->comm(), stream));
      framework::Tensor *cpu_shape_size_tensor =
          new framework::Tensor(shape_dytpe);

      // copy the shape size tensor to cpu
      cpu_shape_size_tensor->Resize({1});
      cpu_shape_size_tensor->mutable_data(platform::CPUPlace(), shape_dytpe);
      framework::TensorCopySync(gpu_shape_size_tensor, platform::CPUPlace(),
                                cpu_shape_size_tensor);
      auto *cpu_data = cpu_shape_size_tensor->data<int64_t>();
      int64_t shape_size = cpu_data[0];
      VLOG(3) << "recv the shape size: " << shape_size << " from peer";

      // step2: recv the shape

      // recv the shape tensor on gpu
      framework::Tensor gpu_shape_tensor(shape_dytpe);
      gpu_shape_tensor.Resize({shape_size});
      gpu_shape_tensor.mutable_data(place, shape_dytpe);
      auto *gpu_shape_data = gpu_shape_tensor.data<int64_t>();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          gpu_shape_data, shape_size, nccl_dtype, peer, comm->comm(), stream));

      // copy the shape tensor to cpu
      framework::Tensor *cpu_shape_tensor = new framework::Tensor(shape_dytpe);
      cpu_shape_tensor->Resize({shape_size});
      cpu_shape_tensor->mutable_data(platform::CPUPlace(), shape_dytpe);
      framework::TensorCopySync(gpu_shape_tensor, platform::CPUPlace(),
                                cpu_shape_tensor);
      auto *cpu_shape_data = cpu_shape_tensor->data<int64_t>();
      std::vector<int> all_shape;
      for (int i = 0; i < shape_size; ++i) {
        all_shape.emplace_back(cpu_shape_data[i]);
      }
      framework::DDim new_dim;
      new_dim = new_dim.reshape(all_shape);
      VLOG(3) << "recv the shape: (" << new_dim << ") from peer";

      // step3: reshape the out tensor and recv the out tensor
      out->Resize(new_dim);
      numel = out->numel();
      out->mutable_data<T>(new_dim, place);
    } else {
      out->mutable_data<T>(out_dims, place);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
        out->data<T>(), numel, dtype, peer, comm->comm(), stream));
    VLOG(3) << "rank " << comm->rank() << " recv " << phi::product(out->dims())
            << " from " << peer;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL and "
        "NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(recv_v2, ops::RecvOpV2CUDAKernel<float>,
                        ops::RecvOpV2CUDAKernel<double>,
                        ops::RecvOpV2CUDAKernel<int>,
                        ops::RecvOpV2CUDAKernel<int64_t>,
                        ops::RecvOpV2CUDAKernel<int8_t>,
                        ops::RecvOpV2CUDAKernel<plat::float16>);
