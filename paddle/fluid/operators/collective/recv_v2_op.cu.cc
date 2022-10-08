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

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
framework::DDim recv_shape_info(const platform::Place &place,
                                const gpuStream_t &stream,
                                platform::NCCLComm *comm,
                                const int &peer,
                                distributed::ProcessGroup *group) {
  if (!group) {
    PADDLE_ENFORCE_EQ((stream != nullptr && comm != nullptr),
                      true,
                      platform::errors::InvalidArgument(
                          "NCCLComm and Stream should be provided if use NCCL "
                          "to send the shape info."));
  }

  paddle::experimental::DataType shape_dytpe =
      paddle::experimental::DataType::INT32;
  ncclDataType_t nccl_dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(shape_dytpe));

  // step1: recv the shape size
  phi::DenseTensor gpu_shape_size_tensor(shape_dytpe);
  if (!group) {
    gpu_shape_size_tensor.Resize({1});
    gpu_shape_size_tensor.mutable_data(place, shape_dytpe);
    auto *gpu_data = gpu_shape_size_tensor.data<int>();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
        gpu_data, 1, nccl_dtype, peer, comm->comm(), stream));
  }

  // copy the shape size tensor to cpu
  phi::DenseTensor *cpu_shape_size_tensor = new phi::DenseTensor(shape_dytpe);
  cpu_shape_size_tensor->Resize({1});
  cpu_shape_size_tensor->mutable_data(platform::CPUPlace(), shape_dytpe);
  if (group) {
    std::vector<phi::DenseTensor> shape_size_tensor;
    shape_size_tensor.emplace_back(*cpu_shape_size_tensor);
    auto shape_size_task = group->Recv(shape_size_tensor, peer);
  } else {
    framework::TensorCopySync(
        gpu_shape_size_tensor, platform::CPUPlace(), cpu_shape_size_tensor);
  }
  auto *cpu_data = cpu_shape_size_tensor->data<int>();
  int shape_size = cpu_data[0];
  VLOG(3) << "recv the shape size: " << shape_size << " from peer";

  // step2: recv the shape
  phi::DenseTensor gpu_shape_tensor(shape_dytpe);
  if (!group) {
    gpu_shape_tensor.Resize({shape_size});
    gpu_shape_tensor.mutable_data(place, shape_dytpe);
    auto *gpu_shape_data = gpu_shape_tensor.data<int>();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
        gpu_shape_data, shape_size, nccl_dtype, peer, comm->comm(), stream));
  }

  // copy the shape tensor to cpu
  phi::DenseTensor *cpu_shape_tensor = new phi::DenseTensor(shape_dytpe);
  cpu_shape_tensor->Resize({shape_size});
  cpu_shape_tensor->mutable_data(platform::CPUPlace(), shape_dytpe);
  if (group) {
    std::vector<phi::DenseTensor> shape_tensor;
    shape_tensor.emplace_back(*cpu_shape_tensor);
    auto shape_task = group->Recv(shape_tensor, peer);
  } else {
    framework::TensorCopySync(
        gpu_shape_tensor, platform::CPUPlace(), cpu_shape_tensor);
  }
  auto *cpu_shape_data = cpu_shape_tensor->data<int>();
  std::vector<int> all_shape;
  for (int i = 0; i < shape_size; ++i) {
    all_shape.emplace_back(cpu_shape_data[i]);
  }
  framework::DDim new_dim;
  new_dim = new_dim.reshape(all_shape);
  VLOG(3) << "recv the shape: (" << new_dim << ") from peer";

  return new_dim;
}
#endif

template <typename T>
class RecvOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    int rid = ctx.Attr<int>("ring_id");
    bool dynamic_shape = ctx.Attr<bool>("dynamic_shape");
    PADDLE_ENFORCE_GE(
        rid,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for recv_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer,
        0,
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

      if (dynamic_shape) {
        VLOG(3) << "recv_v2 will use dynamic shape with send_v2 for switch";
        framework::DDim new_dim = recv_shape_info(ctx.GetPlace(),
                                                  /* gpuStream_t */ nullptr,
                                                  /* NCCLComm* */ nullptr,
                                                  peer,
                                                  pg);
        out->Resize(new_dim);
        out->mutable_data<T>(new_dim, place);
      } else {
        out->mutable_data<T>(out_dims, place);
      }

      out_tensor.emplace_back(*out);
      auto task = pg->Recv(out_tensor, peer);
      return;
    }
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::GPUContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    PADDLE_ENFORCE_LT(
        peer,
        comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer,
                                          comm->nranks()));

    int data_type = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);
    ncclDataType_t dtype = platform::ToNCCLDataType(type);

    auto *out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensorArray>()) {
      PADDLE_ENFORCE_EQ(
          dynamic_shape,
          false,
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
      framework::DDim new_dim = recv_shape_info(place,
                                                stream,
                                                comm,
                                                peer,
                                                /* ProcessGroup* */ nullptr);
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

REGISTER_OP_CUDA_KERNEL(recv_v2,
                        ops::RecvOpV2CUDAKernel<float>,
                        ops::RecvOpV2CUDAKernel<double>,
#if CUDNN_VERSION_MIN(8, 1, 0) && NCCL_VERSION_CODE >= 21000
                        ops::RecvOpV2CUDAKernel<plat::bfloat16>,
#endif
                        ops::RecvOpV2CUDAKernel<int>,
                        ops::RecvOpV2CUDAKernel<int64_t>,
                        ops::RecvOpV2CUDAKernel<int8_t>,
                        ops::RecvOpV2CUDAKernel<plat::float16>);
