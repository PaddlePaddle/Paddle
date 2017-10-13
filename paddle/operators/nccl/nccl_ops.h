#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/operators/nccl/nccl_gpu_common.h"

#include <string.h>

namespace paddle {
namespace operators {

template <typename Type>
class NCCLTypeWrapper;

template <>
class NCCLTypeWrapper<float> {
  static const ncclDataType_t type = ncclFloat;
};

template <>
class NCCLTypeWrapper<double> {
  static const ncclDataType_t type = ncclDouble;
};

template <typename T>
class NCCLAllReduceKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<Tensor>("X");
    auto outs = ctx.MultiOutput<Tensor>("Out");
    std::string reduction = ctx.Attr<std::string>("reduction");
    std::vector<int> gpus = ctx.Attr<std::vector<int>>("gpus");
    ncclRedOp_t op_type;
    if (reduction == "ncclSum") {
      op_type = ncclSum;
    } else if (reduction == "ncclProd") {
      op_type = ncclProd;
    } else if (reduction == "ncclMin") {
      op_type = ncclMin;
    } else
      (reduction == "ncclMax") { op_type = ncclMax; }

    auto dev_ctx =
        static_cast<const platform::CUDADeviceContext>(ctx.device_context());

    NCCLManager* m = NCCLManager::Get();

    auto* comm = m->GetCommunicator(gpus);
    comm->wg_.Add(1);

    auto* stream = &dev_ctx.stream();

    // device id
    int gid = ctx.GetPlace().GetDeviceId();
    int idx = gid % gpus.size();
    comm->streams_[idx] = stream;

    for (size_t i = 0; i < ins.size(); ++i) {
      NCCL_CHECK(ncclAllReduce(ins[i]->data<T>(), outs[i]->mutable_data<T>(),
                               outs[i]->numel() * sizeof(T),
                               NCCLTypeWrapper<T>::type, op_type,
                               &comm->comms_[idx], comm->streams_[idx]));
      NCCL_CHECK(cudaEventRecord(comm->events_[idx], *comms_->streams_[idx]));

      // wait finish
      NCCL_CHECK(
          cudaStreamWaitEvent(comm->streams_[idx], comm->events_[idx], 0));
    }

    comm->wg_.Done();

    wg.Wait();
  }
};
}
}
