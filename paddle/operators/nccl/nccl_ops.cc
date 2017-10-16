#include "paddle/framework/op_registry.h"
#include "paddle/operators/nccl/nccl_gpu_common.h"

namespace paddle {
namespace operators {

// AllreduceOp
class NCCLAllreduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  // allreduce do nothing in infershape
  void InferShape(const framework::InferShapeContext &ctx) const override {}
};

template <typename T>
class NCCLAllreduceOp : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ctx = static_cast<NCCLContext *>(context.device_context());
    // auto *comm = ;
    // auto *src = ;
    // ncclAllReduce(src, dest, )
  }
};

// BcastSendOp
template <typename T>
class NCCLBroadcastSendOp final : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {}
};

// BcastRecvOp
template <typename T>
class NCCLBroadcastRecvOp final : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {}
};
}
}
