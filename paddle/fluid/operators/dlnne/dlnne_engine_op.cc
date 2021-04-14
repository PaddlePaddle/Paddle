#include "paddle/fluid/operators/dlnne/dlnne_engine_op.h"

namespace paddle {
namespace inference {

void CopyTensorDeviceToCpu(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}
void CopyTensorCpuToDevice(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

}  // namespace inference

namespace operators {

class DlnneEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph.");
    AddAttr<std::string>(
        "engine_key",
        "The engine_key here is used to distinguish different DLNNE Engines");
    AddAttr<framework::BlockDesc*>("sub_block", "the trt block");
    AddComment("Dlnne engine operator.");
  }
};

class DlnneEngineInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};

}  // namespace operators
}  // paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(dlnne_engine, ops::DlnneEngineOp, ops::DlnneEngineOpMaker);
