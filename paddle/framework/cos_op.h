#include <map>
#include <functional>

#include "paddle/framework/operator.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

using namespace paddle::platform;

typedef std::function<void(OpContext*)> ComputeFun;

/// simple kernel
template<typename T>
void CosineCPU(OpContext* ctx) {
			printf("run cosin op CPU kernel, scale = %f\n", ctx->op->GetAttr<T>("scale"));
			printf("%s\n", ctx->op->DebugString().c_str());
}

template<typename T>
void CosineGPU(OpContext* ctx) {
	printf("run cosin op GPU kernel, scale = %f\n", ctx->op->GetAttr<T>("scale"));
	printf("%s\n", ctx->op->DebugString().c_str());
}

class CosOp : public OperatorBase {
 public:
	explicit CosOp() {
		kernels_["CPU"] = CosineCPU<float>;
		kernels_["GPU"] = CosineGPU<float>;
	}

  void Run(OpContext* ctx) const override {
		auto dev_ctx = dynamic_cast<CPUDeviceContext*>(ctx->device_context);
		if (dev_ctx != nullptr) {
			kernels_.at("CPU")(ctx);
		} else {
			kernels_.at("GPU")(ctx);
		}
  }

 private:
	std::map<std::string, ComputeFun> kernels_;
};


class CosinOperatorProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
public:
		CosinOperatorProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
						: OpProtoAndCheckerMaker(proto, op_checker) {
			AddInput("input", "input of test op");
			AddOutput("output", "output of test op");
			AddAttr<float>("scale", "scale of cosine op")
							.SetDefault(1.0)
							.LargerThan(0.0);
			AddType("cos");
			AddComment("This is cos op");
		}
};

REGISTER_OP(CosOp, CosinOperatorProtoAndCheckerMaker, cos)

}  // namespace framework
}  // namespace paddle