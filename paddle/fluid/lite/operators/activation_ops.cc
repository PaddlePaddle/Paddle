#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

class ActivationOp : public OpLite {
 public:
  explicit ActivationOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override { return true; }

  bool InferShape() const override {
    param_.Out->Resize(param_.X->dims());
    return true;
  }

  bool AttachImpl(const OpDesc& opdesc, lite::Scope* scope) override {
    auto X_name = opdesc.Input("X").front();
    auto Out_name = opdesc.Output("Out").front();

    param_.X = GetVar<lite::Tensor>(scope, X_name);
    param_.Out = GetMutableVar<Tensor>(scope, Out_name);
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

 private:
  mutable ActivationParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(square, paddle::lite::operators::ActivationOp);
