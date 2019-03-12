#include "paddle/fluid/inference/op_lite/activation_op.h"
#include <Eigen/Eigen>
#include <vector>
#include "paddle/fluid/inference/op_lite/op_lite.h"

namespace paddle {
namespace inference {
namespace op_lite {

bool ReLU::CheckShape() const { return true; }

bool ReLU::InferShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  param_.output->ShareDataWith(*param_.input);
  // share lod
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool ReLU::Run() {
  auto x = framework::EigenVector<float>::Flatten(*param_.input);
  auto out = framework::EigenVector<float>::Flatten(*param_.output);
  param_.functor(param_.eigen_device, x, out);
  return true;
}

bool ReLU::Build(const paddle::framework::OpDesc &opdesc,
                 framework::Scope *scope) {
  const auto &inputs = opdesc.Inputs();
  const auto &outputs = opdesc.Outputs();
  CHECK_OR_FALSE(inputs.count("X"));
  CHECK_OR_FALSE(outputs.count("Out"));
  auto x = scope->FindVar(inputs.at("X").front());
  auto out = scope->FindVar(outputs.at("Out").front());

  param_.input = x->GetMutable<LoDTensor>();
  param_.output = out->GetMutable<LoDTensor>();
  return true;
}

std::string ReLU::DebugString() const { return "relu-lite-op"; }

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
