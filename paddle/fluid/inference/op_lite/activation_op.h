#pragma once

#include <boost/variant.hpp>
#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/op_lite/op_lite.h"
#include "paddle/fluid/operators/activation_op.h"

namespace paddle {
namespace inference {
namespace op_lite {
using framework::LoDTensor;

template<typename T>
struct ReluParam {
  LoDTensor* input{nullptr};
  LoDTensor* output{nullptr};
  // TODO(Superjomn) consider share it in global.
  Eigen::DefaultDevice eigen_device;
  operators::ReluFunctor<T> functor;
};

class ReLU final : public OpLite {
 public:
  bool CheckShape() const override;
  bool InferShape() const override;
  bool Run() override;
  bool Build(const paddle::framework::OpDesc& opdesc,
             framework::Scope* scope) override;
  std::string DebugString() const override;

 private:
  ReluParam<float> param_;
};

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
