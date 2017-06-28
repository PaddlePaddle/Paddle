#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

int Tensor::Rank() const { return arity(dims_); }

int Tensor::Numel() const { return product(dims_); }

void Tensor::Resize(const DDim& dims) {
  dims_ = dims;
  return;
}

void Tensor::Reshape(const DDim& dims) {
  if (product(dims) != product(dims_)) {
    // TODO: error: "Reshape() can not change tensor's numel".
  }
  dims_ = dims;
  return;
}

const std::shared_ptr<Tensor::Placeholder>& Tensor::Holder() const {
  return holder_;
}

const DDim& Tensor::Dims() const { return dims_; }

const paddle::platform::Place& Tensor::Place() const { return place_; }

}  // namespace framework
}  // namespace paddle