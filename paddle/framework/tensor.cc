#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

template <typename T>
const T* Tensor::Data() const {
  PADDLE_ASSERT(holder_ != nullptr);
  PADDLE_ASSERT(holder_->Place() == place_);
  PADDLE_ASSERT(holder_->Size() >= product(dims_) * sizeof(T));
  return static_cast<const T*>(holder->Ptr());
}

bool Tensor::NeedReset() const {
  return (holder_ == nullptr || holder_->Place() != place_ ||
          holder_->Size() < product(dims_) * sizeof(T));
}

template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
T* Tensor::MutableData() {
  if (NeedReset()) {
    holder_.reset(new PlaceholderImpl(place_, product(dims_) * sizeof(T)));
  }
  return static_cast<T*>(holder_->Ptr());
}

template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
T* Tensor::MutableData(const DDim& dims) {
  dims_ = dims;
  return MutableData<T>();
}

template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
T* Tensor::MutableData(const DDim& dims, const Place& place) {
  dims_ = dims;
  place_ = place;
  return MutableData<T>();
}

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

void Tensor::ShareData(const Tensor& src) {
  if (src.NeedReset()) {
    // TODO: error: "Src tensor need to be reseted before calling ShareData()".
  }
  holder_ = src.Holder();
  dims_ = src.Dims();
  place_ = src.Place();
  return;
}

template <typename T>
void Tensor::CopyFrom(const Tensor& src) {
  if ((void*)&src == (void*)this) {
    return;
  }
  int len = product(src.Dims());
  T* src_ptr = src.Data<T>();
  T* dst_ptr = MutableData<T>(src.Dims());
  for (int i = 0; i < len; ++i) {
    dst_ptr[i] = src_ptr[i];
  }
  return;
}

const std::shared_ptr<Tensor::Placeholder>& Tensor::Holder() const {
  return holder_;
}

const DDim& Tensor::Dims() const { return dims_; }

const paddle::platform::Place& Tensor::Place() const { return place_; }

template <typename T>
bool Tensor::IsType() const {
  return typeid(T) == holder_.TypeInfo();
}

}  // namespace framework
}  // namespace paddle