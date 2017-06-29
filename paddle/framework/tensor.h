#pragma once

#include <memory>
#include <type_traits>
#include <typeinfo>
#include "paddle/framework/ddim.h"
#include "paddle/framework/enforce.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace framework {

class Tensor {
  using paddle::platform::Place;
  using paddle::platform::get_place;

 public:
  explicit Tensor(DDim dims) : dims_(dims), place_(get_place()) {}
  explicit Tensor(DDim dims, Place place) : dims_(dims), place_(place) {}

  Tensor& operator=(const Tensor& src) = delete;

  template <typename T>
  const T* data() const {
    PADDLE_ENFORCE(holder_ != nullptr);
    PADDLE_ENFORCE(holder_->Place() == place_);
    PADDLE_ENFORCE(holder_->Size() >= Numel() * sizeof(T));
    return static_cast<const T*>(holder->Ptr());
  }

  template <typename T>
  bool NeedReset() const {
    return (holder_ == nullptr || holder_->Place() != place_ ||
            holder_->Size() < Numel() * sizeof(T));
  }

  // must be POD types
  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* mutable_data() {
    if (NeedReset<T>()) {
      holder_.reset(new PlaceholderImpl(place_, Numel() * sizeof(T)));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* mutable_data(const DDim& dims) {
    dims_ = dims;
    return mutable_data<T>();
  }

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* mutable_data(const DDim& dims, const Place& place) {
    dims_ = dims;
    place_ = place;
    return mutable_data<T>();
  }

  int Rank() const { return arity(dims_); }

  int Numel() const { return product(dims_); }

  void Reshape(const DDim& dims) {
    PADDLE_ENFORCE(product(dims) == Numel(),
                   "Reshape() can not change tensor's numel!");
    dims_ = dims;
  }

  template <typename T>
  void ShareData(const Tensor& src) {
    PADDLE_ENFORCE(!src.NeedReset<T>(),
                   "Src tensor need to be reseted before calling ShareData().");
    holder_ = src.holder_;
    dims_ = src.dims_;
    place_ = src.place_;
  }

  const DDim& Dims() const { return dims_; }

  const paddle::platform::Place& Place() const { return place_; }

  template <typename T>
  bool IsType() const {
    return typeid(T) == holder_.TypeInfo();
  }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual std::type_info TypeInfo() const = 0;
    virtual void* Ptr() const = 0;
    virtual Place Place() const = 0;
    virtual size_t Size() const = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(Place place, size_t size)
        : ptr_(paddle::memory::Alloc(place, size),
               paddle::memory::Deleter(place)),
          place_(place),
          size_(size) {}

    virtual std::type_info TypeInfo() const { return typeid(T); }
    virtual void* Ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual size_t Size() const { return size_; }
    virtual Place Place() const { return place_; }

    std::unique_ptr<T, paddle::memory::Deleter> ptr_;
    Place place_;  // record the place of ptr_.
    size_t size_;  // size of the memory block.
  };

  std::shared_ptr<Placeholder> holder_;  // holds the memory block if allocated.
  DDim dims_;  // could be smallers than the holder_->Size().
  Place place_;
};

}  // namespace framework
}  // namespace paddle
