#pragma once

#include <memory>
#include <type_traits>
#include <typeinfo>
#include "paddle/framework/ddim.h"
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
  const T* Data() const {
    PADDLE_ASSERT(holder_ != nullptr);
    PADDLE_ASSERT(holder_->Place() == place_);
    PADDLE_ASSERT(holder_->Size() >= product(dims_) * sizeof(T));
    return static_cast<const T*>(holder->Ptr());
  }

  template <typename T>
  bool NeedReset() const {
    return (holder_ == nullptr || holder_->Place() != place_ ||
            holder_->Size() < product(dims_) * sizeof(T));
  }

  // must be POD types
  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData() {
    if (NeedReset<T>()) {
      holder_.reset(new PlaceholderImpl(place_, product(dims_) * sizeof(T)));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData(const DDim& dims) {
    dims_ = dims;
    return MutableData<T>();
  }

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData(const DDim& dims, const Place& place) {
    dims_ = dims;
    place_ = place;
    return MutableData<T>();
  }

  int Rank() const;

  int Numel() const;

  void Resize(const DDim& dims);

  void Reshape(const DDim& dims);

  template <typename T>
  void ShareData(const Tensor& src) {
    if (src.NeedReset<T>()) {
      // TODO: error: "Src tensor need to be reseted before calling
      // ShareData()".
    }
    holder_ = src.Holder();
    dims_ = src.Dims();
    place_ = src.Place();
    return;
  }

  template <typename T>
  void CopyFrom(const Tensor& src) {
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

  const std::shared_ptr<Placeholder>& Holder() const;

  const DDim& Dims() const;

  const paddle::platform::Place& Place() const;

  template <typename T>
  bool IsType() const {
    return typeid(T) == holder_.TypeInfo();
  }

  // Placeholder hides type T, so it doesn't appear as a template
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual std::type_info TypeInfo() const = 0;
    virtual void* Ptr() const = 0;
    virtual Place Place() const = 0;
    virtual size_t Size() const = 0;
  };

 private:
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
