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
  const T* Data() const;

  bool NeedReset();

  // must be POD types
  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData();

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData(const DDim& dims);

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>::type>
  T* MutableData(const DDim& dims, const Place& place);

  int Rank() const;

  int Numel() const;

  void Resize(const DDim& dims);

  void Reshape(const DDim& dims);

  void ShareData(const Tensor& src);

  template <typename T>
  void CopyFrom(const Tensor& src);

  const std::shared_ptr<Placeholder>& Holder() const;

  const DDim& Dims() const;

  const paddle::platform::Place& Place() const;

  template <typename T>
  bool IsType() const;

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
