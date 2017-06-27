/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#pragma once

namespace paddle {
namespace framework {

class Tensor {
  using paddle::platform::Place;
  using paddle::platform::get_place;

 public:
  template <typename T>
  const T* data() const {
    PADDLE_ASSERT(holder_ != nullptr,
                  "Tensor::data must be called after Tensor::mutable_data");
    return static_cast<const T*>(holder->Ptr());
  }

  template <typename T,  // must be POD types
            typename = std::enable_if<std::is_pod<T>::value>::type>
  T* mutable_data(DDim dims, Place place) {
    if (holder_ == nullptr || holder_->Place() != place ||
        holder_->Size() < dims.product() * sizeof(T)) {
      holder_.reset(new PlaceholderImpl(place, dims.product() * sizeof(T)));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T,  // must be POD types
            typename = std::enable_if<std::is_pod<T>::value>::type>
  T* mutable_data(DDim dims) {
    return mutable_data<T>(dims, paddle::platform::get_place());
  }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* Ptr() const = 0;
    virtual Place Place() const = 0;
    virtual size_t Size() const = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(Place pl, size_t size)
        : ptr_(paddle::memory::Alloc(pl, size), paddle::memory::Deleter(pl)),
          place_(pl),
          size_(size) {}

    virtual void* Ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual size_t Size() const { return size_; }
    virtual Place Place() const { return place_; }

    std::unique_ptr<T, memory::Deleter> ptr_;
    Place place_;  // record the place of ptr_.
    size_t size_;  // size of the memory block.
  };

  std::unique_ptr<Placeholder> holder_;  // holds the memory block if allocated.
};

}  // namespace framework
}  // namespace paddle
