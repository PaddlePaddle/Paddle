/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <vector>

#include "paddle/framework/ddim.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {

namespace framework {

class Tensor {
 public:
  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenMatrix;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor() : offset_(0) {}

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  inline T* data();

  /*! Return a pointer to constant memory block. */
  template <typename T>
  inline const T* data() const;

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(platform::Place place);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims    The dimensions of the memory block.
   * @param[in] place   The place of the memory block.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(DDim dims, platform::Place place);

  /*! Return the dimensions of the memory block. */
  inline const DDim& dims() const;

  /*! Return the numel of the memory block. */
  inline int64_t numel() const;

  /*! Resize the dimensions of the memory block. */
  inline Tensor& Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  template <typename T>
  inline Tensor& ShareDataWith(const Tensor& src);

  /**
   * @brief   Copy the content of external tensor to a new place.
   *
   * @param[in] src   The external tensor.
   * @param[in] ctx   The device context contains place where to store.
   *
   * @note    CopyFrom supports CPU <-> GPU, GPU <-> GPU.
   */
  template <typename T>
  inline void CopyFrom(const Tensor& src, const platform::Place& dst_place);

  /**
   * @brief   Copy the content of an external vector to a tensor.
   *
   * @param[in] src   The external vector.
   * @param[in] ctx   The device context contains place where to store.
   *
   * * @note    CopyFromVector assumes that the tensor has been resized
   *            before invoking.
   */
  template <typename T>
  inline void CopyFromVector(const std::vector<T>& src,
                             const platform::Place& dst_place);

  /**
   * @brief   Return the slice of the tensor.
   *
   * @param[in] begin_idx   The begin index of the slice.
   * @param[in] end_idx     The end index of the slice.
   */
  template <typename T>
  inline Tensor Slice(const int& begin_idx, const int& end_idx) const;

  platform::Place place() const {
    PADDLE_ENFORCE_NOT_NULL(holder_, "Tensor get place() must contains holder");
    return holder_->place();
  }

  std::type_index type() const { return holder_->type(); }

 private:
  template <typename T>
  inline void check_memory_size() const;

 private:
  /**
   * @note    Placeholder hides type T, so it doesn't appear as a template
   *          parameter of Variable.
   */
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* ptr() const = 0;
    virtual size_t size() const = 0;
    virtual std::type_index type() const = 0;
    virtual platform::Place place() const = 0;
  };

  template <typename T, typename Place>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(Place place, size_t size)
        : ptr_(static_cast<T*>(memory::Alloc(place, size)),
               memory::PODDeleter<T, Place>(place)),
          place_(place),
          size_(size) {
      PADDLE_ENFORCE_NOT_NULL(ptr_, "Insufficient %s memory to allocation.",
                              (is_cpu_place(place_) ? "CPU" : "GPU"));
    }

    virtual size_t size() const { return size_; }
    virtual platform::Place place() const { return place_; }
    virtual void* ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual std::type_index type() const { return std::type_index(typeid(T)); }

    /*! the pointer of memory block. */
    std::unique_ptr<T, memory::PODDeleter<T, Place>> ptr_;

    /*! the place of memory block. */
    platform::Place place_;

    /*! the size of memory block. */
    size_t size_;
  };

  /*! holds the memory block if allocated. */
  std::shared_ptr<Placeholder> holder_;

  /*! points to dimensions of memory block. */
  DDim dims_;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really begins.
   */
  size_t offset_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/framework/tensor_impl.h"
