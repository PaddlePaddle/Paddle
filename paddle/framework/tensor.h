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
#if (!PADDLE_ONLY_CPU)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "paddle/framework/ddim.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace pybind {
namespace details {  // forward declare
template <bool less, size_t i, typename... args>
struct CastToPyBufferImpl;
}  // namespace details
}  // namespace pybind

namespace framework {

class Tensor {
 public:
  template <bool less, size_t i, typename... args>
  friend struct paddle::pybind::details::CastToPyBufferImpl;

  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

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

  /*! Resize the dimensions of the memory block. */
  inline void Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  template <typename T>
  inline void ShareDataWith(const Tensor& src);

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
   * @brief   Return the slice of the tensor.
   *
   * @param[in] begin_idx   The begin index of the slice.
   * @param[in] end_idx     The end index of the slice.
   */
  template <typename T>
  inline Tensor Slice(const int& begin_idx, const int& end_idx) const;

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
      PADDLE_ENFORCE(ptr_ != nullptr, "Insufficient %s memory to allocation.",
                     is_cpu_place(place_) ? "CPU" : "GPU");
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

/*
 * LODTensor
 */
class LODTensor {
 public:
// level_t save offsets of each unit.
#ifdef PADDLE_ONLY_CPU
  using level_t = std::vector<uint32_t>;
#else
  using level_t = thrust::device_vector<uint32_t>;
#endif
  // LOD stores offsets of each level of units, the largest units level first,
  // then the smaller units level. Each level_t stores the offsets of units in
  // Tesor.
  typedef std::vector<level_t> lod_t;

  explicit LODTensor() {}

  LODTensor(LODTensor&& other)
      : lod_start_pos_(std::move(other.lod_start_pos_)),
        tensor_(other.tensor_) {}

  /*
   * Number of LODTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t Levels() const {
    return lod_start_pos_.get() ? lod_start_pos_->size() : 0UL;
  }
  /*
   * Number of elements in a level.
   */
  size_t Elements(uint32_t level = 0) const {
    PADDLE_ENFORCE(level < Levels(), "level [%d] out of range [%d]", level,
                   Levels());
    return lod_start_pos_->at(level).size();
  }

  /*
   * Slice of a level.
   */
  LODTensor Slice(uint32_t level) const;

  /*
   * Slice of elements of a level, [elem_begin: elem_end]
   * NOTE low performance in slice seq_start_positions_.
   */
  template <typename T>
  LODTensor Slice(uint32_t level, uint32_t elem_begin, uint32_t elem_end) const;

  // Slice with tensor's data shared with this.
  // LODTensor LODSliceShared(int level, int elem_begin, int elem_end) const;

  // Copy other's lod_start_pos_, to share LOD info.
  // NOTE the LOD info sould not be changed.
  void ShareConstLODFrom(const LODTensor& other) {
    lod_start_pos_ = other.lod_start_pos_;
  }
  // Copy other's lod_start_pos_'s content, free to mutate.
  void ShareMutableLODFrom(const LODTensor& other) {
    lod_start_pos_ = std::make_shared<lod_t>(*other.lod_start_pos_);
  }
  // Determine whether LODTensor has a valid LOD info.
  bool HasLOD() const { return lod_start_pos_.get(); }

  void set_tensor(std::shared_ptr<Tensor> tensor) { tensor_ = tensor; }
  void set_lod(std::shared_ptr<lod_t> lod) { this->lod_start_pos_ = lod; }
  Tensor* tensor() { return tensor_.get(); }

 private:
  std::shared_ptr<lod_t> lod_start_pos_;
  std::shared_ptr<Tensor> tensor_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/framework/detail/tensor-inl.h"
