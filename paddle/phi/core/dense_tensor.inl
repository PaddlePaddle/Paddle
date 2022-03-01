/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/* --------------------------- */
/*   From framework::Tensor    */
/* --------------------------- */
/* The following members & interfaces were copied from framework::Tensor,
    so as to facilitate the unification of different Tensors

    Will be adjusted/removed/moved in the near future
*/
public:
/* Temporarily put InplaceVersion inside DenseTensor.
Will move to AutogradMeta as soon as we switch to Eager Dygraph.
*/
class InplaceVersion {
public:
  bool IsUnique() const { return inplace_version_ == 0; }
  void Bump() { ++inplace_version_; }
  uint32_t CurrentVersion() const { return inplace_version_; }
  void SetInplaceVersionToZero() { inplace_version_ = 0; }

private:
  uint32_t inplace_version_{0};
};

/* @jim19930609: Remove dependency on protobuf after Tensor Unification.
*/
explicit DenseTensor(paddle::experimental::DataType dtype);

/// \brief Use existing storage space to create dense tensor. This interface
/// can be used to deliberately create an uninitialized dense tensor.
/// \param storage The existing storage.
/// \param meta The meta data of dense tensor.
DenseTensor(intrusive_ptr<Storage> storage, const DenseTensorMeta& meta);

/// \brief Use existing storage space to create dense tensor. This interface
/// can be used to deliberately create an uninitialized dense tensor.
/// \param storage The existing storage.
/// \param meta The meta data of dense tensor.
DenseTensor(intrusive_ptr<Storage> storage, DenseTensorMeta&& meta);

inline bool IsInitialized() const { return holder_ != nullptr; }

template <typename T>
T* mutable_data(const phi::Place& place,
                size_t requested_size = 0);

template <typename T>
T* mutable_data(const DDim& dims,
                const phi::Place& place,
                size_t requested_size = 0);

void* mutable_data(const phi::Place& place,
                    paddle::experimental::DataType type,
                    size_t requested_size = 0);

void* mutable_data(const phi::Place& place,
                    size_t requested_size = 0);

void* mutable_data(const phi::Place& place,
                    paddle::experimental::DataType type,
                    const phi::Stream& stream);

/* @jim19930609: Remove dependency on protobuf after Tensor Unification.
*/
paddle::experimental::DataType type() const;

// memory size returns the holding memory size in byte.
size_t memory_size() const;

void check_memory_size() const;

void set_layout(const paddle::framework::DataLayout layout);

void clear() {
  holder_.reset();
  meta_.offset = 0;
}

void ShareBufferWith(const DenseTensor& tensor);

void ShareDataTypeWith(const DenseTensor& tensor) {
  meta_.dtype = tensor.meta().dtype;
}

bool IsSharedBufferWith(const DenseTensor& src) const {
  return holder_ && holder_ == src.Holder();
}

const std::shared_ptr<phi::Allocation>& Holder() const { return holder_; }

void set_offset(size_t offset) { meta_.offset = offset; }
size_t offset() const { return meta_.offset; }

std::shared_ptr<phi::Allocation> MoveMemoryHolder() {
  return std::move(holder_);
}

void ResetHolder(const std::shared_ptr<phi::Allocation>& holder);

void ResetHolderWithType(const std::shared_ptr<phi::Allocation>& holder,
                        paddle::experimental::DataType type);

void set_type(paddle::experimental::DataType type);

InplaceVersion& InplaceVersionCounter() {
  return *inplace_version_counter_;
}

/*! The internal of two tensors share the same memory block. */
DenseTensor& ShareDataWith(const DenseTensor& src);

/*! The internal of two tensors share the same inplace version counter. */
DenseTensor& ShareInplaceVersionCounterWith(const DenseTensor& src);

DenseTensor Slice(int64_t begin_idx, int64_t end_idx) const;

std::vector<DenseTensor> Split(int64_t split_size, int64_t axis) const;

std::vector<DenseTensor> Chunk(int64_t chunks, int64_t axis) const;

protected:
std::shared_ptr<InplaceVersion> inplace_version_counter_{std::make_shared<InplaceVersion>()};

/* @jim19930609: This is a hack
In general, it is badly designed to fuse MKLDNN-specific objects into a
generic Tensor.
We temporarily leave them here to unblock Tensor Unification progress.
In the final state, we should come up with a MKLDNN_Tensor and move the
following codes there.
*/
#ifdef PADDLE_WITH_MKLDNN

public:
inline dnnl::memory::format_tag format() const { return format_; }

inline void set_format(const dnnl::memory::format_tag format) {
  format_ = format;
}

protected:
/**
 * @brief the detail format of memory block which have layout as kMKLDNN
 *
 * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
 *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
 *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
 *       this field.
 */

dnnl::memory::format_tag format_ = dnnl::memory::format_tag::undef;
#endif

/* ------------------------------ */
/*   From framework::LoDTensor    */
/* ------------------------------ */
/* The following members & interfaces were copied from framework::Tensor,
    so as to facilitate the unification of different Tensors

    Will be adjusted/removed/moved in the near future
*/
public:
explicit DenseTensor(const LoD& lod);

void set_lod(const LoD& lod);

LoD* mutable_lod();

/*
* Get the start offset and end offset of an  element from LoD.
*/
std::pair<size_t, size_t> lod_element(size_t level, size_t elem) const;

size_t NumLevels() const;

size_t NumElements(size_t level = 0) const;
