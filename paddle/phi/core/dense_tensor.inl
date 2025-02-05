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
/*   From phi::DenseTensor    */
/* --------------------------- */
/* The following members & interfaces were copied from phi::DenseTensor,
    so as to facilitate the unification of different Tensors

    Will be adjusted/removed/moved in the near future
*/

public:
/* @jim19930609: Remove dependency on protobuf after Tensor Unification.
 */
explicit DenseTensor(phi::DataType dtype);

inline bool IsInitialized() const { return holder_ != nullptr; }

template <typename T>
T* mutable_data(const phi::Place& place, size_t requested_size = 0);

template <typename T>
T* mutable_data(const DDim& dims,
                const phi::Place& place,
                size_t requested_size = 0);

void* mutable_data(const phi::Place& place,
                   phi::DataType type,
                   size_t requested_size = 0);

void* mutable_data(const phi::Place& place, size_t requested_size = 0);

void* mutable_data(const phi::Place& place,
                   phi::DataType type,
                   const phi::Stream& stream);

/* @jim19930609: Remove dependency on protobuf after Tensor Unification.
 */
phi::DataType type() const;

// memory size returns the holding memory size in byte.
size_t memory_size() const;

void check_memory_size() const;

void set_layout(const DataLayout layout);

void ShareBufferWith(const DenseTensor& tensor, bool only_buffer=false);

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
                         phi::DataType type);

void set_type(phi::DataType type);

InplaceVersion& InplaceVersionCounter() { return *inplace_version_counter_; }

/*! The internal of two tensors share the same memory block. */
DenseTensor& ShareDataWith(const DenseTensor& src);

/*! The internal of two tensors share the same memory block without checking the memory size for dist-tensor. */
DenseTensor& ShareDataNoCheckWith(const DenseTensor& src);

/*! The internal of two tensors share the same inplace version counter. */
DenseTensor& ShareInplaceVersionCounterWith(const DenseTensor& src);

DenseTensor Slice(int64_t begin_idx, int64_t end_idx) const;

std::vector<DenseTensor> Split(int64_t split_size, int64_t axis) const;

std::vector<DenseTensor> Chunk(int64_t chunks, int64_t axis) const;

#ifdef PADDLE_WITH_DNNL

public:
const dnnl::memory::desc& mem_desc() const;

void set_mem_desc(const dnnl::memory::desc& mem_desc);

#endif

/* ------------------------------ */
/*   From phi::DenseTensor    */
/* ------------------------------ */
/* The following members & interfaces were copied from phi::DenseTensor,
    so as to facilitate the unification of different Tensors

    Will be adjusted/removed/moved in the near future
*/
public:
explicit DenseTensor(const LegacyLoD& lod);

void set_lod(const LoD& lod);

LegacyLoD* mutable_lod();

/*
 * Get the start offset and end offset of an  element from LoD.
 */
std::pair<size_t, size_t> lod_element(size_t level, size_t elem) const;

size_t NumLevels() const;

size_t NumElements(size_t level = 0) const;
