// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace phi {

/*
Pad3D is done by using up to 7 reorders. Following example is done
on 2D data for simplicity, but it is straightforward to extend it to 3D case.

Let us consider following example:

          N  C  H  W               L  R  T  B
X_dims = (1, 1, 3, 3), paddings = (1, 2, 3, 4) in order Left, Right, Top, Bottom

We have to copy the X tensor into Out tensor, but except from that we have to
fill the rest of the memory with an additional padding. To avoid looping through
the whole Out memory two times, only these parts of Out memory that won't store
X's memory are filled with pad value. That behavior is achieved by using
oneDNN's submemory descriptors which allows us to set offsets for each dimension
and skip some parts of the memory. For 2D case up to 5 reorders will be used in
Pad3D kernel(if padding=0 reorder is skipped). In the following example i'th
number means, that this part of memory was filled by i'th reorder. 4'th reorder
is copying X memory into Out memory. i&j means that both i'th and j'th reorder
will set the padding at that location:

               INDEX
     | 0   1   2   3   4   5
     |_______________________
   0 |0&2  2   2   2  1&2 1&2
   1 |0&2  2   2   2  1&2 1&2
I  2 |0&2  2   2   2  1&2 1&2
N  3 | 0   4   4   4   1   1
D  4 | 0   4   4   4   1   1
E  5 | 0   4   4   4   1   1
X  6 |0&3  3   3   3  1&3 1&3
   7 |0&3  3   3   3  1&3 1&3
   8 |0&3  3   3   3  1&3 1&3
   9 |0&3  3   3   3  1&3 1&3

Since oneDNN's reorder cannot set the pad value to the memory by itself, we have
to prefill Out's memory and use it as a temporary buffer, which later is copied
into the rest of Out's memory. At the end last reorder is done which copies X
memory into Out memory.

*/

inline int64_t CalculateNumOfPrefillElems(
    const std::vector<int64_t>& out_tz, const std::vector<int64_t>& paddings) {
  int64_t max_elems = 0;
  int64_t independent_dims = out_tz[0] * out_tz[1];

  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    int64_t elems = std::max(paddings[2 * i], paddings[2 * i + 1]);
    for (size_t j = 0; j < paddings.size() / 2; ++j) {
      if (j != i) {
        elems *= out_tz[out_tz.size() - 1 - j];
      }
    }

    if (max_elems < elems) {
      max_elems = elems;
    }
  }
  return independent_dims * max_elems;
}

template <typename T>
void FillPartOfPadding(const dnnl::engine& onednn_engine,
                       T* prefilled_mem_ptr,
                       const std::shared_ptr<dnnl::memory>& out_mem_p,
                       const std::vector<int64_t>& chunk_tz,
                       const std::vector<int64_t>& offsets) {
  auto& astream = OneDNNContext::tls().get_stream();

  dnnl::memory::desc prefilled_mem_desc(
      chunk_tz,
      funcs::OneDNNGetDataType<T>(),
      funcs::GetPlainOneDNNFormat(chunk_tz.size()));
  dnnl::memory prefilled_mem(
      prefilled_mem_desc, onednn_engine, prefilled_mem_ptr);

  dnnl::memory::desc out_slice_md =
      out_mem_p->get_desc().submemory_desc(chunk_tz, {offsets});
  dnnl::memory out_slice_mem(
      out_slice_md, onednn_engine, out_mem_p->get_data_handle());

  auto reorder_p = dnnl::reorder(prefilled_mem, out_slice_mem);
  reorder_p.execute(astream, prefilled_mem, out_slice_mem);
}

template <typename T, typename Context>
void PadOpKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& paddings,
                 float pad_value,
                 DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  auto& astream = OneDNNContext::tls().get_stream();

  std::vector<int64_t> x_tz = vectorize(x.dims());
  // due to the need of supporting NDHWC, inferring out shape
  // must be done inside the kernel
  std::vector<int64_t> out_tz(x_tz);

  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    out_tz[out_tz.size() - 1 - i] += paddings[2 * i] + paddings[2 * i + 1];
  }
  out->Resize(make_ddim(out_tz));

  funcs::ReorderOneDNNHandler reorder_handler(
      x_tz, x.dtype(), funcs::ToOneDNNDataType(x.dtype()), onednn_engine);

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out,
      out_tz,
      funcs::GetPlainOneDNNFormat(out_tz.size()),
      dev_ctx.GetPlace());

  // to avoid allocating new temporary memory, Out's memory is used as a tmp
  // buffer for storing a contiguous memory consisting of pad_value, which
  // later is used as a SRC for reorders that are filling Out with padding
  T* out_ptr = out->data<T>();
  std::fill(out_ptr,
            out_ptr + CalculateNumOfPrefillElems(out_tz, paddings),
            pad_value);

  // paddings are in order: left, right, top, bottom, front, back
  for (size_t i = 0; i < paddings.size(); ++i) {
    if (paddings[i] != 0) {
      std::vector<int64_t> offsets(out_tz.size(), 0);
      std::vector<int64_t> chunk_tz(out_tz.begin(), out_tz.end());

      chunk_tz[out_tz.size() - 1 - i / 2] = paddings[i];
      if (i % 2 == 1) {
        offsets[out_tz.size() - 1 - i / 2] =
            paddings[i - 1] + x_tz[out_tz.size() - 1 - i / 2];
      }

      FillPartOfPadding(
          onednn_engine, out_ptr, reorder_dst_memory_p, chunk_tz, offsets);
    }
  }
  astream.wait();

  std::vector<int64_t> offsets(out_tz.size(), 0);
  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    offsets[out_tz.size() - 1 - i] = paddings[2 * i];
  }

  auto slice_mem_p =
      reorder_handler.AcquireSubmemory(x_tz, offsets, reorder_dst_memory_p);

  auto reorder_p =
      reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
  reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
  astream.wait();

  out->set_mem_desc(reorder_dst_memory_p->get_desc());
}
}  // namespace phi
