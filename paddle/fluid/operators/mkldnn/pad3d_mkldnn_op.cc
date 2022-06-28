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

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

#define PAD3D_SIZE 6

namespace paddle {
namespace operators {

using paddle::framework::Tensor;


/*
Pad3D is done by using up to 7 reorders. Following example is done
on 2D example for simplicity, but it is straightforward to extend it to 3D case.

Let us consider following example:

          N  C  H  W               L  R  T  B
X dims = (1, 1, 3, 3), paddings = (1, 2, 3, 4) in order Left, Right, Top, Bottom

We have to copy the X tensor into Out tensor, but except from that we have to fill the rest of the memory with additional padding.
To avoid looping through the whole Out memory two times, only these parts of Out memory that won't store X's memory are filled with pad value.
That behavior is achieved by using oneDNN's submemory descriptors which allows us to set offsets for each dimension and skip some parts of the memory.
For 2D case up to 5 reorders will be used in Pad3D kernel(if padding=0 reorder is skipped). 
In the following example i'th number means, that this part of memory was filled by i'th reorder. 4'th reorder is copying X memory into Out memory.
i&j means that both i'th and j'th reorder will set the padding at that location:

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

Since oneDNN's reorder cannot set the pad value to the border memory, we have to prefill Out's memory and use it as a temporary buffer, which later is copied
into the rest of Out's memory. At the end last reorder is done which is copying X memory into Out memory.

*/
template <typename T>
class Pad3dMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    std::vector<int> paddings(ctx.Attr<std::vector<int>>("paddings"));

    T pad_value = static_cast<T>(ctx.Attr<float>("value"));

    auto x_tz = phi::vectorize(x->dims());
    auto out_tz = phi::vectorize(out->dims());

    auto paddle_dtype = framework::TransToProtoVarType(x->dtype());

    platform::ReorderMKLDNNHandler reorder_handler(
      x_tz,
      paddle_dtype,
      framework::ToMKLDNNDataType(paddle_dtype),
      onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(x->mem_desc(), platform::to_void_cast(x->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(out, out_tz, platform::GetPlainMKLDNNFormat(5), ctx.GetPlace());

    T* out_ptr = out->data<T>();
    std::fill(out_ptr, out_ptr+CalculatePrefillElems(out_tz, paddings), pad_value);

    // paddings are in order: left, right, top, bottom, front, back
    for(int i = 0; i < 6; ++i) {
      if(paddings[i] != 0) {
        std::vector<int64_t> offsets(5, 0);
        std::vector<int64_t> chunk_tz(out_tz.begin(), out_tz.end());

        chunk_tz[4 - i / 2] = paddings[i];
        if (i % 2 == 1) {
          offsets[4 - i / 2] = paddings[i - 1] + x_tz[4 - i / 2];
        }

        FillPartOfPadding(paddle_dtype, onednn_engine, out_ptr, reorder_dst_memory_p, chunk_tz, offsets);
      }
    }
    
    std::vector<int64_t> offsets(5, 0); // NCDHW     
    for(int i=0; i<3; ++i) {
      offsets[4-i] = paddings[2*i];
    }
    
    auto slice_mem_p = reorder_handler.AcquireSubmemory(x_tz, offsets, reorder_dst_memory_p);

    auto reorder_p =
        reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
    reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
    astream.wait();

    out->set_mem_desc(reorder_dst_memory_p->get_desc());
  }

  int64_t CalculatePrefillElems(const std::vector<int64_t>& out_tz, const std::vector<int>& paddings) const {
    int64_t max_elems = 0;

    int64_t independent_dims = out_tz[0] * out_tz[1];

    for(int i = 0; i < 3; ++i) {
      int64_t elems = std::max(paddings[2*i], paddings[2*i+1]);
      for(int j = 0; j < 3; ++j) {
        if(j != i) {
          elems *= out_tz[4 - j];
        }
      }

      if(max_elems < elems) {
        max_elems = elems;
      }
    }
    return independent_dims * max_elems;
  }

  void FillPartOfPadding(framework::proto::VarType::Type paddle_dtype,
                         const dnnl::engine& onednn_engine,
                         T* prefilled_mem_ptr,
                         const std::shared_ptr<dnnl::memory>&out_mem_p,
                         std::vector<int64_t>& chunk_tz,
                         const std::vector<int64_t>& offsets) const {
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    dnnl::memory::desc prefilled_mem_desc(chunk_tz, platform::MKLDNNGetDataType<T>(), platform::GetPlainMKLDNNFormat(5));
    auto prefilled_mem_p = std::make_shared<dnnl::memory>(prefilled_mem_desc, onednn_engine, prefilled_mem_ptr);

    platform::ReorderMKLDNNHandler reorder_handler(
      chunk_tz,
      paddle_dtype,
      framework::ToMKLDNNDataType(paddle_dtype),
      onednn_engine);

    auto out_slice_mem_p = reorder_handler.AcquireSubmemory(chunk_tz, offsets, out_mem_p);
    auto reorder_p =
        reorder_handler.AcquireReorder(out_slice_mem_p, prefilled_mem_p);
    reorder_p->execute(astream, *prefilled_mem_p, *out_slice_mem_p);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(pad3d, MKLDNN, paddle::platform::CPUPlace,
                   ops::Pad3dMKLDNNKernel<float>);
