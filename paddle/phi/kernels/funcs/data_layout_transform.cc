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

#include "paddle/phi/kernels/funcs/data_layout_transform.h"

#include "glog/logging.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#endif

namespace phi::funcs {

#ifdef PADDLE_WITH_DNNL

void* GetDataFromTensor(const DenseTensor& tensor,
                        dnnl::memory::data_type type) {
  switch (type) {
    case dnnl::memory::data_type::f32:
      return to_void_cast(tensor.data<float>());
    case dnnl::memory::data_type::s8:
      return to_void_cast(tensor.data<int8_t>());
    case dnnl::memory::data_type::u8:
      return to_void_cast(tensor.data<unsigned char>());
    case dnnl::memory::data_type::s32:
      return to_void_cast(tensor.data<int32_t>());
    case dnnl::memory::data_type::bf16:
      return to_void_cast(tensor.data<dtype::bfloat16>());
    default:
      PADDLE_THROW(errors::InvalidArgument("Wrong oneDNN type provided."));
  }
}

// This helper function is used to construct a dnnl memory descriptor from a
// reference dense tensor and a target layout. For 0-D tensor case, we will
// construct a 1-D memory descriptor with shape [1], since oneDNN didn't support
// 0-D now.
dnnl::memory::desc make_memory_desc(const phi::DenseTensor& ref_tensor,
                                    phi::DataLayout target_layout) {
  auto ref_dims = common::vectorize<int64_t>(ref_tensor.dims());
  auto ref_type = ToOneDNNDataType(ref_tensor.dtype());
  PADDLE_ENFORCE_NE(ref_type,
                    OneDNNDataType::undef,
                    errors::InvalidArgument(
                        "Ref tensor type (%s) is not supported by oneDNN.",
                        ref_tensor.dtype()));

  auto md_dims = !ref_dims.empty() ? ref_dims : std::vector<int64_t>{1};
  auto md_format =
      OneDNNFormatForSize(md_dims.size(), ToOneDNNFormat(target_layout));
  dnnl::memory::desc md(md_dims, ref_type, md_format);
  return md;
}

void TransDataLayoutFromOneDNN(DataLayout in_layout,
                               DataLayout out_layout,
                               const DenseTensor& in,
                               DenseTensor* out,
                               Place place,
                               bool always_copy) {
  // Set default as NCHW in case not specified
  out_layout = out_layout == DataLayout::ANY ? DataLayout::NCHW : out_layout;

  auto& pool = DeviceContextPool::Instance();
  auto* dev_ctx = dynamic_cast<OneDNNContext*>(pool.Get(place));
  auto& cpu_engine = dev_ctx->GetEngine();
  auto in_dims = common::vectorize<int64_t>(in.dims());

  auto md_dims = !in_dims.empty() ? in_dims : std::vector<int64_t>{1};
  const auto src_mem_desc =
      !in_dims.empty() ? in.mem_desc()
                       : dnnl::memory::desc(md_dims,
                                            ToOneDNNDataType(in.dtype()),
                                            dnnl::memory::format_tag::x);

  dnnl::memory::desc out_mem_desc = make_memory_desc(in, out_layout);

  // output tensor has the same dims as input. Reorder don't change dims
  out->set_mem_desc(out_mem_desc);
  out->Resize(in.dims());

  // Note(0x45f): Using initialized() to support slice Tensors
  // with shapes like [0, 0, 0].
  if (in.initialized() && ((in.mem_desc() != out->mem_desc()) || always_copy)) {
    auto in_tz = common::vectorize<int64_t>(in.dims());
    auto in_type = ToOneDNNDataType(in.dtype());
    void* in_data = GetDataFromTensor(in, in_type);

    ReorderOneDNNHandler handler(in_tz, in.dtype(), in_type, cpu_engine);

    auto reorder_src_memory_p = handler.AcquireSrcMemory(src_mem_desc, in_data);
    auto reorder_dst_memory_p =
        handler.AcquireDstMemory(out, out->mem_desc(), place);
    auto reorder_p =
        handler.AcquireReorder(reorder_dst_memory_p, reorder_src_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();
  } else {
    out->ShareDataWith(in);
  }
  // For expected NHWC data format we need to reshape the Output tensor
  // As MKL-DNN description was in NCHW and paddle is expecting NHWC
  MatchShapeToLayout(out, in_layout, out_layout);

  out->set_layout(DataLayout::kNCHW);
  VLOG(10) << "out->layout: " << out->layout() << " in->dims: " << in.dims()
           << " out->dims: " << out->dims();
}

#endif

}  // namespace phi::funcs
