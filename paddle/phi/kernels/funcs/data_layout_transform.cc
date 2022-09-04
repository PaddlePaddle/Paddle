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

#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#endif

namespace phi {
namespace funcs {

#ifdef PADDLE_WITH_MKLDNN

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
      PADDLE_THROW(errors::InvalidArgument("Wrong mkldnn type provided."));
  }
}

void innerTransDataLayoutFromOneDNN(DataLayout in_layout,
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

  auto in_tz = vectorize<int64_t>(in.dims());
  auto out_tz = in_tz;

  auto in_type = ToOneDNNDataType(in.dtype());
  PADDLE_ENFORCE_NE(
      in_type,
      OneDNNDataType::undef,
      errors::InvalidArgument("Input tensor type (%s) is not supported.",
                              in.dtype()));

  auto out_format =
      OneDNNFormatForSize(in_tz.size(), ToOneDNNFormat(out_layout));
  dnnl::memory::desc out_mem_desc(out_tz, in_type, out_format);

  // output tensor has the same dims as input. Reorder don't change dims
  out->set_mem_desc(out_mem_desc);
  out->Resize(in.dims());

  if ((in.mem_desc() != out->mem_desc()) || always_copy) {
    void* in_data = GetDataFromTensor(in, in_type);

    ReorderOneDNNHandler handler(in_tz, in.dtype(), in_type, cpu_engine);

    auto reorder_src_memory_p =
        handler.AcquireSrcMemory(in.mem_desc(), in_data);
    auto reorder_dst_memory_p =
        handler.AcquireDstMemory(out, out->mem_desc(), place);
    auto reorder_p =
        handler.AcquireReorder(reorder_dst_memory_p, reorder_src_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();
    ::paddle::platform::RecordEvent record_reorder(
        "ext_reorder",
        ::paddle::platform::TracerEventType::UserDefined,
        2,
        ::paddle::platform::EventRole::kUniqueOp);
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();
  } else {
    out->ShareDataWith(in);
  }
  // For exepected NHWC data format we need to reshape the Output tensor
  // As MKL-DNN description was in NCHW and paddle is expecting NHWC
  MatchShapeToLayout(out, in_layout, out_layout);

  out->set_layout(DataLayout::kNCHW);
  VLOG(10) << "out->layout: " << out->layout() << " in->dims: " << in.dims()
           << " out->dims: " << out->dims();
  // reset format since the out tensor will be feed to non-MKLDNN OPkernel
  out->set_format(OneDNNMemoryFormat::undef);
}

#endif

}  // namespace funcs
}  // namespace phi
