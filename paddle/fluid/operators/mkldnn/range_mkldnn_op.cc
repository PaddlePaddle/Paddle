/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class RangeMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                               dnnl::eltwise_backward> {
 public:
  RangeMKLDNNHandler(const T step, const int tensor_size,
                     const dnnl::engine engine, platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                          dnnl::eltwise_backward>(engine,
                                                                  cpu_place) {
    auto md = dnnl::memory::desc({tensor_size - RANGE_SHIFT},
                                 platform::MKLDNNGetDataType<T>(),
                                 dnnl::memory::format_tag::a);

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_linear, md,
                                            1.0f, RANGE_SHIFT * step);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(framework::Tensor *out) {
    T *ptr = out->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(), ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor *out) {
    T *ptr = out->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(),
                                            ptr + RANGE_SHIFT);
  }

  static constexpr int RANGE_SHIFT = 64 / sizeof(T);  // AVX512 stores 64 bytes
};

template <typename T>
class RangeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto &dnnl_engine = dev_ctx.GetEngine();

    const T start = ctx.Input<Tensor>("Start")->data<T>()[0];
    const T end = ctx.Input<Tensor>("End")->data<T>()[0];
    const T step = ctx.Input<Tensor>("Step")->data<T>()[0];

    const int64_t size = std::ceil(std::abs((end - start) / step));

    Tensor *out = ctx.Output<Tensor>("Out");
    out->Resize(framework::make_ddim({size}));
    out->mutable_data<T>(ctx.GetPlace());

    T *data = out->data<T>();
    data[0] = start;

    if (size <= RangeMKLDNNHandler<T>::RANGE_SHIFT) {
      for (int i = 1; i < size; ++i) {
        data[i] = data[i - 1] + step;
      }
    } else {
      for (int i = 1; i < RangeMKLDNNHandler<T>::RANGE_SHIFT; ++i) {
        data[i] = data[i - 1] + step;
      }

      RangeMKLDNNHandler<T> handler(step, size, dnnl_engine, ctx.GetPlace());

      auto src_memory_p = handler.AcquireSrcMemory(out);
      auto dst_memory_p = handler.AcquireDstMemory(out);

      auto range_p = handler.AcquireForwardPrimitive();

      auto &astream = platform::MKLDNNDeviceContext::tls().get_stream();

#if defined(_OPENMP)
      // range kernel can only work single-threaded since
      // the memories overlaps and execution must be done
      // sequentially because only part of it is initialized
      int prev_num_threads = omp_get_num_threads();
      omp_set_num_threads(1);
#endif

      range_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                 {MKLDNN_ARG_DST, *dst_memory_p}});
      astream.wait();

#if defined(_OPENMP)
      omp_set_num_threads(prev_num_threads);
#endif
    }

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(dnnl::memory::format_tag::a);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(range, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::RangeMKLDNNKernel<float>);
