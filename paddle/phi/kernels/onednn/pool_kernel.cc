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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/onednn/onednn_reuse.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {

namespace funcs {
template <typename T>
class PoolingOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::pooling_forward,
                                     dnnl::pooling_backward> {
 public:
  PoolingOneDNNHandler(const std::string& pooling_type,
                       const std::vector<int>& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool global_pooling,
                       const std::string& padding_algorithm,
                       bool ceil_mode,
                       bool exclusive,
                       bool adaptive,
                       const dnnl::engine engine,
                       Place cpu_place,
                       const DenseTensor* input,
                       DenseTensor* output)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::pooling_forward,
                                dnnl::pooling_backward>(engine, cpu_place) {
    std::vector<int64_t> copied_kernel_size(kernel_size.begin(),
                                            kernel_size.end());
    std::vector<int64_t> copied_strides(strides.begin(), strides.end());
    std::vector<int64_t> copied_paddings(paddings.begin(), paddings.end());
    // Only 2D pooling is supported now
    PADDLE_ENFORCE_EQ(
        copied_kernel_size.size(),
        2,
        errors::InvalidArgument("The copied_kernel_size must be 2D, i.e. 2D "
                                "pooling, but received %dD.",
                                copied_kernel_size.size()));
    PADDLE_ENFORCE_EQ(
        pooling_type == "max" || pooling_type == "avg",
        true,
        errors::InvalidArgument(
            "The pooling_type must be 'max' or 'avg', but received %s.",
            pooling_type));
    PADDLE_ENFORCE_EQ(
        input->dims().size(),
        4,
        errors::InvalidArgument(
            "Input dim must be with 4, i.e. NCHW, but received %d.",
            input->dims().size()));

    const auto input_dims = input->dims();
    DDim data_dims = slice_ddim(input_dims, 2, input_dims.size());

    if (global_pooling) {
      UpdateKernelSize<int64_t>(&copied_kernel_size, data_dims);
    }

    UpdatePadding<int64_t>(&copied_paddings,
                           global_pooling,
                           0,
                           padding_algorithm,
                           data_dims,
                           copied_strides,
                           copied_kernel_size);

    auto onednn_paddings = ToOneDNNPadding(copied_paddings);

    const auto dt = ToOneDNNDataType(input->dtype());
    const auto src_tz = vectorize(input->dims());
    const auto dst_tz = vectorize(output->dims());
    const auto dst_md = OneDNNMemDesc(dst_tz, dt, OneDNNMemoryFormat::any);

    if (ceil_mode) {
      CorrectOutputSize(src_tz,
                        dst_tz,
                        copied_kernel_size,
                        copied_paddings,
                        copied_strides,
                        onednn_paddings[1]);
    }

    if (adaptive) {
      ComputeAdaptivePoolParameters(
          src_tz, &copied_kernel_size, &copied_strides);
    }
    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training,
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        input->mem_desc(),
        dst_md,
        copied_strides,
        copied_kernel_size,
        onednn_paddings[0],
        onednn_paddings[1]);
  }

  PoolingOneDNNHandler(const std::string& pooling_type,
                       const std::vector<int>& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool global_pooling,
                       const std::string& padding_algorithm,
                       bool ceil_mode,
                       bool exclusive,
                       bool adaptive,
                       const dnnl::engine engine,
                       Place cpu_place,
                       const DenseTensor* in_x,
                       const DenseTensor* out_grad,
                       DenseTensor* in_x_grad)

      : OneDNNHandlerNoCachingT<T,
                                dnnl::pooling_forward,
                                dnnl::pooling_backward>(engine, cpu_place) {
    std::vector<int64_t> copied_kernel_size(kernel_size.begin(),
                                            kernel_size.end());
    std::vector<int64_t> copied_strides(strides.begin(), strides.end());
    std::vector<int64_t> copied_paddings(paddings.begin(), paddings.end());
    auto in_x_dims = in_x->dims();
    DDim data_dims = slice_ddim(in_x_dims, 2, in_x_dims.size());
    if (global_pooling) {
      UpdateKernelSize<int64_t>(&copied_kernel_size, data_dims);
    }

    UpdatePadding<int64_t>(&copied_paddings,
                           global_pooling,
                           0,
                           padding_algorithm,
                           data_dims,
                           copied_strides,
                           copied_kernel_size);

    auto src_tz = vectorize<int64_t>(in_x->dims());
    auto diff_src_tz = vectorize<int64_t>(in_x_grad->dims());
    auto diff_dst_tz = vectorize<int64_t>(out_grad->dims());

    const auto dt = ToOneDNNDataType(in_x->dtype());
    auto dst_md = dnnl::memory::desc(diff_dst_tz, dt, OneDNNMemoryFormat::any);
    auto diff_src_md = dnnl::memory::desc(
        diff_src_tz, oneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    auto onednn_paddings = ToOneDNNPadding(copied_paddings);

    if (ceil_mode) {
      CorrectOutputSize(src_tz,
                        diff_dst_tz,
                        copied_kernel_size,
                        copied_paddings,
                        copied_strides,
                        onednn_paddings[1]);
    }

    if (adaptive) {
      ComputeAdaptivePoolParameters(
          diff_src_tz, &copied_kernel_size, &copied_strides);
    }

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training,
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        in_x->mem_desc(),
        dst_md,
        copied_strides,
        copied_kernel_size,
        onednn_paddings[0],
        onednn_paddings[1]);

    this->AcquireBackwardPrimitiveDescriptor(
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        diff_src_md,
        out_grad->mem_desc(),
        copied_strides,
        copied_kernel_size,
        onednn_paddings[0],
        onednn_paddings[1]);
  }

  std::shared_ptr<dnnl::memory> AcquireWorkspaceMemory(
      const OneDNNContext& dev_ctx, const std::string& unique_name) {
    dnnl::memory::desc workspace_md = this->fwd_pd_->workspace_desc();
    // Pooling Workspace has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    std::string workspace_key = CreateKey(dev_ctx,
                                          workspace_md.dims(),
                                          workspace_md.data_type(),
                                          unique_name,
                                          "@wrk");
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(workspace_key));
    if (mem_p == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      mem_p = std::static_pointer_cast<dnnl::memory>(
          dev_ctx.GetBlob(workspace_key));
      if (mem_p == nullptr) {
        mem_p = std::make_shared<dnnl::memory>(workspace_md, this->engine_);
        dev_ctx.SetBlob(workspace_key, mem_p);
      }
    }
    return mem_p;
  }

  static void ComputeAdaptivePoolParameters(const std::vector<int64_t>& src_tz,
                                            std::vector<int64_t>* kernel_size,
                                            std::vector<int64_t>* strides) {
    // https://github.com/oneapi-src/oneDNN/tree/bkocot/adaptive-pooling/rfcs/20200818-adaptive-pooling
    auto IH = static_cast<double>(src_tz[src_tz.size() - 2]);
    auto IW = static_cast<double>(src_tz[src_tz.size() - 1]);
    auto OH = static_cast<double>(kernel_size->at(0));
    auto OW = static_cast<double>(kernel_size->at(1));

    strides->at(0) =
        static_cast<int64_t>(floor((IH * 2.0) / OH) - floor(IH / OH));
    strides->at(1) =
        static_cast<int64_t>(floor((IW * 2.0) / OW) - floor(IW / OW));
    kernel_size->at(0) =
        static_cast<int64_t>(ceil((IH * 2.0) / OH) - floor(IH / OH));
    kernel_size->at(1) =
        static_cast<int64_t>(ceil((IW * 2.0) / OW) - floor(IW / OW));
  }

 private:
  static inline int ComputeCeiledOutput(int input_size,
                                        int kernel_size,
                                        int padding,
                                        int stride) {
    return (input_size - kernel_size + 2 * padding) / stride + 1;
  }

  static inline void CorrectOutputSize(
      const std::vector<int64_t>& src_tz,
      const std::vector<int64_t>& dst_tz,
      const std::vector<int64_t>& kernel_size,
      const std::vector<int64_t>& paddings,
      const std::vector<int64_t>& strides,
      std::vector<int64_t>& right_bot_padding) {  // NOLINT
    for (size_t i = 0; i < right_bot_padding.size(); i++) {
      int desired_size = ComputeCeiledOutput(
          src_tz[i + 2], kernel_size[i], paddings[i], strides[i]);
      if (desired_size != dst_tz[i + 2]) {
        right_bot_padding[i] += strides[i] - 1;
      }
    }
  }
};
}  // namespace funcs

template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  funcs::PoolingOneDNNHandler<T> handler(pooling_type,
                                         kernel_size,
                                         strides,
                                         paddings,
                                         global_pooling,
                                         padding_algorithm,
                                         ceil_mode,
                                         exclusive,
                                         adaptive,
                                         dev_ctx.GetEngine(),
                                         dev_ctx.GetPlace(),
                                         &x,
                                         out);

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto dst_memory = handler.AcquireDstMemory(out);

  auto pool_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (pooling_type == "max") {
    // Training
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory},
                     {DNNL_ARG_DST, *dst_memory},
                     {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Inference
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *dst_memory}});
  }
  astream.wait();

  out->set_mem_desc(dst_memory->get_desc());
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      const std::vector<int>& kernel_size,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
  funcs::PoolingOneDNNHandler<T> handler(pooling_type,
                                         kernel_size,
                                         strides,
                                         paddings,
                                         global_pooling,
                                         padding_algorithm,
                                         ceil_mode,
                                         exclusive,
                                         adaptive,
                                         dev_ctx.GetEngine(),
                                         dev_ctx.GetPlace(),
                                         &x,
                                         &dout,
                                         dx);

  auto diff_dst_memory = handler.AcquireDiffDstMemory(&dout);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(dx);

  auto pool_bwd_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (pooling_type == "max") {
    // Max - pooling needs Workspace
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                         {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Average Pooling
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory}});
  }
  astream.wait();

  dx->set_mem_desc(diff_src_memory->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(pool2d,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::Pool2dKernel,
                   float,
                   int8_t,
                   uint8_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(pool2d_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::Pool2dGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
