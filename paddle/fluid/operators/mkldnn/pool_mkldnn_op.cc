/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;

// Generate keys for storing/retriving primitives for this operator
std::string CreateKey(const paddle::framework::ExecutionContext& ctx,
                      const memory::dims& input_dims,
                      const std::string& pooling_type,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const memory::data_type& dt, const std::string& suffix) {
  std::string key;
  key.reserve(platform::MKLDNNHandler::MaxKeyLength);
  platform::MKLDNNHandler::AppendKeyDims(&key, input_dims);
  platform::MKLDNNHandler::AppendKey(&key, pooling_type);
  platform::MKLDNNHandler::AppendKeyVec(&key, ksize);
  platform::MKLDNNHandler::AppendKeyVec(&key, strides);
  platform::MKLDNNHandler::AppendKeyVec(&key, paddings);
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(dt));
  platform::MKLDNNHandler::AppendKey(&key, suffix);
  return key;
}

static inline int ComputeCeiledOutput(int input_size, int kernel_size,
                                      int padding, int stride) {
  return (input_size - kernel_size + 2 * padding) / stride + 1;
}

static inline void CorrectOutputSize(
    const std::vector<int>& src_tz, const std::vector<int>& dst_tz,
    const std::vector<int>& kernel_size, const std::vector<int>& paddings,
    const std::vector<int>& strides,
    std::vector<int>& right_bot_padding) {  // NOLINT
  for (size_t i = 0; i < right_bot_padding.size(); i++) {
    int desired_size = ComputeCeiledOutput(src_tz[i + 2], kernel_size[i],
                                           paddings[i], strides[i]);
    if (desired_size != dst_tz[i + 2]) {
      right_bot_padding[i] += strides[i];
    }
  }
}

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    bool is_test = ctx.Attr<bool>("is_test");
    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    // Only 2D pooling is supported now
    PADDLE_ENFORCE(ksize.size() == 2, "ksize must be 2D, i.e. 2D pooling");
    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "avg",
                   "pooling_type must be 'max' or 'avg'");
    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input dim must be with 4, i.e. NCHW");

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    auto input_format = input->format();
    memory::format output_format{memory::format::format_undef};

    mkldnn::memory::data_type dt =
        paddle::framework::ToMKLDNNDataType(input->type());
    const std::string key = CreateKey(ctx, src_tz, pooling_type, ksize, strides,
                                      paddings, dt, ctx.op().Output("Out"));
    const std::string key_pool_p = key + "@pool_p";
    const std::string key_pool_pd = key + "@pool_pd";
    const std::string key_pool_src_mem_p = key + "@pool_src_mem_p";
    const std::string key_pool_dst_mem_p = key + "@pool_dst_mem_p";
    const std::string key_pool_workspace_memory =
        key + "@pool_workspace_memory";

    auto pool_p =
        std::static_pointer_cast<pooling_forward>(dev_ctx.GetBlob(key_pool_p));
    if (pool_p == nullptr) {
      const std::vector<int>& padding_left_top(paddings);
      std::vector<int> padding_right_bottom(paddings);
      bool ceil_mode = ctx.Attr<bool>("ceil_mode");
      if (ceil_mode) {
        CorrectOutputSize(src_tz, dst_tz, ksize, paddings, strides,
                          padding_right_bottom);
      }

      auto src_md = platform::MKLDNNMemDesc(src_tz, dt, input_format);

      /* create memory descriptor for pooling without specified format
       * ('any') which lets a primitive (pooling in this case) choose
       * the memory format preferred for best performance
       */
      auto dst_md =
          platform::MKLDNNMemDesc(dst_tz, dt, mkldnn::memory::format::any);
      auto propagation = src_md.data.data_type == mkldnn_f32
                             ? mkldnn::prop_kind::forward_training
                             : mkldnn::prop_kind::forward_scoring;
      std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd =
          CreatePrimitiveDesc(src_md, dst_md, propagation, strides,
                              padding_left_top, padding_right_bottom, ksize,
                              pooling_type, mkldnn_engine, ceil_mode, is_test);

      // save pool_pd into global device context to be referred in backward path
      if (!is_test) dev_ctx.SetBlob(key_pool_pd, pool_pd);

      auto src_memory = std::make_shared<memory>(pool_pd->src_primitive_desc(),
                                                 to_void_cast<T>(input_data));
      auto dst_memory =
          std::make_shared<memory>(pool_pd->dst_primitive_desc(), output_data);

      dev_ctx.SetBlob(key_pool_src_mem_p, src_memory);
      dev_ctx.SetBlob(key_pool_dst_mem_p, dst_memory);

      if (is_test) {
        pool_p = std::make_shared<pooling_forward>(*pool_pd, *src_memory,
                                                   *dst_memory);
      } else {
        std::shared_ptr<mkldnn::memory> workspace_memory =
            CreateWorkspaceMemory(pool_pd, pooling_type, mkldnn_engine);

        // save pool_workspace_memory to be referred in backward path
        dev_ctx.SetBlob(key_pool_workspace_memory, workspace_memory);

        pool_p = std::make_shared<pooling_forward>(
            *pool_pd, *src_memory, *dst_memory, *workspace_memory);
      }

      dev_ctx.SetBlob(key_pool_p, pool_p);

      output_format =
          (memory::format)dst_memory->get_primitive_desc().desc().data.format;
    } else {
      // Primitives already exist
      auto pool_src_memory_p =
          std::static_pointer_cast<memory>(dev_ctx.GetBlob(key_pool_src_mem_p));
      PADDLE_ENFORCE(pool_src_memory_p != nullptr,
                     "Fail to find pooling src mem_p in device context");
      auto pool_dst_memory_p =
          std::static_pointer_cast<memory>(dev_ctx.GetBlob(key_pool_dst_mem_p));
      PADDLE_ENFORCE(pool_dst_memory_p != nullptr,
                     "Fail to find pooling dst mem_p in device context");
      pool_src_memory_p->set_data_handle(to_void_cast<T>(input_data));
      pool_dst_memory_p->set_data_handle(output_data);

      output_format = (memory::format)pool_dst_memory_p->get_primitive_desc()
                          .desc()
                          .data.format;
    }

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{*pool_p};
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(output_format);
  }

 private:
  std::unique_ptr<mkldnn::pooling_forward::primitive_desc> CreatePrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& dst,
      const mkldnn::prop_kind& propagation, const std::vector<int>& stride,
      const std::vector<int>& padding_left_top,
      const std::vector<int>& padding_right_bot, const std::vector<int>& kernel,
      const std::string& pooling_type, const mkldnn::engine& engine,
      bool ceil_mode, bool is_test) const {
    auto mkldnn_forward_prop_kind = is_test
                                        ? mkldnn::prop_kind::forward_inference
                                        : mkldnn::prop_kind::forward_training;
    auto pool_desc = mkldnn::pooling_forward::desc(
        mkldnn_forward_prop_kind,
        pooling_type == "max" ? mkldnn::algorithm::pooling_max
                              : mkldnn::algorithm::pooling_avg,
        src, dst, stride, kernel, padding_left_top, padding_right_bot,
        mkldnn::padding_kind::zero);

    auto p_pool_pd =
        new mkldnn::pooling_forward::primitive_desc(pool_desc, engine);
    return std::unique_ptr<mkldnn::pooling_forward::primitive_desc>(p_pool_pd);
  }

  std::unique_ptr<mkldnn::memory> CreateWorkspaceMemory(
      std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd,
      const std::string& pooling_type, const mkldnn::engine& engine) const {
    mkldnn::memory::primitive_desc workspace_md =
        pooling_type == "max"
            ? pool_pd->workspace_primitive_desc()
            : mkldnn::memory::primitive_desc({{},
                                              platform::MKLDNNGetDataType<T>(),
                                              mkldnn::memory::format::nchw},
                                             engine);

    auto p_workspace_memory = new mkldnn::memory(workspace_md);
    return std::unique_ptr<mkldnn::memory>(p_workspace_memory);
  }
};

template <typename T>
class PoolMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    const Tensor* in_x = ctx.Input<Tensor>("X");
    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE(in_x->layout() == DataLayout::kMKLDNN &&
                       in_x->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input X tensor");
    PADDLE_ENFORCE(out_grad->layout() == DataLayout::kMKLDNN &&
                       out_grad->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input output_grad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const mkldnn::engine& mkldnn_engine = dev_ctx.GetEngine();

    const T* out_grad_data = out_grad->data<T>();
    T* in_x_grad_data = in_x_grad->mutable_data<T>(ctx.GetPlace());
    memory::format in_x_grad_format{memory::format::format_undef};

    std::vector<int> diff_src_tz =
        paddle::framework::vectorize2int(in_x_grad->dims());
    std::vector<int> diff_dst_tz =
        paddle::framework::vectorize2int(out_grad->dims());

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when referring info from device context
    const std::string key =
        CreateKey(ctx, diff_src_tz, pooling_type, ksize, strides, paddings,
                  memory::data_type::f32, ctx.op().Input("Out"));
    const std::string key_pool_bwd_p = key + "@pool_bwd_p";
    const std::string key_pool_diff_src_mem_p = key + "@pool_diff_src_mem_p";
    const std::string key_pool_diff_dst_mem_p = key + "@pool_diff_dst_mem_p";
    const std::string key_pool_src_mem_p = key + "@pool_src_mem_p";
    const std::string key_pool_dst_mem_p = key + "@pool_dst_mem_p";
    const std::string key_pool_pd = key + "@pool_pd";
    const std::string key_pool_workspace_memory =
        key + "@pool_workspace_memory";

    auto user_diff_dst_memory =
        memory({{{diff_dst_tz}, memory::data_type::f32, out_grad->format()},
                mkldnn_engine},
               to_void_cast<T>(out_grad_data));

    std::shared_ptr<memory> diff_src_memory;
    std::shared_ptr<memory> diff_dst_memory;
    auto dst_memory =
        std::static_pointer_cast<memory>(dev_ctx.GetBlob(key_pool_dst_mem_p));
    PADDLE_ENFORCE(dst_memory != nullptr,
                   "Fail to find dst_memory in device context");

    primitive reorder_diff_dst;
    bool is_diff_dst_reordered = false;
    auto pool_bwd_p = std::static_pointer_cast<pooling_backward>(
        dev_ctx.GetBlob(key_pool_bwd_p));
    if (pool_bwd_p == nullptr) {
      // Retrieve src_memory/dst_memory saved in forward pass
      auto src_memory =
          std::static_pointer_cast<memory>(dev_ctx.GetBlob(key_pool_src_mem_p));
      PADDLE_ENFORCE(src_memory != nullptr,
                     "Fail to find src_memory in device context");
      // Retrieve pool_pd/pool_workspace_memory from device context
      auto pool_pd =
          std::static_pointer_cast<mkldnn::pooling_forward::primitive_desc>(
              dev_ctx.GetBlob(key_pool_pd));
      PADDLE_ENFORCE(pool_pd != nullptr,
                     "Fail to find pool_pd in device context");
      auto workspace_memory = std::static_pointer_cast<memory>(
          dev_ctx.GetBlob(key_pool_workspace_memory));
      PADDLE_ENFORCE(workspace_memory != nullptr,
                     "Fail to find workspace_memory in device context");

      // create memory descriptors for pooling
      auto diff_src_md = src_memory.get()->get_primitive_desc().desc();
      auto diff_dst_md = dst_memory.get()->get_primitive_desc().desc();

      auto pool_bwd_desc = mkldnn::pooling_backward::desc(
          pooling_type == "max" ? mkldnn::algorithm::pooling_max
                                : mkldnn::algorithm::pooling_avg,
          diff_src_md, diff_dst_md, strides, ksize, paddings, paddings,
          mkldnn::padding_kind::zero);
      auto pool_bwd_pd = mkldnn::pooling_backward::primitive_desc(
          pool_bwd_desc, mkldnn_engine, *pool_pd);

      // reorder between user_diff_dst and pool diff_dst if needed
      diff_dst_memory = std::make_shared<memory>(user_diff_dst_memory);
      if (memory::primitive_desc(dst_memory->get_primitive_desc()) !=
          user_diff_dst_memory.get_primitive_desc()) {
        diff_dst_memory =
            std::make_shared<memory>(dst_memory.get()->get_primitive_desc());
        reorder_diff_dst = reorder(user_diff_dst_memory, *diff_dst_memory);
        is_diff_dst_reordered = true;
      }

      diff_src_memory = std::make_shared<memory>(
          pool_bwd_pd.diff_src_primitive_desc(), in_x_grad_data);

      dev_ctx.SetBlob(key_pool_diff_src_mem_p, diff_src_memory);
      dev_ctx.SetBlob(key_pool_diff_dst_mem_p, diff_dst_memory);

      pool_bwd_p = std::make_shared<pooling_backward>(
          pool_bwd_pd, *diff_dst_memory, *workspace_memory, *diff_src_memory);
      dev_ctx.SetBlob(key_pool_bwd_p, pool_bwd_p);

    } else {
      // Primitives already exist
      diff_src_memory = std::static_pointer_cast<memory>(
          dev_ctx.GetBlob(key_pool_diff_src_mem_p));
      PADDLE_ENFORCE(diff_src_memory != nullptr,
                     "Fail to find pooling src mem_p in device context");
      diff_dst_memory = std::static_pointer_cast<memory>(
          dev_ctx.GetBlob(key_pool_diff_dst_mem_p));
      PADDLE_ENFORCE(diff_dst_memory != nullptr,
                     "Fail to find pooling dst mem_p in device context");

      diff_src_memory->set_data_handle(reinterpret_cast<void*>(in_x_grad_data));
      diff_dst_memory->set_data_handle(const_cast<T*>(out_grad_data));

      // reorder between user_diff_dst and pool diff_dst if needed
      if (memory::primitive_desc(dst_memory->get_primitive_desc()) !=
          user_diff_dst_memory.get_primitive_desc()) {
        diff_dst_memory =
            std::make_shared<memory>(dst_memory.get()->get_primitive_desc());
        reorder_diff_dst = reorder(user_diff_dst_memory, *diff_dst_memory);
        is_diff_dst_reordered = true;
      }
    }

    in_x_grad_format = (memory::format)diff_src_memory->get_primitive_desc()
                           .desc()
                           .data.format;

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline;
    if (is_diff_dst_reordered) {
      pipeline.push_back(reorder_diff_dst);
    }
    pipeline.push_back(*pool_bwd_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    in_x_grad->set_layout(DataLayout::kMKLDNN);
    in_x_grad->set_format(in_x_grad_format);
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(pool2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNOpKernel<float>,
                   ops::PoolMKLDNNOpKernel<int8_t>,
                   ops::PoolMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNGradOpKernel<float>);
