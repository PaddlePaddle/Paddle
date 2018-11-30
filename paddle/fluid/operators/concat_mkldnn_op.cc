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

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::concat;
using mkldnn::stream;
using platform::to_void_cast;

// Generate keys for storing/retriving primitives for this operator
// TODO(jczaja): Make hashing function more optimial
static std::string gethash(const memory::dims& input_dims,
                           const std::string& pooling_type,
                           const std::vector<int>& ksize,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& suffix) {
  auto dims2str = [](const memory::dims& operand_dims) {
    std::string dstr = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      dstr += std::to_string(operand_dims[i]) + "-";
    }
    return dstr;
  };
  return dims2str(input_dims) + dims2str(ksize) + dims2str(strides) +
         dims2str(paddings) + pooling_type + suffix;
}

static void EnforceLayouts(const std::vector<const Tensor*> inputs) {
  for (auto* input : inputs) {
    const bool is_layout_correct = input->layout() == DataLayout::kMKLDNN;
    const bool is_format_defined = input->format() !=
                                   memory::format::format_undef;
    PADDLE_ENFORCE(is_layout_correct && is_format_defined,
                   "Wrong layout/format set for Input tensor");
  }
}

static memory::primitive_desc CreateMemPrimDesc(
    const framework::Tensor& input, const mkldnn::engine& engine) {
  constexpr auto data_type = mkldnn::memory::f32;
  const auto dims = paddle::framework::vectorize2int(input.dims());
  const auto format = input.format();
  auto description = memory::desc(dims, data_type, format);
  auto mem_prim_desc = memory::primitive_desc(description, engine);
  return mem_prim_desc;
}

static platform::CPUPlace GetCpuPlace(
    const paddle::framework::ExecutionContext& ctx) {
  auto place = ctx.GetPlace();
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(place),
                 "It must use CPUPlace.");
  return boost::get<platform::CPUPlace>(place);
}

template <typename T>
class ConcatMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto place = GetCpuPlace(ctx);
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto multi_input = ctx.MultiInput<framework::Tensor>("X");
    framework::Tensor* output = ctx.Output<framework::Tensor>("Out");
    int64_t concat_axis = static_cast<int64_t>(ctx.Attr<int>("axis"));

    EnforceLayouts(multi_input);

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<memory> srcs;
    for (size_t i = 0; i < multi_input.size(); i++) {
        auto mem_prim_desc = CreateMemPrimDesc(*multi_input[i], mkldnn_engine);
        srcs_pd.push_back(mem_prim_desc);
        srcs.push_back(memory(mem_prim_desc, to_void_cast(multi_input[i]->data<T>())));
    }
    auto dst_dims = paddle::framework::vectorize2int(output->dims());
    auto dst_desc = memory::desc(dst_dims, mkldnn::memory::f32, memory::format::any);
    auto concat_pd = concat::primitive_desc(dst_desc, static_cast<int>(concat_axis), srcs_pd);
    auto dst_mem = memory(concat_pd.dst_primitive_desc(), output->mutable_data<T>(place));

    std::vector<primitive::at> inputs; //= {srcs};
    inputs.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); i++) {
      inputs.push_back(srcs[i]);
    }
    auto concat_prim = concat(concat_pd, inputs, dst_mem);

    std::vector<primitive> pipeline;
    pipeline.push_back(concat_prim);
    stream(stream::kind::eager).submit(pipeline).wait(); // TODO(mgallus): When this is not workin' split into decl and def

    /*
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    auto input_format = input->format();
    memory::format output_format{memory::format::format_undef};

    const std::string key = gethash(src_tz, pooling_type, ksize, strides,
                                    paddings, ctx.op().Output("Out"));
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
      auto src_md = platform::MKLDNNMemDesc(
          src_tz, platform::MKLDNNGetDataType<T>(), input_format);

      auto dst_md = platform::MKLDNNMemDesc(dst_tz, mkldnn::memory::f32,
                                            mkldnn::memory::format::any);

      std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd =
          CreatePrimitiveDesc(src_md, dst_md, strides, padding_left_top,
                              padding_right_bottom, ksize, pooling_type,
                              mkldnn_engine, ceil_mode, is_test);

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
    std::vector<mkldnn::primitive> pipeline{*(pool_p.get())};
    stream(stream::kind::eager).submit(pipeline).wait();
    */
    output->mutable_data(place);
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format((memory::format)dst_mem.get_primitive_desc().desc()
                                              .data.format);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(concat, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatMKLDNNOpKernel<float>)
