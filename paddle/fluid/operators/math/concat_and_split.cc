/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/concat_and_split.h"

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#endif
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#endif
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {}  // namespace framework
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    phi::funcs::ConcatFunctor<phi::CPUContext, T> functor;
    functor(context, input, axis, output);
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    phi::funcs::SplitFunctor<phi::CPUContext, T> functor;
    functor(context, input, ref_inputs, axis, outputs);
  }
};

#ifdef PADDLE_WITH_XPU
/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::XPUDeviceContext, T> {
 public:
  void operator()(const platform::XPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    int dev_id = context.GetPlace().GetDeviceId();
    platform::XPUDeviceGuard guard(dev_id);

    int num = input.size();
    auto input_dims = input[0].dims();

    std::vector<std::vector<int>> xdims_list(num);
    for (int i = 0; i < num; ++i) {
      std::vector<int> tmp_dims(input_dims.size());
      for (int j = 0; j < input_dims.size(); ++j) {
        tmp_dims[j] = input[i].dims()[j];
      }
      xdims_list[i] = tmp_dims;
    }

    std::vector<const T*> ptrs;
    for (int i = 0; i < num; ++i) {
      ptrs.push_back(input[i].data<T>());
    }

    auto r = xpu::concat<T>(context.x_context(), ptrs, output->data<T>(),
                            xdims_list, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r, XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class SplitFunctor<platform::XPUDeviceContext, T> {
 public:
  void operator()(const platform::XPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    int dev_id = context.GetPlace().GetDeviceId();
    platform::XPUDeviceGuard guard(dev_id);

    auto& ins = ref_inputs;

    int num = ins.size();
    auto input_dims = ins[0]->dims();
    std::vector<int> split_list(num);
    std::vector<int> xdims_list(input_dims.size());
    int total_length = 0;
    for (int i = 0; i < num; ++i) {
      split_list[i] = ins[i]->dims()[axis];
      total_length += ins[i]->dims()[axis];
    }

    for (int i = 0; i < input_dims.size(); ++i) {
      if (i == axis) continue;
      xdims_list[i] = input_dims[i];
    }
    xdims_list[axis] = total_length;

    std::vector<T*> ptrs(num);
    for (int i = 0; i < num; ++i) {
      ptrs[i] = outputs->at(i)->data<T>();
    }

    auto r = xpu::split<T>(context.x_context(), input.data<T>(), ptrs,
                           xdims_list, split_list, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r, XPUAPIErrorMsg[r]));
  }
};
#endif

#ifdef PADDLE_WITH_ASCEND_CL
template <typename T>
class ConcatFunctor<platform::NPUDeviceContext, T> {
 public:
  void operator()(const platform::NPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    int dev_id = context.GetPlace().GetDeviceId();
    platform::NPUDeviceGuard guard(dev_id);

    std::vector<std::string> names;
    for (size_t i = 0; i < input.size(); ++i) {
      names.push_back("x" + std::to_string(i));
    }
    NpuOpRunner runner{
        "ConcatD",
        {input},
        {*output},
        {{"concat_dim", axis}, {"N", static_cast<int>(input.size())}}};
    runner.AddInputNames(names);
    runner.Run(context.stream());
  }
};

template <typename T>
class SplitFunctor<platform::NPUDeviceContext, T> {
 public:
  void operator()(const platform::NPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    if (input.numel() == 0) {
      return;
    }

    size_t num = outputs->size();

    int input_rows = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      input_rows *= dim_0[i];
    }

    int input_cols = 0;

    std::vector<int64_t> output_cols(outputs->size());
    for (size_t i = 0; i < num; ++i) {
      int t_cols = ref_inputs[i]->numel() / input_rows;
      input_cols += t_cols;
      output_cols[i] = t_cols;
    }
    auto npu_place = context.GetPlace();

    // computation
    for (int k = 0; k < input_rows; ++k) {
      const T* src_ptr = input.data<T>() + k * input_cols;
      int col_idx = 0;
      for (size_t j = 0; j < num; ++j) {
        int col_len = output_cols[j];
        auto* out_tensor = outputs->at(j);
        if (out_tensor != nullptr) {
          T* dst_ptr = out_tensor->data<T>() + k * col_len;
          memory::Copy(npu_place, dst_ptr, npu_place, src_ptr + col_idx,
                       sizeof(T) * col_len, context.stream());
        }
        col_idx += col_len;
      }
    }
  }
};
#endif

#ifdef PADDLE_WITH_MLU
template <typename T>
class ConcatFunctor<platform::MLUDeviceContext, T> {
 public:
  void operator()(const platform::MLUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    int dev_id = context.GetPlace().GetDeviceId();
    platform::MLUDeviceGuard guard(dev_id);

    auto ins_size = input.size();

    const int axis_t = axis;
    const int ins_size_t = ins_size;

    // mlu should do sth
    // init ins tensors
    std::vector<const void*> inputs;
    std::vector<MLUCnnlTensorDesc> input_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    for (size_t i = 0; i < ins_size; i++) {
      input_descs.emplace_back(MLUCnnlTensorDesc(
          input[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(input[i].dtype())));
      desc_vector.push_back(input_descs.back().get());
      inputs.push_back(input[i].data());
    }
    // init out tensors
    MLUCnnlTensorDesc output_desc(*output, CNNL_LAYOUT_ARRAY,
                                  ToCnnlDataType(output->dtype()));

    // MLU should do sth
    MLUCnnl::Concat(context, ins_size_t, axis_t, desc_vector.data(),
                    inputs.data(), output_desc.get(), GetBasePtr(output));
  }
};

template <typename T>
class SplitFunctor<platform::MLUDeviceContext, T> {
 public:
  void operator()(const platform::MLUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    if (input.numel() == 0) {
      return;
    }

    int dev_id = context.GetPlace().GetDeviceId();
    platform::MLUDeviceGuard guard(dev_id);

    auto in_dims = input.dims();
    auto out_size = outputs->size();

    std::vector<framework::DDim> outs_dims(out_size, in_dims);
    for (size_t i = 0; i < out_size; ++i) {
      outs_dims[i][axis] = ref_inputs[i]->dims()[axis];
    }

    // init out tensors
    std::vector<void*> vct_tensor;
    std::vector<MLUCnnlTensorDesc> output_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    for (size_t i = 0; i < out_size; i++) {
      (*outputs)[i]->Resize(outs_dims[i]);
      output_descs.emplace_back(
          MLUCnnlTensorDesc(*(*outputs)[i], CNNL_LAYOUT_ARRAY,
                            ToCnnlDataType((*outputs)[i]->dtype())));
      desc_vector.push_back(output_descs.back().get());
      vct_tensor.push_back(GetBasePtr((*outputs)[i]));
    }
    // init in tensors
    MLUCnnlTensorDesc input_desc(input, CNNL_LAYOUT_ARRAY,
                                 ToCnnlDataType(input.dtype()));

    // MLU should do sth
    MLUCnnl::Split(context, out_size, axis, input_desc.get(), input.data(),
                   desc_vector.data(), vct_tensor.data());
  }
};
#endif

#define DEFINE_FUNCTOR(type)                                      \
  template class ConcatFunctor<platform::CPUDeviceContext, type>; \
  template class SplitFunctor<platform::CPUDeviceContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

#ifdef PADDLE_WITH_XPU
#define DEFINE_XPU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::XPUDeviceContext, type>; \
  template class SplitFunctor<platform::XPUDeviceContext, type>;

DEFINE_XPU_FUNCTOR(float)
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#define DEFINE_NPU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::NPUDeviceContext, type>; \
  template class SplitFunctor<platform::NPUDeviceContext, type>;

FOR_ALL_TYPES(DEFINE_NPU_FUNCTOR)
#endif

#ifdef PADDLE_WITH_MLU
#define DEFINE_MLU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::MLUDeviceContext, type>; \
  template class SplitFunctor<platform::MLUDeviceContext, type>;
DEFINE_MLU_FUNCTOR(float)
DEFINE_MLU_FUNCTOR(platform::float16)
DEFINE_MLU_FUNCTOR(int64_t)
DEFINE_MLU_FUNCTOR(bool)
DEFINE_MLU_FUNCTOR(int)
DEFINE_MLU_FUNCTOR(int8_t)
DEFINE_MLU_FUNCTOR(int16_t)
DEFINE_MLU_FUNCTOR(uint8_t)
#endif
}  // namespace math
}  // namespace operators
}  // namespace paddle
