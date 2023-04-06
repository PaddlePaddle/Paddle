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
#include "paddle/fluid/platform/device_context.h"

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#endif
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    phi::funcs::ConcatFunctor<phi::CPUContext, T> functor;
    functor(context, input, axis, output);
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
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
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    using XPUType = typename XPUTypeTrait<T>::Type;
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

    std::vector<const XPUType*> ptrs;
    for (int i = 0; i < num; ++i) {
      if (input[i].place() != context.GetPlace()) {
        // data not on xpu, probably on cpu. move it now
        phi::DenseTensor tmp_data = input[i];
        context.template Alloc<T>(&tmp_data);
        ptrs.push_back(reinterpret_cast<const XPUType*>(tmp_data.data<T>()));
      } else {
        ptrs.push_back(reinterpret_cast<const XPUType*>(input[i].data<T>()));
      }
    }
    context.template Alloc<T>(output);

    auto r = xpu::concat<XPUType>(context.x_context(),
                                  ptrs,
                                  reinterpret_cast<XPUType*>(output->data<T>()),
                                  xdims_list,
                                  axis);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r,
            XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class SplitFunctor<platform::XPUDeviceContext, T> {
 public:
  void operator()(const platform::XPUDeviceContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    using XPUType = typename XPUTypeTrait<T>::Type;
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

    std::vector<XPUType*> ptrs(num);
    for (int i = 0; i < num; ++i) {
      context.template Alloc<T>(outputs->at(i));
      ptrs[i] = reinterpret_cast<XPUType*>(outputs->at(i)->data<T>());
    }
    phi::DenseTensor tmp_data = input;
    if (input.place() != context.GetPlace()) {
      // data not on xpu, probably on cpu. move it now
      context.template Alloc<T>(&tmp_data);
    }

    auto r = xpu::split<XPUType>(
        context.x_context(),
        reinterpret_cast<const XPUType*>(tmp_data.data<T>()),
        ptrs,
        xdims_list,
        split_list,
        axis);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r,
            XPUAPIErrorMsg[r]));
  }
};
#endif

#ifdef PADDLE_WITH_MLU
template <typename T>
class ConcatFunctor<platform::MLUDeviceContext, T> {
 public:
  void operator()(const platform::MLUDeviceContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
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
    MLUCnnlTensorDesc output_desc(
        *output, CNNL_LAYOUT_ARRAY, ToCnnlDataType(output->dtype()));

    // MLU should do sth
    MLUCnnl::Concat(context,
                    ins_size_t,
                    axis_t,
                    desc_vector.data(),
                    inputs.data(),
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

template <typename T>
class SplitFunctor<platform::MLUDeviceContext, T> {
 public:
  void operator()(const platform::MLUDeviceContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
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
          MLUCnnlTensorDesc(*(*outputs)[i],
                            CNNL_LAYOUT_ARRAY,
                            ToCnnlDataType((*outputs)[i]->dtype())));
      desc_vector.push_back(output_descs.back().get());
      vct_tensor.push_back(GetBasePtr((*outputs)[i]));
    }
    // init in tensors
    MLUCnnlTensorDesc input_desc(
        input, CNNL_LAYOUT_ARRAY, ToCnnlDataType(input.dtype()));

    // MLU should do sth
    MLUCnnl::Split(context,
                   out_size,
                   axis,
                   input_desc.get(),
                   input.data(),
                   desc_vector.data(),
                   vct_tensor.data());
  }
};
#endif

#define DEFINE_FUNCTOR(type)                           \
  template class ConcatFunctor<phi::CPUContext, type>; \
  template class SplitFunctor<phi::CPUContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

#ifdef PADDLE_WITH_XPU
#define DEFINE_XPU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::XPUDeviceContext, type>; \
  template class SplitFunctor<platform::XPUDeviceContext, type>;

DEFINE_XPU_FUNCTOR(float)
DEFINE_XPU_FUNCTOR(platform::float16)
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
