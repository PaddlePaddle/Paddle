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

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"

namespace phi {
namespace funcs {

using XPUDeviceGuard = phi::backends::xpu::XPUDeviceGuard;

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<XPUContext, T> {
 public:
  void operator()(const XPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int dev_id = context.GetPlace().GetDeviceId();
    XPUDeviceGuard guard(dev_id);

    int num = input.size();
    auto input_dims = input[0].dims();
    // special for 0-dim shape
    if (input_dims.size() == 0) {
      input_dims = {1};
    }

    std::vector<std::vector<int>> xdims_list(num);
    for (int i = 0; i < num; ++i) {
      std::vector<int> tmp_dims(input_dims.size());
      for (int j = 0; j < input_dims.size(); ++j) {
        auto ins_i_dims = input[i].dims();
        // special for 0-dim shape
        if (ins_i_dims.size() == 0) {
          ins_i_dims = {1};
        }
        tmp_dims[j] = ins_i_dims[j];
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
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat");
  }
};

template <typename T>
class SplitFunctor<XPUContext, T> {
 public:
  void operator()(const XPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int dev_id = context.GetPlace().GetDeviceId();
    XPUDeviceGuard guard(dev_id);

    auto& ins = ref_inputs;

    int num = ins.size();
    auto input_dims = ins[0]->dims();
    // special for 0-dim shape
    if (input_dims.size() == 0) {
      input_dims = {1};
    }
    std::vector<int> split_list(num);
    std::vector<int> xdims_list(input_dims.size());
    int total_length = 0;
    for (int i = 0; i < num; ++i) {
      auto ins_i_dims = ins[i]->dims();
      // special for 0-dim shape
      if (ins_i_dims.size() == 0) {
        ins_i_dims = {1};
      }
      split_list[i] = ins_i_dims[axis];
      total_length += ins_i_dims[axis];
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

    // int split(Context* ctx, const T* x, const std::vector<T*>& y_list, const
    // std::vector<int64_t>& xshape, const std::vector<int64_t>& split_list,
    // int64_t axis);
    auto r = xpu::split<XPUType>(
        context.x_context(),
        reinterpret_cast<const XPUType*>(tmp_data.data<T>()),
        ptrs,
        xdims_list,
        split_list,
        axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "split");
  }
};

#define DEFINE_XPU_FUNCTOR(type)                  \
  template class ConcatFunctor<XPUContext, type>; \
  template class SplitFunctor<XPUContext, type>;

DEFINE_XPU_FUNCTOR(float)
DEFINE_XPU_FUNCTOR(phi::dtype::float16)
DEFINE_XPU_FUNCTOR(int32_t)
DEFINE_XPU_FUNCTOR(int64_t)
DEFINE_XPU_FUNCTOR(uint8_t)

}  // namespace funcs
}  // namespace phi
