/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/operators/mkldnn/axpy_handler.h"
#endif

namespace phi {
namespace funcs {
template <typename T>
struct SelectedRowsAdd<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input1,
                  const phi::SelectedRows& input2,
                  phi::SelectedRows* output) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(
        in1_height,
        input2.height(),
        phi::errors::InvalidArgument("The two inputs height must be equal."
                                     "But received first input height  = "
                                     "[%d], second input height = [%d]",
                                     in1_height,
                                     input2.height()));
    output->set_height(in1_height);

    auto& in1_rows = input1.rows();
    auto& in2_rows = input2.rows();
    std::vector<int64_t> out_rows;
    out_rows.reserve(in1_rows.size() + in2_rows.size());

    // concat rows
    out_rows.insert(out_rows.end(), in1_rows.begin(), in1_rows.end());
    out_rows.insert(out_rows.end(), in2_rows.begin(), in2_rows.end());
    output->set_rows(out_rows);

    auto* out_value = output->mutable_value();
    auto& in1_value = input1.value();
    auto& in2_value = input2.value();

    auto in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        in2_value.numel() / in2_rows.size(),
        phi::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But received first input width = [%d], second input width = [%d]",
            in1_row_numel,
            in2_value.numel() / in2_rows.size()));
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        out_value->numel() / out_rows.size(),
        phi::errors::InvalidArgument(
            "The input and oupput width must be equal."
            "But received input width = [%d], output width = [%d]",
            in1_row_numel,
            out_value->numel() / out_rows.size()));

    auto in1_place = input1.place();
    PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(in1_place),
                      true,
                      phi::errors::InvalidArgument(
                          "The running environment is not on the CPU place."));
    auto in2_place = input2.place();
    PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(in2_place),
                      true,
                      phi::errors::InvalidArgument(
                          "The running environment is not on the CPU place."));
    auto out_place = context.GetPlace();
    PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(out_place),
                      true,
                      phi::errors::InvalidArgument(
                          "The running environment is not on the CPU place."));

    auto* out_data = out_value->data<T>();
    auto* in1_data = in1_value.data<T>();
    paddle::memory::Copy(out_place,
                         out_data,
                         in1_place,
                         in1_data,
                         in1_value.numel() * sizeof(T));

    auto* in2_data = in2_value.data<T>();
    paddle::memory::Copy(out_place,
                         out_data + in1_value.numel(),
                         in2_place,
                         in2_data,
                         in2_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAdd<phi::CPUContext, float>;
template struct SelectedRowsAdd<phi::CPUContext, double>;

template <typename T>
struct SelectedRowsAddTensor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input1,
                  const phi::DenseTensor& input2,
                  phi::DenseTensor* output) {
    auto in1_height = input1.height();
    const auto& in2_dims = input2.dims();
    const auto& out_dims = output->dims();
    PADDLE_ENFORCE_EQ(
        in1_height,
        in2_dims[0],
        phi::errors::InvalidArgument("The two inputs height must be equal."
                                     "But received first input height = "
                                     "[%d], second input height = [%d]",
                                     in1_height,
                                     in2_dims[0]));
    PADDLE_ENFORCE_EQ(
        in1_height,
        out_dims[0],
        phi::errors::InvalidArgument(
            "The input and output height must be equal."
            "But received input height = [%d], output height = [%d]",
            in1_height,
            out_dims[0]));

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        input2.numel() / in1_height,
        phi::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But received first input width = [%d], second input width = [%d]",
            in1_row_numel,
            input2.numel() / in1_height));
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        output->numel() / in1_height,
        phi::errors::InvalidArgument(
            "The input and output width must be equal."
            "But received input width = [%d], output width = [%d]",
            in1_row_numel,
            output->numel() / in1_height));

    phi::funcs::SetConstant<phi::CPUContext, T> functor;
    functor(context, output, 0.0);

    auto* in1_data = in1_value.data<T>();
    auto* out_data = output->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        out_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }

    auto out_eigen = EigenVector<T>::Flatten(*output);
    auto in2_eigen = EigenVector<T>::Flatten(input2);
    out_eigen.device(*context.eigen_device()) = out_eigen + in2_eigen;
  }
};

template struct SelectedRowsAddTensor<phi::CPUContext, float>;
template struct SelectedRowsAddTensor<phi::CPUContext, double>;

template <typename T>
struct SelectedRowsAddTo<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input1,
                  const int64_t input2_offset,
                  phi::SelectedRows* input2) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(
        in1_height,
        input2->height(),
        phi::errors::InvalidArgument("The two inputs height must be equal."
                                     "But received first input height = "
                                     "[%d], second input height = [%d]",
                                     in1_height,
                                     input2->height()));

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    paddle::framework::MixVector<int64_t> mixv_in2_rows(&in2_rows);
    mixv_in2_rows.Extend(in1_rows.begin(), in1_rows.end());

    auto in1_place = input1.place();
    PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(in1_place),
                      true,
                      phi::errors::InvalidArgument(
                          "The running environment is not on the CPU place."));
    auto in2_place = input2->place();
    PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(in2_place),
                      true,
                      phi::errors::InvalidArgument(
                          "The running environment is not on the CPU place."));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->data<T>();
    paddle::memory::Copy(in2_place,
                         in2_data + input2_offset,
                         in1_place,
                         in1_data,
                         in1_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAddTo<phi::CPUContext, float>;
template struct SelectedRowsAddTo<phi::CPUContext, double>;
template struct SelectedRowsAddTo<phi::CPUContext, int>;
template struct SelectedRowsAddTo<phi::CPUContext, int64_t>;

template <typename T>
struct SelectedRowsSumTo<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const std::vector<phi::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  phi::SelectedRows* input2) {
    // Ensure all selected rows have the same height
    size_t size = 0u;
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      auto& in_rows = (*iter)->rows();
      size += in_rows.end() - in_rows.begin();
      auto in1_height = (*iter)->height();
      PADDLE_ENFORCE_EQ(in1_height,
                        input2->height(),
                        phi::errors::InvalidArgument(
                            "The two inputs height must be equal."
                            "But received first input height = [%d], second "
                            "input height = [%d]",
                            in1_height,
                            input2->height()));
    }
    // concat rows
    std::vector<int64_t> in2_rows;
    in2_rows.reserve(in2_rows.size() + size);
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      const paddle::framework::Vector<int64_t>& in_rows = (*iter)->rows();
      in2_rows.insert(in2_rows.end(), in_rows.begin(), in_rows.end());
    }
    input2->set_rows(in2_rows);

    auto* in2_value = input2->mutable_value();
    auto* in2_data = in2_value->data<T>();
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(context);
    size_t offset = 0u;
    for (size_t i = 0u; i != input1.size(); ++i) {
      auto& in_value = input1[i]->value();
      const auto* in_data = in_value.data<T>();
      offset += input2_offsets[i];
      blas.VCOPY(in_value.numel(), in_data, in2_data + offset);
    }
  }
};

template struct SelectedRowsSumTo<phi::CPUContext, float>;
template struct SelectedRowsSumTo<phi::CPUContext, double>;

template <typename T>
struct SelectedRowsAddToTensor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input1,
                  phi::DenseTensor* input2) {
    if (UNLIKELY(input1.rows().size() == 0)) {
      LOG(WARNING) << "input selected rows is empty!";
      return;
    }
    auto in1_height = input1.height();
    const auto& in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(
        in1_height,
        in2_dims[0],
        phi::errors::InvalidArgument("The two inputs height must be equal."
                                     "But received first input height = "
                                     "[%d], second input height = [%d]",
                                     in1_height,
                                     in2_dims[0]));

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        input2->numel() / in1_height,
        phi::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But received first input width = [%d], second input width = [%d]",
            in1_row_numel,
            input2->numel() / in1_height));

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }
  }
};

template struct SelectedRowsAddToTensor<phi::CPUContext, float>;
template struct SelectedRowsAddToTensor<phi::CPUContext, double>;
template struct SelectedRowsAddToTensor<phi::CPUContext, int>;
template struct SelectedRowsAddToTensor<phi::CPUContext, int64_t>;
template struct SelectedRowsAddToTensor<phi::CPUContext, phi::dtype::bfloat16>;
// This is a separated namespace for manipulate SelectedRows typed
// data. Like merge duplicated rows, adding two SelectedRows etc.
//
// Another group of functors is called "scatter updates", which means
// use SelectedRows to update a dense tensor with different Ops, like
// add or mul.
namespace scatter {

template <typename T, typename DeviceContext>
typename std::enable_if<!std::is_integral<T>::value>::type elementwise_add_to(
    phi::funcs::BlasT<DeviceContext, T>* blas,
    size_t data_len,
    const T* in,
    T* out) {
  blas->AXPY(data_len, T(1.f), in, out);
}

template <typename T, typename DeviceContext>
typename std::enable_if<std::is_integral<T>::value>::type elementwise_add_to(
    phi::funcs::BlasT<DeviceContext, T>* blas,
    size_t data_len,
    const T* in,
    T* out) {
  for (size_t i = 0; i < data_len; i++) {
    out[i] += in[i];
  }
}

template <typename T, typename DeviceContext>
typename std::enable_if<std::is_same<T, phi::dtype::bfloat16>::value>::type
add_sparse_inputs(const std::vector<const phi::SelectedRows*>& inputs,
                  const std::unordered_map<int64_t, size_t>& rows_to_id,
                  int64_t input_width,
                  const DeviceContext& context,
                  T* out_data) {
#ifndef PADDLE_WITH_MKLDNN
  auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
#endif
  for (auto* input : inputs) {
    if (input->rows().size() == 0) {
      continue;
    }
    auto* input_data = input->value().data<T>();
    auto& input_rows = input->rows();

#ifdef PADDLE_WITH_MKLDNN
    paddle::operators::OneDNNAXPYHandler<T> axpy_handler(input_width, T(1.f));
    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_to_id.at(input_rows[i]);
      axpy_handler(&input_data[i * input_width],
                   &out_data[out_i * input_width]);
    }
#else
    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_to_id.at(input_rows[i]);
      elementwise_add_to<T, DeviceContext>(&blas,
                                           static_cast<size_t>(input_width),
                                           &input_data[i * input_width],
                                           &out_data[out_i * input_width]);
    }
#endif
  }
}

template <typename T, typename DeviceContext>
typename std::enable_if<!std::is_same<T, phi::dtype::bfloat16>::value>::type
add_sparse_inputs(const std::vector<const phi::SelectedRows*>& inputs,
                  const std::unordered_map<int64_t, size_t>& rows_to_id,
                  int64_t input_width,
                  const DeviceContext& context,
                  T* out_data) {
  VLOG(4) << "[CPU] add_sparse_inputs <" << typeid(T).name();
  auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
  for (auto* input : inputs) {
    if (input->rows().size() == 0) {
      continue;
    }
    auto* input_data = input->value().data<T>();
    auto& input_rows = input->rows();

    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_to_id.at(input_rows[i]);
      elementwise_add_to<T, DeviceContext>(&blas,
                                           static_cast<size_t>(input_width),
                                           &input_data[i * input_width],
                                           &out_data[out_i * input_width]);
    }
  }
}

template <typename DeviceContext, typename T>
struct MergeAddImpl {
  phi::SelectedRows operator()(const DeviceContext& context,
                               const phi::SelectedRows& input,
                               const bool sorted_result = false) {
    phi::SelectedRows out;
    (*this)(context, input, &out, sorted_result);
    return out;
  }

  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output,
                  const bool sorted_result = false) {
    std::vector<const phi::SelectedRows*> inputs;
    inputs.push_back(&input);
    (*this)(context, inputs, output, sorted_result);
  }

  void operator()(const DeviceContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output,
                  const bool sorted_result = false) {
    if (inputs.size() == 0) {
      VLOG(3) << "no input! return";
      return;
    }
    const phi::SelectedRows* has_value_input = nullptr;
    for (auto* in : inputs) {
      if (in->rows().size() > 0) {
        has_value_input = in;
        break;
      }
    }
    if (has_value_input == nullptr) {
      VLOG(3) << "no input has value! just return" << std::endl;
      return;
    }
    auto input_width = has_value_input->value().dims()[1];
    auto input_height = has_value_input->height();
    phi::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    size_t row_num = 0;
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          input_width,
          input->value().dims()[1],
          phi::errors::InvalidArgument("All inputs should have same "
                                       "dimension except for the first one."));
      PADDLE_ENFORCE_EQ(
          input_height,
          input->height(),
          phi::errors::InvalidArgument("All inputs should have same height."));
      row_num += input->rows().size();
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }

    out.set_height(input_height);
    out.mutable_value()->mutable_data<T>(
        phi::make_ddim(
            {static_cast<int64_t>(merged_row_set.size()), input_width}),
        context.GetPlace());
    auto* out_data = out.mutable_value()->data<T>();

    if (merged_row_set.size() == row_num && !sorted_result) {
      // no duplicated ids, just concat the result together
      std::vector<int64_t> merge_rows;
      merge_rows.reserve(row_num);
      // concat rows
      for (auto* in : inputs) {
        merge_rows.insert(
            merge_rows.end(), in->rows().begin(), in->rows().end());
      }
      out.set_rows(merge_rows);
      auto in_place = inputs[0]->place();
      auto out_place = out.place();
      int64_t copied_numel = 0;
      for (auto* in : inputs) {
        auto* in_data = in->value().data<T>();
        auto in_numel = in->rows().size() * input_width;
        paddle::memory::Copy(out_place,
                             out_data + copied_numel,
                             in_place,
                             in_data,
                             in_numel * sizeof(T));
        copied_numel += in_numel;
      }
    } else {
      std::vector<int64_t> merge_rows(merged_row_set.begin(),
                                      merged_row_set.end());

      if (sorted_result) {
        std::sort(merge_rows.begin(), merge_rows.end());
      }

      out.set_rows(merge_rows);

      phi::funcs::SetConstant<DeviceContext, T> constant_functor;
      constant_functor(context, out.mutable_value(), static_cast<T>(0.f));

      std::unordered_map<int64_t, size_t> rows_to_id;
      for (size_t i = 0; i < merge_rows.size(); ++i) {
        rows_to_id[merge_rows[i]] = i;
      }

      add_sparse_inputs<T, DeviceContext>(
          inputs, rows_to_id, input_width, context, out_data);
    }
  }
};

template <typename T>
struct MergeAdd<phi::CPUContext, T> {
  // unary functor, merge by adding duplicated rows in
  // the input SelectedRows object.
  phi::SelectedRows operator()(const phi::CPUContext& context,
                               const phi::SelectedRows& input,
                               const bool sorted_result) {
    return MergeAddImpl<phi::CPUContext, T>()(context, input, sorted_result);
  }

  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output,
                  const bool sorted_result) {
    MergeAddImpl<phi::CPUContext, T>()(context, input, output, sorted_result);
  }

  void operator()(const phi::CPUContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output,
                  const bool sorted_result) {
    MergeAddImpl<phi::CPUContext, T>()(context, inputs, output, sorted_result);
  }
};

#define TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(dtype)    \
  template struct MergeAddImpl<phi::CPUContext, dtype>; \
  template struct MergeAdd<phi::CPUContext, dtype>;

TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(float)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(double)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(int)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(int64_t)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(phi::dtype::bfloat16)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(phi::dtype::complex<float>)
TEMPLATE_SPECIALIZED_FOR_MERGEADD_CPU(phi::dtype::complex<double>)

#ifdef PADDLE_WITH_XPU
template <typename T>
struct MergeAdd<phi::XPUContext, T> {
  phi::SelectedRows operator()(const phi::XPUContext& context,
                               const phi::SelectedRows& input,
                               const bool sorted_result = false) {
    phi::SelectedRows out;
    (*this)(context, input, &out, sorted_result);
    return out;
  }

  void operator()(const phi::XPUContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output,
                  const bool sorted_result = false) {
    paddle::framework::Vector<int64_t> input_rows(input.rows());
    if (input_rows.size() == 0) {
      return;
    }

    phi::SelectedRows& out = *output;
    std::set<int64_t> row_set(input_rows.begin(), input_rows.end());
    std::vector<int64_t> merge_rows(row_set.begin(), row_set.end());
    auto input_width = input.value().dims()[1];

    out.set_rows(merge_rows);
    out.set_height(input.height());
    out.mutable_value()->mutable_data<T>(
        phi::make_ddim({static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    std::unordered_map<int64_t, size_t> rows_to_id;
    for (size_t i = 0; i < merge_rows.size(); ++i) {
      rows_to_id[merge_rows[i]] = i;
    }

    auto* y_data = out.mutable_value()->data<T>();
    auto* x_data = input.value().data<T>();
    int xm = input_rows.size();
    int ym = merge_rows.size();
    int n = input_width;

    xpu::ctx_guard RAII_GUARD(context.x_context());
    int64_t* x_rows_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(xm);
    int64_t* y_rows_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(ym);
    paddle::memory::Copy(context.GetPlace(),
                         y_rows_data,
                         phi::CPUPlace(),
                         merge_rows.data(),
                         ym * sizeof(int64_t));
    paddle::memory::Copy(context.GetPlace(),
                         x_rows_data,
                         phi::CPUPlace(),
                         input_rows.data(),
                         xm * sizeof(int64_t));
    int r = xpu::merge_dup_rows<T, int64_t>(context.x_context(),
                                            x_data,
                                            y_data,
                                            x_rows_data,
                                            y_rows_data,
                                            xm,
                                            n,
                                            ym);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "merge_dup_rows");
  }

  void operator()(const phi::XPUContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output,
                  const bool sorted_result = false) {
    if (inputs.size() == 0) {
      VLOG(3) << "no input! return";
      return;
    }
    const phi::SelectedRows* has_value_input = nullptr;
    for (auto* in : inputs) {
      if (in->rows().size() > 0) {
        has_value_input = in;
        break;
      }
    }
    if (has_value_input == nullptr) {
      VLOG(3) << "no input has value! just return" << std::endl;
      return;
    }
    auto input_width = has_value_input->value().dims()[1];
    auto input_height = has_value_input->height();
    phi::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    size_t row_num = 0;
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          input_width,
          input->value().dims()[1],
          phi::errors::InvalidArgument("All inputs should have same "
                                       "dimension except for the first one."));
      PADDLE_ENFORCE_EQ(
          input_height,
          input->height(),
          phi::errors::InvalidArgument("All inputs should have same height."));
      row_num += input->rows().size();
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }

    std::vector<int64_t> merge_rows(merged_row_set.begin(),
                                    merged_row_set.end());

    if (sorted_result) {
      std::sort(merge_rows.begin(), merge_rows.end());
    }

    out.set_rows(merge_rows);
    out.set_height(input_height);
    out.mutable_value()->mutable_data<T>(
        phi::make_ddim(
            {static_cast<int64_t>(merged_row_set.size()), input_width}),
        context.GetPlace());

    float* y_data = reinterpret_cast<float*>(out.mutable_value()->data<T>());

    std::unordered_map<int64_t, size_t> rows_to_id;
    for (size_t i = 0; i < merge_rows.size(); ++i) {
      rows_to_id[merge_rows[i]] = i;
    }

    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      auto& input_rows = input->rows();

      auto* x_data = input->value().data<T>();
      int xm = input_rows.size();
      int ym = merge_rows.size();
      int n = input_width;

      xpu::ctx_guard RAII_GUARD(context.x_context());
      int64_t* x_rows_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(xm);
      int64_t* y_rows_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(ym);
      paddle::memory::Copy(context.GetPlace(),
                           y_rows_data,
                           phi::CPUPlace(),
                           merge_rows.data(),
                           ym * sizeof(int64_t));
      paddle::memory::Copy(context.GetPlace(),
                           x_rows_data,
                           phi::CPUPlace(),
                           input_rows.data(),
                           xm * sizeof(int64_t));
      int r = xpu::merge_dup_rows<T, int64_t>(context.x_context(),
                                              x_data,
                                              y_data,
                                              x_rows_data,
                                              y_rows_data,
                                              xm,
                                              n,
                                              ym);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "merge_dup_rows");
    }
  }
};

#endif
template <typename T>
struct MergeAverage<phi::CPUContext, T> {
  phi::SelectedRows operator()(const phi::CPUContext& context,
                               const phi::SelectedRows& input) {
    phi::SelectedRows out;
    (*this)(context, input, &out);
    return out;
  }

  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output) {
    std::vector<const phi::SelectedRows*> inputs;
    inputs.push_back(&input);
    (*this)(context, inputs, output);
  }

  void operator()(const phi::CPUContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output) {
    if (inputs.size() == 0) {
      VLOG(3) << "no input! return";
      return;
    }
    const phi::SelectedRows* has_value_input = nullptr;
    for (auto* in : inputs) {
      if (in->rows().size() > 0) {
        has_value_input = in;
        break;
      }
    }
    if (has_value_input == nullptr) {
      VLOG(3) << "no input has value! just return" << std::endl;
      return;
    }
    auto input_width = has_value_input->value().dims()[1];
    auto input_height = has_value_input->height();
    phi::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    size_t row_num = 0;
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          input_width,
          input->value().dims()[1],
          phi::errors::InvalidArgument("All inputs should have same "
                                       "dimension except for the first one."));
      PADDLE_ENFORCE_EQ(
          input_height,
          input->height(),
          phi::errors::InvalidArgument("All input should have same height."));
      row_num += input->rows().size();
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }

    out.set_height(input_height);
    out.mutable_value()->mutable_data<T>(
        phi::make_ddim(
            {static_cast<int64_t>(merged_row_set.size()), input_width}),
        context.GetPlace());
    auto* out_data = out.mutable_value()->data<T>();

    std::vector<int64_t> merge_rows(merged_row_set.begin(),
                                    merged_row_set.end());
    std::sort(merge_rows.begin(), merge_rows.end());

    out.set_rows(merge_rows);

    phi::funcs::SetConstant<phi::CPUContext, T> constant_functor;
    constant_functor(context, out.mutable_value(), 0.0);

    std::unordered_map<int64_t, size_t> rows_to_id;
    for (size_t i = 0; i < merge_rows.size(); ++i) {
      rows_to_id[merge_rows[i]] = i;
    }

    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(context);
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      auto* input_data = input->value().data<T>();
      auto& input_rows = input->rows();

      for (size_t i = 0; i < input_rows.size(); i++) {
        size_t out_i = rows_to_id[input_rows[i]];
        elementwise_add_to<T>(&blas,
                              static_cast<size_t>(input_width),
                              &input_data[i * input_width],
                              &out_data[out_i * input_width]);
      }
    }
    size_t input_width_cast = static_cast<size_t>(input_width);
    T count = static_cast<T>(inputs.size());
    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (size_t j = 0; j < input_width_cast; j++) {
        out_data[i * input_width + j] = out_data[i * input_width + j] / count;
      }
    }
  }
};

#ifdef PADDLE_WITH_XPU
template struct MergeAdd<phi::XPUContext, float>;
#endif

template struct MergeAverage<phi::CPUContext, int>;
template struct MergeAverage<phi::CPUContext, int64_t>;
template struct MergeAverage<phi::CPUContext, float>;
template struct MergeAverage<phi::CPUContext, double>;

template <typename T>
struct UpdateToTensor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const ScatterOps& op,
                  const phi::SelectedRows& input1,
                  phi::DenseTensor* input2) {
    auto in1_height = input1.height();
    const auto& in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(
        in1_height,
        in2_dims[0],
        phi::errors::InvalidArgument("The two inputs height must be equal."
                                     "But received first input height = "
                                     "[%d], second input height = [%d]",
                                     in1_height,
                                     in2_dims[0]));

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel,
        input2->numel() / in1_height,
        phi::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But received first input width = [%d], second input width = [%d]",
            in1_row_numel,
            input2->numel() / in1_height));

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->data<T>();

    // FIXME(typhoonzero): use macro fix the below messy code.
    switch (op) {
      case ScatterOps::ASSIGN:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::ADD:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUB:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] -=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUBBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] -
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
      case ScatterOps::MUL:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] *=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIV:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] /=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIVBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] /
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
    }
  }
};

}  // namespace scatter
}  // namespace funcs
}  // namespace phi
