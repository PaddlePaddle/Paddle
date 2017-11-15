/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/selected_rows_functor.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
struct SelectedRowsAdd<platform::CPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(in1_height, input2.height());
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
    PADDLE_ENFORCE_EQ(in1_row_numel, in2_value.numel() / in2_rows.size());
    PADDLE_ENFORCE_EQ(in1_row_numel, out_value->numel() / out_rows.size());

    auto in1_place = input1.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in1_place));
    auto in2_place = input2.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in2_place));
    auto out_place = context.GetPlace();
    PADDLE_ENFORCE(platform::is_cpu_place(out_place));

    auto* out_data = out_value->data<T>();
    auto* in1_data = in1_value.data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(out_place), out_data,
                 boost::get<platform::CPUPlace>(in1_place), in1_data,
                 in1_value.numel() * sizeof(T));

    auto* in2_data = in2_value.data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(out_place),
                 out_data + in1_value.numel(),
                 boost::get<platform::CPUPlace>(in2_place), in2_data,
                 in2_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAdd<platform::CPUPlace, float>;
template struct SelectedRowsAdd<platform::CPUPlace, double>;

template <typename T>
struct SelectedRowsAddTensor<platform::CPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output) {
    auto in1_height = input1.height();
    auto in2_dims = input2.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);
    PADDLE_ENFORCE_EQ(in1_height, out_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2.numel() / in1_height);
    PADDLE_ENFORCE_EQ(in1_row_numel, output->numel() / in1_height);

    SetConstant<platform::CPUPlace, T> functor;
    functor(context, output, 0.0);

    auto* in1_data = in1_value.data<T>();
    auto* out_data = output->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        out_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }

    auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    auto in2_eigen = framework::EigenVector<T>::Flatten(input2);
    out_eigen.device(*context.GetEigenDevice<platform::CPUPlace>()) =
        out_eigen + in2_eigen;
  }
};

template struct SelectedRowsAddTensor<platform::CPUPlace, float>;
template struct SelectedRowsAddTensor<platform::CPUPlace, double>;

template <typename T>
struct SelectedRowsAddTo<platform::CPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const int64_t input2_offset,
                  framework::SelectedRows* input2) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(in1_height, input2->height());

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    in2_rows.insert(in2_rows.end(), in1_rows.begin(), in1_rows.end());

    auto in1_place = input1.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in1_place));
    auto in2_place = input2->place();
    PADDLE_ENFORCE(platform::is_cpu_place(in2_place));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(in2_place),
                 in2_data + input2_offset,
                 boost::get<platform::CPUPlace>(in1_place), in1_data,
                 in1_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAddTo<platform::CPUPlace, float>;
template struct SelectedRowsAddTo<platform::CPUPlace, double>;

template <typename T>
struct SelectedRowsAddToTensor<platform::CPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2->numel() / in1_height);

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

template struct SelectedRowsAddToTensor<platform::CPUPlace, float>;
template struct SelectedRowsAddToTensor<platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
