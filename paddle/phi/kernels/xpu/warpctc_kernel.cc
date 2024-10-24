// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/warpctc_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WarpctcKernel(const Context& dev_ctx,
                   const DenseTensor& logits,
                   const DenseTensor& label,
                   const paddle::optional<DenseTensor>& logits_length,
                   const paddle::optional<DenseTensor>& labels_length,
                   int blank,
                   bool norm_by_times,
                   DenseTensor* loss,
                   DenseTensor* warpctcgrad) {
  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(common::errors::External(
        "XPU only support logits_length is_initialized"));
  }
  bool has_labels_length = labels_length.is_initialized();
  if (!has_labels_length) {
    PADDLE_THROW(common::errors::External(
        "XPU only support labels_length is_initialized"));
  }

  int max_sequence_length = logits.dims()[0];
  int num_sequences = logits.dims()[1];
  int sequence_width = logits.dims()[2];
  int max_target_seq_length = label.dims()[1];

  PADDLE_ENFORCE_GT(max_sequence_length,
                    0,
                    common::errors::InvalidArgument(
                        "The first dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        max_sequence_length));
  PADDLE_ENFORCE_GT(num_sequences,
                    0,
                    common::errors::InvalidArgument(
                        "The second dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        num_sequences));
  PADDLE_ENFORCE_GT(sequence_width,
                    0,
                    common::errors::InvalidArgument(
                        "The third dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        sequence_width));
  PADDLE_ENFORCE_GE(
      blank,
      0,
      common::errors::InvalidArgument("Input(blank) should be "
                                      "equal or greater than zero "
                                      "but received %d. ",
                                      blank));
  PADDLE_ENFORCE_LT(blank,
                    sequence_width,
                    common::errors::InvalidArgument("Input(blank) should be "
                                                    "less than %d "
                                                    "but received %d. ",
                                                    sequence_width,
                                                    blank));

  auto logits_length_dtype = logits_length.get_ptr()->dtype();
  auto labels_length_dtype = labels_length.get_ptr()->dtype();
  PADDLE_ENFORCE_EQ(logits_length_dtype == labels_length_dtype,
                    true,
                    common::errors::InvalidArgument(
                        "The data type of Input(logits_length) and "
                        "Input(labels_length) should be equal. "));
  PADDLE_ENFORCE_EQ(logits_length_dtype == DataType::INT32 ||
                        logits_length_dtype == DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "The data type of Input(logits_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64),
                        DataTypeToString(logits_length_dtype)));
  PADDLE_ENFORCE_EQ(labels_length_dtype == DataType::INT32 ||
                        labels_length_dtype == DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "The data type of Input(labels_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64),
                        DataTypeToString(labels_length_dtype)));

  warpctcgrad->Resize(
      common::make_ddim({max_sequence_length, num_sequences, sequence_width}));
  dev_ctx.template Alloc<T>(warpctcgrad);
  T* warpctcgrad_data = warpctcgrad->data<T>();

  int sm_workspace, lm_workspace;
  int64_t max_S = 2 * max_target_seq_length + 1;
  if (warpctcgrad_data == nullptr) {
    sm_workspace = sizeof(T) * sequence_width +
                   sizeof(int) * max_target_seq_length + sizeof(int);
    lm_workspace = 2 * max_S * sizeof(T) + 2 * max_S * sizeof(int);
  } else {
    sm_workspace = sizeof(T) * sequence_width +
                   sizeof(int) * max_target_seq_length + sizeof(int);
    lm_workspace = 4 * max_S * sizeof(T) + 2 * max_S * sizeof(int) +
                   sequence_width * sizeof(T);
  }

  PADDLE_ENFORCE_LE(
      sm_workspace + lm_workspace,
      256 * 1024,
      common::errors::InvalidArgument(
          "Input size should be equal or less than %d for xpu warpctc kernel, "
          "but size %d is received. ",
          256 * 1024,
          sm_workspace + lm_workspace));

  loss->Resize(common::make_ddim({num_sequences, 1}));
  dev_ctx.template Alloc<T>(loss);
  T* loss_data = loss->data<T>();

  const T* logits_data = logits.data<T>();
  const int* label_data = label.data<int>();
  int r;
  if (logits_length_dtype == DataType::INT32) {
    auto logits_length_data = logits_length.get_ptr()->data<int>();
    auto labels_length_data = labels_length.get_ptr()->data<int>();
    r = xpu::ctc_loss<T, int>(dev_ctx.x_context(),
                              logits_data,
                              label_data,
                              loss_data,
                              warpctcgrad_data,
                              logits_length_data,
                              labels_length_data,
                              max_sequence_length,
                              num_sequences,
                              sequence_width,
                              max_target_seq_length,
                              blank);
  } else {
    auto logits_length_data = logits_length.get_ptr()->data<int64_t>();
    auto labels_length_data = labels_length.get_ptr()->data<int64_t>();
    r = xpu::ctc_loss<T, int64_t>(dev_ctx.x_context(),
                                  logits_data,
                                  label_data,
                                  loss_data,
                                  warpctcgrad_data,
                                  logits_length_data,
                                  labels_length_data,
                                  max_sequence_length,
                                  num_sequences,
                                  sequence_width,
                                  max_target_seq_length,
                                  blank);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "ctc_loss");
}

}  // namespace phi

PD_REGISTER_KERNEL(warpctc, XPU, ALL_LAYOUT, phi::WarpctcKernel, float) {}
