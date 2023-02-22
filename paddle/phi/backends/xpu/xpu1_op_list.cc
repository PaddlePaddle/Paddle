/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/phi/backends/xpu/xpu_op_list.h"

namespace phi {
namespace backends {
namespace xpu {

XPUOpMap& get_kl1_ops() {
  // KL1支持的op，通过op_name, data_type
  static XPUOpMap s_xpu1_kernels{
      {"abs", XPUKernelSet({phi::DataType::FLOAT32})},
      {"accuracy", XPUKernelSet({phi::DataType::FLOAT32})},
      {"adam", XPUKernelSet({phi::DataType::FLOAT32})},
      {"adamw", XPUKernelSet({phi::DataType::FLOAT32})},
      {"affine_channel_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"affine_channel", XPUKernelSet({phi::DataType::FLOAT32})},
      {"arg_max", XPUKernelSet({phi::DataType::FLOAT32})},
      {"assign",
       XPUKernelSet({phi::DataType::FLOAT32,
                     phi::DataType::FLOAT64,
                     phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::BOOL})},
      {"batch_norm_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"batch_norm", XPUKernelSet({phi::DataType::FLOAT32})},
      {"bilinear_interp", XPUKernelSet({phi::DataType::FLOAT32})},
      {"bilinear_interp_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"bilinear_interp_v2", XPUKernelSet({phi::DataType::FLOAT32})},
      {"bilinear_interp_v2_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"broadcast",
       XPUKernelSet({phi::DataType::FLOAT32,
                     phi::DataType::FLOAT64,
                     phi::DataType::INT32,
                     phi::DataType::INT64})},
      {"cast",
       XPUKernelSet({phi::DataType::FLOAT32,
                     phi::DataType::INT64,
                     phi::DataType::INT32})},
      {"clip_by_norm", XPUKernelSet({phi::DataType::FLOAT32})},
      {"coalesce_tensor",
       XPUKernelSet({phi::DataType::FLOAT32,
                     phi::DataType::FLOAT64,
                     phi::DataType::INT32})},
      {"concat", XPUKernelSet({phi::DataType::FLOAT32})},
      {"concat_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"conv2d", XPUKernelSet({phi::DataType::FLOAT32})},
      {"conv2d_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"deformable_conv", XPUKernelSet({phi::DataType::FLOAT32})},
      {"deformable_conv_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"depthwise_conv2d", XPUKernelSet({phi::DataType::FLOAT32})},
      {"depthwise_conv2d_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"dropout", XPUKernelSet({phi::DataType::FLOAT32})},
      {"dropout_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"c_allreduce_sum", XPUKernelSet({phi::DataType::FLOAT32})},
      {"c_reduce_sum", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_add", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_add_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_div_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_div", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_floordiv", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_max_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_max", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_min_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_min", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_mul_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_mul", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_pow", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_sub_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_sub", XPUKernelSet({phi::DataType::FLOAT32})},
      {"equal", XPUKernelSet({phi::DataType::INT64})},
      {"expand_as_v2",
       XPUKernelSet({phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT16,
                     phi::DataType::FLOAT32})},
      {"expand_v2",
       XPUKernelSet({phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT16,
                     phi::DataType::FLOAT32})},
      {"fc_xpu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"fill_any_like", XPUKernelSet({phi::DataType::INT64})},
      {"fill_constant",
       XPUKernelSet({phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::FLOAT64,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT32})},
      {"gather_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"gather", XPUKernelSet({phi::DataType::FLOAT32})},
      {"gaussian_random", XPUKernelSet({phi::DataType::FLOAT32})},
      {"gelu_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"gelu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"generate_sequence_xpu",
       XPUKernelSet({
           phi::DataType::FLOAT32,
           phi::DataType::INT32,
           phi::DataType::INT64,
       })},
      {"hard_switch_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"hard_switch", XPUKernelSet({phi::DataType::FLOAT32})},
      {"iou_similarity", XPUKernelSet({phi::DataType::FLOAT32})},
      {"lamb", XPUKernelSet({phi::DataType::FLOAT32})},
      {"layer_norm_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"layer_norm", XPUKernelSet({phi::DataType::FLOAT32})},
      {"leaky_relu_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"leaky_relu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"load",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT8,
                     phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::FLOAT32})},
      {"logicaland",
       XPUKernelSet({phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::INT16,
                     phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::FLOAT32})},
      {"logicalnot",
       XPUKernelSet({phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::INT16,
                     phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::FLOAT32})},
      {"logicalor",
       XPUKernelSet({phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::INT16,
                     phi::DataType::INT32,
                     phi::DataType::INT64,
                     phi::DataType::FLOAT32})},
      {"log_loss_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"log_loss", XPUKernelSet({phi::DataType::FLOAT32})},
      {"logsumexp", XPUKernelSet({phi::DataType::FLOAT32})},
      {"log", XPUKernelSet({phi::DataType::FLOAT32})},
      {"lookup_table_v2_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"lookup_table_v2", XPUKernelSet({phi::DataType::FLOAT32})},
      {"matmul_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"matmul_v2_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"matmul_v2", XPUKernelSet({phi::DataType::FLOAT32})},
      {"matmul", XPUKernelSet({phi::DataType::FLOAT32})},
      {"mean_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"mean", XPUKernelSet({phi::DataType::FLOAT32})},
      {"momentum", XPUKernelSet({phi::DataType::FLOAT32})},
      {"mul_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"mul", XPUKernelSet({phi::DataType::FLOAT32})},
      {"nearest_interp_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"nearest_interp_v2_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"nearest_interp_v2", XPUKernelSet({phi::DataType::FLOAT32})},
      {"nearest_interp", XPUKernelSet({phi::DataType::FLOAT32})},
      {"one_hot_v2",
       XPUKernelSet({phi::DataType::INT32, phi::DataType::INT64})},
      {"one_hot", XPUKernelSet({phi::DataType::INT32, phi::DataType::INT64})},
      {"pool2d_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"pool2d", XPUKernelSet({phi::DataType::FLOAT32})},
      {"pow", XPUKernelSet({phi::DataType::FLOAT32})},
      {"range",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::FLOAT32})},
      {"reduce_max_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_max", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_mean", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_mean_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_prod", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_sum_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reduce_sum", XPUKernelSet({phi::DataType::FLOAT32})},
      {"relu_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"relu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reshape2_grad",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT32})},
      {"reshape2",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT32})},
      {"rmsprop", XPUKernelSet({phi::DataType::FLOAT32})},
      {"rnn_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"rnn", XPUKernelSet({phi::DataType::FLOAT32})},
      {"roi_align_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"roi_align", XPUKernelSet({phi::DataType::FLOAT32})},
      {"scale", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sgd", XPUKernelSet({phi::DataType::FLOAT32})},
      {"shape",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::FLOAT32})},
      {"sigmoid_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sigmoid", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sign", XPUKernelSet({phi::DataType::FLOAT32})},
      {"slice_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"slice", XPUKernelSet({phi::DataType::FLOAT32, phi::DataType::INT32})},
      {"softmax_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"softmax_with_cross_entropy", XPUKernelSet({phi::DataType::FLOAT32})},
      {"softmax_with_cross_entropy_grad",
       XPUKernelSet({phi::DataType::FLOAT32})},
      {"softmax", XPUKernelSet({phi::DataType::FLOAT32})},
      {"split", XPUKernelSet({phi::DataType::FLOAT32, phi::DataType::INT32})},
      {"sqrt_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sqrt", XPUKernelSet({phi::DataType::FLOAT32})},
      {"square_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"square", XPUKernelSet({phi::DataType::FLOAT32})},
      {"squeeze2_grad",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"squeeze2",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"squeeze_grad",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"squeeze",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"stack", XPUKernelSet({phi::DataType::FLOAT32})},
      {"stack_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sum", XPUKernelSet({phi::DataType::FLOAT32})},
      {"tanh_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"tanh", XPUKernelSet({phi::DataType::FLOAT32})},
      {"top_k", XPUKernelSet({phi::DataType::FLOAT32})},
      {"transpose2_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"transpose2", XPUKernelSet({phi::DataType::FLOAT32})},
      {"transpose_grad", XPUKernelSet({phi::DataType::FLOAT32})},
      {"transpose", XPUKernelSet({phi::DataType::FLOAT32})},
      {"truncated_gaussian_random", XPUKernelSet({phi::DataType::FLOAT32})},
      {"uniform_random", XPUKernelSet({phi::DataType::FLOAT32})},
      {"unsqueeze2_grad",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"unsqueeze2",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"unsqueeze_grad",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"unsqueeze",
       XPUKernelSet({phi::DataType::FLOAT64,
                     phi::DataType::INT64,
                     phi::DataType::INT32,
                     phi::DataType::BOOL,
                     phi::DataType::INT8,
                     phi::DataType::UINT8,
                     phi::DataType::FLOAT32})},
      {"where_index", XPUKernelSet({phi::DataType::BOOL})},
      // AddMore
  };

  return s_xpu1_kernels;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
