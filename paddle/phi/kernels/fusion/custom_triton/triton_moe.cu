// // Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <cuda_fp16.h>
// #include <vector>
// #include "generated/moe/moe.h"
// #include "paddle/extension.h"


// void TritonMoe(
//     const paddle::Tensor& a_ptr,
//     const paddle::Tensor& b_ptr,
//     const paddle::Tensor& c_ptr,
//     const paddle::Tensor& topk_weights_ptr,
//     const paddle::Tensor& sorted_token_ids_ptr,
//     const paddle::Tensor& expert_ids_ptr,
//     const paddle::Tensor& num_tokens_post_padded_ptr,
//     int N,
//     int K,
//     int EM,
//     int num_valid_tokens,
//     int stride_am,
//     int stride_ak,
//     int stride_be,
//     int stride_bk,
//     int stride_bn,
//     int stride_cm,
//     int stride_cn,
//     int stride_weight,
//     int stride_token_id) {


//   auto status = moe_kernel(
//     a_ptr.stream(),
//     (CUdeviceptr)(a_ptr.data<phi::dtype::float16>()),
//     (CUdeviceptr)(b_ptr.data<phi::dtype::float16>()),
//     (CUdeviceptr)(c_ptr.data<phi::dtype::float16>()),
//     (CUdeviceptr)(topk_weights_ptr.data<phi::dtype::float16>()),
//     (CUdeviceptr)(sorted_token_ids_ptr.data<int>()),
//     (CUdeviceptr)(expert_ids_ptr.data<int>()),
//     (CUdeviceptr)(num_tokens_post_padded_ptr.data<int>()),
//     N,
//     K,
//     EM,    // EM
//     num_valid_tokens, // valid_token_num
//     stride_am, stride_ak,
//     stride_be, stride_bk, stride_bn,
//     stride_cm, stride_cn,
//     stride_weight, stride_token_id, 0);
//   assert(status == CUDA_SUCCESS);
// }

// PD_BUILD_OP(triton_moe)
//     .Inputs({"a_ptr", "b_ptr", "c_ptr", "topk_weights_ptr", "sorted_token_ids_ptr", "expert_ids_ptr", "num_tokens_post_padded_ptr"})
//     .Outputs({"c_out"})
//     .SetInplaceMap({{"c_ptr", "c_out"}})
//     .Attrs({"N: int", "K: int", "EM: int", "num_valid_tokens: int", "stride_am: int", "stride_ak: int", "stride_be: int", "stride_bk: int", "stride_bn: int", "stride_cm: int", "stride_cn: int", "stride_weight: int", "stride_token_id: int"}) 
//     .SetKernelFn(PD_KERNEL(TritonMoe));






#include "paddle/extension.h"
#include "generated/moe/moe.h"


void* get_tensor_ptr(const paddle::Tensor& input)
{
  if (input.type() == paddle::DataType::FLOAT16) {
    return (void*)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (void*)(input.data<int>());
  } else {
    assert(false);
    return nullptr;
  }
}

void triton_moe_func(const paddle::Tensor& para0,
const paddle::Tensor& para1,
const paddle::Tensor& para2,
const paddle::Tensor& para3,
const paddle::Tensor& para4,
const paddle::Tensor& para5,
const paddle::Tensor& para6,
int attr0,
int attr1,
int attr2,
int attr3,
int attr4,
int attr5,
int attr6,
int attr7,
int attr8,
int attr9,
int attr10,
int attr11,
int attr12) {

  auto status = moe_kernel(para0.stream(),(CUdeviceptr)(get_tensor_ptr(para0)),
(CUdeviceptr)(get_tensor_ptr(para1)),
(CUdeviceptr)(get_tensor_ptr(para2)),
(CUdeviceptr)(get_tensor_ptr(para3)),
(CUdeviceptr)(get_tensor_ptr(para4)),
(CUdeviceptr)(get_tensor_ptr(para5)),
(CUdeviceptr)(get_tensor_ptr(para6)),
attr0,
attr1,
attr2,
attr3,
attr4,
attr5,
attr6,
attr7,
attr8,
attr9,
attr10,
attr11,
attr12,
0);
  assert(status == CUDA_SUCCESS);
}

PD_BUILD_OP(triton_moe)
    .Inputs({"para0", "para1", "para2", "para3", "para4", "para5", "para6"})
    .Outputs({"out2"})
    .SetInplaceMap({{"para2","out2"}})
    .Attrs({"attr0 : int", "attr1 : int", "attr2 : int", "attr3 : int", "attr4 : int", "attr5 : int", "attr6 : int", "attr7 : int", "attr8 : int", "attr9 : int", "attr10 : int", "attr11 : int", "attr12 : int"}) 
    .SetKernelFn(PD_KERNEL(triton_moe_func));