// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/prim/api/manual/utils/utils.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

namespace paddle {
namespace prim {

template<>
Tensor empty<Tensor>(const paddle::experimental::IntArray& shape,
             paddle::experimental::DataType dtype,
             const paddle::Place& place){
    if(dtype == paddle::experimental::DataType::UNDEFINED){
        dtype = paddle::experimental::DataType::FLOAT32;
    }
    return empty_ad_func(shape, dtype, place);
}

template<>
Tensor empty_like<Tensor>(const paddle::experimental::Tensor& x,
                  paddle::experimental::DataType dtype,
                  const paddle::Place& place){
    if(dtype == paddle::experimental::DataType::UNDEFINED){
        dtype = paddle::experimental::DataType::FLOAT32;
    }   
    return empty_like_ad_func(x, dtype, place);
}

}  // namespace prim
}  // namespace paddle