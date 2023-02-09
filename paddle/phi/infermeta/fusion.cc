/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/fusion.h"
#include <vector>
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

void FcXPUInferMeta(const MetaTensor& x,
                    const MetaTensor& w,
                    const MetaTensor& w_max,
                    const MetaTensor& bias,
                    int in_num_col_dims,
                    bool transpose_x,
                    float alpha,
                    float beta,
                    int act_type,
                    float act_alpha,
                    MetaTensor* out) {
  std::vector<int> out_shape(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; i++) {
    out_shape[i] = x.dims()[i];
  }
  out_shape[in_num_col_dims] = w.dims()[0];
  out->set_dims(DDim(out_shape.data(), out_shape.size()));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

}  // namespace phi
