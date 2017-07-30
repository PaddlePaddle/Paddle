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

#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

LODTensor LODTensor::Slice(uint32_t level) const {
  LODTensor res;
  auto new_lod = std::make_shared<lod_t>(lod_start_pos_->begin() + level,
                                         lod_start_pos_->end());
  res.set_tensor(tensor_);
  res.set_lod(new_lod);
  return res;
}

}  // namespace framework
}  // namespace paddle
