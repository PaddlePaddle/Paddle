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

#include "glog/logging.h"

#include "paddle/phi/backends/device_code.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace fusion {

template <typename DeviceContext>
static void MutableMultiTypeData(std::vector<phi::DenseTensor*>* var,
                                 const std::vector<int>& data_type,
                                 const DeviceContext& dev_ctx) {
  for (size_t i = 0; i < var->size(); i++) {
    if (data_type[i] == phi::TransToProtoVarType(phi::DataType::FLOAT32)) {
      dev_ctx.template Alloc<float>((*var)[i],
                                    (*var)[i]->numel() * sizeof(float));
    } else if (data_type[i] ==
               phi::TransToProtoVarType(phi::DataType::FLOAT16)) {
      dev_ctx.template Alloc<phi::dtype::float16>(
          (*var)[i], (*var)[i]->numel() * sizeof(phi::dtype::float16));
    } else if (data_type[i] ==
               phi::TransToProtoVarType(phi::DataType::FLOAT64)) {
      dev_ctx.template Alloc<double>((*var)[i],
                                     (*var)[i]->numel() * sizeof(double));
    }
  }
}

template <typename T, typename Context>
void FusionGroupKernel(const Context& dev_ctx,
                       const std::vector<const DenseTensor*>& ins,
                       const std::vector<int>& outs_dtype,
                       const std::vector<int>& inputs_dtype,
                       const std::string& func_name,
                       int type,
                       std::vector<DenseTensor*> outs) {
  size_t num_ins = ins.size();
  size_t num_outs = outs.size();

  MutableMultiTypeData(&outs, outs_dtype, dev_ctx);

  phi::DeviceCode* dev_code =
      phi::DeviceCodePool::Instance().Get(dev_ctx.GetPlace(), func_name);
  VLOG(3) << "func_name: " << func_name;

  if (type == 0) {
    size_t n = ins[0]->numel();
    std::vector<void*> args;
    args.push_back(&n);
    std::vector<const void*> ptrs(num_ins + num_outs);
    for (size_t i = 0; i < num_ins; ++i) {
      if (inputs_dtype[i] == phi::TransToProtoVarType(phi::DataType::FLOAT16)) {
        ptrs[i] = ins[i]->data<phi::dtype::float16>();
      } else if (inputs_dtype[i] ==
                 phi::TransToProtoVarType(phi::DataType::FLOAT32)) {
        ptrs[i] = ins[i]->data<float>();
      } else if (inputs_dtype[i] ==
                 phi::TransToProtoVarType(phi::DataType::FLOAT64)) {
        ptrs[i] = ins[i]->data<double>();
      }
      args.push_back(&ptrs[i]);
    }
    for (size_t j = 0; j < num_outs; ++j) {
      if (outs_dtype[j] == phi::TransToProtoVarType(phi::DataType::FLOAT16)) {
        ptrs[num_ins + j] = outs[j]->data<phi::dtype::float16>();
      } else if (outs_dtype[j] ==
                 phi::TransToProtoVarType(phi::DataType::FLOAT32)) {
        ptrs[num_ins + j] = outs[j]->data<float>();
      } else if (outs_dtype[j] ==
                 phi::TransToProtoVarType(phi::DataType::FLOAT64)) {
        ptrs[num_ins + j] = outs[j]->data<double>();
      }
      args.push_back(&ptrs[num_ins + j]);
    }
    dev_code->Launch(n, &args);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fusion_group,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusionGroupKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
