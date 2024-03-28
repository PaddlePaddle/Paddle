// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/reshard_function_desc.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

namespace phi {
class DeviceContext;

namespace distributed {

inline pir::Operation* Build(ReshardFuncDesc* base_desc, pir::Builder& builder, const std::vector<pir::Value>& inputs) {
    if (base_desc->name == "AllReduce") {
        AllReduceOpDesc* desc = dynamic_cast<AllReduceOpDesc*>(base_desc);
        CreateOrGetCommContext(desc->process_ids);

        std::string ring_id = GenUniqueCommKey(desc->process_ids);
        return builder.Build<paddle::dialect::AllReduceOp>(inputs[0], ring_id, desc->reduce_type);
    }
    return nullptr;
}

}  // namespace distributed
}  // namespace phi
