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

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace distributed {

struct BaseOpDesc {
  ReshardFuncDesc(const std::string& n, DataType d) : name(n), dtype(d) {}
  ReshardFuncDesc(const std::string& n,
                  DataType d,
                  const std::vector<int64_t>& pids)
      : name(n), dtype(d), process_ids(pids) {}
  virtual ~ReshardFuncDesc() {}

  std::string name;
  DataType dtype;
  std::vector<int64_t> process_ids;
};

struct AllReduceOpDesc : public BaseOpDesc {
  AllReduceOpDesc(DataType dt, const std::vector<int64_t>& pids, int red_type)
      : ReshardFuncDesc("AllReduce", dt, pids), reduce_type(red_type) {}
  int reduce_type;
};

struct AllGatherOpDesc : public BaseOpDesc {
  AllGatherOpDesc(DataType dt, const std::vector<int64_t>& pids)
      : ReshardFuncDesc("Split", dt, pids) {}
};

struct SplitOpDesc : public BaseOpDesc {
  SplitOpDesc(DataType dt, const std::vector<int64_t>& sts, int64_t ax)
      : ReshardFuncDesc("Split", dt), sections(sts), axis(ax) {}

  std::vector<int64_t> sections;
  int64_t axis;
};

struct ConcatOpDesc : public BaseOpDesc {
  ConcatOpDesc(DataType dt, int64_t ax)
      : ReshardFuncDesc("Concat", dt), axis(ax) {}
  int64_t axis;
};

//  struct FullOpDesc : public ReshardFuncDesc {
//      phi::IntArray shape;
//      int64_t value;
//      FullOpDesc(DataType dt, const phi::IntArray& array, int64_t val):
//      ReshardFuncDesc("Full", dt), shape(array), value(val) { }

//      pir::Operation* Build(pir::Builder& builder, pir::Value& input) override
//      {
//          return builder.Build<paddle::dialect::FullOp>(shape, value);
//      }
//  };

//  struct DivideOpDesc : public ReshardFuncDesc {
//      DivideOpDesc(DataType dt): ReshardFuncDesc("Full", dt) {}
//      pir::Operation* Build(pir::Builder& builder, pir::Value& input) override
//      {
//          return builder.Build<paddle::dialect::DivideOp>(input, ring_id,
//          reduce_type);
//      }
//  };

struct SendOpDesc : public BaseOpDesc {
  SendOpDesc(DataType dt,
             const std::vector<int64_t>& pids,
             int peer_rank,
             bool dyn_shape)
      : ReshardFuncDesc("Send", dt, pids),
        peer(peer_rank),
        dynamic_shape(dyn_shape) {}

  int peer;
  bool dynamic_shape;
};

struct RecvOpDesc : public BaseOpDesc {
  RecvOpDesc(DataType dt,
             const std::vector<int64_t>& pids,
             int peer_rank,
             bool dyn_shape)
      : ReshardFuncDesc("Recv", dt, pids),
        peer(peer_rank),
        dynamic_shape(dyn_shape) {}

  int peer;
  bool dynamic_shape;
};

}  // namespace distributed
}  // namespace phi
