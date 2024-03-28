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
#include <vector>
#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
//#include "paddle/pir/include/core/builder.h"
//#include "paddle/pir/include/core/value.h"

namespace phi {
namespace distributed {

class ReshardFuncDesc {
public:
    ReshardFuncDesc(const std::string& n, DataType d): name(n), dtype(d){}
    virtual ~ReshardFuncDesc() {}
    //virtual pir::Operation* Build(pir::Builder& builder, pir::Value& input) = 0;

    std::string name;
    DataType dtype;

};

class AllReduceOpDesc : public ReshardFuncDesc {
public:
    AllReduceOpDesc(DataType dt, const std::vector<int64_t>& pids, int red_type): ReshardFuncDesc("AllReduce", dt), process_ids(pids), reduce_type(red_type) {}
    //fir::Operation* Build(pir::Builder& builder, pir::Value& input) override { }

    std::vector<int64_t> process_ids;
    int reduce_type;
};

class AllGatherOpDesc : public ReshardFuncDesc {
public:
    AllGatherOpDesc(DataType dt, const std::vector<int64_t>& pids): ReshardFuncDesc("Split", dt), process_ids(pids) {}

private:
    std::vector<int64_t> process_ids;
};

class SplitOpDesc : public ReshardFuncDesc {
public:
    SplitOpDesc(DataType dt, const std::vector<int64_t>& sts, int64_t ax): ReshardFuncDesc("Split", dt), sections(sts), axis(ax) { }
private:
    std::vector<int64_t> sections;
    int64_t axis;
};

class ConcatOpDesc : public ReshardFuncDesc {
public:
    ConcatOpDesc(DataType dt, int64_t ax): ReshardFuncDesc("Concat", dt), axis(ax) { }
private:
    int64_t axis;
};

//  struct FullOpDesc : public ReshardFuncDesc {
//      phi::IntArray shape;
//      int64_t value;
//      FullOpDesc(DataType dt, const phi::IntArray& array, int64_t val): ReshardFuncDesc("Full", dt), shape(array), value(val) { }

//      pir::Operation* Build(pir::Builder& builder, pir::Value& input) override {
//          return builder.Build<paddle::dialect::FullOp>(shape, value);
//      }
//  };

//  struct DivideOpDesc : public ReshardFuncDesc {
//      DivideOpDesc(DataType dt): ReshardFuncDesc("Full", dt) {}
//      pir::Operation* Build(pir::Builder& builder, pir::Value& input) override {
//          return builder.Build<paddle::dialect::DivideOp>(input, ring_id, reduce_type);
//      }
//  };

}  // namespace distributed
}  // namespace phi
