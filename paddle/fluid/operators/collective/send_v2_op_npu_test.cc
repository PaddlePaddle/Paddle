/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <string>
#include <thread>  // NOLINT
#include <vector>
#include <stdio.h>
#include "gtest/gtest.h"

#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(send_v2);
USE_NO_KERNEL_OP(c_comm_init_hcom);
USE_OP_DEVICE_KERNEL(send_v2, NPU);

void Prepare(f::Scope* scope, const p::DeviceContext& ctx){

    std::string rank_table_file = getenv("RANK_TABLE_FILE");
    int rank_id = atoi(getenv("RANK_ID"));
    int device_id = atoi(getenv("DEVICE_ID"));
    int src_rank = atoi(getenv("SRC_RANK"));
    int dest_rank = atoi(getenv("DEST_RANK"));
    VLOG(3)<<"rank_id "<< rank_id << "src_rank"<< src_rank <<"dest_rank" <<dest_rank;

    std::vector<int> rank_ids = {0, 1};
    f::AttributeMap comm_init_attrs;
    comm_init_attrs["ring_id"] = 0;
    comm_init_attrs["nranks"] = 2;
    comm_init_attrs["rank"] = rank_id;
    comm_init_attrs["device_id"] = device_id;
    comm_init_attrs["rank_ids"] = rank_ids;
    auto comm_init_op = f::OpRegistry::CreateOp("c_comm_init_hcom", {}, {}, comm_init_attrs);
    auto place = ctx.GetPlace();
    comm_init_op->Run(*scope, place);
    ctx.Wait();
}

void TestHcomSendOp(f::Scope* scope, const p::DeviceContext& ctx){
    std::cout<< "BEGIN TEST:"<< __FUNCTION__ <<std::endl;
    auto x = scope->Var("X");
    auto tensor_x = x->GetMutable<f::LoDTensor>();
    int num = atoi(getenv("DATA_SIZE"));;
    EXPECT_GT(num, 0);
    EXPECT_LT(num, 1 << 15);
    std::vector<float> init(num*num, 1.0 * atoi(getenv("DEST_RANK")));
    int rank_id = atoi(getenv("RANK_ID"));
    VLOG(3)<<"rank id:"<<rank_id;
    TensorFromVector(init, ctx, tensor_x);
    tensor_x->Resize({num, num});
    ctx.Wait();
    auto place = ctx.GetPlace();
    ctx.Wait();

    f::AttributeMap attrs;
    attrs["tag"]=std::string("srtest");
    attrs["peer"]=atoi(getenv("DEST_RANK"));
    attrs["ring_id"]=0;
    attrs["srTag"]=0;

    auto op = f::OpRegistry::CreateOp("send_v2", {{"X", {"X"}}}, {}, attrs);
    
    op->Run(*scope, place);
    VLOG(3)<<"send run over";
    ctx.Wait();    
}

TEST(send_v2, NPU){
    f::Scope scope;
    char * npu_id=getenv("FLAGS_selected_npus");
    VLOG(3) << "Select npu:" << npu_id;
    p::NPUDeviceContext ctx(p::NPUPlace(atoi(npu_id)));
    VLOG(3) << "Place over";
    Prepare(&scope, ctx);
    VLOG(3) << "Prepare over";
    TestHcomSendOp(&scope, ctx);
    VLOG(3) << "Test over";

}
