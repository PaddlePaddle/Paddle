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

#pragma once

#ifndef _WIN32
#include <unistd.h>
#endif

#include <stdio.h>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#if defined(PADDLE_WITH_ECCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/eccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_NO_KERNEL_OP(c_gen_eccl_id);
USE_NO_KERNEL_OP(c_comm_init_eccl);

template <typename T>
void PrintDebugInfo(const std::string preStr, const std::vector<T>& data) {
  std::string debugstring = "";
  for (auto ele : data) {
    debugstring += std::to_string(ele) + std::string(",");
  }
  VLOG(2) << preStr << ":" << std::endl << debugstring;
}

void PrepareUniqueId(f::Scope* scope, const p::DeviceContext& ctx,
                     PaddleEcclCommGroupIdType group_id) {
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));
  int rank_count = atoi(getenv("RANK_COUNT"));
  int split_index = atoi(getenv("SPLIT_INDEX"));
  std::string endpoint = std::string(getenv("ENDPOINT"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_count = " << rank_count << "; endpoint = " << endpoint
          << "; split_index = " << split_index;

  f::AttributeMap gen_eccl_id;

  gen_eccl_id["rank"] = rank_id;
  gen_eccl_id["rank_count"] = rank_count;
  gen_eccl_id["endpoint"] = std::string(endpoint);
  gen_eccl_id["group_name"] = std::string(group_id);
  gen_eccl_id["split_index"] = split_index;

  auto comm_init_op =
      f::OpRegistry::CreateOp("c_gen_eccl_id", {}, {}, gen_eccl_id);
  VLOG(3) << "break";
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}

void Prepare(f::Scope* scope, const p::DeviceContext& ctx,
             PaddleEcclCommGroupIdType group_id) {
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));
  int rank_count = atoi(getenv("RANK_COUNT"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_count = " << rank_count;

  // std::vector<int> rank_ids{0, 1};
  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["rank_ids"] = rank_count;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  comm_init_attrs["group_name"] = group_id;
  // comm_init_attrs["rank_ids"] = rank_ids;
  auto comm_init_op =
      f::OpRegistry::CreateOp("c_comm_init_eccl", {}, {}, comm_init_attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}
