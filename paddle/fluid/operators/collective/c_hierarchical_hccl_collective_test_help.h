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

#if defined(PADDLE_WITH_HIERARCHICAL_HCCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_NO_KERNEL_OP(c_gen_hierarchical_hccl_id);
USE_NO_KERNEL_OP(c_comm_init_hierarchical_hccl);

DEFINE_int32(rank_id, -1, "Specify test rank id!");
DEFINE_int32(device_id, -1, "Specify test device id!");
DEFINE_int32(rank_count, -1, "Specify test rank count!");
DEFINE_int32(split_index, -1, "Specify test split index!");
DEFINE_int32(root_id, -1, "Specify test root id!");
DEFINE_int32(data_size, -1, "Specify test data size!");
DEFINE_int32(src_rank, -1, "Specify test src rank!");
DEFINE_int32(dest_rank, -1, "Specify test dest rank!");
DEFINE_string(endpoint, "", "Specify test end point!");

DECLARE_int32(rank_id);
DECLARE_int32(device_id);
DECLARE_int32(rank_count);
DECLARE_int32(split_index);
DECLARE_int32(root_id);
DECLARE_int32(data_size);
DECLARE_int32(dest_rank);
DECLARE_int32(src_rank);
DECLARE_string(endpoint);

DECLARE_string(selected_npus);

template <typename T>
void PrintDebugInfo(const std::string preStr, const std::vector<T>& data) {
  std::string debugstring = "";
  for (auto ele : data) {
    debugstring += std::to_string(ele) + std::string(",");
  }
  VLOG(2) << preStr << ":" << std::endl << debugstring;
}

// [min, max)
void check_test_int_env(int value, std::string env_int_flags, int min,
                        int max) {
  PADDLE_ENFORCE_GE(value, min,
                    paddle::platform::errors::InvalidArgument(
                        "Input env (%s) should be great or equal than (%d)!",
                        env_int_flags.c_str(), min));

  if (max != -1) {
    PADDLE_ENFORCE_LT(value, max,
                      paddle::platform::errors::InvalidArgument(
                          "Input env (%s) should be great or equal than (%d)!",
                          env_int_flags.c_str(), max));
  }
}

void check_test_comm_env() {
  PADDLE_ENFORCE_NOT_NULL(
      FLAGS_endpoint.c_str(),
      paddle::platform::errors::InvalidArgument("Input env (%s) should be set!",
                                                FLAGS_endpoint.c_str()));
  VLOG(1) << "FLAGS_endpoint" << FLAGS_endpoint << std::endl;
  check_test_int_env(FLAGS_rank_count, "FLAGS_rank_count", 0, -1);
  check_test_int_env(FLAGS_rank_id, "FLAGS_rank_id", 0, FLAGS_rank_count);
  check_test_int_env(FLAGS_root_id, "FLAGS_root_id", 0, FLAGS_rank_count);
  check_test_int_env(FLAGS_device_id, "FLAGS_device_id", 0, FLAGS_rank_count);
  check_test_int_env(FLAGS_split_index, "FLAGS_split_index", 0, -1);
}

void check_test_sendrecv_env() {
  check_test_int_env(FLAGS_src_rank, "FLAGS_src_rank", 0, FLAGS_rank_count);
  check_test_int_env(FLAGS_dest_rank, "FLAGS_dest_rank", 0, FLAGS_rank_count);
  check_test_int_env(FLAGS_data_size, "FLAGS_data_size", 0, -1);
}

void prepare(f::Scope* scope, const p::DeviceContext& ctx,
             HierarchicalHcclCommGroupIdType group_id) {
  check_test_comm_env();

  int rank_id = FLAGS_rank_id;
  int device_id = FLAGS_device_id;
  int rank_count = FLAGS_rank_count;
  int split_index = FLAGS_split_index;
  std::string endpoint = FLAGS_endpoint;

  VLOG(3) << "Get input parameter from env: "
          << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_count = " << rank_count << "; endpoint = " << endpoint
          << "; split_index = " << split_index;

  VLOG(3) << "Begin gen hierarchical hccl id!";
  f::AttributeMap gen_hierarchical_hccl_id;

  gen_hierarchical_hccl_id["rank"] = rank_id;
  gen_hierarchical_hccl_id["rank_count"] = rank_count;
  gen_hierarchical_hccl_id["endpoint"] = std::string(endpoint);
  gen_hierarchical_hccl_id["group_name"] = std::string(group_id);
  gen_hierarchical_hccl_id["split_index"] = split_index;

  auto comm_hccl_id = f::OpRegistry::CreateOp("c_gen_hierarchical_hccl_id", {},
                                              {}, gen_hierarchical_hccl_id);
  auto place = ctx.GetPlace();
  comm_hccl_id->Run(*scope, place);
  ctx.Wait();
  VLOG(3) << "Gen hierarchical hccl id successfully!";

  VLOG(3) << "Begin init hierarchical hccl comm!";
  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["rank_ids"] = rank_count;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  comm_init_attrs["group_name"] = group_id;
  // comm_init_attrs["rank_ids"] = rank_ids;

  auto comm_init_op = f::OpRegistry::CreateOp("c_comm_init_hierarchical_hccl",
                                              {}, {}, comm_init_attrs);
  comm_init_op->Run(*scope, place);
  ctx.Wait();
  VLOG(3) << "Init hierarchical hccl comm successfully!";
}
