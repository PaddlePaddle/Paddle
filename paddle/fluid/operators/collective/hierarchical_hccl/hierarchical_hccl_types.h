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

#include <hccl/hccl_types.h>
#include <string>

// unique id
const int HIERARCHICALHCCL_UNIQUE_ID_BYTES = 8192;
using HierarchicalHcclUniqueId = struct HierarchicalHcclUniqueId_t {
  char internal[HIERARCHICALHCCL_UNIQUE_ID_BYTES];
};

using HierarchicalHcclBackend = std::string;
using HierarchicalHcclScope = std::string;
using HierarchicalHcclRank = int;

// init configs
using HierarchicalHcclMemberRange = struct HierarchicalHcclMemberRange_t {
  HierarchicalHcclRank start;
  HierarchicalHcclRank end;
};

using HierarchicalHcclRankList = struct HierarchicalHcclRankList_t {
  HierarchicalHcclRank *ranks;
  int rank_count;
};

using HierarchicalHcclLayerMemberType = enum HierarchicalHcclLayerMemberType_t {
  UNKNOWN = 0,
  RANGE = 1,
  RANK_LIST = 2,
};

using HierarchicalHcclLayerConfig = struct HierarchicalHcclLayerConfig_t {
  union HierarchicalHcclLayerMembers_t {
    HierarchicalHcclMemberRange *range;
    HierarchicalHcclRankList *members;
  } members;
  HierarchicalHcclLayerMemberType member_type;
  HierarchicalHcclBackend backend;
  int level;
  HierarchicalHcclScope scope;
};

using HierarchicalHcclInitConfig = struct HierarchicalHcclInitConfig_t {
  HierarchicalHcclBackend backend;
  HierarchicalHcclScope scope;
  HierarchicalHcclLayerConfig *layers;
  int layer_count;
};

// group id
using HierarchicalHcclCommGroupIdType = std::string;

#define HierarchicalHccl_COMM_GLOBAL_GROUP_ID \
  "__HierarchicalHccl_COMM_GLOBAL_GROUP__"
#define HCOM_GROUP_PREFIX "HCOM_GROUP_"

// result
using HierarchicalHcclResult = enum HierarchicalHcclResult_t {
  SUCCESS = 0,
  INTERNAL_ERROR = 1,
};

// reduction operations
using HierarchicalHcclReductionOp = HcclReduceOp;

// data types
using HierarchicalHcclDataType = HcclDataType;

// stream
using HierarchicalHcclRuntimeStream = void *;
