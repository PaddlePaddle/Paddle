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
#include "glog/logging.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"

namespace pir {
/**
 * IMPORTANT!!!
 * all those defining strings can't be changed, otherwise the deserialization
 * will failed. define all keys in serialized files to ensure accuracy for
 * deserialization make sure all the key mutually exclusive
 */

// all IR structure's identifier (region, block, op, attr, type value etc)
// which can be string , int64_t etc.
#define ID "#"
#define VALUE_ID "%"
// program's key:
#define REGIONS "regions"

// region's key:
// which is json array with block json object(ID and BLOCKARGS and BLOCKOPS)
#define BLOCKS "blocks"

// block's key:
// which is json array with value json object
#define BLOCKARGS "args"
#define KEYWORDBLOCKARGS "kwargs"

#define KEYWORDNAME "key"

// which is json array with operation json object
#define BLOCKOPS "ops"

// operation's key:
// input
// which is json array with opoperand json object(ID)
#define OPOPERANDS "I"

// output
// which is json array with value json object(ID and TYPE_TYPE)
#define OPRESULTS "O"

// which is json array with json object(NAME and ATTR_TYPE)
#define ATTRS "A"
#define OPRESULTS_ATTRS "OA"
#define DIST_ATTRS "DA"
#define QUANT_ATTRS "QA"

// value's key:
//  value's type which should be pir::Type's json object(ID or ID and DATA).
#define TYPE_TYPE "TT"

// attr's name which is operation's feature.
#define NAME "N"

// attr's value which is pir::Attribute's json object(ID and DATA).
#define ATTR_TYPE "AT"

// type/attr's contents which is json::array.
#define DATA "D"
// float/double data with nan, inf, -inf
#define VOID_DATA "VD"

// NULL_TYPE
#define NULL_TYPE "NULL"

// special op compress
#define PARAMETEROP "p"

// actions for patch
#define DELETE "DEL"
#define ADD "ADD"
#define UPDATE "UPD"
#define NEW_NAME "NN"
#define ADD_ATTRS "ADD_A"
#define ADD_OPRESULTS_ATTRS "ADD_OA"
#define PATCH "patch"

std::pair<std::string, std::string> GetContentSplitByDot(
    const std::string& str);

std::vector<std::string> GetOpDistAttr();
std::vector<std::string> GetOpQuantAttr();

void GetCompressOpName(std::string* op_name);

void GetDecompressOpName(std::string* op_name);
class DialectIdMap {
 public:
  static DialectIdMap* Instance();
  DialectIdMap();
  void insert(const std::string& key, const std::string& value);

  std::string GetCompressDialectId(const std::string& name);

  std::string GetDecompressDialectId(const std::string& id);

 private:
  std::unordered_map<std::string, std::string> CompressDialect;
  std::unordered_map<std::string, std::string> DecompressDialect;
};

}  // namespace pir
