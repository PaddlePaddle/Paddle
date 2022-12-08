// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/paddle2cinn/transform_desc.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using PbVarType = framework::proto::VarType;
namespace cpp = ::cinn::frontend::paddle::cpp;

::cinn::frontend::paddle::cpp::VarDescAPI::Type TransformVarTypeToCinn(
    const ::paddle::framework::proto::VarType::Type &type) {
#define SET_TYPE_CASE_ITEM(type__)                                  \
  case ::paddle::framework::proto::VarType::type__:                 \
    return ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var type"));
  }
#undef SET_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarTypeFromCinn(
    const ::cinn::frontend::paddle::cpp::VarDescAPI::Type &type) {
#define SET_TYPE_CASE_ITEM(type__)                              \
  case ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__: \
    return ::paddle::framework::proto::VarType::type__;         \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var type"));
  }
#undef SET_TYPE_CASE_ITEM
}

::cinn::frontend::paddle::cpp::VarDescAPI::Type TransformVarDataTypeToCinn(
    const ::paddle::framework::proto::VarType::Type &type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                             \
  case ::paddle::framework::proto::VarType::type__:                 \
    return ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var data type"));
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarDataTypeFromCpp(
    const ::cinn::frontend::paddle::cpp::VarDescAPI::Type &type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                         \
  case ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__: \
    return ::paddle::framework::proto::VarType::type__;         \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var data type"));
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

void TransformVarDescToCinn(framework::VarDesc *pb_desc,
                            cpp::VarDesc *cpp_desc) {
  cpp_desc->SetName(pb_desc->Name());
  cpp_desc->SetType(TransformVarTypeToCinn(pb_desc->GetType()));
  cpp_desc->SetPersistable(pb_desc->Persistable());
  if (pb_desc->Name() != "feed" && pb_desc->Name() != "fetch") {
    cpp_desc->SetDataType(TransformVarDataTypeToCinn(pb_desc->GetDataType()));
    cpp_desc->SetShape(pb_desc->GetShape());
  }
}

void TransformVarDescFromCinn(const cpp::VarDesc &cpp_desc,
                              framework::VarDesc *pb_desc) {
  pb_desc->Proto()->Clear();
  pb_desc->SetName(cpp_desc.Name());
  pb_desc->SetType(TransformVarTypeFromCinn(cpp_desc.GetType()));
  pb_desc->SetPersistable(cpp_desc.Persistable());
  if (cpp_desc.Name() != "feed" && cpp_desc.Name() != "fetch") {
    pb_desc->SetShape(cpp_desc.GetShape());
    pb_desc->SetDataType(TransformVarDataTypeFromCpp(cpp_desc.GetDataType()));
  }
}

/// For OpDesc transform
void OpInputsToCinn(framework::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc->InputNames()) {
    cpp_desc->SetInput(param, pb_desc->Input(param));
  }
}

void OpInputsFromCinn(const cpp::OpDesc &cpp_desc, framework::OpDesc *pb_desc) {
  pb_desc->MutableInputs()->clear();
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    pb_desc->SetInput(param, cpp_desc.Input(param));
  }
}

void OpOutputsToCinn(framework::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc->OutputNames()) {
    cpp_desc->SetOutput(param, pb_desc->Output(param));
  }
}

void OpOutputsFromCinn(const cpp::OpDesc &cpp_desc,
                       framework::OpDesc *pb_desc) {
  pb_desc->MutableOutputs()->clear();
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    pb_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

void OpAttrsToCinn(framework::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  using AttrType = framework::proto::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                                        \
  case AttrType::type__:                                           \
    cpp_desc->SetAttr<T>(name, pb_desc->GetAttrIfExists<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      IMPL_ONE(LONG, int64_t);
      IMPL_ONE(LONGS, std::vector<int64_t>);
      case AttrType::BLOCK: {
        auto i = pb_desc->GetAttrIfExists<int32_t>(name);
        cpp_desc->SetAttr<int32_t>(name, i);
        break;
      }
      default:
        PADDLE_THROW(platform::errors::NotFound(
            "Unsupported attr type %d found ", static_cast<int>(type)));
    }
  };
#undef IMPL_ONE

  for (const auto &attr_name : pb_desc->AttrNames()) {
    auto type = pb_desc->GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void OpAttrsFromCinn(const cpp::OpDesc &cpp_desc, framework::OpDesc *pb_desc) {
  pb_desc->MutableAttrMap()->clear();
  using AttrType = cpp::OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                            \
  case AttrType::type__:                               \
    pb_desc->SetAttr(name, cpp_desc.GetAttr<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      IMPL_ONE(LONG, int64_t);
      IMPL_ONE(LONGS, std::vector<int64_t>);
      default:
        PADDLE_THROW(platform::errors::NotFound(
            "Unsupported attr type %d found ", static_cast<int>(type)));
    }
  };
#undef IMPL_ONE

  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void TransformOpDescToCinn(framework::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  cpp_desc->SetType(pb_desc->Type());
  OpInputsToCinn(pb_desc, cpp_desc);
  OpOutputsToCinn(pb_desc, cpp_desc);
  OpAttrsToCinn(pb_desc, cpp_desc);
}

void TransformOpDescFromCinn(const cpp::OpDesc &cpp_desc,
                             framework::OpDesc *pb_desc) {
  pb_desc->Proto()->Clear();
  pb_desc->SetType(cpp_desc.Type());
  OpInputsFromCinn(cpp_desc, pb_desc);
  OpOutputsFromCinn(cpp_desc, pb_desc);
  OpAttrsFromCinn(cpp_desc, pb_desc);
}

/// For BlockDesc transform
void TransformBlockDescToCinn(framework::BlockDesc *pb_desc,
                              cpp::BlockDesc *cpp_desc) {
  cpp_desc->SetIdx(pb_desc->ID());
  cpp_desc->SetParentIdx(pb_desc->Parent());
  cpp_desc->SetForwardBlockIdx(pb_desc->ForwardBlockID());

  cpp_desc->ClearOps();
  const auto &all_ops = pb_desc->AllOps();
  for (const auto &op : all_ops) {
    auto *cpp_op_desc = cpp_desc->AddOp<cpp::OpDesc>();
    TransformOpDescToCinn(op, cpp_op_desc);
  }

  cpp_desc->ClearVars();
  const auto &all_vars = pb_desc->AllVars();
  for (const auto &var : all_vars) {
    auto *cpp_var_desc = cpp_desc->AddVar<cpp::VarDesc>();
    TransformVarDescToCinn(var, cpp_var_desc);
  }
}

void TransformBlockDescFromCinn(const cpp::BlockDesc &cpp_desc,
                                framework::BlockDesc *pb_desc) {
  pb_desc->Proto()->Clear();

  pb_desc->Proto()->set_idx(cpp_desc.Idx());
  pb_desc->Proto()->set_parent_idx(cpp_desc.ParentIdx());
  pb_desc->Proto()->set_forward_block_idx(cpp_desc.ForwardBlockIdx());

  for (size_t i = 0; i < cpp_desc.OpsSize(); ++i) {
    const auto &cpp_op_desc =
        cpp_desc.template GetConstOp<cpp::OpDesc>(static_cast<int32_t>(i));
    auto *pb_op_desc = pb_desc->AppendOp();
    TransformOpDescFromCinn(cpp_op_desc, pb_op_desc);
  }

  for (size_t i = 0; i < cpp_desc.VarsSize(); ++i) {
    const auto &cpp_var_desc =
        cpp_desc.template GetConstVar<cpp::VarDesc>(static_cast<int32_t>(i));
    auto *pb_var_desc = pb_desc->Var(cpp_var_desc.Name());
    TransformVarDescFromCinn(cpp_var_desc, pb_var_desc);
  }
}

/// For ProgramDesc transform
void TransformProgramDescToCinn(framework::ProgramDesc *pb_desc,
                                cpp::ProgramDesc *cpp_desc) {
  if (pb_desc->Proto()->version().has_version()) {
    cpp_desc->SetVersion(pb_desc->Version());
  }

  cpp_desc->ClearBlocks();
  for (size_t i = 0; i < pb_desc->Size(); ++i) {
    auto *pb_block_desc = pb_desc->MutableBlock(i);
    auto *cpp_block_desc = cpp_desc->AddBlock<cpp::BlockDesc>();
    TransformBlockDescToCinn(pb_block_desc, cpp_block_desc);
  }
}

void TransformProgramDescFromCinn(const cpp::ProgramDesc &cpp_desc,
                                  framework::ProgramDesc *pb_desc) {
  pb_desc->Proto()->Clear();

  if (cpp_desc.HasVersion()) {
    pb_desc->SetVersion(cpp_desc.Version());
  }

  // For paddle proto program, the only way to add block is invoke
  // AppendBlock(),
  // the AppendBlock need one necessary parameter: const BlockDesc &parent,
  // but the only function of parent is set the block's parent_idx value.
  // Meanwhile a program has at least one block, so we set block0 to all
  // sub-block's parent in initial and cannot remove.
  // Don't worry, it will be change in "TransformBlockDescFromCinn".
  auto *block0 = pb_desc->MutableBlock(0);

  for (size_t i = 0; i < cpp_desc.BlocksSize(); ++i) {
    const auto &cpp_block_desc = cpp_desc.GetConstBlock<cpp::BlockDesc>(i);
    framework::BlockDesc *pb_block_desc = nullptr;
    if (i < pb_desc->Size()) {
      pb_block_desc = pb_desc->MutableBlock(i);
    } else {
      pb_block_desc = pb_desc->AppendBlock(*block0);
    }
    TransformBlockDescFromCinn(cpp_block_desc, pb_block_desc);
  }
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
