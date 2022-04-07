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

#pragma once

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_HANDLER(op_type, handler)                         \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                  \
      __reg_ipu_op_handler__##op_type,                             \
      "REGISTER_HANDLER must be called in global namespace");      \
  struct __PaddleRegisterIpuOpHandler_##op_type {                  \
    __PaddleRegisterIpuOpHandler_##op_type() {                     \
      ::paddle::platform::ipu::RegisterHandler(                    \
          #op_type, paddle::platform::ipu::handler);               \
    }                                                              \
    int Touch() const { return 0; }                                \
  };                                                               \
  static __PaddleRegisterIpuOpHandler_##op_type                    \
      __PaddleRegisterIpuOpHandler_instance##op_type;              \
  int TouchPaddleIpuOpHandlerRegister_##op_type() {                \
    return __PaddleRegisterIpuOpHandler_instance##op_type.Touch(); \
  }

#define USE_HANDLER(op_type)                              \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                         \
      __use_ipu_op_handler__##op_type,                    \
      "USE_HANDLER must be called in global namespace");  \
  extern int TouchPaddleIpuOpHandlerRegister_##op_type(); \
  UNUSED static int use_handler__itself_##op_type##_ =    \
      TouchPaddleIpuOpHandlerRegister_##op_type()

using SymbolHandler = std::function<Node *(Graph *, Node *)>;

std::unordered_map<std::string, SymbolHandler> &SymbolHandlers();

bool RegisterHandler(const std::string &, const SymbolHandler &);

SymbolHandler GetHandler(const std::string &);

void ConnectNodes(Node *first_node, Node *next_node);
void DisConnectNodes(Node *first_node, Node *next_node);
void ClearNode(Node *node);
void CopyOpAttr(const std::string &attr_name, OpDesc *op, OpDesc *new_op,
                bool override = false);

const int VarType2OnnxDtype(const int type);
const std::string VarType2PopStr(const int type);

Node *GetInputVarNode(const std::string &input_name, const Node *op_node,
                      const int id = 0);
Node *GetOutputVarNode(const std::string &output_name, const Node *op_node,
                       const int id = 0);
Node *GetInputVarNodeByVarName(const std::string &var_name,
                               const Node *op_node);
Node *GetOutputVarNodeByVarName(const std::string &var_name,
                                const Node *op_node);

const bool is_float_equal(float a, float b, float eps = 1e-8);
const int GetOutputVarDtype(const Node *node,
                            const std::string &output_name = "Out");

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
