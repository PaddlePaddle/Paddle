/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace platform {
namespace gcu {
static bool IsPersistant(const std::string& var_name) {
  // TODO(xuelei.wan ): ctrl var?
  return var_name.find("tmp") == std::string::npos &&
         var_name.find("auto_") == std::string::npos &&
         var_name.find("generated_") == std::string::npos &&
         var_name.find("@GRAD") == std::string::npos;
}

static std::vector<int64_t> GetTensorShape(const Tensor* pdtensor) {
  std::vector<int64_t> shape;
  for (int i = 0; i < pdtensor->dims().size(); i++) {
    shape.push_back(pdtensor->dims().at(i));
  }
  return shape;
}

static LayoutType ChooseGcuKernelType(GraphPtr graph) {
  // only single op graph is valid
  // prefer choose if match the principle of continuity prefer
  // channellast prefer if not match
  // if not register clast ops, cfirst ops prefer and last selection is
  // insensetive kernel
  for (Node* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto op_desc = node->Op();
      auto op_type = op_desc->Type();

      if (EquivalenceTransformer::GetInstance().Get(op_type, CHANNELLAST)) {
        return CHANNELLAST;
      } else if (EquivalenceTransformer::GetInstance().Get(op_type,
                                                           CHANNELFIRST)) {
        return CHANNELFIRST;
      }
      return INSENSITIVE;
    }
  }
  return INSENSITIVE;
}

static int InplaceTransposeOnGcu(const framework::ExecutionContext& ctx,
                                 Variable* p_var,
                                 const std::string& var_name,
                                 const Layout& layout) {
  if (!p_var->IsInitialized()) {
    VLOG(3) << "var " << var_name << " is not init when do trans.skip it!";
    return 0;
  }
  auto tensor = p_var->GetMutable<Tensor>();
  if (tensor->dims().size() < 4) {
    VLOG(3) << "var:" << var_name << " addr:" << reinterpret_cast<void*>(tensor)
            << " rank[%zu] is smaller than 4, skip trans format!"
            << tensor->dims().size();
    return 0;
  } else {
    std::vector<int64_t> org_shape = GetTensorShape(tensor);
    std::vector<int64_t> transed_shape;
    Layout org_layout;
    auto already_transed_gcu_tensor =
        GcuTensorTable::GetInstance()->GetTransedGcuTensor(
            reinterpret_cast<void*>(tensor));
    if (!already_transed_gcu_tensor) {
      org_layout = GetFormatByPdShape(org_shape);
      transed_shape = TransShapeByFormat(
          org_shape, LayoutToString(org_layout), LayoutToString(layout));
    } else {
      org_layout = already_transed_gcu_tensor->GetFormat();
      transed_shape = already_transed_gcu_tensor->GetShape();
      VLOG(1) << "[warn]var:" << var_name
              << "tensor addr:" << reinterpret_cast<void*>(tensor)
              << " has already do trans, but now trans again! curr format is :"
              << LayoutToString(org_layout)
              << " target format is:" << LayoutToString(layout);
    }

    auto transed_tensor = std::make_shared<Tensor>();
    transed_tensor->Resize(phi::make_ddim(transed_shape));
    transed_tensor->mutable_data(ctx.device_context().GetPlace(),
                                 tensor->dtype());

    auto gcu_src_tensor = std::make_shared<paddle::platform::gcu::GcuTensor>(
        org_shape,
        org_layout,
        paddle::platform::gcu::TransformUtil::ConvertDataType(tensor->dtype()));
    gcu_src_tensor->SetData(const_cast<void*>(tensor->data()));
    auto gcu_trans_tensor = std::make_shared<paddle::platform::gcu::GcuTensor>(
        transed_shape,
        layout,
        paddle::platform::gcu::TransformUtil::ConvertDataType(
            transed_tensor->dtype()));
    gcu_trans_tensor->SetData(const_cast<void*>(transed_tensor->data()));
    auto perm =
        GetPermByFormat(LayoutToString(org_layout), LayoutToString(layout));
    paddle::platform::gcu::GcuTranspose(
        *gcu_src_tensor, *gcu_trans_tensor, perm);
    VLOG(3) << "var:" << var_name << " addr:" << reinterpret_cast<void*>(tensor)
            << " trans format success from " << LayoutToString(org_layout)
            << " to " << LayoutToString(layout);
    paddle::framework::TensorCopySync(
        *transed_tensor, ctx.device_context().GetPlace(), tensor);
    GcuTensorTable::GetInstance()->BuffTensor(reinterpret_cast<void*>(tensor),
                                              gcu_trans_tensor,
                                              IsPersistant(var_name));
  }
  return 1;
}

static bool JudgeAndGet(const std::vector<VarNameValuePair>& v,
                        const std::string& name,
                        Variable*& var_ptr) {  // NOLINT
  for (const auto& pair : v) {
    if (name == pair.first) {
      var_ptr = pair.second;
      return true;
    }
  }
  var_ptr = nullptr;
  return false;
}

static void InputsLayoutProcess(
    const framework::ExecutionContext& ctx,
    const std::vector<VarNameValuePair>& input_vars,
    const std::vector<VarNameValuePair>& input_params,
    Graph* graph) {
  Variable* p_var = nullptr;
  auto op_type = ctx.Type();
  VLOG(6) << "==== " << op_type << " start inputs layout process ===";
  auto op_layout_type = ChooseGcuKernelType(graph);
  if (op_layout_type == CHANNELLAST) {
    auto op_infos =
        paddle::platform::gcu::EquivalenceTransformer::GetInstance().GetOpInfo(
            op_type, op_layout_type);
    if (op_infos.ins_layouts.empty() && op_infos.ins_layouts.empty()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s registered op info when register CHANNELLAST gcu op!",
          op_type.c_str()));
    }
    //
    for (Node* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto op_desc = node->Op();
        for (const auto& e : op_desc->Inputs()) {
          op_desc->SetAttr(kGcuLayoutType, static_cast<int>(op_layout_type));
          auto archetype = e.first;
          auto layout = op_infos.ins_layouts[archetype];
          for (auto& var_name : e.second) {
            if (JudgeAndGet(input_vars, var_name, p_var)) {
              if (InplaceTransposeOnGcu(ctx, p_var, var_name, layout) == 0) {
                continue;
              }
              // modify graph input var
              bool is_in = false;
              for (auto& in : node->inputs) {
                if (in->Name() == var_name) {
                  auto transed_gcu_tensor =
                      GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                          reinterpret_cast<void*>(p_var->GetMutable<Tensor>()));
                  PADDLE_ENFORCE_NE(transed_gcu_tensor,
                                    nullptr,
                                    platform::errors::Fatal(
                                        "var %s addr is %p has no transed op",
                                        var_name.c_str(),
                                        p_var->GetMutable<Tensor>()));
                  in->Var()->SetShape(transed_gcu_tensor->GetShape());
                  is_in = true;
                  break;
                }
              }
              PADDLE_ENFORCE_EQ(is_in,
                                true,
                                platform::errors::Fatal(
                                    "var %s is not in op inputs list when "
                                    "modify graph input shape on do transpose!",
                                    var_name.c_str()));
            } else if (JudgeAndGet(input_params, var_name, p_var)) {
              if (InplaceTransposeOnGcu(ctx, p_var, var_name, layout) == 0) {
                continue;
              }
              // modify graph input var
              bool is_in = false;
              for (auto& in : node->inputs) {
                if (in->Name() == var_name) {
                  auto transed_gcu_tensor =
                      GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                          reinterpret_cast<void*>(p_var->GetMutable<Tensor>()));
                  PADDLE_ENFORCE_NE(transed_gcu_tensor,
                                    nullptr,
                                    platform::errors::Fatal(
                                        "var %s addr is %p has no transed op",
                                        var_name.c_str(),
                                        p_var->GetMutable<Tensor>()));
                  in->Var()->SetShape(transed_gcu_tensor->GetShape());
                  is_in = true;
                  break;
                }
              }
              PADDLE_ENFORCE_EQ(is_in,
                                true,
                                platform::errors::Fatal(
                                    "var %s is not in op inputs list when "
                                    "modify graph input shape on do transpose!",
                                    var_name.c_str()));
            }
          }
        }
      }
    }
  } else if (op_layout_type == CHANNELFIRST) {
  } else {
    // establish main format
    // if meet mis-match format, norm according to the first format
    bool is_capture = false;
    Layout op_format = Layout::NCHW;
    for (Node* node : graph->Nodes()) {
      if (!node->IsOp()) continue;
      auto op_desc = node->Op();
      if (!op_desc->HasAttr(kGcuLayoutType)) {
        op_desc->SetAttr(kGcuLayoutType, static_cast<int>(op_layout_type));
      }
      for (const auto& e : op_desc->Inputs()) {
        auto archetype = e.first;
        for (auto& var_name : e.second) {
          if (JudgeAndGet(input_vars, var_name, p_var)) {
            auto pd_tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(pd_tensor));
            if (transed_gcu_tensor == nullptr) {
              continue;
            }
            // when get, shape rank must be equal or larger than 4
            op_format = transed_gcu_tensor->GetFormat();
            is_capture = true;
            break;
          } else if (JudgeAndGet(input_params, var_name, p_var)) {
            auto pd_tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(pd_tensor));
            if (transed_gcu_tensor == nullptr) {
              continue;
            }
            // when get, shape rank must be equal or larger than 4
            op_format = transed_gcu_tensor->GetFormat();
            is_capture = true;
            break;
          } else {
            VLOG(1) << "[warn]" << ctx.Type() << " in tensor " << var_name
                    << " has no memory!";
          }
        }
        if (is_capture) {
          op_desc->SetAttr(kGcuMainFormat, static_cast<int>(op_format));
          break;
        }
      }
    }
    if (!is_capture) {
      // no need to do format conflict process
      return;
    }
    // solve conflict between main format and current format
    // Mark output format according to input format
    for (Node* node : graph->Nodes()) {
      if (!node->IsOp()) continue;
      auto op_desc = node->Op();
      op_desc->SetAttr(kGcuLayoutType, static_cast<int>(op_layout_type));
      for (const auto& e : op_desc->Inputs()) {
        auto archetype = e.first;
        for (auto& var_name : e.second) {
          if (JudgeAndGet(input_vars, var_name, p_var)) {
            auto pd_tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(pd_tensor));
            if (transed_gcu_tensor == nullptr) {
              continue;
            }
            if (op_format == transed_gcu_tensor->GetFormat()) {
              continue;
            }
            InplaceTransposeOnGcu(ctx, p_var, var_name, op_format);
          } else if (JudgeAndGet(input_params, var_name, p_var)) {
            auto pd_tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(pd_tensor));
            if (transed_gcu_tensor == nullptr) {
              continue;
            }
            if (op_format == transed_gcu_tensor->GetFormat()) {
              continue;
            }
            InplaceTransposeOnGcu(ctx, p_var, var_name, op_format);
          } else {
            VLOG(1) << "[warn]" << ctx.Type() << " in tensor " << var_name
                    << " has no memory!";
          }
        }
      }
    }
  }
}

static void OutputsLayoutProcess(
    const framework::ExecutionContext& ctx,
    const std::vector<VarNameValuePair>& output_vars,
    Graph* graph) {
  Variable* p_var = nullptr;
  auto op_type = ctx.Type();
  VLOG(6) << "==== " << op_type << " start outputs layout process ===";
  auto op_layout_type = ChooseGcuKernelType(graph);
  if (op_layout_type == CHANNELLAST) {
    auto op_infos =
        paddle::platform::gcu::EquivalenceTransformer::GetInstance().GetOpInfo(
            op_type, op_layout_type);
    if (op_infos.ins_layouts.empty() && op_infos.ins_layouts.empty()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s registered op info when register CHANNELLAST gcu op!",
          op_type.c_str()));
    }
    //
    for (Node* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto op_desc = node->Op();
        for (const auto& e : op_desc->Outputs()) {
          // op_desc->SetAttr(kGcuLayoutType, static_cast<int>(op_layout_type));
          auto archetype = e.first;
          for (auto& var_name : e.second) {
            if (!JudgeAndGet(output_vars, var_name, p_var)) continue;
            if (!p_var->IsInitialized()) {
              // TODO(xuelei.wan ): maybe empty tensor only convert shape
              VLOG(3) << "var " << var_name
                      << " is not init when do trans.skip it!";
              continue;
            }
            auto tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(tensor));
            if (transed_gcu_tensor) {
              auto record_format_str = transed_gcu_tensor->GetFormatStr();
              auto curr_format_str =
                  LayoutToString(op_infos.outs_layouts[archetype]);
              PADDLE_ENFORCE_EQ(
                  record_format_str,
                  curr_format_str,
                  platform::errors::Fatal("var %s saved format is %s curr "
                                          "format is %s, should be same!",
                                          var_name.c_str(),
                                          record_format_str.c_str(),
                                          curr_format_str.c_str()));
              VLOG(1) << "[warn]var:" << var_name
                      << " addr:" << reinterpret_cast<void*>(tensor)
                      << " has already record trans format! curr format is:"
                      << curr_format_str;
            } else {
              auto org_layout =
                  tensor->dims().size() >= 5 ? Layout::NCDHW : Layout::NCHW;
              auto transed_layout = op_infos.outs_layouts[archetype];
              auto transed_shape =
                  TransShapeByFormat(GetTensorShape(tensor),
                                     LayoutToString(org_layout),
                                     LayoutToString(transed_layout));
              transed_gcu_tensor =
                  std::make_shared<paddle::platform::gcu::GcuTensor>(
                      transed_shape,
                      transed_layout,
                      paddle::platform::gcu::TransformUtil::ConvertDataType(
                          tensor->dtype()));
              GcuTensorTable::GetInstance()->BuffTensor(
                  reinterpret_cast<void*>(tensor),
                  transed_gcu_tensor,
                  IsPersistant(var_name));
              VLOG(3) << "var:" << var_name << " op_type:" << op_type
                      << " record trans format info.Transed Layout is: "
                      << LayoutToString(transed_layout);
            }

            // modify graph output var
            bool is_in = false;
            for (auto& out : node->outputs) {
              if (out->Name() == var_name) {
                out->Var()->SetShape(transed_gcu_tensor->GetShape());
                is_in = true;
                break;
              }
            }
            PADDLE_ENFORCE_EQ(is_in,
                              true,
                              platform::errors::Fatal(
                                  "var %s is not in op outputs list when "
                                  "modify graph output shape on do transpose!",
                                  var_name.c_str()));
          }
        }
      }
    }
  } else if (op_layout_type == CHANNELFIRST) {
  } else {
    for (Node* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto op_desc = node->Op();
        if (!op_desc->HasAttr(kGcuMainFormat)) {
          VLOG(3)
              << "op type:" << op_desc->Type()
              << " does not mark main format, skip mark output format process!";
          break;
        }

        auto value = PADDLE_GET_CONST(int, op_desc->GetAttr(kGcuMainFormat));
        auto main_format = static_cast<Layout>(value);
        for (const auto& e : op_desc->Outputs()) {
          auto archetype = e.first;
          for (auto& var_name : e.second) {
            if (!JudgeAndGet(output_vars, var_name, p_var)) continue;
            if (!p_var->IsInitialized()) {
              // TODO(xuelei.wan ): maybe empty tensor only convert shape
              VLOG(3) << "var " << var_name
                      << " is not init when do trans.skip it!";
              continue;
            }
            auto tensor = p_var->GetMutable<Tensor>();
            auto transed_gcu_tensor =
                GcuTensorTable::GetInstance()->GetTransedGcuTensor(
                    reinterpret_cast<void*>(tensor));
            if (transed_gcu_tensor) {
              auto record_format_str = transed_gcu_tensor->GetFormatStr();
              auto curr_format_str = LayoutToString(main_format);
              PADDLE_ENFORCE_EQ(
                  record_format_str,
                  curr_format_str,
                  platform::errors::Fatal("var %s saved format is %s curr "
                                          "format is %s, should be same!",
                                          var_name.c_str(),
                                          record_format_str.c_str(),
                                          curr_format_str.c_str()));
              VLOG(1) << "[warn]var:" << var_name
                      << " addr:" << reinterpret_cast<void*>(tensor)
                      << " has already record trans format! curr format is:"
                      << curr_format_str;
            } else {
              // only over rank 4 need to record output format
              if (tensor->dims().size() < 4) {
                continue;
              }
              auto org_layout =
                  tensor->dims().size() == 5 ? Layout::NCDHW : Layout::NCHW;
              auto transed_shape =
                  TransShapeByFormat(GetTensorShape(tensor),
                                     LayoutToString(org_layout),
                                     LayoutToString(main_format));
              transed_gcu_tensor =
                  std::make_shared<paddle::platform::gcu::GcuTensor>(
                      transed_shape,
                      main_format,
                      paddle::platform::gcu::TransformUtil::ConvertDataType(
                          tensor->dtype()));
              GcuTensorTable::GetInstance()->BuffTensor(
                  reinterpret_cast<void*>(tensor),
                  transed_gcu_tensor,
                  IsPersistant(var_name));
              VLOG(3) << "var:" << var_name << " op_type:" << op_type
                      << " record trans format info.Transed Layout is: "
                      << LayoutToString(main_format);
            }

            // modify graph output var
            bool is_in = false;
            for (auto& out : node->outputs) {
              if (out->Name() == var_name) {
                out->Var()->SetShape(transed_gcu_tensor->GetShape());
                is_in = true;
                break;
              }
            }
            PADDLE_ENFORCE_EQ(is_in,
                              true,
                              platform::errors::Fatal(
                                  "var %s is not in op outputs list when "
                                  "modify graph output shape on do transpose!",
                                  var_name.c_str()));
          }
        }
      }
    }
  }
}
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
