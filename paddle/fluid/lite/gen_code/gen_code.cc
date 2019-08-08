// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/gen_code/gen_code.h"
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace gencode {

void Module::AddWeight(const std::string &name, const TensorRepr &tensor) {
  auto w_name = WeightUniqueName();
  Line(string_format("// Create weight: %s", name.c_str()));
  // auto* w0 = scope.Var("w0")->GetMutable<lite::Tensor>();
  Line(string_format("auto* %s = scope->Var(%s)->GetMutable<lite::Tensor>();",
                     w_name.c_str(), Repr(name).c_str()));
  // lite::DDim w_ddim({1, 2})
  Line(string_format("lite::DDim %s_ddim(std::vector<int64_t>(%s));",
                     w_name.c_str(), tensor.ddim.repr().c_str()));
  // std::vector<float> w_data({});
  auto w_data_repr = DataRepr(
      std::string(static_cast<const char *>(tensor.raw_data), tensor.num_bytes),
      tensor.dtype);
  Line(string_format("std::vector<%s> %s_data({%s});",
                     PrecisionToStr(tensor.dtype).c_str(), w_name.c_str(),
                     w_data_repr.c_str()));
  // w0->Assign<float, lite::DDim, TARGET(kX86)>(w0_data.data(), w0_ddim);
  Line(string_format(
      "%s->Assign<%s, lite::DDim, TARGET(kX86)>(%s_data.data(), %s_ddim);",
      w_name.c_str(), PrecisionToStr(tensor.dtype).c_str(), w_name.c_str(),
      w_name.c_str()));
  Line("");
}

void Module::AddHeaderIncludeGenCode() {
  Line("");
  Line("#include <string>");
  Line("#include <vector>");
  Line("#include \"paddle/fluid/lite/core/compatible_tensor.h\"");
  Line("#include \"paddle/fluid/lite/core/context.h\"");
  Line("#include \"paddle/fluid/lite/gen_code/paddle_infer.h\"");
  Line("#include \"paddle/fluid/lite/core/op_registry.h\"");
  Line("#include \"paddle/fluid/lite/core/scope.h\"");
  Line("#include \"paddle/fluid/lite/model_parser/cpp/op_desc.h\"");
  Line("");
  Line("");
}

std::string Module::DataRepr(const std::string &raw_data, PrecisionType dtype) {
  std::stringstream ss;
  switch (dtype) {
    case PRECISION(kFloat): {
      const float *raw = reinterpret_cast<const float *>(raw_data.c_str());
      int num_elems = raw_data.size() / sizeof(float);
      if (num_elems) {
        for (int i = 0; i < num_elems - 1; i++) {
          ss << raw[i] << ",";
        }
        ss << raw[num_elems - 1];
      }
    } break;

    default:
      LOG(FATAL) << "Unsupported type " << PrecisionToStr(dtype);
  }
  return ss.str();
}

void Module::AddOpDescHelper(const std::string &op_id,
                             const cpp::OpDesc &desc) {
  std::string desc_var = op_id + "_desc";
  Line(string_format("lite::cpp::OpDesc %s;", desc_var.c_str()));
  auto vec_str_repr = [](const std::vector<std::string> &vec) {
    return Repr(vec);
  };
  for (auto &item : desc.inputs()) {
    Line(string_format("%s.SetInput(%s, %s);", desc_var.c_str(),
                       Repr(item.first).c_str(),
                       vec_str_repr(item.second).c_str()));
  }

  for (auto &item : desc.outputs()) {
    Line(string_format("%s.SetOutput(%s, %s);", desc_var.c_str(),
                       Repr(item.first).c_str(),
                       vec_str_repr(item.second).c_str()));
  }

  auto attr_repr = [&](const std::string &name) -> std::string {
    using AttrType = OpDescAPI::AttrType;
    auto type = desc.GetAttrType(name);

    switch (type) {
      case AttrType::INT:
        return std::to_string(desc.GetAttr<int>(name));
      case AttrType::FLOAT:
        return std::to_string(desc.GetAttr<float>(name));
      case AttrType::BOOLEAN:
        return std::to_string(desc.GetAttr<bool>(name));
      case AttrType::STRING:
        return "\"" + desc.GetAttr<std::string>(name) + "\"";
      case AttrType::FLOATS: {
        auto vals = desc.GetAttr<std::vector<float>>(name);
        return "{" + Join(vals, ",") + "}";
      }
      case AttrType::INTS: {
        auto vals = desc.GetAttr<std::vector<int>>(name);
        return "{" + Join(vals, ",") + "}";
      }

      case AttrType::STRINGS: {
        std::vector<std::string> tmp;
        auto vals = desc.GetAttr<std::vector<std::string>>(name);
        std::transform(vals.begin(), vals.end(), std::back_inserter(tmp),
                       [](const std::string &x) { return Repr(x); });
        return "{" + Join(tmp, ",") + "}";
      }
      default:
        LOG(FATAL) << "Unsupported attribute type: " << static_cast<int>(type);
    }
    return "";
  };

  auto attr_type_repr = [&](const std::string &name) -> std::string {
    using AttrType = OpDescAPI::AttrType;
    auto type = desc.GetAttrType(name);

    switch (type) {
      case AttrType::INT:
        return "int";
      case AttrType::FLOAT:
        return "float";
      case AttrType::BOOLEAN:
        return "bool";
      case AttrType::STRING:
        return "std::string";
      case AttrType::FLOATS:
        return "std::vector<float>";
      case AttrType::STRINGS:
        return "std::vector<std::string>";
      case AttrType::INTS:
        return "std::vector<int>";
      default:
        LOG(FATAL) << "Unsupported attribute type: " << static_cast<int>(type);
    }

    return "unk_t";
  };
  for (auto &item : desc.AttrNames()) {
    // Drop the python information.
    if (item == "op_callstack") continue;
    auto attr_type = attr_type_repr(item);
    auto attr_val = attr_repr(item);
    Line(string_format("%s.SetAttr<%s>(%s, %s);",  //
                       desc_var.c_str(), attr_type.c_str(), Repr(item).c_str(),
                       attr_val.c_str()));
  }
}

void Module::AddOp(const cpp::OpDesc &op) {
  auto op_name = OpUniqueName();
  AddOpDescHelper(op_name, op);

  LOG(INFO) << "add op " << op_name;

  Line(string_format("// Create Op: %s", op.Type().c_str()));

  Line(string_format("auto %s = lite::LiteOpRegistry::Global().Create(\"%s\");",
                     op_name.c_str(), op.Type().c_str()));

  CHECK(op.HasAttr(kKernelTypeAttr))
      << "the kernel type should be specified before generate code.";
  auto kernel_type = op.GetAttr<std::string>(kKernelTypeAttr);
  Line(string_format("%s->Attach(%s, exec_scope);", op_name.c_str(),
                     (op_name + "_desc").c_str()));

  // Create kernel
  auto kernel_name = KernelUniqueName();
  Line(string_format(
      "auto %s = std::move(%s->CreateKernels(valid_places, \"%s\").front());",
      kernel_name.c_str(), op_name.c_str(), kernel_type.c_str()));

  // Set Context for kernel
  // clang-format off
  Line(string_format("%s->SetContext(lite::ContextScheduler::Global().NewContext(%s->target()));", kernel_name.c_str(), kernel_name.c_str()));  // NOLINT
  // clang-format on

  Line(string_format("ops.push_back(%s);", op_name.c_str()));
  Line(string_format("kernels.push_back(std::move(%s));", kernel_name.c_str()));

  op_kinds_.insert(op.Type());
  kernel_kinds_.insert(kernel_type);
}
}  // namespace gencode
}  // namespace lite
}  // namespace paddle
