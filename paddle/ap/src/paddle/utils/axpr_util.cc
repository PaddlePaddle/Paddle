// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/utils/axpr_util.h"
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/common/enforce.h"

namespace ap::paddle {

namespace {

class AxprValueImpl : public AxprValueImplBase {
 public:
  explicit AxprValueImpl(const axpr::Value& value) : value_(value) {}
  ~AxprValueImpl() override = default;

  const axpr::Value& value() const { return value_; }

 private:
  axpr::Value value_;
};

AxprValue ToAxprValueFacade(const axpr::Value& axpr_value) {
  std::shared_ptr<AxprValueImplBase> impl{
      std::make_unique<AxprValueImpl>(axpr_value)};
  return AxprValue{impl};
}

adt::Result<axpr::Value> FromAxprValueFacade(const AxprValue& axpr_value) {
  ADT_CHECK(dynamic_cast<AxprValueImpl*>(&*axpr_value.value) != nullptr);
  return dynamic_cast<AxprValueImpl*>(&*axpr_value.value)->value();
}

AxprValue AxprValueNoneImpl() { return ToAxprValueFacade(axpr::Nothing{}); }
AxprValue AxprValueFromBoolImpl(bool data) { return ToAxprValueFacade(data); }
AxprValue AxprValueFromIntImpl(int64_t data) { return ToAxprValueFacade(data); }
AxprValue AxprValueFromFloatImpl(double data) {
  return ToAxprValueFacade(data);
}
AxprValue AxprValueFromStrImpl(const std::string& data) {
  return ToAxprValueFacade(data);
}
adt::Result<adt::List<axpr::Value>> AxprListFromListImpl(
    const std::vector<AxprValue>& data) {
  adt::List<axpr::Value> ret;
  ret->reserve(data.size());
  for (const auto& elt : data) {
    ADT_LET_CONST_REF(value, FromAxprValueFacade(elt));
    ret->emplace_back(value);
  }
  return ret;
}
adt::Result<AxprValue> AxprValueFromListImpl(
    const std::vector<AxprValue>& data) {
  ADT_LET_CONST_REF(ret, AxprListFromListImpl(data));
  return ToAxprValueFacade(ret);
}
adt::Result<bool> AxprValueIsNoneImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastableTo<adt::Nothing>();
}
adt::Result<bool> AxprValueIsBoolImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastableTo<bool>();
}
adt::Result<bool> AxprValueIsIntImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastableTo<int64_t>();
}
adt::Result<bool> AxprValueIsFloatImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastableTo<double>();
}
adt::Result<bool> AxprValueIsStrImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastableTo<std::string>();
}
adt::Result<bool> AxprValueIsListImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return axpr::AbstractList<axpr::Value>::CastableFrom(value);
}
adt::Result<bool> AxprValueToBoolImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastTo<bool>();
}
adt::Result<int64_t> AxprValueToIntImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastTo<int64_t>();
}
adt::Result<double> AxprValueToFloatImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastTo<double>();
}
adt::Result<std::string> AxprValueToStrImpl(const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  return value.template CastTo<std::string>();
}
adt::Result<std::vector<AxprValue>> AxprValueToListImpl(
    const AxprValue& axpr_value) {
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(value));
  ADT_LET_CONST_REF(lst_size, lst.size());
  std::vector<AxprValue> ret;
  ret.reserve(lst_size);
  for (int i = 0; i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt, lst.at(i));
    ret.emplace_back(ToAxprValueFacade(elt));
  }
  return ret;
}
adt::Result<AxprValue> AxprValueImportModuleImpl(
    const std::string& module_name) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr =
        lmd.Lambda({"module_name"}, [&](auto& ctx) {
          return ctx.Var("__builtin__import").Call(ctx.Var("module_name"));
        });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      ret,
      interpreter.Interpret(lambda, std::vector<axpr::Value>{module_name}));
  return ToAxprValueFacade(ret);
}
adt::Result<AxprValue> AxprValueTypeImpl(const AxprValue& axpr_value) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda({"value"}, [&](auto& ctx) {
      return ctx.Var("type").Call(ctx.Var("value"));
    });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      ret, interpreter.Interpret(lambda, std::vector<axpr::Value>{value}));
  return ToAxprValueFacade(ret);
}
adt::Result<AxprValue> AxprValueGetAttrImpl(const AxprValue& axpr_value,
                                            const std::string& attr_name) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr =
        lmd.Lambda({"value", "attr_name"}, [&](auto& ctx) {
          return ctx.Var("getattr").Call(ctx.Var("value"),
                                         ctx.Var("attr_name"));
        });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(ret,
                    interpreter.Interpret(
                        lambda, std::vector<axpr::Value>{value, attr_name}));
  return ToAxprValueFacade(ret);
}
adt::Result<AxprValue> AxprValueGetItemImpl(const AxprValue& axpr_value,
                                            const AxprValue& item_name) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda(
        {"value", "item"},
        [&](auto& ctx) { return ctx.Var("value").At(ctx.Var("item")); });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  ADT_LET_CONST_REF(item, FromAxprValueFacade(item_name));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      ret,
      interpreter.Interpret(lambda, std::vector<axpr::Value>{value, item}));
  return ToAxprValueFacade(ret);
}
adt::Result<AxprValue> AxprValueCallImpl(
    const AxprValue& axpr_value, const std::vector<AxprValue>& axpr_args) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr =
        lmd.Lambda({"func", "args"}, [&](auto& ctx) {
          return ctx.Var("__builtin__apply")
              .Call(ctx.Var("func"), ctx.Var("args"));
        });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  ADT_LET_CONST_REF(args, AxprListFromListImpl(axpr_args));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      ret,
      interpreter.Interpret(lambda, std::vector<axpr::Value>{value, args}));
  return ToAxprValueFacade(ret);
}
adt::Result<std::string> AxprValueStrImpl(const AxprValue& axpr_value) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda({"value"}, [&](auto& ctx) {
      return ctx.Var(axpr::kBuiltinToString()).Call(ctx.Var("value"));
    });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      str_value,
      interpreter.Interpret(lambda, std::vector<axpr::Value>{value}));
  ADT_LET_CONST_REF(str, str_value.template CastTo<std::string>());
  return str;
}
adt::Result<int64_t> AxprValueLenImpl(const AxprValue& axpr_value) {
  static axpr::Lambda<axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd{};
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda({"value"}, [&](auto& ctx) {
      return ctx.Var(axpr::kBuiltinLength()).Call(ctx.Var("value"));
    });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ADT_LET_CONST_REF(value, FromAxprValueFacade(axpr_value));
  static ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(
      len_value,
      interpreter.Interpret(lambda, std::vector<axpr::Value>{value}));
  ADT_LET_CONST_REF(len, len_value.template CastTo<int64_t>());
  return len;
}

}  // namespace

AxprValue AxprValueNone() { return AxprValueNoneImpl(); }
AxprValue AxprValueFromBool(bool data) { return AxprValueFromBoolImpl(data); }
AxprValue AxprValueFromInt(int64_t data) { return AxprValueFromIntImpl(data); }
AxprValue AxprValueFromFloat(double data) {
  return AxprValueFromFloatImpl(data);
}
AxprValue AxprValueFromStr(const std::string& data) {
  return AxprValueFromStrImpl(data);
}
AxprValue AxprValueFromList(const std::vector<AxprValue>& data) {
  const auto& ret = AxprValueFromListImpl(data);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueFromList failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsNone(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsNoneImpl(axpr_value);
  PADDLE_ENFORCE_EQ(ret.HasOkValue(),
                    true,
                    phi::errors::Fatal(
                        "AxprValueIsNone failed. \nTraceback (most recent call "
                        "last):\n%s\n%s: %s. ",
                        ret.GetError().CallStackToString(),
                        ret.GetError().class_name(),
                        ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsBool(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsBoolImpl(axpr_value);
  PADDLE_ENFORCE_EQ(ret.HasOkValue(),
                    true,
                    phi::errors::Fatal(
                        "AxprValueIsBool failed. \nTraceback (most recent call "
                        "last):\n%s\n%s: %s. ",
                        ret.GetError().CallStackToString(),
                        ret.GetError().class_name(),
                        ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsInt(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsIntImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueIsInt failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsFloat(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsFloatImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueIsFloat failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsStr(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsStrImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueIsStr failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueIsList(const AxprValue& axpr_value) {
  const auto& ret = AxprValueIsListImpl(axpr_value);
  PADDLE_ENFORCE_EQ(ret.HasOkValue(),
                    true,
                    phi::errors::Fatal(
                        "AxprValueIsList failed. \nTraceback (most recent call "
                        "last):\n%s\n%s: %s. ",
                        ret.GetError().CallStackToString(),
                        ret.GetError().class_name(),
                        ret.GetError().msg()));
  return ret.GetOkValue();
}
bool AxprValueToBool(const AxprValue& axpr_value) {
  const auto& ret = AxprValueToBoolImpl(axpr_value);
  PADDLE_ENFORCE_EQ(ret.HasOkValue(),
                    true,
                    phi::errors::Fatal(
                        "AxprValueToBool failed. \nTraceback (most recent call "
                        "last):\n%s\n%s: %s. ",
                        ret.GetError().CallStackToString(),
                        ret.GetError().class_name(),
                        ret.GetError().msg()));
  return ret.GetOkValue();
}
int64_t AxprValueToInt(const AxprValue& axpr_value) {
  const auto& ret = AxprValueToIntImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueToInt failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
double AxprValueToFloat(const AxprValue& axpr_value) {
  const auto& ret = AxprValueToFloatImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueToFloat failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
std::string AxprValueToStr(const AxprValue& axpr_value) {
  const auto& ret = AxprValueToStrImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueToStr failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
std::vector<AxprValue> AxprValueToList(const AxprValue& axpr_value) {
  const auto& ret = AxprValueToListImpl(axpr_value);
  PADDLE_ENFORCE_EQ(ret.HasOkValue(),
                    true,
                    phi::errors::Fatal(
                        "AxprValueToList failed. \nTraceback (most recent call "
                        "last):\n%s\n%s: %s. ",
                        ret.GetError().CallStackToString(),
                        ret.GetError().class_name(),
                        ret.GetError().msg()));
  return ret.GetOkValue();
}
AxprValue AxprValueImportModule(const std::string& module_name) {
  const auto& ret = AxprValueImportModuleImpl(module_name);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueImportModule failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
AxprValue AxprValueType(const AxprValue& axpr_value) {
  const auto& ret = AxprValueTypeImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueType failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
AxprValue AxprValueGetAttr(const AxprValue& axpr_value,
                           const std::string& attr_name) {
  const auto& ret = AxprValueGetAttrImpl(axpr_value, attr_name);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueGetAttr failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
AxprValue AxprValueGetItem(const AxprValue& axpr_value,
                           const AxprValue& item_name) {
  const auto& ret = AxprValueGetItemImpl(axpr_value, item_name);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal(
          "AxprValueGetItem failed. \nTraceback (most recent call "
          "last):\n%s\n%s: %s. ",
          ret.GetError().CallStackToString(),
          ret.GetError().class_name(),
          ret.GetError().msg()));
  return ret.GetOkValue();
}
AxprValue AxprValueCall(const AxprValue& axpr_value,
                        const std::vector<AxprValue>& args) {
  const auto& ret = AxprValueCallImpl(axpr_value, args);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueCall failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
std::string AxprValueStr(const AxprValue& axpr_value) {
  const auto& ret = AxprValueStrImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueStr failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}
int64_t AxprValueLen(const AxprValue& axpr_value) {
  const auto& ret = AxprValueLenImpl(axpr_value);
  PADDLE_ENFORCE_EQ(
      ret.HasOkValue(),
      true,
      phi::errors::Fatal("AxprValueLen failed. \nTraceback (most recent call "
                         "last):\n%s\n%s: %s. ",
                         ret.GetError().CallStackToString(),
                         ret.GetError().class_name(),
                         ret.GetError().msg()));
  return ret.GetOkValue();
}

}  // namespace ap::paddle
