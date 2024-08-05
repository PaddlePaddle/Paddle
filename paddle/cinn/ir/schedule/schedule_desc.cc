// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/schedule/schedule_desc.h"

#include <glog/logging.h>

#include <functional>
#include <typeinfo>
#include <utility>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

// ------ Following codes are about `Apply` functions registry of various types
// of ScheduleDesc::Step
class PackedStepContext;
// uniformed function prototype of a scheduling operation in IRSchedule
using StepApplyFunc = std::vector<Expr> (*)(PackedStepContext*);

// format the inputs, attrs, uniformed function of a scheduling step
class StepKindInfo {
 public:
  // compatible for Registry::EntryType
  std::string name;

  // format: {"<name1>", "<name2>", ...}
  StepKindInfo& Inputs(std::vector<std::string>&& inputs) {
    inputs_ = inputs;
    return *this;
  }
  // format: {"<name1>", "<name2>", ...}
  StepKindInfo& Attrs(std::vector<std::string>&& attrs) {
    attrs_ = attrs;
    return *this;
  }
  // format: APPLY_FUNC_UNIFORM(...)
  StepKindInfo& SetApplyFn(StepApplyFunc&& func) {
    apply_func_ = func;
    return *this;
  }

  // execute the Apply function of this type
  std::vector<Expr> Apply(PackedStepContext* context) const {
    return apply_func_(context);
  }

 private:
  friend class PackedStepContext;

  std::vector<std::string> inputs_;
  std::vector<std::string> attrs_;
  StepApplyFunc apply_func_{nullptr};
};

// StepKindInfo register for all scheduling steps
class StepKindRegistry : public Registry<StepKindInfo> {
 public:
  StepKindRegistry() = default;

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(StepKindRegistry);
};

// PackedStepContext is the param of a uniformed `Apply` function, which is used
// to be an auxiliary structure to interact with in/out arguments of the
// original scheduling function in IRSchedule
class PackedStepContext {
 public:
  explicit PackedStepContext(const ScheduleDesc::Step& desc,
                             const StepKindInfo* step_kind,
                             IRSchedule* schedule)
      : ir_schedule_(schedule) {
    Build(desc, step_kind);
  }

  // get the pointer of current IRSchedule object
  IRSchedule* ScheduleHandler() const { return ir_schedule_; }

  // get the idx-th input whose signature is Expr
  Expr InputAt(size_t idx) const {
    PADDLE_ENFORCE_LT(idx,
                      input_range_.size(),
                      ::common::errors::InvalidArgument("idx overranges"));
    const auto& range = input_range_.at(idx);

    PADDLE_ENFORCE_EQ(range.second - range.first,
                      1,
                      ::common::errors::InvalidArgument(
                          "Input is not single param, idx: %d.", idx));
    return inputs_[range.first];
  }

  // get the idx-th input whose signature is `std::vector<Expr>`
  std::vector<Expr> InputsAt(size_t idx) const {
    PADDLE_ENFORCE_LT(idx,
                      input_range_.size(),
                      ::common::errors::InvalidArgument("idx overranges"));
    const auto& range = input_range_.at(idx);
    std::vector<Expr> results;
    for (size_t s = range.first; s < range.second; ++s) {
      results.emplace_back(inputs_[s]);
    }
    return results;
  }

  // get the idx-th attribute value with correct type
  template <typename AttrType>
  const AttrType& AttrAt(size_t idx) const {
    try {
      return absl::get<AttrType>(attrs_.at(idx));
    } catch (absl::bad_variant_access& ex) {
      std::stringstream ss;
      ss << "Attribute cast error, idx:" << idx
         << ", get type:" << typeid(AttrType).name()
         << ", real index:" << attrs_.at(idx).index();
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      throw ex;
    }
  }

 private:
  void Build(const ScheduleDesc::Step& desc, const StepKindInfo* step_kind) {
    // build inputs
    size_t input_idx = 0;
    for (auto&& param_name : step_kind->inputs_) {
      auto arg_it = desc.inputs.find(param_name);
      PADDLE_ENFORCE_NE(
          arg_it,
          desc.inputs.end(),
          ::common::errors::InvalidArgument(
              "Can't find param: %s while building inputs", param_name));
      auto&& args = arg_it->second;
      inputs_.insert(inputs_.end(),
                     std::make_move_iterator(args.begin()),
                     std::make_move_iterator(args.end()));
      input_range_.emplace_back(input_idx, input_idx + args.size());
      input_idx += args.size();
    }

    // build attrs
    size_t attr_idx = 0;
    for (auto&& attr_name : step_kind->attrs_) {
      auto attr_it = desc.attrs.find(attr_name);
      PADDLE_ENFORCE_NE(attr_it,
                        desc.attrs.end(),
                        ::common::errors::InvalidArgument(
                            "Can't find attribute: %s", attr_name));
      attrs_.emplace_back(attr_it->second);
      ++attr_idx;
    }
  }

  IRSchedule* ir_schedule_;
  std::vector<Expr> inputs_;
  std::vector<std::pair<size_t, size_t>> input_range_;
  std::vector<utils::Attribute> attrs_;
};

#define CINN_SPECIALIZE_ApplyCallHelper(attr_type)                             \
  template <typename... Tail>                                                  \
  struct ApplyCallHelper<attr_type, Tail...> {                                 \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs> \
    static std::vector<Expr> Apply(PackedStepContext* ctx,                     \
                                   PreviousArgs... pargs) {                    \
      using rf_attr_type = std::remove_reference<attr_type>::type;             \
      using rc_attr_type = std::remove_const<rf_attr_type>::type;              \
      const auto& arg = ctx->AttrAt<rc_attr_type>(attr_idx);                   \
      return ApplyCallHelper<Tail...>::                                        \
          template Apply<in_idx, attr_idx + 1, out_idx>(                       \
              ctx, std::forward<PreviousArgs>(pargs)..., arg);                 \
    }                                                                          \
  }

template <typename T>
struct TypeTag {};

// used for converting a member function of the IRSchedule to be a free function
// with the first parameter is a pointer to the IRSchedule.
template <typename F, F f>
struct FreeFuncConverter;

template <typename Return,
          typename... Args,
          Return (IRSchedule::*impl_fn)(Args...)>
struct FreeFuncConverter<Return (IRSchedule::*)(Args...), impl_fn> {
  static Return Apply(IRSchedule* sch, Args... args) {
    return (sch->*impl_fn)(std::forward<Args>(args)...);
  }
};

template <typename Return,
          typename... Args,
          Return (IRSchedule::*impl_fn)(Args...) const>
struct FreeFuncConverter<Return (IRSchedule::*)(Args...) const, impl_fn> {
  static Return Apply(IRSchedule* sch, Args... args) {
    return (sch->*impl_fn)(std::forward<Args>(args)...);
  }
};

// used for formatting scheduling functions with various function signatures to
// be uniformed form
template <typename F, F f>
struct ApplyFuncImpl;

template <typename Return, typename... Args, Return(impl_fn)(Args...)>
struct ApplyFuncImpl<Return (*)(Args...), impl_fn> {
  static std::vector<Expr> Apply(PackedStepContext* ctx) {
    return ApplyCallHelper<Args..., TypeTag<int>>::template Apply<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct ApplyCallHelper;

  // the signature of input parameters of a scheduling operation only can
  // be one of IRSchedule, Expr or std::vector<Expr>
  template <typename... Tail>
  struct ApplyCallHelper<IRSchedule*, Tail...> {
    template <int in_idx, int attr_idx, int out_idx>
    static std::vector<Expr> Apply(PackedStepContext* ctx) {
      static_assert(in_idx == 0, "IRSchedule* must be the first argument");
      IRSchedule* ir_schedule = ctx->ScheduleHandler();
      return ApplyCallHelper<
          Tail...>::template Apply<in_idx + 1, attr_idx, out_idx>(ctx,
                                                                  ir_schedule);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<Expr&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx,
                                   PreviousArgs... pargs) {
      auto arg = ctx->InputAt(in_idx - 1);
      return ApplyCallHelper<Tail...>::
          template Apply<in_idx + 1, attr_idx, out_idx>(
              ctx, std::forward<PreviousArgs>(pargs)..., arg);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const Expr&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx,
                                   PreviousArgs... pargs) {
      auto arg = ctx->InputAt(in_idx - 1);
      return ApplyCallHelper<Tail...>::
          template Apply<in_idx + 1, attr_idx, out_idx>(
              ctx, std::forward<PreviousArgs>(pargs)..., arg);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const std::vector<Expr>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx,
                                   PreviousArgs... pargs) {
      auto arg = ctx->InputsAt(in_idx - 1);
      return ApplyCallHelper<Tail...>::
          template Apply<in_idx + 1, attr_idx, out_idx>(
              ctx, std::forward<PreviousArgs>(pargs)..., arg);
    }
  };

  CINN_SPECIALIZE_ApplyCallHelper(bool);
  CINN_SPECIALIZE_ApplyCallHelper(int);
  CINN_SPECIALIZE_ApplyCallHelper(float);
  CINN_SPECIALIZE_ApplyCallHelper(const std::string&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<bool>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<int>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<float>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<std::string>&);
  CINN_SPECIALIZE_ApplyCallHelper(int64_t);
  CINN_SPECIALIZE_ApplyCallHelper(double);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<int64_t>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<double>&);

  template <int out_idx, typename T>
  struct ApplyReturnHelper;

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, void> {
    static std::vector<Expr> Apply(Args... args) {
      impl_fn(std::forward<Args>(args)...);
      return {};
    }
  };

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, Expr> {
    static std::vector<Expr> Apply(Args... args) {
      auto ret = impl_fn(std::forward<Args>(args)...);
      return {ret};
    }
  };

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, std::vector<Expr>> {
    static std::vector<Expr> Apply(Args... args) {
      return impl_fn(std::forward<Args>(args)...);
    }
  };

  // end: base template
  template <typename T>
  struct ApplyCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx,
                                   PreviousArgs... pargs) {
      static_assert(out_idx == 0, "Output is exported from return value");
      return ApplyReturnHelper<out_idx, Return>::Apply(
          std::forward<Args>(pargs)...);
    }
  };
};

#define APPLY_FUNC_UNIFORM(...) \
  ::cinn::ir::ApplyFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Apply
#define FREE_FUNCTION_CONVERTER(...) \
  ::cinn::ir::FreeFuncConverter<decltype(__VA_ARGS__), __VA_ARGS__>::Apply

#define CINN_BUILD_STEP_KIND(TypeName)                                \
  static ::cinn::ir::StepKindInfo& __step_kind_registrar_##TypeName = \
      ::cinn::ir::StepKindRegistry::Global()->__REGISTER_OR_GET__(#TypeName)

// register StepKindInfo for every type of scheduling operation
CINN_BUILD_STEP_KIND(GetAllBlocks)
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<std::vector<Expr> (IRSchedule::*)() const>(
            &IRSchedule::GetAllBlocks))));

CINN_BUILD_STEP_KIND(GetChildBlocks)
    .Inputs({"expr"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<std::vector<Expr> (IRSchedule::*)(const Expr&) const>(
            &IRSchedule::GetChildBlocks))));

CINN_BUILD_STEP_KIND(GetLoops).Inputs({"block"}).SetApplyFn(
    APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<std::vector<Expr> (IRSchedule::*)(const Expr&) const>(
            &IRSchedule::GetLoops))));

CINN_BUILD_STEP_KIND(GetLoopsWithName)
    .Attrs({"block_name"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<std::vector<Expr> (IRSchedule::*)(const std::string&)
                        const>(&IRSchedule::GetLoops))));

CINN_BUILD_STEP_KIND(GetBlock)
    .Attrs({"block_name"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const std::string&) const>(
            &IRSchedule::GetBlock))));

CINN_BUILD_STEP_KIND(Split)
    .Inputs({"loop", "factors"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<std::vector<Expr> (IRSchedule::*)(
            const Expr&, const std::vector<Expr>&)>(&IRSchedule::Split))));

CINN_BUILD_STEP_KIND(Fuse).Inputs({"loops"}).SetApplyFn(
    APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const std::vector<Expr>&)>(
            &IRSchedule::Fuse))));

CINN_BUILD_STEP_KIND(FuseWithName)
    .Attrs({"block_name", "loops_index"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(
            const std::string&, const std::vector<int>&)>(&IRSchedule::Fuse))));

CINN_BUILD_STEP_KIND(FuseWithBlock)
    .Inputs({"block"})
    .Attrs({"loops_index"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const Expr&, const std::vector<int>&)>(
            &IRSchedule::Fuse))));

CINN_BUILD_STEP_KIND(ComputeAt)
    .Inputs({"block", "loop"})
    .Attrs({"keep_unit_loops"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::ComputeAt)));

CINN_BUILD_STEP_KIND(SimpleComputeAt)
    .Inputs({"block", "loop"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::SimpleComputeAt)));

CINN_BUILD_STEP_KIND(ReverseComputeAt)
    .Inputs({"block", "loop"})
    .Attrs({"keep_unit_loops"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::ReverseComputeAt)));

CINN_BUILD_STEP_KIND(GetRootBlock)
    .Inputs({"expr"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::GetRootBlock)));

CINN_BUILD_STEP_KIND(CacheRead)
    .Inputs({"block"})
    .Attrs({"read_buffer_index", "memory_type"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::CacheRead)));

CINN_BUILD_STEP_KIND(CacheWrite)
    .Inputs({"block"})
    .Attrs({"write_buffer_index", "memory_type"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::CacheWrite)));

CINN_BUILD_STEP_KIND(SyncThreads)
    .Inputs({"ir_node"})
    .Attrs({"after_node"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::SyncThreads)));

CINN_BUILD_STEP_KIND(SetBuffer)
    .Inputs({"block"})
    .Attrs({"memory_type", "fixed"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::SetBuffer)));

CINN_BUILD_STEP_KIND(AddUnitLoop)
    .Inputs({"block"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(static_cast<Expr (IRSchedule::*)(const Expr&)>(
            &IRSchedule::AddUnitLoop))));

CINN_BUILD_STEP_KIND(Reorder).Inputs({"loops"}).SetApplyFn(
    APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const std::vector<Expr>&)>(
            &IRSchedule::Reorder))));

CINN_BUILD_STEP_KIND(ReorderWithBlock)
    .Inputs({"block"})
    .Attrs({"loops_index"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const Expr&, const std::vector<int>&)>(
            &IRSchedule::Reorder))));

CINN_BUILD_STEP_KIND(ReorderWithName)
    .Attrs({"block_name", "loops_index"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(
        static_cast<Expr (IRSchedule::*)(const std::string&,
                                         const std::vector<int>&)>(
            &IRSchedule::Reorder))));

CINN_BUILD_STEP_KIND(Parallel).Inputs({"loop"}).SetApplyFn(
    APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Parallel)));

CINN_BUILD_STEP_KIND(Vectorize)
    .Inputs({"loop"})
    .Attrs({"factor"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Vectorize)));

CINN_BUILD_STEP_KIND(Unroll).Inputs({"loop"}).SetApplyFn(
    APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Unroll)));

CINN_BUILD_STEP_KIND(ComputeInline)
    .Inputs({"schedule_block"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::ComputeInline)));

CINN_BUILD_STEP_KIND(ReverseComputeInline)
    .Inputs({"schedule_block"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::ReverseComputeInline)));

CINN_BUILD_STEP_KIND(Bind)
    .Inputs({"loop"})
    .Attrs({"thread_axis"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Bind)));

CINN_BUILD_STEP_KIND(Rfactor)
    .Inputs({"rf_loop"})
    .Attrs({"rf_axis"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Rfactor)));

CINN_BUILD_STEP_KIND(FactorizeReduction)
    .Inputs({"rf_loop"})
    .Attrs({"rf_axis"})
    .Attrs({"with_write_back_block_init"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::FactorizeReduction)));

CINN_BUILD_STEP_KIND(MergeExprs)
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::MergeExprs)));

template <typename AttrType>
void Annotate(IRSchedule* ir_sch, const Expr&, const std::string&, AttrType);
template <>
void Annotate<int>(IRSchedule* ir_sch,
                   const Expr& block,
                   const std::string& key,
                   int value) {
  ir_sch->Annotate(block, key, value);
}
template <>
void Annotate<bool>(IRSchedule* ir_sch,
                    const Expr& block,
                    const std::string& key,
                    bool value) {
  ir_sch->Annotate(block, key, value);
}
template <>
void Annotate<float>(IRSchedule* ir_sch,
                     const Expr& block,
                     const std::string& key,
                     float value) {
  ir_sch->Annotate(block, key, value);
}
void AnnotateStringAttr(IRSchedule* ir_sch,
                        const Expr& block,
                        const std::string& key,
                        const std::string& value) {
  ir_sch->Annotate(block, key, value);
}

CINN_BUILD_STEP_KIND(AnnotateIntAttr)
    .Inputs({"block"})
    .Attrs({"key", "value"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(Annotate<int>));

CINN_BUILD_STEP_KIND(AnnotateBoolAttr)
    .Inputs({"block"})
    .Attrs({"key", "value"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(Annotate<bool>));

CINN_BUILD_STEP_KIND(AnnotateFloatAttr)
    .Inputs({"block"})
    .Attrs({"key", "value"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(Annotate<float>));

CINN_BUILD_STEP_KIND(AnnotateStringAttr)
    .Inputs({"block"})
    .Attrs({"key", "value"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(AnnotateStringAttr));

CINN_BUILD_STEP_KIND(Unannotate)
    .Inputs({"block"})
    .Attrs({"key"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::Unannotate)));

CINN_BUILD_STEP_KIND(FlattenLoops)
    .Inputs({"loops"})
    .Attrs({"force_flat"})
    .SetApplyFn(
        APPLY_FUNC_UNIFORM(FREE_FUNCTION_CONVERTER(&IRSchedule::FlattenLoops)));

CINN_BUILD_STEP_KIND(SamplePerfectTile)
    .Inputs({"loop"})
    .Attrs({"n", "max_innermost_factor", "decision"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::SamplePerfectTile)));

CINN_BUILD_STEP_KIND(TagPostSchedule)
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::TagPostSchedule)));

CINN_BUILD_STEP_KIND(SampleCategorical)
    .Attrs({"candidates", "probs", "decision"})
    .SetApplyFn(APPLY_FUNC_UNIFORM(
        FREE_FUNCTION_CONVERTER(&IRSchedule::SampleCategorical)));

// ------ Following codes are about member function implement of the
// ScheduleDesc class
void AttrVariantToProto(const utils::Attribute& attr,
                        proto::ScheduleDesc_Attr* attr_proto) {
#define SET_DESC_SINGLE_ITEM(index, built_type, proto_type, proto_field)   \
  case index:                                                              \
    attr_proto->set_dtype(proto::ScheduleDesc_Attr_DataType_##proto_type); \
    attr_proto->set_##proto_field(absl::get<built_type>(attr));            \
    break;

#define SET_DESC_REPEATED_ITEM(index, built_type, proto_type, proto_field) \
  case index: {                                                            \
    attr_proto->set_dtype(proto::ScheduleDesc_Attr_DataType_##proto_type); \
    const auto& values = absl::get<built_type>(attr);                      \
    attr_proto->mutable_##proto_field()->Reserve(values.size());           \
    *attr_proto->mutable_##proto_field() = {values.begin(), values.end()}; \
    break;                                                                 \
  }

  switch (attr.index()) {
    SET_DESC_SINGLE_ITEM(0, bool, BOOLEAN, b);
    SET_DESC_SINGLE_ITEM(1, float, FLOAT, f);
    SET_DESC_SINGLE_ITEM(2, int, INT, i);
    SET_DESC_SINGLE_ITEM(3, std::string, STRING, s);
    SET_DESC_REPEATED_ITEM(4, std::vector<bool>, BOOLEANS, bools);
    SET_DESC_REPEATED_ITEM(5, std::vector<int>, INTS, ints);
    SET_DESC_REPEATED_ITEM(6, std::vector<float>, FLOATS, floats);
    SET_DESC_REPEATED_ITEM(7, std::vector<std::string>, STRINGS, strings);
    SET_DESC_SINGLE_ITEM(8, int64_t, LONG, l);
    SET_DESC_SINGLE_ITEM(9, double, DOUBLE, d);
    SET_DESC_REPEATED_ITEM(10, std::vector<int64_t>, LONGS, longs);
    SET_DESC_REPEATED_ITEM(11, std::vector<double>, DOUBLES, doubles);
    default:
      std::stringstream ss;
      ss << "Invalid index:" << attr.index();
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

#undef SET_DESC_SINGLE_ITEM
#undef SET_DESC_REPEATED_ITEM
}

utils::Attribute AttrProtoToVariant(const proto::ScheduleDesc_Attr& attr) {
  utils::Attribute value;
#define PARSE_DESC_SINGLE_ITEM(proto_type, proto_field, built_type) \
  case proto::ScheduleDesc_Attr_DataType_##proto_type:              \
    value = built_type(attr.proto_field());                         \
    break;

#define PARSE_DESC_REPEATED_ITEM(proto_type, proto_field, built_type)       \
  case proto::ScheduleDesc_Attr_DataType_##proto_type:                      \
    value =                                                                 \
        built_type({attr.proto_field().begin(), attr.proto_field().end()}); \
    break;

  switch (attr.dtype()) {
    PARSE_DESC_SINGLE_ITEM(BOOLEAN, b, bool);
    PARSE_DESC_SINGLE_ITEM(INT, i, int);
    PARSE_DESC_SINGLE_ITEM(FLOAT, f, float);
    PARSE_DESC_SINGLE_ITEM(STRING, s, std::string);
    PARSE_DESC_REPEATED_ITEM(BOOLEANS, bools, std::vector<bool>);
    PARSE_DESC_REPEATED_ITEM(INTS, ints, std::vector<int>);
    PARSE_DESC_REPEATED_ITEM(FLOATS, floats, std::vector<float>);
    PARSE_DESC_REPEATED_ITEM(STRINGS, strings, std::vector<std::string>);
    PARSE_DESC_SINGLE_ITEM(LONG, l, int64_t);
    PARSE_DESC_SINGLE_ITEM(DOUBLE, d, double);
    PARSE_DESC_REPEATED_ITEM(LONGS, longs, std::vector<int64_t>);
    PARSE_DESC_REPEATED_ITEM(DOUBLES, doubles, std::vector<double>);
    default:
      std::stringstream ss;
      ss << "Invalid type:" << attr.DebugString();
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

#undef PARSE_DESC_SINGLE_ITEM
#undef PARSE_DESC_REPEATED_ITEM
  return value;
}

// Expr hash functor, presents how to hash an Expr
struct ExprHash {
  size_t operator()(const Expr& e) const {
    return std::hash<IrNode*>()(e.ptr());
  }
};
// Expr equal functor, presents whether a Expr pair is equal
struct ExprEqual {
  bool operator()(const Expr& lhs, const Expr& rhs) const {
    return lhs.get() == rhs.get();
  }
};

void ScheduleDesc::Append(Step&& step) { steps_.emplace_back(std::move(step)); }

void ScheduleDesc::Pop() {
  if (!steps_.empty()) {
    steps_.pop_back();
  }
}

void ScheduleDesc::Replay(IRSchedule* schedule,
                          bool without_post_schedule) const {
  ReplayWithProto(this->ToProto(), schedule, without_post_schedule);
}

proto::ScheduleDesc ScheduleDesc::ToProto() const {
  // map each Expr to a formatted name (e1, e2, ...)
  absl::flat_hash_map<Expr, std::string, ExprHash, ExprEqual> expr2name;
  proto::ScheduleDesc desc_proto;

  for (auto&& step : steps_) {
    auto* step_proto = desc_proto.add_steps();
    step_proto->set_type(step.type);
    // inputs of a step must refer to Exprs resulted by preceding steps
    for (auto&& param2exprs : step.inputs) {
      const std::string& param_name = param2exprs.first;
      auto* expr_desc = step_proto->add_inputs();
      expr_desc->set_parameter(param_name);
      for (auto&& expr : param2exprs.second) {
        auto expr_it = expr2name.find(expr);
        PADDLE_ENFORCE_NE(expr_it,
                          expr2name.end(),
                          ::common::errors::InvalidArgument(
                              "Can't find expr of param_name: %s", param_name));
        expr_desc->add_arguments(expr_it->second);
      }
    }

    // each output Expr is represented by a formatted name, to be referred by
    // succeeding steps
    for (auto&& expr : step.outputs) {
      std::string local_name = "e" + std::to_string(expr2name.size());
      expr2name.emplace(expr, local_name);
      step_proto->add_outputs(expr2name.at(expr));
    }

    for (auto&& attr2value : step.attrs) {
      auto* attr_proto = step_proto->add_attrs();
      const auto& attr_value = attr2value.second;
      VLOG(5) << "Attr.index:" << attr_value.index();
      attr_proto->set_name(attr2value.first);
      AttrVariantToProto(attr_value, attr_proto);
    }
  }
  return desc_proto;
}

std::vector<Expr> ScheduleDesc::ReplayWithProto(
    const proto::ScheduleDesc& desc_proto,
    IRSchedule* sch,
    bool without_post_schedule) {
  VLOG(4) << "proto::ScheduleDesc:\n" << desc_proto.DebugString();
  if (desc_proto.steps().empty()) {
    LOG(WARNING) << "Input proto::ScheduleDesc is empty";
    return {};
  }

  // map a formatted name (e1, e2, ...) to an Expr
  absl::flat_hash_map<std::string, Expr> name2expr;
  std::vector<Expr> last_outputs;

  // restore each scheduling step and apply to the new IRSchedule object
  for (auto&& step_proto : desc_proto.steps()) {
    VLOG(4) << "Replay step:\n" << step_proto.DebugString();
    ScheduleDesc::Step step;
    step.type = step_proto.type();
    PADDLE_ENFORCE_NE(
        step.type.empty(),
        true,
        ::common::errors::InvalidArgument("Name of StepKind is empty"));
    if (without_post_schedule && step.type == "TagPostSchedule") {
      break;
    }
    const StepKindInfo* step_kind = StepKindRegistry::Global()->Find(step.type);
    PADDLE_ENFORCE_NE(step_kind,
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "Can't find StepKind: %s", step.type));

    for (auto&& param2args : step_proto.inputs()) {
      for (auto&& arg : param2args.arguments()) {
        auto arg_it = name2expr.find(arg);
        PADDLE_ENFORCE_NE(
            arg_it,
            name2expr.end(),
            ::common::errors::InvalidArgument("Cant't find argument: %s", arg));
        step.inputs[param2args.parameter()].emplace_back(arg_it->second);
      }
    }
    for (auto&& attr : step_proto.attrs()) {
      step.attrs[attr.name()] = AttrProtoToVariant(attr);
    }

    PackedStepContext context(step, step_kind, sch);
    step.outputs = step_kind->Apply(&context);
    PADDLE_ENFORCE_EQ(
        step_proto.outputs().size(),
        step.outputs.size(),
        ::common::errors::InvalidArgument("Output size not matched"));
    for (size_t i = 0; i < step.outputs.size(); ++i) {
      name2expr[step_proto.outputs(i)] = step.outputs.at(i);
    }
    last_outputs = std::move(step.outputs);
  }
  return last_outputs;
}

ScheduleDesc ScheduleDesc::ForkAndUpdate(int step_idx,
                                         utils::Attribute decision,
                                         bool without_post_schedule) const {
  int n_valid_step = 0;
  if (!without_post_schedule) {
    n_valid_step = steps_.size();
  } else {
    for (const auto& step : steps_) {
      if (step.type != "TagPostSchedule") {
        ++n_valid_step;
      } else {
        break;
      }
    }
  }
  std::vector<ScheduleDesc::Step> new_steps(steps_.begin(),
                                            steps_.begin() + n_valid_step);
  new_steps[step_idx].attrs["decision"] = decision;
  return ScheduleDesc(std::move(new_steps));
}

}  // namespace ir
}  // namespace cinn
