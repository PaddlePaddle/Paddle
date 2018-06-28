/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

static constexpr char kInputs[] = "inputs";
static constexpr char kParameters[] = "parameters";
static constexpr char kPlaces[] = "places";

static constexpr char kOutputs[] = "outputs";
static constexpr char kParallelScopes[] = "parallel_scopes";

static constexpr char kParallelBlock[] = "sub_block";
static constexpr char kUseNCCL[] = "use_nccl";

using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

static void SplitTensorAndMoveTensorToScopes(
    const framework::Scope &scope, std::vector<framework::Scope *> *sub_scopes,
    const std::vector<platform::Place> &places,
    const std::vector<std::string> &names) {
  size_t num_sub_scopes = 0;
  for (auto &argu : names) {
    const auto &tensor =
        detail::Ref(scope.FindVar(argu),
                    "Cannot find variable %s in the parent scope", argu)
            .Get<LoDTensor>();
    auto lod_tensors = tensor.SplitLoDTensor(places);

    for (auto &lod : lod_tensors) {
      VLOG(3) << lod.dims();
    }
    if (num_sub_scopes == 0) {
      num_sub_scopes = lod_tensors.size();
    } else {
      PADDLE_ENFORCE_EQ(num_sub_scopes, lod_tensors.size());
    }
    PADDLE_ENFORCE_NE(num_sub_scopes, 0);
    if (sub_scopes->size() == 0) {
      sub_scopes->reserve(num_sub_scopes);
      for (size_t i = 0; i < num_sub_scopes; ++i) {
        sub_scopes->emplace_back(&scope.NewScope());
      }
    }

    for (size_t i = 0; i < lod_tensors.size(); ++i) {
      *detail::Ref(sub_scopes->at(i)->Var(argu),
                   "Cannot find variable in the sub-scope", argu)
           .GetMutable<LoDTensor>() = lod_tensors[i];
    }
  }
}

inline void CopyOrShare(const framework::Variable &src,
                        const platform::Place &dst_place,
                        framework::Variable *dst) {
  if (src.IsType<LoDTensor>()) {
    if (src.Get<LoDTensor>().place() == dst_place) {
      dst->GetMutable<LoDTensor>()->ShareDataWith(src.Get<LoDTensor>());
      dst->GetMutable<LoDTensor>()->set_lod(src.Get<LoDTensor>().lod());
    } else {
      TensorCopy(src.Get<LoDTensor>(), dst_place, dst->GetMutable<LoDTensor>());
    }
  } else if (src.IsType<SelectedRows>()) {
    auto &src_sr = src.Get<SelectedRows>();
    auto *dst_sr = dst->GetMutable<SelectedRows>();
    dst_sr->set_height(src_sr.height());
    if (src_sr.value().place() == dst_place) {
      dst_sr->mutable_value()->ShareDataWith(src_sr.value());
      dst_sr->set_rows(src_sr.rows());
    } else {
      TensorCopy(src_sr.value(), dst_place, dst_sr->mutable_value());
    }
  } else {
    PADDLE_THROW("Expect LoDTensor/SelectedRows, get %s", src.Type().name());
  }
}

void WaitOnPlace(const platform::Place place) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);
  dev_ctx.Wait();
}

void WaitOnPlaces(const std::vector<platform::Place> places) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();

  for (auto &place : places) {
    auto &dev_ctx = *pool.Get(place);
    dev_ctx.Wait();
  }
}

class ParallelDoOp : public framework::OperatorBase {
 public:
  ParallelDoOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    auto *block = Attr<framework::BlockDesc *>(kParallelBlock);
    auto *program = block->Program();

    auto &places = scope.FindVar(Input(kPlaces))->Get<platform::PlaceList>();

    auto &sub_scopes = *scope.FindVar(Output(kParallelScopes))
                            ->GetMutable<std::vector<framework::Scope *>>();

    // split input
    SplitTensorAndMoveTensorToScopes(scope, &sub_scopes, places,
                                     Inputs(kInputs));

    // copy parameter
    for (auto &param : Inputs(kParameters)) {
      PADDLE_ENFORCE(scope.FindVar(param)->IsType<LoDTensor>(),
                     "Only support parameter type as LoDTensor");
      auto &src = scope.FindVar(param)->Get<LoDTensor>();

      auto *sub_scope0 = sub_scopes[0];
      auto *dst0 = sub_scope0->Var(param)->GetMutable<LoDTensor>();
      dst0->ShareDataWith(src);

      for (size_t i = 1; i < sub_scopes.size(); ++i) {
        auto &place = places[i];
        auto *sub_scope = sub_scopes[i];
        auto *dst = sub_scope->Var(param)->GetMutable<LoDTensor>();
        framework::TensorCopy(src, place, dst);
      }
    }
    WaitOnPlaces(places);

    std::vector<std::future<void>> workers;
    workers.reserve(places.size());
    for (size_t place_idx = 0; place_idx < sub_scopes.size(); ++place_idx) {
      auto &place = places[place_idx];
      auto *cur_scope = sub_scopes[place_idx];

      workers.emplace_back(
          framework::Async([program, cur_scope, place, block, place_idx] {
            // Give the thread an id to distinguish parallel block with same id.
            platform::RecordThread rt(static_cast<int>(place_idx) + 1);
            framework::Executor executor(place);
            executor.Run(*program, cur_scope, block->ID(),
                         false /*create_local_scope*/);
          }));
    }
    for (auto &worker : workers) {
      worker.wait();
    }
    WaitOnPlaces(places);

    // merge output
    for (auto &o_name : Outputs(kOutputs)) {
      std::vector<const framework::LoDTensor *> lod_tensors;
      lod_tensors.reserve(sub_scopes.size());
      for (auto *sub_scope : sub_scopes) {
        lod_tensors.emplace_back(&sub_scope->FindVar(o_name)->Get<LoDTensor>());
      }

      auto *lod_tensor_to_be_merged =
          scope.FindVar(o_name)->GetMutable<LoDTensor>();
      lod_tensor_to_be_merged->MergeLoDTensor(lod_tensors, dev_ctx.GetPlace());
    }
    WaitOnPlaces(places);
  }
};

class ParallelDoOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kInputs, "").AsDuplicable();
    AddInput(kParameters, "").AsDuplicable();
    AddInput(kPlaces, "");
    AddOutput(kOutputs, "").AsDuplicable();
    AddOutput(kParallelScopes, "");
    AddAttr<framework::BlockDesc *>(kParallelBlock, "");
    AddAttr<bool>(kUseNCCL, "true if we use nccl on backward")
        .SetDefault(false);
    AddComment(R"DOC(
ParallelDo Operator.
)DOC");
  }
};

class ParallelDoGradOp : public framework::OperatorBase {
 public:
  ParallelDoGradOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *block = Attr<framework::BlockDesc *>(kParallelBlock);
    auto *program = block->Program();

    auto &sub_scopes = scope.FindVar(Input(kParallelScopes))
                           ->Get<std::vector<framework::Scope *>>();
    auto &places = scope.FindVar(Input(kPlaces))->Get<platform::PlaceList>();

    // feed output@grad
    SplitTensorAndMoveTensorToScopes(
        scope, const_cast<std::vector<framework::Scope *> *>(&sub_scopes),
        places, Inputs(framework::GradVarName(kOutputs)));
    WaitOnPlaces(places);

    // exe run
    std::vector<std::future<void>> workers;
    for (size_t i = 0; i < sub_scopes.size(); ++i) {
      auto &place = places[i];
      auto *cur_scope = sub_scopes[i];

      // execute
      workers.emplace_back(
          framework::Async([program, cur_scope, place, block, i] {
            // Give the thread an id to distinguish parallel block with same id.
            platform::RecordThread rt(static_cast<int>(i) + 1);
            framework::Executor executor(place);
            executor.Run(*program, cur_scope, block->ID(),
                         false /*create_local_scope*/);
          }));
    }
    for (auto &worker : workers) {
      worker.wait();
    }
    WaitOnPlaces(places);

    // NCCL allreduce op will be added by backward,
    // so no need to explicitly accumulate grad
    if (!(Attr<bool>(kUseNCCL))) {
      AccumulateGrad(scope, place, sub_scopes, places);
    } else {
      for (auto &place : places) {
        PADDLE_ENFORCE(platform::is_gpu_place(place),
                       "NCCL only supports cuda place");
      }
    }
    for (auto &s : Outputs(framework::GradVarName(kParameters))) {
      if (s == framework::kEmptyVarName) {
        continue;
      }
      VLOG(3) << "Moving " << s;
      CopyOrShare(*sub_scopes[0]->FindVar(s), place, scope.FindVar(s));
    }
    WaitOnPlaces(places);
  }

  void AccumulateGrad(const framework::Scope &scope,
                      const platform::Place &place,
                      const std::vector<framework::Scope *> &sub_scopes,
                      const platform::PlaceList &places) const {
    for (auto &s : Outputs(framework::GradVarName(kParameters))) {
      if (s == framework::kEmptyVarName) {
        continue;
      }
      VLOG(3) << "Accumulating " << s;
      if (s == framework::kEmptyVarName) continue;
      std::string tmp_name;
      auto *tmp = sub_scopes[0]->Var(&tmp_name);

      for (size_t i = 1; i < sub_scopes.size(); ++i) {
        CopyOrShare(*sub_scopes[i]->FindVar(s), places[0], tmp);
        WaitOnPlaces(places);

        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {s, tmp_name}}}, {{"Out", {s}}},
            framework::AttributeMap{{"use_mkldnn", {false}}});
        VLOG(10) << sum_op->DebugStringEx(sub_scopes[0]);
        sum_op->Run(*sub_scopes[0], places[0]);
        WaitOnPlace(places[0]);
      }

      CopyOrShare(*sub_scopes[0]->FindVar(s), place, scope.FindVar(s));
    }
    WaitOnPlaces(places);
  }
};

std::ostream &operator<<(std::ostream &sout,
                         const std::vector<std::string> &strs) {
  std::copy(strs.begin(), strs.end(),
            std::ostream_iterator<std::string>(sout, ","));
  return sout;
}

class ParallelDoGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDesc> Apply() const {
    auto *grad = new framework::OpDesc();
    grad->SetType("parallel_do_grad");
    for (auto &input_param : this->InputNames()) {
      VLOG(3) << input_param;
      grad->SetInput(input_param, this->Input(input_param));
      if (input_param != kPlaces) {
        grad->SetOutput(framework::GradVarName(input_param),
                        this->InputGrad(input_param, false));
      }
    }
    auto *g_block = this->grad_block_[0];

    // All variable name that needed by gradient operators
    std::unordered_set<std::string> all_inputs_in_grad_blocks;

    for (size_t i = 0; i < g_block->OpSize(); ++i) {
      auto *op = g_block->Op(i);
      for (auto &var_name : op->InputArgumentNames()) {
        all_inputs_in_grad_blocks.insert(var_name);
      }
    }

    for (auto &output_param : this->OutputNames()) {
      if (output_param == kParallelScopes) {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->Output(output_param));
      } else {
        grad->SetInput(output_param, this->Output(output_param));
        std::vector<std::string> og_names;
        for (auto &og_name : this->OutputGrad(output_param)) {
          if (all_inputs_in_grad_blocks.count(og_name) != 0) {
            // there are some gradient operators who need the OG. So make this
            // OG as an input of parallel.do
            og_names.push_back(og_name);
          }
          // else, there is no operator who need the OG. Do not use this OG as
          // an input
        }
        grad->SetInput(framework::GradVarName(output_param), og_names);
      }
    }
    grad->SetAttrMap(this->Attrs());
    grad->SetBlockAttr(kParallelBlock, grad_block_[0]);

    return std::unique_ptr<framework::OpDesc>(grad);
  }
};

class ParallelDoGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs(kParameters));
    PADDLE_ENFORCE(ctx->HasInputs(kInputs));
    PADDLE_ENFORCE(ctx->HasInputs(kOutputs));

    ctx->SetOutputsDim(framework::GradVarName(kParameters),
                       ctx->GetInputsDim(kParameters));

    auto i_dims = ctx->GetInputsDim(kInputs);
    auto ig_names = ctx->Outputs(framework::GradVarName(kInputs));

    for (size_t i = 0; i < ig_names.size(); ++i) {
      auto &ig_name = ig_names[i];
      if (ig_name == framework::kEmptyVarName) {
        continue;
      }

      ctx->SetDims({ig_name}, {i_dims[i]});
    }

    auto p_dims = ctx->GetInputsDim(kParameters);
    auto pg_names = ctx->Outputs(framework::GradVarName(kParameters));
    for (size_t i = 0; i < pg_names.size(); ++i) {
      auto &pg_name = pg_names[i];
      if (pg_name == framework::kEmptyVarName) {
        continue;
      }
      ctx->SetDims({pg_name}, {p_dims[i]});
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(parallel_do, paddle::operators::ParallelDoOp,
                  paddle::operators::ParallelDoOpProtoMaker,
                  paddle::operators::ParallelDoGradOpDescMaker);
REGISTER_OPERATOR(parallel_do_grad, paddle::operators::ParallelDoGradOp,
                  paddle::operators::ParallelDoGradOpShapeInference);
