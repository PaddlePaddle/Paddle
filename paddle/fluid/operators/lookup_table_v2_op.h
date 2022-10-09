/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = phi::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename InT, typename OutT>
static std::vector<OutT> CopyIdsToVector(const Tensor &ids) {
  auto numel = ids.numel();
  const auto *src = ids.data<InT>();
  std::vector<OutT> ret(numel);
  if (std::is_same<InT, OutT>::value) {
    std::memcpy(ret.data(), src, numel * sizeof(InT));
  } else {
    for (decltype(numel) i = 0; i < numel; ++i) {
      ret[i] = src[i];
    }
  }
  return ret;
}

template <typename T>
struct LookupTableV2CPUFunctor {
  LookupTableV2CPUFunctor(const framework::ExecutionContext &context,
                          const Tensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto *output_t = context_.Output<LoDTensor>("Out");  // float tensor
    auto *table_var = context_.InputVar("W");

    int64_t padding_idx = context_.Attr<int64_t>("padding_idx");

    auto ids = CopyIdsToVector<IdT, int64_t>(*ids_t_);
    auto ids_numel = static_cast<int64_t>(ids.size());

    if (table_var->template IsType<LoDTensor>()) {
      const auto &table_t = table_var->template Get<LoDTensor>();
      int64_t row_number = table_t.dims()[0];
      int64_t row_width = table_t.dims()[1];

      auto *table = table_t.template data<T>();
      auto *output = output_t->template mutable_data<T>(context_.GetPlace());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_LT(
              ids[i],
              row_number,
              platform::errors::InvalidArgument(
                  "Variable value (input) of OP(fluid.layers.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  row_number,
                  ids[i]));
          PADDLE_ENFORCE_GE(
              ids[i],
              0,
              platform::errors::InvalidArgument(
                  "Variable value (input) of OP(fluid.layers.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  row_number,
                  ids[i]));
          memcpy(output + i * row_width,
                 table + ids[i] * row_width,
                 row_width * sizeof(T));
        }
      }
    } else if (table_var->template IsType<phi::SelectedRows>()) {
      const auto &table_t = table_var->template Get<phi::SelectedRows>();
      int64_t row_width = table_t.value().dims()[1];
      const auto *table = table_t.value().template data<T>();
      auto *output = output_t->template mutable_data<T>(context_.GetPlace());
      auto input_data_type =
          framework::TransToProtoVarType(table_t.value().dtype());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_GE(
              ids[i],
              0,
              platform::errors::InvalidArgument(
                  "Variable value (input) of OP(fluid.layers.embedding) "
                  "expected >= 0. But received %ld",
                  ids[i]));
          auto id_index = table_t.Index(ids[i]);
          PADDLE_ENFORCE_GE(
              id_index,
              0,
              platform::errors::InvalidArgument(
                  "the input key should be exists. But received %d.",
                  id_index));

          if (input_data_type == framework::proto::VarType::BF16) {
            memcpy(output + i * row_width,
                   table + id_index * row_width,
                   row_width * sizeof(T));
          } else {
            auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(context_);
            blas.VCOPY(row_width,
                       table + id_index * row_width,
                       output + i * row_width);
          }
        }
      }
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const Tensor *ids_t_;
};

template <typename T>
class LookupTableV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids = context.Input<phi::DenseTensor>("Ids");
    LookupTableV2CPUFunctor<T> functor(context, ids);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids->dtype()),
                                functor);
  }
};

template <typename T>
struct LookupTableV2GradCPUFunctor {
  LookupTableV2GradCPUFunctor(const framework::ExecutionContext &context,
                              const Tensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto *table_var = context_.InputVar("W");
    DDim table_dim;
    if (table_var->template IsType<LoDTensor>()) {
      table_dim = context_.Input<LoDTensor>("W")->dims();
    } else if (table_var->template IsType<phi::SelectedRows>()) {
      auto *table_t = context_.Input<phi::SelectedRows>("W");
      table_dim = table_t->value().dims();
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The parameter W of a LookupTableV2 "
          "must be either LoDTensor or SelectedRows"));
    }

    int64_t padding_idx = context_.Attr<int64_t>("padding_idx");
    bool is_sparse = context_.Attr<bool>("is_sparse");

    auto ids = CopyIdsToVector<IdT, int64_t>(*ids_t_);
    auto ids_num = static_cast<int64_t>(ids.size());

    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *d_output = context_.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table =
          context_.Output<phi::SelectedRows>(framework::GradVarName("W"));

      d_table->set_rows(ids);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});

      d_table_value->template mutable_data<T>(context_.GetPlace());

      d_table->set_height(table_dim[0]);

      auto *d_output_data = d_output->template data<T>();
      auto *d_table_data = d_table_value->template data<T>();

      auto d_output_dims = d_output->dims();
      auto d_output_dims_2d =
          phi::flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
      PADDLE_ENFORCE_EQ(d_table_value->dims(),
                        d_output_dims_2d,
                        platform::errors::InvalidArgument(
                            "ShapeError: The shape of lookup_table@Grad and "
                            "output@Grad should be same. "
                            "But received lookup_table@Grad's shape = [%s], "
                            "output@Grad's shape = [%s].",
                            d_table_value->dims(),
                            d_output_dims_2d));
      memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());

    } else {
      auto *d_output = context_.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context_.Output<LoDTensor>(framework::GradVarName("W"));
      auto *ids_data = ids.data();

      int64_t N = table_dim[0];
      int64_t D = table_dim[1];

      auto *d_output_data = d_output->template data<T>();
      auto *d_table_data =
          d_table->template mutable_data<T>(context_.GetPlace());

      memset(d_table_data, 0, d_table->numel() * sizeof(T));

      for (int64_t i = 0; i < ids_num; ++i) {
        if (padding_idx != kNoPadding && ids_data[i] == padding_idx) {
          // the gradient of padding_idx should be 0, already done by memset, so
          // do nothing.
        } else {
          PADDLE_ENFORCE_LT(
              ids_data[i],
              N,
              platform::errors::InvalidArgument(
                  "Variable value (input) of OP(fluid.layers.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  N,
                  ids_data[i]));
          PADDLE_ENFORCE_GE(
              ids_data[i],
              0,
              platform::errors::InvalidArgument(
                  "Variable value (input) of OP(fluid.layers.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  N,
                  ids_data[i]));
          for (int j = 0; j < D; ++j) {
            d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
          }
        }
      }
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const Tensor *ids_t_;
};

template <typename T>
class LookupTableV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids = context.Input<phi::DenseTensor>("Ids");
    LookupTableV2GradCPUFunctor<T> functor(context, ids);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids->dtype()),
                                functor);
  }
};

}  // namespace operators
}  // namespace paddle
