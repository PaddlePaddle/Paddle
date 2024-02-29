

#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"

namespace paddle {

namespace dialect {

pir::OpResult abs(const pir::Value& x) {
  CheckValueDataType(x, "x", "abs");
  paddle::dialect::AbsOp abs_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AbsOp>(x);
  return abs_op.result(0);
}

pir::OpResult abs_(const pir::Value& x) {
  CheckValueDataType(x, "x", "abs_");
  paddle::dialect::Abs_Op abs__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Abs_Op>(x);
  return abs__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> accuracy(
    const pir::Value& x, const pir::Value& indices, const pir::Value& label) {
  CheckValueDataType(x, "x", "accuracy");
  paddle::dialect::AccuracyOp accuracy_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AccuracyOp>(
          x, indices, label);
  return std::make_tuple(
      accuracy_op.result(0), accuracy_op.result(1), accuracy_op.result(2));
}

pir::OpResult acos(const pir::Value& x) {
  CheckValueDataType(x, "x", "acos");
  paddle::dialect::AcosOp acos_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcosOp>(x);
  return acos_op.result(0);
}

pir::OpResult acos_(const pir::Value& x) {
  CheckValueDataType(x, "x", "acos_");
  paddle::dialect::Acos_Op acos__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Acos_Op>(x);
  return acos__op.result(0);
}

pir::OpResult acosh(const pir::Value& x) {
  CheckValueDataType(x, "x", "acosh");
  paddle::dialect::AcoshOp acosh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcoshOp>(x);
  return acosh_op.result(0);
}

pir::OpResult acosh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "acosh_");
  paddle::dialect::Acosh_Op acosh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Acosh_Op>(x);
  return acosh__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
adagrad_(const pir::Value& param,
         const pir::Value& grad,
         const pir::Value& moment,
         const pir::Value& learning_rate,
         const paddle::optional<pir::Value>& master_param,
         float epsilon,
         bool multi_precision) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      moment.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adagrad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::Adagrad_Op adagrad__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adagrad_Op>(
            param,
            grad,
            moment,
            learning_rate,
            optional_master_param.get(),
            epsilon,
            multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adagrad__op.result(2))) {
      optional_master_param_out =
          paddle::make_optional<pir::OpResult>(adagrad__op.result(2));
    }
    if (!master_param) {
      adagrad__op.result(2).set_type(pir::Type());
    }
    return std::make_tuple(adagrad__op.result(0),
                           adagrad__op.result(1),
                           optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      moment.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adagrad_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::AdagradDenseParamSparseGrad_Op
        adagrad_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::AdagradDenseParamSparseGrad_Op>(
                    param,
                    grad,
                    moment,
                    learning_rate,
                    optional_master_param.get(),
                    epsilon,
                    multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adagrad_dense_param_sparse_grad__op.result(2))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          adagrad_dense_param_sparse_grad__op.result(2));
    }
    if (!master_param) {
      adagrad_dense_param_sparse_grad__op.result(2).set_type(pir::Type());
    }
    return std::make_tuple(adagrad_dense_param_sparse_grad__op.result(0),
                           adagrad_dense_param_sparse_grad__op.result(1),
                           optional_master_param_out);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (adagrad_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adam_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      float beta1,
      float beta2,
      float epsilon,
      bool lazy_mode,
      int64_t min_row_size_to_use_multithread,
      bool multi_precision,
      bool use_global_beta_pow) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adam_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::Adam_Op adam__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adam_Op>(
            param,
            grad,
            learning_rate,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            optional_master_param.get(),
            optional_skip_update.get(),
            beta1,
            beta2,
            epsilon,
            lazy_mode,
            min_row_size_to_use_multithread,
            multi_precision,
            use_global_beta_pow);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adam__op.result(5))) {
      optional_master_param_out =
          paddle::make_optional<pir::OpResult>(adam__op.result(5));
    }
    if (!master_param) {
      adam__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(adam__op.result(0),
                           adam__op.result(1),
                           adam__op.result(2),
                           adam__op.result(3),
                           adam__op.result(4),
                           optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adam_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::AdamDenseParamSparseGrad_Op
        adam_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::AdamDenseParamSparseGrad_Op>(
                    param,
                    grad,
                    learning_rate,
                    moment1,
                    moment2,
                    beta1_pow,
                    beta2_pow,
                    optional_master_param.get(),
                    optional_skip_update.get(),
                    beta1,
                    beta2,
                    epsilon,
                    lazy_mode,
                    min_row_size_to_use_multithread,
                    multi_precision,
                    use_global_beta_pow);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adam_dense_param_sparse_grad__op.result(5))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          adam_dense_param_sparse_grad__op.result(5));
    }
    if (!master_param) {
      adam_dense_param_sparse_grad__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(adam_dense_param_sparse_grad__op.result(0),
                           adam_dense_param_sparse_grad__op.result(1),
                           adam_dense_param_sparse_grad__op.result(2),
                           adam_dense_param_sparse_grad__op.result(3),
                           adam_dense_param_sparse_grad__op.result(4),
                           optional_master_param_out);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (adam_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adam_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      pir::Value beta1,
      pir::Value beta2,
      pir::Value epsilon,
      bool lazy_mode,
      int64_t min_row_size_to_use_multithread,
      bool multi_precision,
      bool use_global_beta_pow) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adam_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::Adam_Op adam__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adam_Op>(
            param,
            grad,
            learning_rate,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            optional_master_param.get(),
            optional_skip_update.get(),
            beta1,
            beta2,
            epsilon,
            lazy_mode,
            min_row_size_to_use_multithread,
            multi_precision,
            use_global_beta_pow);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adam__op.result(5))) {
      optional_master_param_out =
          paddle::make_optional<pir::OpResult>(adam__op.result(5));
    }
    if (!master_param) {
      adam__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(adam__op.result(0),
                           adam__op.result(1),
                           adam__op.result(2),
                           adam__op.result(3),
                           adam__op.result(4),
                           optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "adam_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::AdamDenseParamSparseGrad_Op
        adam_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::AdamDenseParamSparseGrad_Op>(
                    param,
                    grad,
                    learning_rate,
                    moment1,
                    moment2,
                    beta1_pow,
                    beta2_pow,
                    optional_master_param.get(),
                    optional_skip_update.get(),
                    beta1,
                    beta2,
                    epsilon,
                    lazy_mode,
                    min_row_size_to_use_multithread,
                    multi_precision,
                    use_global_beta_pow);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(adam_dense_param_sparse_grad__op.result(5))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          adam_dense_param_sparse_grad__op.result(5));
    }
    if (!master_param) {
      adam_dense_param_sparse_grad__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(adam_dense_param_sparse_grad__op.result(0),
                           adam_dense_param_sparse_grad__op.result(1),
                           adam_dense_param_sparse_grad__op.result(2),
                           adam_dense_param_sparse_grad__op.result(3),
                           adam_dense_param_sparse_grad__op.result(4),
                           optional_master_param_out);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (adam_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamax_(const pir::Value& param,
        const pir::Value& grad,
        const pir::Value& learning_rate,
        const pir::Value& moment,
        const pir::Value& inf_norm,
        const pir::Value& beta1_pow,
        const paddle::optional<pir::Value>& master_param,
        float beta1,
        float beta2,
        float epsilon,
        bool multi_precision) {
  CheckValueDataType(param, "param", "adamax_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::dialect::Adamax_Op adamax__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adamax_Op>(
          param,
          grad,
          learning_rate,
          moment,
          inf_norm,
          beta1_pow,
          optional_master_param.get(),
          beta1,
          beta2,
          epsilon,
          multi_precision);
  paddle::optional<pir::OpResult> optional_master_param_out;
  if (!IsEmptyValue(adamax__op.result(3))) {
    optional_master_param_out =
        paddle::make_optional<pir::OpResult>(adamax__op.result(3));
  }
  if (!master_param) {
    adamax__op.result(3).set_type(pir::Type());
  }
  return std::make_tuple(adamax__op.result(0),
                         adamax__op.result(1),
                         adamax__op.result(2),
                         optional_master_param_out);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamw_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& learning_rate,
       const pir::Value& moment1,
       const pir::Value& moment2,
       const pir::Value& beta1_pow,
       const pir::Value& beta2_pow,
       const paddle::optional<pir::Value>& master_param,
       const paddle::optional<pir::Value>& skip_update,
       float beta1,
       float beta2,
       float epsilon,
       float lr_ratio,
       float coeff,
       bool with_decay,
       bool lazy_mode,
       int64_t min_row_size_to_use_multithread,
       bool multi_precision,
       bool use_global_beta_pow) {
  CheckValueDataType(param, "param", "adamw_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::optional<pir::Value> optional_skip_update;
  if (!skip_update) {
    optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_skip_update = skip_update;
  }
  paddle::dialect::Adamw_Op adamw__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adamw_Op>(
          param,
          grad,
          learning_rate,
          moment1,
          moment2,
          beta1_pow,
          beta2_pow,
          optional_master_param.get(),
          optional_skip_update.get(),
          beta1,
          beta2,
          epsilon,
          lr_ratio,
          coeff,
          with_decay,
          lazy_mode,
          min_row_size_to_use_multithread,
          multi_precision,
          use_global_beta_pow);
  paddle::optional<pir::OpResult> optional_master_param_out;
  if (!IsEmptyValue(adamw__op.result(5))) {
    optional_master_param_out =
        paddle::make_optional<pir::OpResult>(adamw__op.result(5));
  }
  if (!master_param) {
    adamw__op.result(5).set_type(pir::Type());
  }
  return std::make_tuple(adamw__op.result(0),
                         adamw__op.result(1),
                         adamw__op.result(2),
                         adamw__op.result(3),
                         adamw__op.result(4),
                         optional_master_param_out);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamw_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& learning_rate,
       const pir::Value& moment1,
       const pir::Value& moment2,
       const pir::Value& beta1_pow,
       const pir::Value& beta2_pow,
       const paddle::optional<pir::Value>& master_param,
       const paddle::optional<pir::Value>& skip_update,
       pir::Value beta1,
       pir::Value beta2,
       pir::Value epsilon,
       float lr_ratio,
       float coeff,
       bool with_decay,
       bool lazy_mode,
       int64_t min_row_size_to_use_multithread,
       bool multi_precision,
       bool use_global_beta_pow) {
  CheckValueDataType(param, "param", "adamw_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::optional<pir::Value> optional_skip_update;
  if (!skip_update) {
    optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_skip_update = skip_update;
  }
  paddle::dialect::Adamw_Op adamw__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adamw_Op>(
          param,
          grad,
          learning_rate,
          moment1,
          moment2,
          beta1_pow,
          beta2_pow,
          optional_master_param.get(),
          optional_skip_update.get(),
          beta1,
          beta2,
          epsilon,
          lr_ratio,
          coeff,
          with_decay,
          lazy_mode,
          min_row_size_to_use_multithread,
          multi_precision,
          use_global_beta_pow);
  paddle::optional<pir::OpResult> optional_master_param_out;
  if (!IsEmptyValue(adamw__op.result(5))) {
    optional_master_param_out =
        paddle::make_optional<pir::OpResult>(adamw__op.result(5));
  }
  if (!master_param) {
    adamw__op.result(5).set_type(pir::Type());
  }
  return std::make_tuple(adamw__op.result(0),
                         adamw__op.result(1),
                         adamw__op.result(2),
                         adamw__op.result(3),
                         adamw__op.result(4),
                         optional_master_param_out);
}

pir::OpResult addmm(const pir::Value& input,
                    const pir::Value& x,
                    const pir::Value& y,
                    float beta,
                    float alpha) {
  CheckValueDataType(x, "x", "addmm");
  paddle::dialect::AddmmOp addmm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddmmOp>(
          input, x, y, beta, alpha);
  return addmm_op.result(0);
}

pir::OpResult addmm_(const pir::Value& input,
                     const pir::Value& x,
                     const pir::Value& y,
                     float beta,
                     float alpha) {
  CheckValueDataType(x, "x", "addmm_");
  paddle::dialect::Addmm_Op addmm__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Addmm_Op>(
          input, x, y, beta, alpha);
  return addmm__op.result(0);
}

pir::OpResult affine_grid(const pir::Value& input,
                          const std::vector<int64_t>& output_shape,
                          bool align_corners) {
  CheckValueDataType(input, "input", "affine_grid");
  paddle::dialect::AffineGridOp affine_grid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AffineGridOp>(
          input, output_shape, align_corners);
  return affine_grid_op.result(0);
}

pir::OpResult affine_grid(const pir::Value& input,
                          pir::Value output_shape,
                          bool align_corners) {
  CheckValueDataType(input, "input", "affine_grid");
  paddle::dialect::AffineGridOp affine_grid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AffineGridOp>(
          input, output_shape, align_corners);
  return affine_grid_op.result(0);
}

pir::OpResult affine_grid(const pir::Value& input,
                          std::vector<pir::Value> output_shape,
                          bool align_corners) {
  CheckValueDataType(input, "input", "affine_grid");
  auto output_shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_shape);
  paddle::dialect::AffineGridOp affine_grid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AffineGridOp>(
          input, output_shape_combine_op.out(), align_corners);
  return affine_grid_op.result(0);
}

pir::OpResult allclose(const pir::Value& x,
                       const pir::Value& y,
                       float rtol,
                       float atol,
                       bool equal_nan) {
  CheckValueDataType(x, "x", "allclose");
  paddle::dialect::AllcloseOp allclose_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AllcloseOp>(
          x, y, rtol, atol, equal_nan);
  return allclose_op.result(0);
}

pir::OpResult allclose(const pir::Value& x,
                       const pir::Value& y,
                       pir::Value rtol,
                       pir::Value atol,
                       bool equal_nan) {
  CheckValueDataType(x, "x", "allclose");
  paddle::dialect::AllcloseOp allclose_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AllcloseOp>(
          x, y, rtol, atol, equal_nan);
  return allclose_op.result(0);
}

pir::OpResult angle(const pir::Value& x) {
  CheckValueDataType(x, "x", "angle");
  paddle::dialect::AngleOp angle_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AngleOp>(x);
  return angle_op.result(0);
}

pir::OpResult argmax(const pir::Value& x,
                     int64_t axis,
                     bool keepdims,
                     bool flatten,
                     phi::DataType dtype) {
  CheckValueDataType(x, "x", "argmax");
  paddle::dialect::ArgmaxOp argmax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArgmaxOp>(
          x, axis, keepdims, flatten, dtype);
  return argmax_op.result(0);
}

pir::OpResult argmax(const pir::Value& x,
                     pir::Value axis,
                     bool keepdims,
                     bool flatten,
                     phi::DataType dtype) {
  CheckValueDataType(x, "x", "argmax");
  paddle::dialect::ArgmaxOp argmax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArgmaxOp>(
          x, axis, keepdims, flatten, dtype);
  return argmax_op.result(0);
}

pir::OpResult argmin(const pir::Value& x,
                     int64_t axis,
                     bool keepdims,
                     bool flatten,
                     phi::DataType dtype) {
  CheckValueDataType(x, "x", "argmin");
  paddle::dialect::ArgminOp argmin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArgminOp>(
          x, axis, keepdims, flatten, dtype);
  return argmin_op.result(0);
}

pir::OpResult argmin(const pir::Value& x,
                     pir::Value axis,
                     bool keepdims,
                     bool flatten,
                     phi::DataType dtype) {
  CheckValueDataType(x, "x", "argmin");
  paddle::dialect::ArgminOp argmin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArgminOp>(
          x, axis, keepdims, flatten, dtype);
  return argmin_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> argsort(const pir::Value& x,
                                                 int axis,
                                                 bool descending) {
  CheckValueDataType(x, "x", "argsort");
  paddle::dialect::ArgsortOp argsort_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArgsortOp>(
          x, axis, descending);
  return std::make_tuple(argsort_op.result(0), argsort_op.result(1));
}

pir::OpResult as_complex(const pir::Value& x) {
  CheckValueDataType(x, "x", "as_complex");
  paddle::dialect::AsComplexOp as_complex_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsComplexOp>(
          x);
  return as_complex_op.result(0);
}

pir::OpResult as_real(const pir::Value& x) {
  CheckValueDataType(x, "x", "as_real");
  paddle::dialect::AsRealOp as_real_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsRealOp>(x);
  return as_real_op.result(0);
}

pir::OpResult as_strided(const pir::Value& input,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& stride,
                         int64_t offset) {
  CheckValueDataType(input, "input", "as_strided");
  paddle::dialect::AsStridedOp as_strided_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsStridedOp>(
          input, dims, stride, offset);
  return as_strided_op.result(0);
}

pir::OpResult asin(const pir::Value& x) {
  CheckValueDataType(x, "x", "asin");
  paddle::dialect::AsinOp asin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinOp>(x);
  return asin_op.result(0);
}

pir::OpResult asin_(const pir::Value& x) {
  CheckValueDataType(x, "x", "asin_");
  paddle::dialect::Asin_Op asin__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Asin_Op>(x);
  return asin__op.result(0);
}

pir::OpResult asinh(const pir::Value& x) {
  CheckValueDataType(x, "x", "asinh");
  paddle::dialect::AsinhOp asinh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinhOp>(x);
  return asinh_op.result(0);
}

pir::OpResult asinh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "asinh_");
  paddle::dialect::Asinh_Op asinh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Asinh_Op>(x);
  return asinh__op.result(0);
}

pir::OpResult atan(const pir::Value& x) {
  CheckValueDataType(x, "x", "atan");
  paddle::dialect::AtanOp atan_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanOp>(x);
  return atan_op.result(0);
}

pir::OpResult atan_(const pir::Value& x) {
  CheckValueDataType(x, "x", "atan_");
  paddle::dialect::Atan_Op atan__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Atan_Op>(x);
  return atan__op.result(0);
}

pir::OpResult atan2(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "atan2");
  paddle::dialect::Atan2Op atan2_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Atan2Op>(x,
                                                                           y);
  return atan2_op.result(0);
}

pir::OpResult atanh(const pir::Value& x) {
  CheckValueDataType(x, "x", "atanh");
  paddle::dialect::AtanhOp atanh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanhOp>(x);
  return atanh_op.result(0);
}

pir::OpResult atanh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "atanh_");
  paddle::dialect::Atanh_Op atanh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Atanh_Op>(x);
  return atanh__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> auc(
    const pir::Value& x,
    const pir::Value& label,
    const pir::Value& stat_pos,
    const pir::Value& stat_neg,
    const paddle::optional<pir::Value>& ins_tag_weight,
    const std::string& curve,
    int num_thresholds,
    int slide_steps) {
  CheckValueDataType(x, "x", "auc");
  paddle::optional<pir::Value> optional_ins_tag_weight;
  if (!ins_tag_weight) {
    optional_ins_tag_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ins_tag_weight = ins_tag_weight;
  }
  paddle::dialect::AucOp auc_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AucOp>(
          x,
          label,
          stat_pos,
          stat_neg,
          optional_ins_tag_weight.get(),
          curve,
          num_thresholds,
          slide_steps);
  return std::make_tuple(auc_op.result(0), auc_op.result(1), auc_op.result(2));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
average_accumulates_(const pir::Value& param,
                     const pir::Value& in_sum_1,
                     const pir::Value& in_sum_2,
                     const pir::Value& in_sum_3,
                     const pir::Value& in_num_accumulates,
                     const pir::Value& in_old_num_accumulates,
                     const pir::Value& in_num_updates,
                     float average_window,
                     int64_t max_average_window,
                     int64_t min_average_window) {
  CheckValueDataType(param, "param", "average_accumulates_");
  paddle::dialect::AverageAccumulates_Op average_accumulates__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AverageAccumulates_Op>(
              param,
              in_sum_1,
              in_sum_2,
              in_sum_3,
              in_num_accumulates,
              in_old_num_accumulates,
              in_num_updates,
              average_window,
              max_average_window,
              min_average_window);
  return std::make_tuple(average_accumulates__op.result(0),
                         average_accumulates__op.result(1),
                         average_accumulates__op.result(2),
                         average_accumulates__op.result(3),
                         average_accumulates__op.result(4),
                         average_accumulates__op.result(5));
}

pir::OpResult bce_loss(const pir::Value& input, const pir::Value& label) {
  CheckValueDataType(input, "input", "bce_loss");
  paddle::dialect::BceLossOp bce_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BceLossOp>(
          input, label);
  return bce_loss_op.result(0);
}

pir::OpResult bce_loss_(const pir::Value& input, const pir::Value& label) {
  CheckValueDataType(input, "input", "bce_loss_");
  paddle::dialect::BceLoss_Op bce_loss__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BceLoss_Op>(
          input, label);
  return bce_loss__op.result(0);
}

pir::OpResult bernoulli(const pir::Value& x) {
  CheckValueDataType(x, "x", "bernoulli");
  paddle::dialect::BernoulliOp bernoulli_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BernoulliOp>(
          x);
  return bernoulli_op.result(0);
}

pir::OpResult bicubic_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(x, "x", "bicubic_interp");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::BicubicInterpOp bicubic_interp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BicubicInterpOp>(x,
                                                    optional_out_size.get(),
                                                    optional_size_tensor.get(),
                                                    optional_scale_tensor.get(),
                                                    data_layout,
                                                    out_d,
                                                    out_h,
                                                    out_w,
                                                    scale,
                                                    interp_method,
                                                    align_corners,
                                                    align_mode);
  return bicubic_interp_op.result(0);
}

pir::OpResult bilinear(const pir::Value& x,
                       const pir::Value& y,
                       const pir::Value& weight,
                       const paddle::optional<pir::Value>& bias) {
  if (bias) {
    CheckValueDataType(bias.get(), "bias", "bilinear");
  } else {
    CheckValueDataType(weight, "weight", "bilinear");
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::BilinearOp bilinear_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BilinearOp>(
          x, y, weight, optional_bias.get());
  return bilinear_op.result(0);
}

pir::OpResult bilinear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(x, "x", "bilinear_interp");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::BilinearInterpOp bilinear_interp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BilinearInterpOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return bilinear_interp_op.result(0);
}

pir::OpResult bincount(const pir::Value& x,
                       const paddle::optional<pir::Value>& weights,
                       int minlength) {
  if (weights) {
    CheckValueDataType(weights.get(), "weights", "bincount");
  } else {
    CheckValueDataType(x, "x", "bincount");
  }
  paddle::optional<pir::Value> optional_weights;
  if (!weights) {
    optional_weights = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_weights = weights;
  }
  paddle::dialect::BincountOp bincount_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BincountOp>(
          x, optional_weights.get(), minlength);
  return bincount_op.result(0);
}

pir::OpResult bincount(const pir::Value& x,
                       const paddle::optional<pir::Value>& weights,
                       pir::Value minlength) {
  if (weights) {
    CheckValueDataType(weights.get(), "weights", "bincount");
  } else {
    CheckValueDataType(x, "x", "bincount");
  }
  paddle::optional<pir::Value> optional_weights;
  if (!weights) {
    optional_weights = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_weights = weights;
  }
  paddle::dialect::BincountOp bincount_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BincountOp>(
          x, optional_weights.get(), minlength);
  return bincount_op.result(0);
}

pir::OpResult binomial(const pir::Value& count, const pir::Value& prob) {
  CheckValueDataType(prob, "prob", "binomial");
  paddle::dialect::BinomialOp binomial_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BinomialOp>(
          count, prob);
  return binomial_op.result(0);
}

pir::OpResult bitwise_and(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_and");
  paddle::dialect::BitwiseAndOp bitwise_and_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BitwiseAndOp>(
          x, y);
  return bitwise_and_op.result(0);
}

pir::OpResult bitwise_and_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_and_");
  paddle::dialect::BitwiseAnd_Op bitwise_and__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BitwiseAnd_Op>(x, y);
  return bitwise_and__op.result(0);
}

pir::OpResult bitwise_not(const pir::Value& x) {
  CheckValueDataType(x, "x", "bitwise_not");
  paddle::dialect::BitwiseNotOp bitwise_not_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BitwiseNotOp>(
          x);
  return bitwise_not_op.result(0);
}

pir::OpResult bitwise_not_(const pir::Value& x) {
  CheckValueDataType(x, "x", "bitwise_not_");
  paddle::dialect::BitwiseNot_Op bitwise_not__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BitwiseNot_Op>(x);
  return bitwise_not__op.result(0);
}

pir::OpResult bitwise_or(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_or");
  paddle::dialect::BitwiseOrOp bitwise_or_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BitwiseOrOp>(
          x, y);
  return bitwise_or_op.result(0);
}

pir::OpResult bitwise_or_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_or_");
  paddle::dialect::BitwiseOr_Op bitwise_or__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BitwiseOr_Op>(
          x, y);
  return bitwise_or__op.result(0);
}

pir::OpResult bitwise_xor(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_xor");
  paddle::dialect::BitwiseXorOp bitwise_xor_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BitwiseXorOp>(
          x, y);
  return bitwise_xor_op.result(0);
}

pir::OpResult bitwise_xor_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bitwise_xor_");
  paddle::dialect::BitwiseXor_Op bitwise_xor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BitwiseXor_Op>(x, y);
  return bitwise_xor__op.result(0);
}

pir::OpResult bmm(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "bmm");
  paddle::dialect::BmmOp bmm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BmmOp>(x, y);
  return bmm_op.result(0);
}

pir::OpResult box_coder(const pir::Value& prior_box,
                        const paddle::optional<pir::Value>& prior_box_var,
                        const pir::Value& target_box,
                        const std::string& code_type,
                        bool box_normalized,
                        int axis,
                        const std::vector<float>& variance) {
  CheckValueDataType(target_box, "target_box", "box_coder");
  paddle::optional<pir::Value> optional_prior_box_var;
  if (!prior_box_var) {
    optional_prior_box_var = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_prior_box_var = prior_box_var;
  }
  paddle::dialect::BoxCoderOp box_coder_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BoxCoderOp>(
          prior_box,
          optional_prior_box_var.get(),
          target_box,
          code_type,
          box_normalized,
          axis,
          variance);
  return box_coder_op.result(0);
}

std::vector<pir::OpResult> broadcast_tensors(
    const std::vector<pir::Value>& input) {
  CheckVectorOfValueDataType(input, "input", "broadcast_tensors");
  auto input_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(input);
  paddle::dialect::BroadcastTensorsOp broadcast_tensors_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BroadcastTensorsOp>(input_combine_op.out());
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      broadcast_tensors_op.result(0));
  return out_split_op.outputs();
}

pir::OpResult ceil(const pir::Value& x) {
  CheckValueDataType(x, "x", "ceil");
  paddle::dialect::CeilOp ceil_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeilOp>(x);
  return ceil_op.result(0);
}

pir::OpResult ceil_(const pir::Value& x) {
  CheckValueDataType(x, "x", "ceil_");
  paddle::dialect::Ceil_Op ceil__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Ceil_Op>(x);
  return ceil__op.result(0);
}

pir::OpResult celu(const pir::Value& x, float alpha) {
  CheckValueDataType(x, "x", "celu");
  paddle::dialect::CeluOp celu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeluOp>(
          x, alpha);
  return celu_op.result(0);
}

std::tuple<std::vector<pir::OpResult>, pir::OpResult> check_finite_and_unscale_(
    const std::vector<pir::Value>& x, const pir::Value& scale) {
  CheckVectorOfValueDataType(x, "x", "check_finite_and_unscale_");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::CheckFiniteAndUnscale_Op check_finite_and_unscale__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CheckFiniteAndUnscale_Op>(x_combine_op.out(),
                                                             scale);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      check_finite_and_unscale__op.result(0));
  return std::make_tuple(out_split_op.outputs(),
                         check_finite_and_unscale__op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> check_numerics(
    const pir::Value& tensor,
    const std::string& op_type,
    const std::string& var_name,
    int check_nan_inf_level,
    int stack_height_limit,
    const std::string& output_dir) {
  CheckValueDataType(tensor, "tensor", "check_numerics");
  paddle::dialect::CheckNumericsOp check_numerics_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CheckNumericsOp>(tensor,
                                                    op_type,
                                                    var_name,
                                                    check_nan_inf_level,
                                                    stack_height_limit,
                                                    output_dir);
  return std::make_tuple(check_numerics_op.result(0),
                         check_numerics_op.result(1));
}

pir::OpResult cholesky(const pir::Value& x, bool upper) {
  CheckValueDataType(x, "x", "cholesky");
  paddle::dialect::CholeskyOp cholesky_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CholeskyOp>(
          x, upper);
  return cholesky_op.result(0);
}

pir::OpResult cholesky_solve(const pir::Value& x,
                             const pir::Value& y,
                             bool upper) {
  CheckValueDataType(y, "y", "cholesky_solve");
  paddle::dialect::CholeskySolveOp cholesky_solve_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CholeskySolveOp>(x, y, upper);
  return cholesky_solve_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> class_center_sample(
    const pir::Value& label,
    int num_classes,
    int num_samples,
    int ring_id,
    int rank,
    int nranks,
    bool fix_seed,
    int seed) {
  CheckValueDataType(label, "label", "class_center_sample");
  paddle::dialect::ClassCenterSampleOp class_center_sample_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ClassCenterSampleOp>(label,
                                                        num_classes,
                                                        num_samples,
                                                        ring_id,
                                                        rank,
                                                        nranks,
                                                        fix_seed,
                                                        seed);
  return std::make_tuple(class_center_sample_op.result(0),
                         class_center_sample_op.result(1));
}

pir::OpResult clip(const pir::Value& x, float min, float max) {
  CheckValueDataType(x, "x", "clip");
  paddle::dialect::ClipOp clip_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipOp>(
          x, min, max);
  return clip_op.result(0);
}

pir::OpResult clip(const pir::Value& x, pir::Value min, pir::Value max) {
  CheckValueDataType(x, "x", "clip");
  paddle::dialect::ClipOp clip_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipOp>(
          x, min, max);
  return clip_op.result(0);
}

pir::OpResult clip_(const pir::Value& x, float min, float max) {
  CheckValueDataType(x, "x", "clip_");
  paddle::dialect::Clip_Op clip__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Clip_Op>(
          x, min, max);
  return clip__op.result(0);
}

pir::OpResult clip_(const pir::Value& x, pir::Value min, pir::Value max) {
  CheckValueDataType(x, "x", "clip_");
  paddle::dialect::Clip_Op clip__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Clip_Op>(
          x, min, max);
  return clip__op.result(0);
}

pir::OpResult clip_by_norm(const pir::Value& x, float max_norm) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "clip_by_norm");
    paddle::dialect::ClipByNormOp clip_by_norm_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::ClipByNormOp>(x, max_norm);
    return clip_by_norm_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "clip_by_norm_sr");
    paddle::dialect::ClipByNormSrOp clip_by_norm_sr_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::ClipByNormSrOp>(x, max_norm);
    return clip_by_norm_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (clip_by_norm) for input Value is unimplemented, please "
      "check the type of input Value."));
}

std::tuple<std::vector<pir::OpResult>, pir::OpResult> coalesce_tensor(
    const std::vector<pir::Value>& input,
    phi::DataType dtype,
    bool copy_data,
    bool set_constant,
    bool persist_output,
    float constant,
    bool use_align,
    int align_size,
    int size_of_dtype,
    const std::vector<int64_t>& concated_shapes,
    const std::vector<int64_t>& concated_ranks) {
  CheckDataType(dtype, "dtype", "coalesce_tensor");
  auto input_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(input);
  paddle::dialect::CoalesceTensorOp coalesce_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CoalesceTensorOp>(input_combine_op.out(),
                                                     dtype,
                                                     copy_data,
                                                     set_constant,
                                                     persist_output,
                                                     constant,
                                                     use_align,
                                                     align_size,
                                                     size_of_dtype,
                                                     concated_shapes,
                                                     concated_ranks);
  auto output_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          coalesce_tensor_op.result(0));
  return std::make_tuple(output_split_op.outputs(),
                         coalesce_tensor_op.result(1));
}

pir::OpResult complex(const pir::Value& real, const pir::Value& imag) {
  CheckValueDataType(real, "real", "complex");
  paddle::dialect::ComplexOp complex_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ComplexOp>(
          real, imag);
  return complex_op.result(0);
}

pir::OpResult concat(const std::vector<pir::Value>& x, int axis) {
  CheckVectorOfValueDataType(x, "x", "concat");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::ConcatOp concat_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ConcatOp>(
          x_combine_op.out(), axis);
  return concat_op.result(0);
}

pir::OpResult concat(const std::vector<pir::Value>& x, pir::Value axis) {
  CheckVectorOfValueDataType(x, "x", "concat");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::ConcatOp concat_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ConcatOp>(
          x_combine_op.out(), axis);
  return concat_op.result(0);
}

pir::OpResult conj(const pir::Value& x) {
  CheckValueDataType(x, "x", "conj");
  paddle::dialect::ConjOp conj_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ConjOp>(x);
  return conj_op.result(0);
}

pir::OpResult conv2d(const pir::Value& input,
                     const pir::Value& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations,
                     int groups,
                     const std::string& data_format) {
  CheckValueDataType(filter, "filter", "conv2d");
  paddle::dialect::Conv2dOp conv2d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv2dOp>(
          input,
          filter,
          strides,
          paddings,
          padding_algorithm,
          dilations,
          groups,
          data_format);
  return conv2d_op.result(0);
}

pir::OpResult conv3d(const pir::Value& input,
                     const pir::Value& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::string& padding_algorithm,
                     int groups,
                     const std::vector<int>& dilations,
                     const std::string& data_format) {
  CheckValueDataType(filter, "filter", "conv3d");
  paddle::dialect::Conv3dOp conv3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv3dOp>(
          input,
          filter,
          strides,
          paddings,
          padding_algorithm,
          groups,
          dilations,
          data_format);
  return conv3d_op.result(0);
}

pir::OpResult conv3d_transpose(const pir::Value& x,
                               const pir::Value& filter,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format) {
  CheckValueDataType(x, "x", "conv3d_transpose");
  paddle::dialect::Conv3dTransposeOp conv3d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv3dTransposeOp>(x,
                                                      filter,
                                                      strides,
                                                      paddings,
                                                      output_padding,
                                                      output_size,
                                                      padding_algorithm,
                                                      groups,
                                                      dilations,
                                                      data_format);
  return conv3d_transpose_op.result(0);
}

pir::OpResult cos(const pir::Value& x) {
  CheckValueDataType(x, "x", "cos");
  paddle::dialect::CosOp cos_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CosOp>(x);
  return cos_op.result(0);
}

pir::OpResult cos_(const pir::Value& x) {
  CheckValueDataType(x, "x", "cos_");
  paddle::dialect::Cos_Op cos__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cos_Op>(x);
  return cos__op.result(0);
}

pir::OpResult cosh(const pir::Value& x) {
  CheckValueDataType(x, "x", "cosh");
  paddle::dialect::CoshOp cosh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CoshOp>(x);
  return cosh_op.result(0);
}

pir::OpResult cosh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "cosh_");
  paddle::dialect::Cosh_Op cosh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cosh_Op>(x);
  return cosh__op.result(0);
}

pir::OpResult crop(const pir::Value& x,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& offsets) {
  CheckValueDataType(x, "x", "crop");
  paddle::dialect::CropOp crop_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropOp>(
          x, shape, offsets);
  return crop_op.result(0);
}

pir::OpResult crop(const pir::Value& x, pir::Value shape, pir::Value offsets) {
  CheckValueDataType(x, "x", "crop");
  paddle::dialect::CropOp crop_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropOp>(
          x, shape, offsets);
  return crop_op.result(0);
}

pir::OpResult crop(const pir::Value& x,
                   std::vector<pir::Value> shape,
                   std::vector<pir::Value> offsets) {
  CheckValueDataType(x, "x", "crop");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  auto offsets_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(offsets);
  paddle::dialect::CropOp crop_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropOp>(
          x, shape_combine_op.out(), offsets_combine_op.out());
  return crop_op.result(0);
}

pir::OpResult cross(const pir::Value& x, const pir::Value& y, int axis) {
  CheckValueDataType(x, "x", "cross");
  paddle::dialect::CrossOp cross_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CrossOp>(
          x, y, axis);
  return cross_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> cross_entropy_with_softmax(
    const pir::Value& input,
    const pir::Value& label,
    bool soft_label,
    bool use_softmax,
    bool numeric_stable_mode,
    int ignore_index,
    int axis) {
  CheckValueDataType(input, "input", "cross_entropy_with_softmax");
  paddle::dialect::CrossEntropyWithSoftmaxOp cross_entropy_with_softmax_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CrossEntropyWithSoftmaxOp>(
              input,
              label,
              soft_label,
              use_softmax,
              numeric_stable_mode,
              ignore_index,
              axis);
  return std::make_tuple(cross_entropy_with_softmax_op.result(0),
                         cross_entropy_with_softmax_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> cross_entropy_with_softmax_(
    const pir::Value& input,
    const pir::Value& label,
    bool soft_label,
    bool use_softmax,
    bool numeric_stable_mode,
    int ignore_index,
    int axis) {
  CheckValueDataType(input, "input", "cross_entropy_with_softmax_");
  paddle::dialect::CrossEntropyWithSoftmax_Op cross_entropy_with_softmax__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CrossEntropyWithSoftmax_Op>(
              input,
              label,
              soft_label,
              use_softmax,
              numeric_stable_mode,
              ignore_index,
              axis);
  return std::make_tuple(cross_entropy_with_softmax__op.result(0),
                         cross_entropy_with_softmax__op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> cummax(const pir::Value& x,
                                                int axis,
                                                phi::DataType dtype) {
  CheckValueDataType(x, "x", "cummax");
  paddle::dialect::CummaxOp cummax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CummaxOp>(
          x, axis, dtype);
  return std::make_tuple(cummax_op.result(0), cummax_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> cummin(const pir::Value& x,
                                                int axis,
                                                phi::DataType dtype) {
  CheckValueDataType(x, "x", "cummin");
  paddle::dialect::CumminOp cummin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumminOp>(
          x, axis, dtype);
  return std::make_tuple(cummin_op.result(0), cummin_op.result(1));
}

pir::OpResult cumprod(const pir::Value& x, int dim) {
  CheckValueDataType(x, "x", "cumprod");
  paddle::dialect::CumprodOp cumprod_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumprodOp>(
          x, dim);
  return cumprod_op.result(0);
}

pir::OpResult cumprod_(const pir::Value& x, int dim) {
  CheckValueDataType(x, "x", "cumprod_");
  paddle::dialect::Cumprod_Op cumprod__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cumprod_Op>(
          x, dim);
  return cumprod__op.result(0);
}

pir::OpResult cumsum(
    const pir::Value& x, int axis, bool flatten, bool exclusive, bool reverse) {
  CheckValueDataType(x, "x", "cumsum");
  paddle::dialect::CumsumOp cumsum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumsumOp>(
          x, axis, flatten, exclusive, reverse);
  return cumsum_op.result(0);
}

pir::OpResult cumsum(const pir::Value& x,
                     pir::Value axis,
                     bool flatten,
                     bool exclusive,
                     bool reverse) {
  CheckValueDataType(x, "x", "cumsum");
  paddle::dialect::CumsumOp cumsum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumsumOp>(
          x, axis, flatten, exclusive, reverse);
  return cumsum_op.result(0);
}

pir::OpResult cumsum_(
    const pir::Value& x, int axis, bool flatten, bool exclusive, bool reverse) {
  CheckValueDataType(x, "x", "cumsum_");
  paddle::dialect::Cumsum_Op cumsum__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cumsum_Op>(
          x, axis, flatten, exclusive, reverse);
  return cumsum__op.result(0);
}

pir::OpResult cumsum_(const pir::Value& x,
                      pir::Value axis,
                      bool flatten,
                      bool exclusive,
                      bool reverse) {
  CheckValueDataType(x, "x", "cumsum_");
  paddle::dialect::Cumsum_Op cumsum__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cumsum_Op>(
          x, axis, flatten, exclusive, reverse);
  return cumsum__op.result(0);
}

pir::OpResult data(const std::string& name,
                   const std::vector<int64_t>& shape,
                   phi::DataType dtype,
                   const Place& place) {
  CheckDataType(dtype, "dtype", "data");
  paddle::dialect::DataOp data_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DataOp>(
          name, shape, dtype, place);
  return data_op.result(0);
}

pir::OpResult depthwise_conv2d(const pir::Value& input,
                               const pir::Value& filter,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format) {
  CheckValueDataType(filter, "filter", "depthwise_conv2d");
  paddle::dialect::DepthwiseConv2dOp depthwise_conv2d_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dOp>(input,
                                                      filter,
                                                      strides,
                                                      paddings,
                                                      padding_algorithm,
                                                      groups,
                                                      dilations,
                                                      data_format);
  return depthwise_conv2d_op.result(0);
}

pir::OpResult det(const pir::Value& x) {
  CheckValueDataType(x, "x", "determinant");
  paddle::dialect::DetOp det_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DetOp>(x);
  return det_op.result(0);
}

pir::OpResult diag(const pir::Value& x, int offset, float padding_value) {
  CheckValueDataType(x, "x", "diag");
  paddle::dialect::DiagOp diag_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DiagOp>(
          x, offset, padding_value);
  return diag_op.result(0);
}

pir::OpResult diag_embed(const pir::Value& input,
                         int offset,
                         int dim1,
                         int dim2) {
  CheckValueDataType(input, "input", "diag_embed");
  paddle::dialect::DiagEmbedOp diag_embed_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DiagEmbedOp>(
          input, offset, dim1, dim2);
  return diag_embed_op.result(0);
}

pir::OpResult diagonal(const pir::Value& x, int offset, int axis1, int axis2) {
  CheckValueDataType(x, "x", "diagonal");
  paddle::dialect::DiagonalOp diagonal_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DiagonalOp>(
          x, offset, axis1, axis2);
  return diagonal_op.result(0);
}

pir::OpResult digamma(const pir::Value& x) {
  CheckValueDataType(x, "x", "digamma");
  paddle::dialect::DigammaOp digamma_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DigammaOp>(x);
  return digamma_op.result(0);
}

pir::OpResult digamma_(const pir::Value& x) {
  CheckValueDataType(x, "x", "digamma_");
  paddle::dialect::Digamma_Op digamma__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Digamma_Op>(
          x);
  return digamma__op.result(0);
}

pir::OpResult dirichlet(const pir::Value& alpha) {
  CheckValueDataType(alpha, "alpha", "dirichlet");
  paddle::dialect::DirichletOp dirichlet_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DirichletOp>(
          alpha);
  return dirichlet_op.result(0);
}

pir::OpResult dist(const pir::Value& x, const pir::Value& y, float p) {
  CheckValueDataType(y, "y", "dist");
  paddle::dialect::DistOp dist_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DistOp>(
          x, y, p);
  return dist_op.result(0);
}

pir::OpResult dot(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "dot");
  paddle::dialect::DotOp dot_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DotOp>(x, y);
  return dot_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> edit_distance(
    const pir::Value& hyps,
    const pir::Value& refs,
    const paddle::optional<pir::Value>& hypslength,
    const paddle::optional<pir::Value>& refslength,
    bool normalized) {
  paddle::optional<pir::Value> optional_hypslength;
  if (!hypslength) {
    optional_hypslength = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_hypslength = hypslength;
  }
  paddle::optional<pir::Value> optional_refslength;
  if (!refslength) {
    optional_refslength = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_refslength = refslength;
  }
  paddle::dialect::EditDistanceOp edit_distance_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::EditDistanceOp>(hyps,
                                                   refs,
                                                   optional_hypslength.get(),
                                                   optional_refslength.get(),
                                                   normalized);
  return std::make_tuple(edit_distance_op.result(0),
                         edit_distance_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> eig(const pir::Value& x) {
  CheckValueDataType(x, "x", "eig");
  paddle::dialect::EigOp eig_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EigOp>(x);
  return std::make_tuple(eig_op.result(0), eig_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> eigh(const pir::Value& x,
                                              const std::string& UPLO) {
  CheckValueDataType(x, "x", "eigh");
  paddle::dialect::EighOp eigh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EighOp>(x,
                                                                          UPLO);
  return std::make_tuple(eigh_op.result(0), eigh_op.result(1));
}

pir::OpResult eigvals(const pir::Value& x) {
  CheckValueDataType(x, "x", "eigvals");
  paddle::dialect::EigvalsOp eigvals_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EigvalsOp>(x);
  return eigvals_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> eigvalsh(const pir::Value& x,
                                                  const std::string& uplo,
                                                  bool is_test) {
  CheckValueDataType(x, "x", "eigvalsh");
  paddle::dialect::EigvalshOp eigvalsh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EigvalshOp>(
          x, uplo, is_test);
  return std::make_tuple(eigvalsh_op.result(0), eigvalsh_op.result(1));
}

pir::OpResult elu(const pir::Value& x, float alpha) {
  CheckValueDataType(x, "x", "elu");
  paddle::dialect::EluOp elu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EluOp>(x,
                                                                         alpha);
  return elu_op.result(0);
}

pir::OpResult elu_(const pir::Value& x, float alpha) {
  CheckValueDataType(x, "x", "elu_");
  paddle::dialect::Elu_Op elu__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Elu_Op>(
          x, alpha);
  return elu__op.result(0);
}

pir::OpResult equal_all(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "equal_all");
  paddle::dialect::EqualAllOp equal_all_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EqualAllOp>(
          x, y);
  return equal_all_op.result(0);
}

pir::OpResult erf(const pir::Value& x) {
  CheckValueDataType(x, "x", "erf");
  paddle::dialect::ErfOp erf_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ErfOp>(x);
  return erf_op.result(0);
}

pir::OpResult erf_(const pir::Value& x) {
  CheckValueDataType(x, "x", "erf_");
  paddle::dialect::Erf_Op erf__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Erf_Op>(x);
  return erf__op.result(0);
}

pir::OpResult erfinv(const pir::Value& x) {
  CheckValueDataType(x, "x", "erfinv");
  paddle::dialect::ErfinvOp erfinv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ErfinvOp>(x);
  return erfinv_op.result(0);
}

pir::OpResult erfinv_(const pir::Value& x) {
  CheckValueDataType(x, "x", "erfinv_");
  paddle::dialect::Erfinv_Op erfinv__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Erfinv_Op>(x);
  return erfinv__op.result(0);
}

pir::OpResult exp(const pir::Value& x) {
  CheckValueDataType(x, "x", "exp");
  paddle::dialect::ExpOp exp_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpOp>(x);
  return exp_op.result(0);
}

pir::OpResult exp_(const pir::Value& x) {
  CheckValueDataType(x, "x", "exp_");
  paddle::dialect::Exp_Op exp__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Exp_Op>(x);
  return exp__op.result(0);
}

pir::OpResult expand(const pir::Value& x, const std::vector<int64_t>& shape) {
  CheckValueDataType(x, "x", "expand");
  paddle::dialect::ExpandOp expand_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandOp>(
          x, shape);
  return expand_op.result(0);
}

pir::OpResult expand(const pir::Value& x, pir::Value shape) {
  CheckValueDataType(x, "x", "expand");
  paddle::dialect::ExpandOp expand_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandOp>(
          x, shape);
  return expand_op.result(0);
}

pir::OpResult expand(const pir::Value& x, std::vector<pir::Value> shape) {
  CheckValueDataType(x, "x", "expand");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::ExpandOp expand_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandOp>(
          x, shape_combine_op.out());
  return expand_op.result(0);
}

pir::OpResult expand_as(const pir::Value& x,
                        const paddle::optional<pir::Value>& y,
                        const std::vector<int>& target_shape) {
  CheckValueDataType(x, "x", "expand_as");
  paddle::optional<pir::Value> optional_y;
  if (!y) {
    optional_y = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_y = y;
  }
  paddle::dialect::ExpandAsOp expand_as_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandAsOp>(
          x, optional_y.get(), target_shape);
  return expand_as_op.result(0);
}

pir::OpResult expm1(const pir::Value& x) {
  CheckValueDataType(x, "x", "expm1");
  paddle::dialect::Expm1Op expm1_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Expm1Op>(x);
  return expm1_op.result(0);
}

pir::OpResult expm1_(const pir::Value& x) {
  CheckValueDataType(x, "x", "expm1_");
  paddle::dialect::Expm1_Op expm1__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Expm1_Op>(x);
  return expm1__op.result(0);
}

pir::OpResult fft_c2c(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward) {
  CheckValueDataType(x, "x", "fft_c2c");
  paddle::dialect::FftC2cOp fft_c2c_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftC2cOp>(
          x, axes, normalization, forward);
  return fft_c2c_op.result(0);
}

pir::OpResult fft_c2r(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      int64_t last_dim_size) {
  CheckValueDataType(x, "x", "fft_c2r");
  paddle::dialect::FftC2rOp fft_c2r_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftC2rOp>(
          x, axes, normalization, forward, last_dim_size);
  return fft_c2r_op.result(0);
}

pir::OpResult fft_r2c(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      bool onesided) {
  CheckValueDataType(x, "x", "fft_r2c");
  paddle::dialect::FftR2cOp fft_r2c_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftR2cOp>(
          x, axes, normalization, forward, onesided);
  return fft_r2c_op.result(0);
}

pir::OpResult fill(const pir::Value& x, float value) {
  CheckValueDataType(x, "x", "fill");
  paddle::dialect::FillOp fill_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillOp>(
          x, value);
  return fill_op.result(0);
}

pir::OpResult fill(const pir::Value& x, pir::Value value) {
  CheckValueDataType(x, "x", "fill");
  paddle::dialect::FillOp fill_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillOp>(
          x, value);
  return fill_op.result(0);
}

pir::OpResult fill_(const pir::Value& x, float value) {
  CheckValueDataType(x, "x", "fill_");
  paddle::dialect::Fill_Op fill__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Fill_Op>(
          x, value);
  return fill__op.result(0);
}

pir::OpResult fill_(const pir::Value& x, pir::Value value) {
  CheckValueDataType(x, "x", "fill_");
  paddle::dialect::Fill_Op fill__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Fill_Op>(
          x, value);
  return fill__op.result(0);
}

pir::OpResult fill_diagonal(const pir::Value& x,
                            float value,
                            int offset,
                            bool wrap) {
  CheckValueDataType(x, "x", "fill_diagonal");
  paddle::dialect::FillDiagonalOp fill_diagonal_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalOp>(x, value, offset, wrap);
  return fill_diagonal_op.result(0);
}

pir::OpResult fill_diagonal_(const pir::Value& x,
                             float value,
                             int offset,
                             bool wrap) {
  CheckValueDataType(x, "x", "fill_diagonal_");
  paddle::dialect::FillDiagonal_Op fill_diagonal__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonal_Op>(x, value, offset, wrap);
  return fill_diagonal__op.result(0);
}

pir::OpResult fill_diagonal_tensor(const pir::Value& x,
                                   const pir::Value& y,
                                   int64_t offset,
                                   int dim1,
                                   int dim2) {
  CheckValueDataType(y, "y", "fill_diagonal_tensor");
  paddle::dialect::FillDiagonalTensorOp fill_diagonal_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalTensorOp>(
              x, y, offset, dim1, dim2);
  return fill_diagonal_tensor_op.result(0);
}

pir::OpResult fill_diagonal_tensor_(const pir::Value& x,
                                    const pir::Value& y,
                                    int64_t offset,
                                    int dim1,
                                    int dim2) {
  CheckValueDataType(y, "y", "fill_diagonal_tensor_");
  paddle::dialect::FillDiagonalTensor_Op fill_diagonal_tensor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalTensor_Op>(
              x, y, offset, dim1, dim2);
  return fill_diagonal_tensor__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
flash_attn(const pir::Value& q,
           const pir::Value& k,
           const pir::Value& v,
           const paddle::optional<pir::Value>& fixed_seed_offset,
           const paddle::optional<pir::Value>& attn_mask,
           float dropout,
           bool causal,
           bool return_softmax,
           bool is_test,
           const std::string& rng_name) {
  CheckValueDataType(q, "q", "flash_attn");
  paddle::optional<pir::Value> optional_fixed_seed_offset;
  if (!fixed_seed_offset) {
    optional_fixed_seed_offset =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fixed_seed_offset = fixed_seed_offset;
  }
  paddle::optional<pir::Value> optional_attn_mask;
  if (!attn_mask) {
    optional_attn_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_attn_mask = attn_mask;
  }
  paddle::dialect::FlashAttnOp flash_attn_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FlashAttnOp>(
          q,
          k,
          v,
          optional_fixed_seed_offset.get(),
          optional_attn_mask.get(),
          dropout,
          causal,
          return_softmax,
          is_test,
          rng_name);
  return std::make_tuple(flash_attn_op.result(0),
                         flash_attn_op.result(1),
                         flash_attn_op.result(2),
                         flash_attn_op.result(3));
}

std::tuple<pir::OpResult, pir::OpResult> flash_attn_unpadded(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const pir::Value& cu_seqlens_q,
    const pir::Value& cu_seqlens_k,
    const paddle::optional<pir::Value>& fixed_seed_offset,
    const paddle::optional<pir::Value>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name) {
  CheckValueDataType(q, "q", "flash_attn_unpadded");
  paddle::optional<pir::Value> optional_fixed_seed_offset;
  if (!fixed_seed_offset) {
    optional_fixed_seed_offset =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fixed_seed_offset = fixed_seed_offset;
  }
  paddle::optional<pir::Value> optional_attn_mask;
  if (!attn_mask) {
    optional_attn_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_attn_mask = attn_mask;
  }
  paddle::dialect::FlashAttnUnpaddedOp flash_attn_unpadded_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FlashAttnUnpaddedOp>(
              q,
              k,
              v,
              cu_seqlens_q,
              cu_seqlens_k,
              optional_fixed_seed_offset.get(),
              optional_attn_mask.get(),
              max_seqlen_q,
              max_seqlen_k,
              scale,
              dropout,
              causal,
              return_softmax,
              is_test,
              rng_name);
  return std::make_tuple(flash_attn_unpadded_op.result(0),
                         flash_attn_unpadded_op.result(1));
}

pir::OpResult flatten(const pir::Value& x, int start_axis, int stop_axis) {
  CheckValueDataType(x, "x", "flatten");
  paddle::dialect::FlattenOp flatten_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FlattenOp>(
          x, start_axis, stop_axis);
  return flatten_op.result(0);
}

pir::OpResult flatten_(const pir::Value& x, int start_axis, int stop_axis) {
  CheckValueDataType(x, "x", "flatten_");
  paddle::dialect::Flatten_Op flatten__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Flatten_Op>(
          x, start_axis, stop_axis);
  return flatten__op.result(0);
}

pir::OpResult flip(const pir::Value& x, const std::vector<int>& axis) {
  CheckValueDataType(x, "x", "flip");
  paddle::dialect::FlipOp flip_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FlipOp>(x,
                                                                          axis);
  return flip_op.result(0);
}

pir::OpResult floor(const pir::Value& x) {
  CheckValueDataType(x, "x", "floor");
  paddle::dialect::FloorOp floor_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FloorOp>(x);
  return floor_op.result(0);
}

pir::OpResult floor_(const pir::Value& x) {
  CheckValueDataType(x, "x", "floor_");
  paddle::dialect::Floor_Op floor__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Floor_Op>(x);
  return floor__op.result(0);
}

pir::OpResult fmax(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "fmax");
  paddle::dialect::FmaxOp fmax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FmaxOp>(x, y);
  return fmax_op.result(0);
}

pir::OpResult fmin(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "fmin");
  paddle::dialect::FminOp fmin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FminOp>(x, y);
  return fmin_op.result(0);
}

pir::OpResult fold(const pir::Value& x,
                   const std::vector<int>& output_sizes,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations) {
  CheckValueDataType(x, "x", "fold");
  paddle::dialect::FoldOp fold_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FoldOp>(
          x, output_sizes, kernel_sizes, strides, paddings, dilations);
  return fold_op.result(0);
}

pir::OpResult frame(const pir::Value& x,
                    int frame_length,
                    int hop_length,
                    int axis) {
  CheckValueDataType(x, "x", "frame");
  paddle::dialect::FrameOp frame_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FrameOp>(
          x, frame_length, hop_length, axis);
  return frame_op.result(0);
}

pir::OpResult full_int_array(const std::vector<int64_t>& value,
                             phi::DataType dtype,
                             const Place& place) {
  CheckDataType(dtype, "dtype", "full_int_array");
  paddle::dialect::FullIntArrayOp full_int_array_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FullIntArrayOp>(value, dtype, place);
  return full_int_array_op.result(0);
}

pir::OpResult gammaln(const pir::Value& x) {
  CheckValueDataType(x, "x", "gammaln");
  paddle::dialect::GammalnOp gammaln_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GammalnOp>(x);
  return gammaln_op.result(0);
}

pir::OpResult gammaln_(const pir::Value& x) {
  CheckValueDataType(x, "x", "gammaln_");
  paddle::dialect::Gammaln_Op gammaln__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Gammaln_Op>(
          x);
  return gammaln__op.result(0);
}

pir::OpResult gather(const pir::Value& x, const pir::Value& index, int axis) {
  CheckValueDataType(x, "x", "gather");
  paddle::dialect::GatherOp gather_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherOp>(
          x, index, axis);
  return gather_op.result(0);
}

pir::OpResult gather(const pir::Value& x,
                     const pir::Value& index,
                     pir::Value axis) {
  CheckValueDataType(x, "x", "gather");
  paddle::dialect::GatherOp gather_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherOp>(
          x, index, axis);
  return gather_op.result(0);
}

pir::OpResult gather_nd(const pir::Value& x, const pir::Value& index) {
  CheckValueDataType(x, "x", "gather_nd");
  paddle::dialect::GatherNdOp gather_nd_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherNdOp>(
          x, index);
  return gather_nd_op.result(0);
}

pir::OpResult gather_tree(const pir::Value& ids, const pir::Value& parents) {
  CheckValueDataType(ids, "ids", "gather_tree");
  paddle::dialect::GatherTreeOp gather_tree_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherTreeOp>(
          ids, parents);
  return gather_tree_op.result(0);
}

pir::OpResult gaussian_inplace(const pir::Value& x,
                               float mean,
                               float std,
                               int seed) {
  CheckValueDataType(x, "x", "gaussian_inplace");
  paddle::dialect::GaussianInplaceOp gaussian_inplace_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GaussianInplaceOp>(x, mean, std, seed);
  return gaussian_inplace_op.result(0);
}

pir::OpResult gaussian_inplace_(const pir::Value& x,
                                float mean,
                                float std,
                                int seed) {
  CheckValueDataType(x, "x", "gaussian_inplace_");
  paddle::dialect::GaussianInplace_Op gaussian_inplace__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GaussianInplace_Op>(x, mean, std, seed);
  return gaussian_inplace__op.result(0);
}

pir::OpResult gelu(const pir::Value& x, bool approximate) {
  CheckValueDataType(x, "x", "gelu");
  paddle::dialect::GeluOp gelu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GeluOp>(
          x, approximate);
  return gelu_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> generate_proposals(
    const pir::Value& scores,
    const pir::Value& bbox_deltas,
    const pir::Value& im_shape,
    const pir::Value& anchors,
    const pir::Value& variances,
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset) {
  CheckValueDataType(anchors, "anchors", "generate_proposals");
  paddle::dialect::GenerateProposalsOp generate_proposals_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GenerateProposalsOp>(scores,
                                                        bbox_deltas,
                                                        im_shape,
                                                        anchors,
                                                        variances,
                                                        pre_nms_top_n,
                                                        post_nms_top_n,
                                                        nms_thresh,
                                                        min_size,
                                                        eta,
                                                        pixel_offset);
  return std::make_tuple(generate_proposals_op.result(0),
                         generate_proposals_op.result(1),
                         generate_proposals_op.result(2));
}

pir::OpResult grid_sample(const pir::Value& x,
                          const pir::Value& grid,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners) {
  CheckValueDataType(x, "x", "grid_sample");
  paddle::dialect::GridSampleOp grid_sample_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GridSampleOp>(
          x, grid, mode, padding_mode, align_corners);
  return grid_sample_op.result(0);
}

pir::OpResult group_norm(const pir::Value& x,
                         const paddle::optional<pir::Value>& scale,
                         const paddle::optional<pir::Value>& bias,
                         float epsilon,
                         int groups,
                         const std::string& data_layout) {
  if (bias) {
    CheckValueDataType(bias.get(), "bias", "group_norm");
  } else if (scale) {
    CheckValueDataType(scale.get(), "scale", "group_norm");
  } else {
    CheckValueDataType(x, "x", "group_norm");
  }
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::GroupNormOp group_norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GroupNormOp>(
          x,
          optional_scale.get(),
          optional_bias.get(),
          epsilon,
          groups,
          data_layout);
  return group_norm_op.result(0);
}

pir::OpResult gumbel_softmax(const pir::Value& x,
                             float temperature,
                             bool hard,
                             int axis) {
  CheckValueDataType(x, "x", "gumbel_softmax");
  paddle::dialect::GumbelSoftmaxOp gumbel_softmax_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GumbelSoftmaxOp>(x, temperature, hard, axis);
  return gumbel_softmax_op.result(0);
}

pir::OpResult hardshrink(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "hard_shrink");
  paddle::dialect::HardshrinkOp hardshrink_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HardshrinkOp>(
          x, threshold);
  return hardshrink_op.result(0);
}

pir::OpResult hardsigmoid(const pir::Value& x, float slope, float offset) {
  CheckValueDataType(x, "x", "hardsigmoid");
  paddle::dialect::HardsigmoidOp hardsigmoid_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardsigmoidOp>(x, slope, offset);
  return hardsigmoid_op.result(0);
}

pir::OpResult hardtanh(const pir::Value& x, float t_min, float t_max) {
  CheckValueDataType(x, "x", "hardtanh");
  paddle::dialect::HardtanhOp hardtanh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HardtanhOp>(
          x, t_min, t_max);
  return hardtanh_op.result(0);
}

pir::OpResult hardtanh_(const pir::Value& x, float t_min, float t_max) {
  CheckValueDataType(x, "x", "hardtanh_");
  paddle::dialect::Hardtanh_Op hardtanh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Hardtanh_Op>(
          x, t_min, t_max);
  return hardtanh__op.result(0);
}

pir::OpResult heaviside(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "heaviside");
  paddle::dialect::HeavisideOp heaviside_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HeavisideOp>(
          x, y);
  return heaviside_op.result(0);
}

pir::OpResult histogram(const pir::Value& input,
                        int64_t bins,
                        int min,
                        int max) {
  CheckValueDataType(input, "input", "histogram");
  paddle::dialect::HistogramOp histogram_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HistogramOp>(
          input, bins, min, max);
  return histogram_op.result(0);
}

pir::OpResult huber_loss(const pir::Value& input,
                         const pir::Value& label,
                         float delta) {
  CheckValueDataType(label, "label", "huber_loss");
  paddle::dialect::HuberLossOp huber_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HuberLossOp>(
          input, label, delta);
  return huber_loss_op.result(0);
}

pir::OpResult i0(const pir::Value& x) {
  CheckValueDataType(x, "x", "i0");
  paddle::dialect::I0Op i0_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I0Op>(x);
  return i0_op.result(0);
}

pir::OpResult i0_(const pir::Value& x) {
  CheckValueDataType(x, "x", "i0_");
  paddle::dialect::I0_Op i0__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I0_Op>(x);
  return i0__op.result(0);
}

pir::OpResult i0e(const pir::Value& x) {
  CheckValueDataType(x, "x", "i0e");
  paddle::dialect::I0eOp i0e_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I0eOp>(x);
  return i0e_op.result(0);
}

pir::OpResult i1(const pir::Value& x) {
  CheckValueDataType(x, "x", "i1");
  paddle::dialect::I1Op i1_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I1Op>(x);
  return i1_op.result(0);
}

pir::OpResult i1e(const pir::Value& x) {
  CheckValueDataType(x, "x", "i1e");
  paddle::dialect::I1eOp i1e_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I1eOp>(x);
  return i1e_op.result(0);
}

pir::OpResult identity_loss(const pir::Value& x, int reduction) {
  CheckValueDataType(x, "x", "identity_loss");
  paddle::dialect::IdentityLossOp identity_loss_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IdentityLossOp>(x, reduction);
  return identity_loss_op.result(0);
}

pir::OpResult identity_loss_(const pir::Value& x, int reduction) {
  CheckValueDataType(x, "x", "identity_loss_");
  paddle::dialect::IdentityLoss_Op identity_loss__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IdentityLoss_Op>(x, reduction);
  return identity_loss__op.result(0);
}

pir::OpResult imag(const pir::Value& x) {
  CheckValueDataType(x, "x", "imag");
  paddle::dialect::ImagOp imag_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ImagOp>(x);
  return imag_op.result(0);
}

pir::OpResult index_add(const pir::Value& x,
                        const pir::Value& index,
                        const pir::Value& add_value,
                        int axis) {
  CheckValueDataType(x, "x", "index_add");
  paddle::dialect::IndexAddOp index_add_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IndexAddOp>(
          x, index, add_value, axis);
  return index_add_op.result(0);
}

pir::OpResult index_add_(const pir::Value& x,
                         const pir::Value& index,
                         const pir::Value& add_value,
                         int axis) {
  CheckValueDataType(x, "x", "index_add_");
  paddle::dialect::IndexAdd_Op index_add__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IndexAdd_Op>(
          x, index, add_value, axis);
  return index_add__op.result(0);
}

pir::OpResult index_put(const pir::Value& x,
                        const std::vector<pir::Value>& indices,
                        const pir::Value& value,
                        bool accumulate) {
  CheckValueDataType(x, "x", "index_put");
  auto indices_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(indices);
  paddle::dialect::IndexPutOp index_put_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IndexPutOp>(
          x, indices_combine_op.out(), value, accumulate);
  return index_put_op.result(0);
}

pir::OpResult index_put_(const pir::Value& x,
                         const std::vector<pir::Value>& indices,
                         const pir::Value& value,
                         bool accumulate) {
  CheckValueDataType(x, "x", "index_put_");
  auto indices_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(indices);
  paddle::dialect::IndexPut_Op index_put__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IndexPut_Op>(
          x, indices_combine_op.out(), value, accumulate);
  return index_put__op.result(0);
}

pir::OpResult index_sample(const pir::Value& x, const pir::Value& index) {
  CheckValueDataType(x, "x", "index_sample");
  paddle::dialect::IndexSampleOp index_sample_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSampleOp>(x, index);
  return index_sample_op.result(0);
}

pir::OpResult index_select(const pir::Value& x,
                           const pir::Value& index,
                           int axis) {
  CheckValueDataType(x, "x", "index_select");
  paddle::dialect::IndexSelectOp index_select_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSelectOp>(x, index, axis);
  return index_select_op.result(0);
}

pir::OpResult index_select_strided(const pir::Value& x,
                                   int64_t index,
                                   int axis) {
  CheckValueDataType(x, "x", "index_select_strided");
  paddle::dialect::IndexSelectStridedOp index_select_strided_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSelectStridedOp>(x, index, axis);
  return index_select_strided_op.result(0);
}

pir::OpResult instance_norm(const pir::Value& x,
                            const paddle::optional<pir::Value>& scale,
                            const paddle::optional<pir::Value>& bias,
                            float epsilon) {
  CheckValueDataType(x, "x", "instance_norm");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::InstanceNormOp instance_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::InstanceNormOp>(
              x, optional_scale.get(), optional_bias.get(), epsilon);
  return instance_norm_op.result(0);
}

pir::OpResult inverse(const pir::Value& x) {
  CheckValueDataType(x, "x", "inverse");
  paddle::dialect::InverseOp inverse_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::InverseOp>(x);
  return inverse_op.result(0);
}

pir::OpResult is_empty(const pir::Value& x) {
  CheckValueDataType(x, "x", "is_empty");
  paddle::dialect::IsEmptyOp is_empty_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsEmptyOp>(x);
  return is_empty_op.result(0);
}

pir::OpResult isclose(const pir::Value& x,
                      const pir::Value& y,
                      double rtol,
                      double atol,
                      bool equal_nan) {
  CheckValueDataType(x, "x", "isclose");
  paddle::dialect::IscloseOp isclose_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IscloseOp>(
          x, y, rtol, atol, equal_nan);
  return isclose_op.result(0);
}

pir::OpResult isclose(const pir::Value& x,
                      const pir::Value& y,
                      pir::Value rtol,
                      pir::Value atol,
                      bool equal_nan) {
  CheckValueDataType(x, "x", "isclose");
  paddle::dialect::IscloseOp isclose_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IscloseOp>(
          x, y, rtol, atol, equal_nan);
  return isclose_op.result(0);
}

pir::OpResult isfinite(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "isfinite");
    paddle::dialect::IsfiniteOp isfinite_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsfiniteOp>(
            x);
    return isfinite_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "isfinite_sr");
    paddle::dialect::IsfiniteSrOp isfinite_sr_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::IsfiniteSrOp>(x);
    return isfinite_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (isfinite) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult isinf(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "isinf");
    paddle::dialect::IsinfOp isinf_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsinfOp>(x);
    return isinf_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "isinf_sr");
    paddle::dialect::IsinfSrOp isinf_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsinfSrOp>(
            x);
    return isinf_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (isinf) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult isnan(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "isnan");
    paddle::dialect::IsnanOp isnan_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsnanOp>(x);
    return isnan_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "isnan_sr");
    paddle::dialect::IsnanSrOp isnan_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IsnanSrOp>(
            x);
    return isnan_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (isnan) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult kldiv_loss(const pir::Value& x,
                         const pir::Value& label,
                         const std::string& reduction) {
  CheckValueDataType(x, "x", "kldiv_loss");
  paddle::dialect::KldivLossOp kldiv_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::KldivLossOp>(
          x, label, reduction);
  return kldiv_loss_op.result(0);
}

pir::OpResult kron(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "kron");
  paddle::dialect::KronOp kron_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::KronOp>(x, y);
  return kron_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> kthvalue(const pir::Value& x,
                                                  int k,
                                                  int axis,
                                                  bool keepdim) {
  CheckValueDataType(x, "x", "kthvalue");
  paddle::dialect::KthvalueOp kthvalue_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::KthvalueOp>(
          x, k, axis, keepdim);
  return std::make_tuple(kthvalue_op.result(0), kthvalue_op.result(1));
}

pir::OpResult label_smooth(const pir::Value& label,
                           const paddle::optional<pir::Value>& prior_dist,
                           float epsilon) {
  CheckValueDataType(label, "label", "label_smooth");
  paddle::optional<pir::Value> optional_prior_dist;
  if (!prior_dist) {
    optional_prior_dist = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_prior_dist = prior_dist;
  }
  paddle::dialect::LabelSmoothOp label_smooth_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LabelSmoothOp>(
              label, optional_prior_dist.get(), epsilon);
  return label_smooth_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
lamb_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      float weight_decay,
      float beta1,
      float beta2,
      float epsilon,
      bool always_adapt,
      bool multi_precision) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "lamb_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::Lamb_Op lamb__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Lamb_Op>(
            param,
            grad,
            learning_rate,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            optional_master_param.get(),
            optional_skip_update.get(),
            weight_decay,
            beta1,
            beta2,
            epsilon,
            always_adapt,
            multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_outs;
    if (!IsEmptyValue(lamb__op.result(5))) {
      optional_master_param_outs =
          paddle::make_optional<pir::OpResult>(lamb__op.result(5));
    }
    if (!master_param) {
      lamb__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(lamb__op.result(0),
                           lamb__op.result(1),
                           lamb__op.result(2),
                           lamb__op.result(3),
                           lamb__op.result(4),
                           optional_master_param_outs);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      moment1.type().isa<paddle::dialect::DenseTensorType>() &&
      moment2.type().isa<paddle::dialect::DenseTensorType>() &&
      beta1_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      beta2_pow.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!skip_update ||
       skip_update->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "lamb_sr_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::optional<pir::Value> optional_skip_update;
    if (!skip_update) {
      optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_skip_update = skip_update;
    }
    paddle::dialect::LambSr_Op lamb_sr__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LambSr_Op>(
            param,
            grad,
            learning_rate,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            optional_master_param.get(),
            optional_skip_update.get(),
            weight_decay,
            beta1,
            beta2,
            epsilon,
            always_adapt,
            multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_outs;
    if (!IsEmptyValue(lamb_sr__op.result(5))) {
      optional_master_param_outs =
          paddle::make_optional<pir::OpResult>(lamb_sr__op.result(5));
    }
    if (!master_param) {
      lamb_sr__op.result(5).set_type(pir::Type());
    }
    return std::make_tuple(lamb_sr__op.result(0),
                           lamb_sr__op.result(1),
                           lamb_sr__op.result(2),
                           lamb_sr__op.result(3),
                           lamb_sr__op.result(4),
                           optional_master_param_outs);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (lamb_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult layer_norm(const pir::Value& x,
                         const paddle::optional<pir::Value>& scale,
                         const paddle::optional<pir::Value>& bias,
                         float epsilon,
                         int begin_norm_axis) {
  CheckValueDataType(x, "x", "layer_norm");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::LayerNormOp layer_norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LayerNormOp>(
          x,
          optional_scale.get(),
          optional_bias.get(),
          epsilon,
          begin_norm_axis);
  return layer_norm_op.result(0);
}

pir::OpResult leaky_relu(const pir::Value& x, float negative_slope) {
  CheckValueDataType(x, "x", "leaky_relu");
  paddle::dialect::LeakyReluOp leaky_relu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LeakyReluOp>(
          x, negative_slope);
  return leaky_relu_op.result(0);
}

pir::OpResult leaky_relu_(const pir::Value& x, float negative_slope) {
  CheckValueDataType(x, "x", "leaky_relu_");
  paddle::dialect::LeakyRelu_Op leaky_relu__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LeakyRelu_Op>(
          x, negative_slope);
  return leaky_relu__op.result(0);
}

pir::OpResult lerp(const pir::Value& x,
                   const pir::Value& y,
                   const pir::Value& weight) {
  CheckValueDataType(weight, "weight", "lerp");
  paddle::dialect::LerpOp lerp_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LerpOp>(
          x, y, weight);
  return lerp_op.result(0);
}

pir::OpResult lerp_(const pir::Value& x,
                    const pir::Value& y,
                    const pir::Value& weight) {
  CheckValueDataType(weight, "weight", "lerp_");
  paddle::dialect::Lerp_Op lerp__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Lerp_Op>(
          x, y, weight);
  return lerp__op.result(0);
}

pir::OpResult lgamma(const pir::Value& x) {
  CheckValueDataType(x, "x", "lgamma");
  paddle::dialect::LgammaOp lgamma_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LgammaOp>(x);
  return lgamma_op.result(0);
}

pir::OpResult lgamma_(const pir::Value& x) {
  CheckValueDataType(x, "x", "lgamma_");
  paddle::dialect::Lgamma_Op lgamma__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Lgamma_Op>(x);
  return lgamma__op.result(0);
}

pir::OpResult linear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(x, "x", "linear_interp");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::LinearInterpOp linear_interp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LinearInterpOp>(x,
                                                   optional_out_size.get(),
                                                   optional_size_tensor.get(),
                                                   optional_scale_tensor.get(),
                                                   data_layout,
                                                   out_d,
                                                   out_h,
                                                   out_w,
                                                   scale,
                                                   interp_method,
                                                   align_corners,
                                                   align_mode);
  return linear_interp_op.result(0);
}

pir::OpResult llm_int8_linear(const pir::Value& x,
                              const pir::Value& weight,
                              const paddle::optional<pir::Value>& bias,
                              const pir::Value& weight_scale,
                              float threshold) {
  CheckValueDataType(x, "x", "llm_int8_linear");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::LlmInt8LinearOp llm_int8_linear_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LlmInt8LinearOp>(
              x, weight, optional_bias.get(), weight_scale, threshold);
  return llm_int8_linear_op.result(0);
}

pir::OpResult log(const pir::Value& x) {
  CheckValueDataType(x, "x", "log");
  paddle::dialect::LogOp log_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogOp>(x);
  return log_op.result(0);
}

pir::OpResult log_(const pir::Value& x) {
  CheckValueDataType(x, "x", "log_");
  paddle::dialect::Log_Op log__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log_Op>(x);
  return log__op.result(0);
}

pir::OpResult log10(const pir::Value& x) {
  CheckValueDataType(x, "x", "log10");
  paddle::dialect::Log10Op log10_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log10Op>(x);
  return log10_op.result(0);
}

pir::OpResult log10_(const pir::Value& x) {
  CheckValueDataType(x, "x", "log10_");
  paddle::dialect::Log10_Op log10__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log10_Op>(x);
  return log10__op.result(0);
}

pir::OpResult log1p(const pir::Value& x) {
  CheckValueDataType(x, "x", "log1p");
  paddle::dialect::Log1pOp log1p_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log1pOp>(x);
  return log1p_op.result(0);
}

pir::OpResult log1p_(const pir::Value& x) {
  CheckValueDataType(x, "x", "log1p_");
  paddle::dialect::Log1p_Op log1p__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log1p_Op>(x);
  return log1p__op.result(0);
}

pir::OpResult log2(const pir::Value& x) {
  CheckValueDataType(x, "x", "log2");
  paddle::dialect::Log2Op log2_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log2Op>(x);
  return log2_op.result(0);
}

pir::OpResult log2_(const pir::Value& x) {
  CheckValueDataType(x, "x", "log2_");
  paddle::dialect::Log2_Op log2__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log2_Op>(x);
  return log2__op.result(0);
}

pir::OpResult log_loss(const pir::Value& input,
                       const pir::Value& label,
                       float epsilon) {
  CheckValueDataType(label, "label", "log_loss");
  paddle::dialect::LogLossOp log_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogLossOp>(
          input, label, epsilon);
  return log_loss_op.result(0);
}

pir::OpResult log_softmax(const pir::Value& x, int axis) {
  CheckValueDataType(x, "x", "log_softmax");
  paddle::dialect::LogSoftmaxOp log_softmax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogSoftmaxOp>(
          x, axis);
  return log_softmax_op.result(0);
}

pir::OpResult logcumsumexp(
    const pir::Value& x, int axis, bool flatten, bool exclusive, bool reverse) {
  CheckValueDataType(x, "x", "logcumsumexp");
  paddle::dialect::LogcumsumexpOp logcumsumexp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogcumsumexpOp>(
              x, axis, flatten, exclusive, reverse);
  return logcumsumexp_op.result(0);
}

pir::OpResult logical_and(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_and");
  paddle::dialect::LogicalAndOp logical_and_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogicalAndOp>(
          x, y);
  return logical_and_op.result(0);
}

pir::OpResult logical_and_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_and_");
  paddle::dialect::LogicalAnd_Op logical_and__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogicalAnd_Op>(x, y);
  return logical_and__op.result(0);
}

pir::OpResult logical_not(const pir::Value& x) {
  CheckValueDataType(x, "x", "logical_not");
  paddle::dialect::LogicalNotOp logical_not_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogicalNotOp>(
          x);
  return logical_not_op.result(0);
}

pir::OpResult logical_not_(const pir::Value& x) {
  CheckValueDataType(x, "x", "logical_not_");
  paddle::dialect::LogicalNot_Op logical_not__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogicalNot_Op>(x);
  return logical_not__op.result(0);
}

pir::OpResult logical_or(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_or");
  paddle::dialect::LogicalOrOp logical_or_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogicalOrOp>(
          x, y);
  return logical_or_op.result(0);
}

pir::OpResult logical_or_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_or_");
  paddle::dialect::LogicalOr_Op logical_or__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogicalOr_Op>(
          x, y);
  return logical_or__op.result(0);
}

pir::OpResult logical_xor(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_xor");
  paddle::dialect::LogicalXorOp logical_xor_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogicalXorOp>(
          x, y);
  return logical_xor_op.result(0);
}

pir::OpResult logical_xor_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "logical_xor_");
  paddle::dialect::LogicalXor_Op logical_xor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogicalXor_Op>(x, y);
  return logical_xor__op.result(0);
}

pir::OpResult logit(const pir::Value& x, float eps) {
  CheckValueDataType(x, "x", "logit");
  paddle::dialect::LogitOp logit_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogitOp>(x,
                                                                           eps);
  return logit_op.result(0);
}

pir::OpResult logit_(const pir::Value& x, float eps) {
  CheckValueDataType(x, "x", "logit_");
  paddle::dialect::Logit_Op logit__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Logit_Op>(
          x, eps);
  return logit__op.result(0);
}

pir::OpResult logsigmoid(const pir::Value& x) {
  CheckValueDataType(x, "x", "logsigmoid");
  paddle::dialect::LogsigmoidOp logsigmoid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogsigmoidOp>(
          x);
  return logsigmoid_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> lstsq(
    const pir::Value& x,
    const pir::Value& y,
    float rcond,
    const std::string& driver) {
  CheckValueDataType(x, "x", "lstsq");
  paddle::dialect::LstsqOp lstsq_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LstsqOp>(
          x, y, rcond, driver);
  return std::make_tuple(lstsq_op.result(0),
                         lstsq_op.result(1),
                         lstsq_op.result(2),
                         lstsq_op.result(3));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> lstsq(
    const pir::Value& x,
    const pir::Value& y,
    pir::Value rcond,
    const std::string& driver) {
  CheckValueDataType(x, "x", "lstsq");
  paddle::dialect::LstsqOp lstsq_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LstsqOp>(
          x, y, rcond, driver);
  return std::make_tuple(lstsq_op.result(0),
                         lstsq_op.result(1),
                         lstsq_op.result(2),
                         lstsq_op.result(3));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu(const pir::Value& x,
                                                           bool pivot) {
  CheckValueDataType(x, "x", "lu");
  paddle::dialect::LuOp lu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LuOp>(x,
                                                                        pivot);
  return std::make_tuple(lu_op.result(0), lu_op.result(1), lu_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu_(const pir::Value& x,
                                                            bool pivot) {
  CheckValueDataType(x, "x", "lu_");
  paddle::dialect::Lu_Op lu__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Lu_Op>(x,
                                                                         pivot);
  return std::make_tuple(lu__op.result(0), lu__op.result(1), lu__op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu_unpack(
    const pir::Value& x,
    const pir::Value& y,
    bool unpack_ludata,
    bool unpack_pivots) {
  CheckValueDataType(x, "x", "lu_unpack");
  paddle::dialect::LuUnpackOp lu_unpack_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LuUnpackOp>(
          x, y, unpack_ludata, unpack_pivots);
  return std::make_tuple(
      lu_unpack_op.result(0), lu_unpack_op.result(1), lu_unpack_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> margin_cross_entropy(
    const pir::Value& logits,
    const pir::Value& label,
    bool return_softmax,
    int ring_id,
    int rank,
    int nranks,
    float margin1,
    float margin2,
    float margin3,
    float scale) {
  CheckValueDataType(logits, "logits", "margin_cross_entropy");
  paddle::dialect::MarginCrossEntropyOp margin_cross_entropy_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MarginCrossEntropyOp>(logits,
                                                         label,
                                                         return_softmax,
                                                         ring_id,
                                                         rank,
                                                         nranks,
                                                         margin1,
                                                         margin2,
                                                         margin3,
                                                         scale);
  return std::make_tuple(margin_cross_entropy_op.result(0),
                         margin_cross_entropy_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
masked_multihead_attention_(
    const pir::Value& x,
    const pir::Value& cache_kv,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& cum_offsets,
    const paddle::optional<pir::Value>& sequence_lengths,
    const paddle::optional<pir::Value>& rotary_tensor,
    const paddle::optional<pir::Value>& beam_cache_offset,
    const paddle::optional<pir::Value>& qkv_out_scale,
    const paddle::optional<pir::Value>& out_shift,
    const paddle::optional<pir::Value>& out_smooth,
    int seq_len,
    int rotary_emb_dims,
    bool use_neox_rotary_style,
    const std::string& compute_dtype,
    float out_scale,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound) {
  CheckValueDataType(x, "x", "masked_multihead_attention_");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_src_mask;
  if (!src_mask) {
    optional_src_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask = src_mask;
  }
  paddle::optional<pir::Value> optional_cum_offsets;
  if (!cum_offsets) {
    optional_cum_offsets = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cum_offsets = cum_offsets;
  }
  paddle::optional<pir::Value> optional_sequence_lengths;
  if (!sequence_lengths) {
    optional_sequence_lengths = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sequence_lengths = sequence_lengths;
  }
  paddle::optional<pir::Value> optional_rotary_tensor;
  if (!rotary_tensor) {
    optional_rotary_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rotary_tensor = rotary_tensor;
  }
  paddle::optional<pir::Value> optional_beam_cache_offset;
  if (!beam_cache_offset) {
    optional_beam_cache_offset =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_beam_cache_offset = beam_cache_offset;
  }
  paddle::optional<pir::Value> optional_qkv_out_scale;
  if (!qkv_out_scale) {
    optional_qkv_out_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_out_scale = qkv_out_scale;
  }
  paddle::optional<pir::Value> optional_out_shift;
  if (!out_shift) {
    optional_out_shift = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_shift = out_shift;
  }
  paddle::optional<pir::Value> optional_out_smooth;
  if (!out_smooth) {
    optional_out_smooth = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_smooth = out_smooth;
  }
  paddle::dialect::MaskedMultiheadAttention_Op masked_multihead_attention__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaskedMultiheadAttention_Op>(
              x,
              cache_kv,
              optional_bias.get(),
              optional_src_mask.get(),
              optional_cum_offsets.get(),
              optional_sequence_lengths.get(),
              optional_rotary_tensor.get(),
              optional_beam_cache_offset.get(),
              optional_qkv_out_scale.get(),
              optional_out_shift.get(),
              optional_out_smooth.get(),
              seq_len,
              rotary_emb_dims,
              use_neox_rotary_style,
              compute_dtype,
              out_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
  paddle::optional<pir::OpResult> optional_beam_cache_offset_out;
  if (!IsEmptyValue(masked_multihead_attention__op.result(2))) {
    optional_beam_cache_offset_out = paddle::make_optional<pir::OpResult>(
        masked_multihead_attention__op.result(2));
  }
  if (!beam_cache_offset) {
    masked_multihead_attention__op.result(2).set_type(pir::Type());
  }
  return std::make_tuple(masked_multihead_attention__op.result(0),
                         masked_multihead_attention__op.result(1),
                         optional_beam_cache_offset_out);
}

pir::OpResult masked_select(const pir::Value& x, const pir::Value& mask) {
  CheckValueDataType(x, "x", "masked_select");
  paddle::dialect::MaskedSelectOp masked_select_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaskedSelectOp>(x, mask);
  return masked_select_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> matrix_nms(
    const pir::Value& bboxes,
    const pir::Value& scores,
    float score_threshold,
    int nms_top_k,
    int keep_top_k,
    float post_threshold,
    bool use_gaussian,
    float gaussian_sigma,
    int background_label,
    bool normalized) {
  CheckValueDataType(scores, "scores", "matrix_nms");
  paddle::dialect::MatrixNmsOp matrix_nms_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MatrixNmsOp>(
          bboxes,
          scores,
          score_threshold,
          nms_top_k,
          keep_top_k,
          post_threshold,
          use_gaussian,
          gaussian_sigma,
          background_label,
          normalized);
  return std::make_tuple(matrix_nms_op.result(0),
                         matrix_nms_op.result(1),
                         matrix_nms_op.result(2));
}

pir::OpResult matrix_power(const pir::Value& x, int n) {
  CheckValueDataType(x, "x", "matrix_power");
  paddle::dialect::MatrixPowerOp matrix_power_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatrixPowerOp>(x, n);
  return matrix_power_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> max_pool2d_with_index(
    const pir::Value& x,
    const std::vector<int>& kernel_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool global_pooling,
    bool adaptive) {
  CheckValueDataType(x, "x", "max_pool2d_with_index");
  paddle::dialect::MaxPool2dWithIndexOp max_pool2d_with_index_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool2dWithIndexOp>(
              x, kernel_size, strides, paddings, global_pooling, adaptive);
  return std::make_tuple(max_pool2d_with_index_op.result(0),
                         max_pool2d_with_index_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> max_pool3d_with_index(
    const pir::Value& x,
    const std::vector<int>& kernel_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool global_pooling,
    bool adaptive) {
  CheckValueDataType(x, "x", "max_pool3d_with_index");
  paddle::dialect::MaxPool3dWithIndexOp max_pool3d_with_index_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool3dWithIndexOp>(
              x, kernel_size, strides, paddings, global_pooling, adaptive);
  return std::make_tuple(max_pool3d_with_index_op.result(0),
                         max_pool3d_with_index_op.result(1));
}

pir::OpResult maxout(const pir::Value& x, int groups, int axis) {
  CheckValueDataType(x, "x", "maxout");
  paddle::dialect::MaxoutOp maxout_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxoutOp>(
          x, groups, axis);
  return maxout_op.result(0);
}

pir::OpResult mean_all(const pir::Value& x) {
  CheckValueDataType(x, "x", "mean_all");
  paddle::dialect::MeanAllOp mean_all_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeanAllOp>(x);
  return mean_all_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
memory_efficient_attention(const pir::Value& query,
                           const pir::Value& key,
                           const pir::Value& value,
                           const paddle::optional<pir::Value>& bias,
                           const paddle::optional<pir::Value>& cu_seqlens_q,
                           const paddle::optional<pir::Value>& cu_seqlens_k,
                           const paddle::optional<pir::Value>& causal_diagonal,
                           const paddle::optional<pir::Value>& seqlen_k,
                           float max_seqlen_q,
                           float max_seqlen_k,
                           bool causal,
                           double dropout_p,
                           float scale,
                           bool is_test) {
  CheckValueDataType(query, "query", "memory_efficient_attention");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_cu_seqlens_q;
  if (!cu_seqlens_q) {
    optional_cu_seqlens_q = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cu_seqlens_q = cu_seqlens_q;
  }
  paddle::optional<pir::Value> optional_cu_seqlens_k;
  if (!cu_seqlens_k) {
    optional_cu_seqlens_k = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cu_seqlens_k = cu_seqlens_k;
  }
  paddle::optional<pir::Value> optional_causal_diagonal;
  if (!causal_diagonal) {
    optional_causal_diagonal = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_causal_diagonal = causal_diagonal;
  }
  paddle::optional<pir::Value> optional_seqlen_k;
  if (!seqlen_k) {
    optional_seqlen_k = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seqlen_k = seqlen_k;
  }
  paddle::dialect::MemoryEfficientAttentionOp memory_efficient_attention_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MemoryEfficientAttentionOp>(
              query,
              key,
              value,
              optional_bias.get(),
              optional_cu_seqlens_q.get(),
              optional_cu_seqlens_k.get(),
              optional_causal_diagonal.get(),
              optional_seqlen_k.get(),
              max_seqlen_q,
              max_seqlen_k,
              causal,
              dropout_p,
              scale,
              is_test);
  return std::make_tuple(memory_efficient_attention_op.result(0),
                         memory_efficient_attention_op.result(1),
                         memory_efficient_attention_op.result(2));
}

pir::OpResult merge_selected_rows(const pir::Value& x) {
  CheckValueDataType(x, "x", "merge_selected_rows");
  paddle::dialect::MergeSelectedRowsOp merge_selected_rows_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MergeSelectedRowsOp>(x);
  return merge_selected_rows_op.result(0);
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_adam_(const std::vector<pir::Value>& param,
             const std::vector<pir::Value>& grad,
             const std::vector<pir::Value>& learning_rate,
             const std::vector<pir::Value>& moment1,
             const std::vector<pir::Value>& moment2,
             const std::vector<pir::Value>& beta1_pow,
             const std::vector<pir::Value>& beta2_pow,
             const paddle::optional<std::vector<pir::Value>>& master_param,
             float beta1,
             float beta2,
             float epsilon,
             bool multi_precision,
             bool use_global_beta_pow) {
  CheckVectorOfValueDataType(param, "param", "merged_adam_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_param_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_param.get());
    optional_master_param = paddle::make_optional<pir::Value>(
        optional_master_param_combine_op.out());
  }
  auto param_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(param);
  auto grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grad);
  auto learning_rate_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(learning_rate);
  auto moment1_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moment1);
  auto moment2_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moment2);
  auto beta1_pow_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta1_pow);
  auto beta2_pow_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta2_pow);
  paddle::dialect::MergedAdam_Op merged_adam__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MergedAdam_Op>(
              param_combine_op.out(),
              grad_combine_op.out(),
              learning_rate_combine_op.out(),
              moment1_combine_op.out(),
              moment2_combine_op.out(),
              beta1_pow_combine_op.out(),
              beta2_pow_combine_op.out(),
              optional_master_param.get(),
              beta1,
              beta2,
              epsilon,
              multi_precision,
              use_global_beta_pow);
  paddle::optional<std::vector<pir::OpResult>> optional_master_param_out;
  if (!IsEmptyValue(merged_adam__op.result(5))) {
    auto optional_master_param_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            merged_adam__op.result(5));
    optional_master_param_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_param_out_slice_op.outputs());
  }
  if (!master_param) {
    merged_adam__op.result(5).set_type(pir::Type());
  }
  auto param_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(0));
  auto moment1_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(1));
  auto moment2_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(2));
  auto beta1_pow_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(3));
  auto beta2_pow_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(4));
  return std::make_tuple(param_out_split_op.outputs(),
                         moment1_out_split_op.outputs(),
                         moment2_out_split_op.outputs(),
                         beta1_pow_out_split_op.outputs(),
                         beta2_pow_out_split_op.outputs(),
                         optional_master_param_out);
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_adam_(const std::vector<pir::Value>& param,
             const std::vector<pir::Value>& grad,
             const std::vector<pir::Value>& learning_rate,
             const std::vector<pir::Value>& moment1,
             const std::vector<pir::Value>& moment2,
             const std::vector<pir::Value>& beta1_pow,
             const std::vector<pir::Value>& beta2_pow,
             const paddle::optional<std::vector<pir::Value>>& master_param,
             pir::Value beta1,
             pir::Value beta2,
             pir::Value epsilon,
             bool multi_precision,
             bool use_global_beta_pow) {
  CheckVectorOfValueDataType(param, "param", "merged_adam_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_param_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_param.get());
    optional_master_param = paddle::make_optional<pir::Value>(
        optional_master_param_combine_op.out());
  }
  auto param_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(param);
  auto grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grad);
  auto learning_rate_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(learning_rate);
  auto moment1_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moment1);
  auto moment2_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moment2);
  auto beta1_pow_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta1_pow);
  auto beta2_pow_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta2_pow);
  paddle::dialect::MergedAdam_Op merged_adam__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MergedAdam_Op>(
              param_combine_op.out(),
              grad_combine_op.out(),
              learning_rate_combine_op.out(),
              moment1_combine_op.out(),
              moment2_combine_op.out(),
              beta1_pow_combine_op.out(),
              beta2_pow_combine_op.out(),
              optional_master_param.get(),
              beta1,
              beta2,
              epsilon,
              multi_precision,
              use_global_beta_pow);
  paddle::optional<std::vector<pir::OpResult>> optional_master_param_out;
  if (!IsEmptyValue(merged_adam__op.result(5))) {
    auto optional_master_param_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            merged_adam__op.result(5));
    optional_master_param_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_param_out_slice_op.outputs());
  }
  if (!master_param) {
    merged_adam__op.result(5).set_type(pir::Type());
  }
  auto param_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(0));
  auto moment1_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(1));
  auto moment2_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(2));
  auto beta1_pow_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(3));
  auto beta2_pow_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_adam__op.result(4));
  return std::make_tuple(param_out_split_op.outputs(),
                         moment1_out_split_op.outputs(),
                         moment2_out_split_op.outputs(),
                         beta1_pow_out_split_op.outputs(),
                         beta2_pow_out_split_op.outputs(),
                         optional_master_param_out);
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_momentum_(const std::vector<pir::Value>& param,
                 const std::vector<pir::Value>& grad,
                 const std::vector<pir::Value>& velocity,
                 const std::vector<pir::Value>& learning_rate,
                 const paddle::optional<std::vector<pir::Value>>& master_param,
                 float mu,
                 bool use_nesterov,
                 const std::vector<std::string>& regularization_method,
                 const std::vector<float>& regularization_coeff,
                 bool multi_precision,
                 float rescale_grad) {
  CheckVectorOfValueDataType(param, "param", "merged_momentum_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_param_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_param.get());
    optional_master_param = paddle::make_optional<pir::Value>(
        optional_master_param_combine_op.out());
  }
  auto param_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(param);
  auto grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grad);
  auto velocity_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(velocity);
  auto learning_rate_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(learning_rate);
  paddle::dialect::MergedMomentum_Op merged_momentum__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MergedMomentum_Op>(
              param_combine_op.out(),
              grad_combine_op.out(),
              velocity_combine_op.out(),
              learning_rate_combine_op.out(),
              optional_master_param.get(),
              mu,
              use_nesterov,
              regularization_method,
              regularization_coeff,
              multi_precision,
              rescale_grad);
  paddle::optional<std::vector<pir::OpResult>> optional_master_param_out;
  if (!IsEmptyValue(merged_momentum__op.result(2))) {
    auto optional_master_param_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            merged_momentum__op.result(2));
    optional_master_param_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_param_out_slice_op.outputs());
  }
  if (!master_param) {
    merged_momentum__op.result(2).set_type(pir::Type());
  }
  auto param_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_momentum__op.result(0));
  auto velocity_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          merged_momentum__op.result(1));
  return std::make_tuple(param_out_split_op.outputs(),
                         velocity_out_split_op.outputs(),
                         optional_master_param_out);
}

std::vector<pir::OpResult> meshgrid(const std::vector<pir::Value>& inputs) {
  CheckVectorOfValueDataType(inputs, "inputs", "meshgrid");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::MeshgridOp meshgrid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeshgridOp>(
          inputs_combine_op.out());
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      meshgrid_op.result(0));
  return out_split_op.outputs();
}

std::tuple<pir::OpResult, pir::OpResult> mode(const pir::Value& x,
                                              int axis,
                                              bool keepdim) {
  CheckValueDataType(x, "x", "mode");
  paddle::dialect::ModeOp mode_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ModeOp>(
          x, axis, keepdim);
  return std::make_tuple(mode_op.result(0), mode_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
momentum_(const pir::Value& param,
          const pir::Value& grad,
          const pir::Value& velocity,
          const pir::Value& learning_rate,
          const paddle::optional<pir::Value>& master_param,
          float mu,
          bool use_nesterov,
          const std::string& regularization_method,
          float regularization_coeff,
          bool multi_precision,
          float rescale_grad) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      velocity.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "momentum_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::Momentum_Op momentum__op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::Momentum_Op>(param,
                                                  grad,
                                                  velocity,
                                                  learning_rate,
                                                  optional_master_param.get(),
                                                  mu,
                                                  use_nesterov,
                                                  regularization_method,
                                                  regularization_coeff,
                                                  multi_precision,
                                                  rescale_grad);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(momentum__op.result(2))) {
      optional_master_param_out =
          paddle::make_optional<pir::OpResult>(momentum__op.result(2));
    }
    if (!master_param) {
      momentum__op.result(2).set_type(pir::Type());
    }
    return std::make_tuple(momentum__op.result(0),
                           momentum__op.result(1),
                           optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      velocity.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "momentum_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::MomentumDenseParamSparseGrad_Op
        momentum_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::MomentumDenseParamSparseGrad_Op>(
                    param,
                    grad,
                    velocity,
                    learning_rate,
                    optional_master_param.get(),
                    mu,
                    use_nesterov,
                    regularization_method,
                    regularization_coeff,
                    multi_precision,
                    rescale_grad);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(momentum_dense_param_sparse_grad__op.result(2))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          momentum_dense_param_sparse_grad__op.result(2));
    }
    if (!master_param) {
      momentum_dense_param_sparse_grad__op.result(2).set_type(pir::Type());
    }
    return std::make_tuple(momentum_dense_param_sparse_grad__op.result(0),
                           momentum_dense_param_sparse_grad__op.result(1),
                           optional_master_param_out);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (momentum_) for input Value is unimplemented, please "
      "check the type of input Value."));
}

pir::OpResult multi_dot(const std::vector<pir::Value>& x) {
  CheckVectorOfValueDataType(x, "x", "multi_dot");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::MultiDotOp multi_dot_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MultiDotOp>(
          x_combine_op.out());
  return multi_dot_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiclass_nms3(
    const pir::Value& bboxes,
    const pir::Value& scores,
    const paddle::optional<pir::Value>& rois_num,
    float score_threshold,
    int nms_top_k,
    int keep_top_k,
    float nms_threshold,
    bool normalized,
    float nms_eta,
    int background_label) {
  CheckValueDataType(scores, "scores", "multiclass_nms3");
  paddle::optional<pir::Value> optional_rois_num;
  if (!rois_num) {
    optional_rois_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rois_num = rois_num;
  }
  paddle::dialect::MulticlassNms3Op multiclass_nms3_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MulticlassNms3Op>(bboxes,
                                                     scores,
                                                     optional_rois_num.get(),
                                                     score_threshold,
                                                     nms_top_k,
                                                     keep_top_k,
                                                     nms_threshold,
                                                     normalized,
                                                     nms_eta,
                                                     background_label);
  return std::make_tuple(multiclass_nms3_op.result(0),
                         multiclass_nms3_op.result(1),
                         multiclass_nms3_op.result(2));
}

pir::OpResult multinomial(const pir::Value& x,
                          int num_samples,
                          bool replacement) {
  CheckValueDataType(x, "x", "multinomial");
  paddle::dialect::MultinomialOp multinomial_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultinomialOp>(x, num_samples, replacement);
  return multinomial_op.result(0);
}

pir::OpResult multinomial(const pir::Value& x,
                          pir::Value num_samples,
                          bool replacement) {
  CheckValueDataType(x, "x", "multinomial");
  paddle::dialect::MultinomialOp multinomial_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultinomialOp>(x, num_samples, replacement);
  return multinomial_op.result(0);
}

pir::OpResult multiplex(const std::vector<pir::Value>& inputs,
                        const pir::Value& index) {
  CheckVectorOfValueDataType(inputs, "inputs", "multiplex");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::MultiplexOp multiplex_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MultiplexOp>(
          inputs_combine_op.out(), index);
  return multiplex_op.result(0);
}

pir::OpResult mv(const pir::Value& x, const pir::Value& vec) {
  CheckValueDataType(vec, "vec", "mv");
  paddle::dialect::MvOp mv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MvOp>(x, vec);
  return mv_op.result(0);
}

pir::OpResult nanmedian(const pir::Value& x,
                        const std::vector<int64_t>& axis,
                        bool keepdim) {
  CheckValueDataType(x, "x", "nanmedian");
  paddle::dialect::NanmedianOp nanmedian_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NanmedianOp>(
          x, axis, keepdim);
  return nanmedian_op.result(0);
}

pir::OpResult nearest_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(x, "x", "nearest_interp");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::NearestInterpOp nearest_interp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NearestInterpOp>(x,
                                                    optional_out_size.get(),
                                                    optional_size_tensor.get(),
                                                    optional_scale_tensor.get(),
                                                    data_layout,
                                                    out_d,
                                                    out_h,
                                                    out_w,
                                                    scale,
                                                    interp_method,
                                                    align_corners,
                                                    align_mode);
  return nearest_interp_op.result(0);
}

pir::OpResult nextafter(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "nextafter");
  paddle::dialect::NextafterOp nextafter_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NextafterOp>(
          x, y);
  return nextafter_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> nll_loss(
    const pir::Value& input,
    const pir::Value& label,
    const paddle::optional<pir::Value>& weight,
    int64_t ignore_index,
    const std::string& reduction) {
  CheckValueDataType(input, "input", "nll_loss");
  paddle::optional<pir::Value> optional_weight;
  if (!weight) {
    optional_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_weight = weight;
  }
  paddle::dialect::NllLossOp nll_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NllLossOp>(
          input, label, optional_weight.get(), ignore_index, reduction);
  return std::make_tuple(nll_loss_op.result(0), nll_loss_op.result(1));
}

pir::OpResult nms(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "nms");
  paddle::dialect::NmsOp nms_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NmsOp>(
          x, threshold);
  return nms_op.result(0);
}

pir::OpResult nonzero(const pir::Value& condition) {
  CheckValueDataType(condition, "condition", "nonzero");
  paddle::dialect::NonzeroOp nonzero_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NonzeroOp>(
          condition);
  return nonzero_op.result(0);
}

pir::OpResult npu_identity(const pir::Value& x, int format) {
  CheckValueDataType(x, "x", "npu_identity");
  paddle::dialect::NpuIdentityOp npu_identity_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NpuIdentityOp>(x, format);
  return npu_identity_op.result(0);
}

pir::OpResult numel(const pir::Value& x) {
  CheckValueDataType(x, "x", "numel");
  paddle::dialect::NumelOp numel_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NumelOp>(x);
  return numel_op.result(0);
}

pir::OpResult overlap_add(const pir::Value& x, int hop_length, int axis) {
  CheckValueDataType(x, "x", "overlap_add");
  paddle::dialect::OverlapAddOp overlap_add_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::OverlapAddOp>(
          x, hop_length, axis);
  return overlap_add_op.result(0);
}

pir::OpResult p_norm(const pir::Value& x,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector) {
  CheckValueDataType(x, "x", "p_norm");
  paddle::dialect::PNormOp p_norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PNormOp>(
          x, porder, axis, epsilon, keepdim, asvector);
  return p_norm_op.result(0);
}

pir::OpResult pad3d(const pir::Value& x,
                    const std::vector<int64_t>& paddings,
                    const std::string& mode,
                    float pad_value,
                    const std::string& data_format) {
  CheckValueDataType(x, "x", "pad3d");
  paddle::dialect::Pad3dOp pad3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dOp>(
          x, paddings, mode, pad_value, data_format);
  return pad3d_op.result(0);
}

pir::OpResult pad3d(const pir::Value& x,
                    pir::Value paddings,
                    const std::string& mode,
                    float pad_value,
                    const std::string& data_format) {
  CheckValueDataType(x, "x", "pad3d");
  paddle::dialect::Pad3dOp pad3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dOp>(
          x, paddings, mode, pad_value, data_format);
  return pad3d_op.result(0);
}

pir::OpResult pad3d(const pir::Value& x,
                    std::vector<pir::Value> paddings,
                    const std::string& mode,
                    float pad_value,
                    const std::string& data_format) {
  CheckValueDataType(x, "x", "pad3d");
  auto paddings_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(paddings);
  paddle::dialect::Pad3dOp pad3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dOp>(
          x, paddings_combine_op.out(), mode, pad_value, data_format);
  return pad3d_op.result(0);
}

pir::OpResult pixel_shuffle(const pir::Value& x,
                            int upscale_factor,
                            const std::string& data_format) {
  CheckValueDataType(x, "x", "pixel_shuffle");
  paddle::dialect::PixelShuffleOp pixel_shuffle_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PixelShuffleOp>(
              x, upscale_factor, data_format);
  return pixel_shuffle_op.result(0);
}

pir::OpResult pixel_unshuffle(const pir::Value& x,
                              int downscale_factor,
                              const std::string& data_format) {
  CheckValueDataType(x, "x", "pixel_unshuffle");
  paddle::dialect::PixelUnshuffleOp pixel_unshuffle_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PixelUnshuffleOp>(
              x, downscale_factor, data_format);
  return pixel_unshuffle_op.result(0);
}

pir::OpResult poisson(const pir::Value& x) {
  CheckValueDataType(x, "x", "poisson");
  paddle::dialect::PoissonOp poisson_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PoissonOp>(x);
  return poisson_op.result(0);
}

pir::OpResult polygamma(const pir::Value& x, int n) {
  CheckValueDataType(x, "x", "polygamma");
  paddle::dialect::PolygammaOp polygamma_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PolygammaOp>(
          x, n);
  return polygamma_op.result(0);
}

pir::OpResult polygamma_(const pir::Value& x, int n) {
  CheckValueDataType(x, "x", "polygamma_");
  paddle::dialect::Polygamma_Op polygamma__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Polygamma_Op>(
          x, n);
  return polygamma__op.result(0);
}

pir::OpResult pow(const pir::Value& x, float y) {
  CheckValueDataType(x, "x", "pow");
  paddle::dialect::PowOp pow_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PowOp>(x, y);
  return pow_op.result(0);
}

pir::OpResult pow_(const pir::Value& x, float y) {
  CheckValueDataType(x, "x", "pow_");
  paddle::dialect::Pow_Op pow__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pow_Op>(x, y);
  return pow__op.result(0);
}

pir::OpResult prelu(const pir::Value& x,
                    const pir::Value& alpha,
                    const std::string& data_format,
                    const std::string& mode) {
  CheckValueDataType(x, "x", "prelu");
  paddle::dialect::PreluOp prelu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PreluOp>(
          x, alpha, data_format, mode);
  return prelu_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> prior_box(
    const pir::Value& input,
    const pir::Value& image,
    const std::vector<float>& min_sizes,
    const std::vector<float>& max_sizes,
    const std::vector<float>& aspect_ratios,
    const std::vector<float>& variances,
    bool flip,
    bool clip,
    float step_w,
    float step_h,
    float offset,
    bool min_max_aspect_ratios_order) {
  CheckValueDataType(input, "input", "prior_box");
  paddle::dialect::PriorBoxOp prior_box_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PriorBoxOp>(
          input,
          image,
          min_sizes,
          max_sizes,
          aspect_ratios,
          variances,
          flip,
          clip,
          step_w,
          step_h,
          offset,
          min_max_aspect_ratios_order);
  return std::make_tuple(prior_box_op.result(0), prior_box_op.result(1));
}

pir::OpResult psroi_pool(const pir::Value& x,
                         const pir::Value& boxes,
                         const paddle::optional<pir::Value>& boxes_num,
                         int pooled_height,
                         int pooled_width,
                         int output_channels,
                         float spatial_scale) {
  CheckValueDataType(x, "x", "psroi_pool");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::PsroiPoolOp psroi_pool_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PsroiPoolOp>(
          x,
          boxes,
          optional_boxes_num.get(),
          pooled_height,
          pooled_width,
          output_channels,
          spatial_scale);
  return psroi_pool_op.result(0);
}

pir::OpResult put_along_axis(const pir::Value& arr,
                             const pir::Value& indices,
                             const pir::Value& values,
                             int axis,
                             const std::string& reduce,
                             bool include_self) {
  CheckValueDataType(arr, "arr", "put_along_axis");
  paddle::dialect::PutAlongAxisOp put_along_axis_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PutAlongAxisOp>(
              arr, indices, values, axis, reduce, include_self);
  return put_along_axis_op.result(0);
}

pir::OpResult put_along_axis_(const pir::Value& arr,
                              const pir::Value& indices,
                              const pir::Value& values,
                              int axis,
                              const std::string& reduce,
                              bool include_self) {
  CheckValueDataType(arr, "arr", "put_along_axis_");
  paddle::dialect::PutAlongAxis_Op put_along_axis__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PutAlongAxis_Op>(
              arr, indices, values, axis, reduce, include_self);
  return put_along_axis__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> qr(const pir::Value& x,
                                            const std::string& mode) {
  CheckValueDataType(x, "x", "qr");
  paddle::dialect::QrOp qr_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::QrOp>(x,
                                                                        mode);
  return std::make_tuple(qr_op.result(0), qr_op.result(1));
}

pir::OpResult real(const pir::Value& x) {
  CheckValueDataType(x, "x", "real");
  paddle::dialect::RealOp real_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RealOp>(x);
  return real_op.result(0);
}

pir::OpResult reciprocal(const pir::Value& x) {
  CheckValueDataType(x, "x", "reciprocal");
  paddle::dialect::ReciprocalOp reciprocal_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReciprocalOp>(
          x);
  return reciprocal_op.result(0);
}

pir::OpResult reciprocal_(const pir::Value& x) {
  CheckValueDataType(x, "x", "reciprocal_");
  paddle::dialect::Reciprocal_Op reciprocal__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Reciprocal_Op>(x);
  return reciprocal__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> reindex_graph(
    const pir::Value& x,
    const pir::Value& neighbors,
    const pir::Value& count,
    const paddle::optional<pir::Value>& hashtable_value,
    const paddle::optional<pir::Value>& hashtable_index) {
  CheckValueDataType(x, "x", "graph_reindex");
  paddle::optional<pir::Value> optional_hashtable_value;
  if (!hashtable_value) {
    optional_hashtable_value = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_hashtable_value = hashtable_value;
  }
  paddle::optional<pir::Value> optional_hashtable_index;
  if (!hashtable_index) {
    optional_hashtable_index = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_hashtable_index = hashtable_index;
  }
  paddle::dialect::ReindexGraphOp reindex_graph_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReindexGraphOp>(
              x,
              neighbors,
              count,
              optional_hashtable_value.get(),
              optional_hashtable_index.get());
  return std::make_tuple(reindex_graph_op.result(0),
                         reindex_graph_op.result(1),
                         reindex_graph_op.result(2));
}

pir::OpResult relu(const pir::Value& x) {
  CheckValueDataType(x, "x", "relu");
  paddle::dialect::ReluOp relu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReluOp>(x);
  return relu_op.result(0);
}

pir::OpResult relu_(const pir::Value& x) {
  CheckValueDataType(x, "x", "relu_");
  paddle::dialect::Relu_Op relu__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Relu_Op>(x);
  return relu__op.result(0);
}

pir::OpResult relu6(const pir::Value& x) {
  CheckValueDataType(x, "x", "relu6");
  paddle::dialect::Relu6Op relu6_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Relu6Op>(x);
  return relu6_op.result(0);
}

pir::OpResult renorm(const pir::Value& x, float p, int axis, float max_norm) {
  CheckValueDataType(x, "x", "renorm");
  paddle::dialect::RenormOp renorm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RenormOp>(
          x, p, axis, max_norm);
  return renorm_op.result(0);
}

pir::OpResult renorm_(const pir::Value& x, float p, int axis, float max_norm) {
  CheckValueDataType(x, "x", "renorm_");
  paddle::dialect::Renorm_Op renorm__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Renorm_Op>(
          x, p, axis, max_norm);
  return renorm__op.result(0);
}

pir::OpResult reverse(const pir::Value& x, const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "reverse");
  paddle::dialect::ReverseOp reverse_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReverseOp>(
          x, axis);
  return reverse_op.result(0);
}

pir::OpResult reverse(const pir::Value& x, pir::Value axis) {
  CheckValueDataType(x, "x", "reverse");
  paddle::dialect::ReverseOp reverse_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReverseOp>(
          x, axis);
  return reverse_op.result(0);
}

pir::OpResult reverse(const pir::Value& x, std::vector<pir::Value> axis) {
  CheckValueDataType(x, "x", "reverse");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::ReverseOp reverse_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReverseOp>(
          x, axis_combine_op.out());
  return reverse_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> rms_norm(
    const pir::Value& x,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& residual,
    const pir::Value& norm_weight,
    const paddle::optional<pir::Value>& norm_bias,
    float epsilon,
    int begin_norm_axis,
    float quant_scale,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound) {
  CheckValueDataType(x, "x", "rms_norm");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_residual;
  if (!residual) {
    optional_residual = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_residual = residual;
  }
  paddle::optional<pir::Value> optional_norm_bias;
  if (!norm_bias) {
    optional_norm_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_norm_bias = norm_bias;
  }
  paddle::dialect::RmsNormOp rms_norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RmsNormOp>(
          x,
          optional_bias.get(),
          optional_residual.get(),
          norm_weight,
          optional_norm_bias.get(),
          epsilon,
          begin_norm_axis,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
  return std::make_tuple(rms_norm_op.result(0), rms_norm_op.result(1));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>,
           paddle::optional<pir::OpResult>>
rmsprop_(const pir::Value& param,
         const pir::Value& mean_square,
         const pir::Value& grad,
         const pir::Value& moment,
         const pir::Value& learning_rate,
         const paddle::optional<pir::Value>& mean_grad,
         const paddle::optional<pir::Value>& master_param,
         float epsilon,
         float decay,
         float momentum,
         bool centered,
         bool multi_precision) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      mean_square.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      moment.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!mean_grad ||
       mean_grad->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "rmsprop_");
    paddle::optional<pir::Value> optional_mean_grad;
    if (!mean_grad) {
      optional_mean_grad = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_mean_grad = mean_grad;
    }
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::Rmsprop_Op rmsprop__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Rmsprop_Op>(
            param,
            mean_square,
            grad,
            moment,
            learning_rate,
            optional_mean_grad.get(),
            optional_master_param.get(),
            epsilon,
            decay,
            momentum,
            centered,
            multi_precision);
    paddle::optional<pir::OpResult> optional_mean_grad_out;
    if (!IsEmptyValue(rmsprop__op.result(3))) {
      optional_mean_grad_out =
          paddle::make_optional<pir::OpResult>(rmsprop__op.result(3));
    }
    paddle::optional<pir::OpResult> optional_master_param_outs;
    if (!IsEmptyValue(rmsprop__op.result(4))) {
      optional_master_param_outs =
          paddle::make_optional<pir::OpResult>(rmsprop__op.result(4));
    }
    if (!mean_grad) {
      rmsprop__op.result(3).set_type(pir::Type());
    }
    if (!master_param) {
      rmsprop__op.result(4).set_type(pir::Type());
    }
    return std::make_tuple(rmsprop__op.result(0),
                           rmsprop__op.result(1),
                           rmsprop__op.result(2),
                           optional_mean_grad_out,
                           optional_master_param_outs);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      mean_square.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      moment.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      (!mean_grad ||
       mean_grad->type().isa<paddle::dialect::DenseTensorType>()) &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "rmsprop_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_mean_grad;
    if (!mean_grad) {
      optional_mean_grad = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_mean_grad = mean_grad;
    }
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::RmspropDenseParamSparseGrad_Op
        rmsprop_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::RmspropDenseParamSparseGrad_Op>(
                    param,
                    mean_square,
                    grad,
                    moment,
                    learning_rate,
                    optional_mean_grad.get(),
                    optional_master_param.get(),
                    epsilon,
                    decay,
                    momentum,
                    centered,
                    multi_precision);
    paddle::optional<pir::OpResult> optional_mean_grad_out;
    if (!IsEmptyValue(rmsprop_dense_param_sparse_grad__op.result(3))) {
      optional_mean_grad_out = paddle::make_optional<pir::OpResult>(
          rmsprop_dense_param_sparse_grad__op.result(3));
    }
    paddle::optional<pir::OpResult> optional_master_param_outs;
    if (!IsEmptyValue(rmsprop_dense_param_sparse_grad__op.result(4))) {
      optional_master_param_outs = paddle::make_optional<pir::OpResult>(
          rmsprop_dense_param_sparse_grad__op.result(4));
    }
    if (!mean_grad) {
      rmsprop_dense_param_sparse_grad__op.result(3).set_type(pir::Type());
    }
    if (!master_param) {
      rmsprop_dense_param_sparse_grad__op.result(4).set_type(pir::Type());
    }
    return std::make_tuple(rmsprop_dense_param_sparse_grad__op.result(0),
                           rmsprop_dense_param_sparse_grad__op.result(1),
                           rmsprop_dense_param_sparse_grad__op.result(2),
                           optional_mean_grad_out,
                           optional_master_param_outs);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (rmsprop_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult roi_align(const pir::Value& x,
                        const pir::Value& boxes,
                        const paddle::optional<pir::Value>& boxes_num,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned) {
  CheckValueDataType(x, "x", "roi_align");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::RoiAlignOp roi_align_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RoiAlignOp>(
          x,
          boxes,
          optional_boxes_num.get(),
          pooled_height,
          pooled_width,
          spatial_scale,
          sampling_ratio,
          aligned);
  return roi_align_op.result(0);
}

pir::OpResult roi_pool(const pir::Value& x,
                       const pir::Value& boxes,
                       const paddle::optional<pir::Value>& boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale) {
  CheckValueDataType(x, "x", "roi_pool");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::RoiPoolOp roi_pool_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RoiPoolOp>(
          x,
          boxes,
          optional_boxes_num.get(),
          pooled_height,
          pooled_width,
          spatial_scale);
  return roi_pool_op.result(0);
}

pir::OpResult roll(const pir::Value& x,
                   const std::vector<int64_t>& shifts,
                   const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll");
  paddle::dialect::RollOp roll_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollOp>(
          x, shifts, axis);
  return roll_op.result(0);
}

pir::OpResult roll(const pir::Value& x,
                   pir::Value shifts,
                   const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll");
  paddle::dialect::RollOp roll_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollOp>(
          x, shifts, axis);
  return roll_op.result(0);
}

pir::OpResult roll(const pir::Value& x,
                   std::vector<pir::Value> shifts,
                   const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll");
  auto shifts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shifts);
  paddle::dialect::RollOp roll_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollOp>(
          x, shifts_combine_op.out(), axis);
  return roll_op.result(0);
}

pir::OpResult round(const pir::Value& x) {
  CheckValueDataType(x, "x", "round");
  paddle::dialect::RoundOp round_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RoundOp>(x);
  return round_op.result(0);
}

pir::OpResult round_(const pir::Value& x) {
  CheckValueDataType(x, "x", "round_");
  paddle::dialect::Round_Op round__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Round_Op>(x);
  return round__op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
rprop_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& prev,
       const pir::Value& learning_rate,
       const paddle::optional<pir::Value>& master_param,
       const pir::Value& learning_rate_range,
       const pir::Value& etas,
       bool multi_precision) {
  CheckValueDataType(param, "param", "rprop_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::dialect::Rprop_Op rprop__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Rprop_Op>(
          param,
          grad,
          prev,
          learning_rate,
          optional_master_param.get(),
          learning_rate_range,
          etas,
          multi_precision);
  paddle::optional<pir::OpResult> optional_master_param_out;
  if (!IsEmptyValue(rprop__op.result(3))) {
    optional_master_param_out =
        paddle::make_optional<pir::OpResult>(rprop__op.result(3));
  }
  if (!master_param) {
    rprop__op.result(3).set_type(pir::Type());
  }
  return std::make_tuple(rprop__op.result(0),
                         rprop__op.result(1),
                         rprop__op.result(2),
                         optional_master_param_out);
}

pir::OpResult rsqrt(const pir::Value& x) {
  CheckValueDataType(x, "x", "rsqrt");
  paddle::dialect::RsqrtOp rsqrt_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RsqrtOp>(x);
  return rsqrt_op.result(0);
}

pir::OpResult rsqrt_(const pir::Value& x) {
  CheckValueDataType(x, "x", "rsqrt_");
  paddle::dialect::Rsqrt_Op rsqrt__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Rsqrt_Op>(x);
  return rsqrt__op.result(0);
}

pir::OpResult scale(const pir::Value& x,
                    float scale,
                    float bias,
                    bool bias_after_scale) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "scale");
    paddle::dialect::ScaleOp scale_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleOp>(
            x, scale, bias, bias_after_scale);
    return scale_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "scale_sr");
    paddle::dialect::ScaleSrOp scale_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleSrOp>(
            x, scale, bias, bias_after_scale);
    return scale_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (scale) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult scale(const pir::Value& x,
                    pir::Value scale,
                    float bias,
                    bool bias_after_scale) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "scale");
    paddle::dialect::ScaleOp scale_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleOp>(
            x, scale, bias, bias_after_scale);
    return scale_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "scale_sr");
    paddle::dialect::ScaleSrOp scale_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleSrOp>(
            x, scale, bias, bias_after_scale);
    return scale_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (scale) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult scale_(const pir::Value& x,
                     float scale,
                     float bias,
                     bool bias_after_scale) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "scale_");
    paddle::dialect::Scale_Op scale__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Scale_Op>(
            x, scale, bias, bias_after_scale);
    return scale__op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "scale_sr_");
    paddle::dialect::ScaleSr_Op scale_sr__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleSr_Op>(
            x, scale, bias, bias_after_scale);
    return scale_sr__op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (scale_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult scale_(const pir::Value& x,
                     pir::Value scale,
                     float bias,
                     bool bias_after_scale) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "scale_");
    paddle::dialect::Scale_Op scale__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Scale_Op>(
            x, scale, bias, bias_after_scale);
    return scale__op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "scale_sr_");
    paddle::dialect::ScaleSr_Op scale_sr__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleSr_Op>(
            x, scale, bias, bias_after_scale);
    return scale_sr__op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (scale_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult scatter(const pir::Value& x,
                      const pir::Value& index,
                      const pir::Value& updates,
                      bool overwrite) {
  CheckValueDataType(x, "x", "scatter");
  paddle::dialect::ScatterOp scatter_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScatterOp>(
          x, index, updates, overwrite);
  return scatter_op.result(0);
}

pir::OpResult scatter_(const pir::Value& x,
                       const pir::Value& index,
                       const pir::Value& updates,
                       bool overwrite) {
  CheckValueDataType(x, "x", "scatter_");
  paddle::dialect::Scatter_Op scatter__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Scatter_Op>(
          x, index, updates, overwrite);
  return scatter__op.result(0);
}

pir::OpResult scatter_nd_add(const pir::Value& x,
                             const pir::Value& index,
                             const pir::Value& updates) {
  CheckValueDataType(x, "x", "scatter_nd_add");
  paddle::dialect::ScatterNdAddOp scatter_nd_add_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ScatterNdAddOp>(x, index, updates);
  return scatter_nd_add_op.result(0);
}

pir::OpResult searchsorted(const pir::Value& sorted_sequence,
                           const pir::Value& values,
                           bool out_int32,
                           bool right) {
  CheckValueDataType(sorted_sequence, "sorted_sequence", "searchsorted");
  paddle::dialect::SearchsortedOp searchsorted_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SearchsortedOp>(
              sorted_sequence, values, out_int32, right);
  return searchsorted_op.result(0);
}

pir::OpResult segment_pool(const pir::Value& x,
                           const pir::Value& segment_ids,
                           const std::string& pooltype) {
  CheckValueDataType(x, "x", "segment_pool");
  paddle::dialect::SegmentPoolOp segment_pool_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SegmentPoolOp>(x, segment_ids, pooltype);
  return segment_pool_op.result(0);
}

pir::OpResult selu(const pir::Value& x, float scale, float alpha) {
  CheckValueDataType(x, "x", "selu");
  paddle::dialect::SeluOp selu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SeluOp>(
          x, scale, alpha);
  return selu_op.result(0);
}

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          const std::string& reduce_op,
                          const std::vector<int64_t>& out_size) {
  CheckValueDataType(x, "x", "send_u_recv");
  paddle::dialect::SendURecvOp send_u_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendURecvOp>(
          x, src_index, dst_index, reduce_op, out_size);
  return send_u_recv_op.result(0);
}

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          pir::Value out_size,
                          const std::string& reduce_op) {
  CheckValueDataType(x, "x", "send_u_recv");
  paddle::dialect::SendURecvOp send_u_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendURecvOp>(
          x, src_index, dst_index, out_size, reduce_op);
  return send_u_recv_op.result(0);
}

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          std::vector<pir::Value> out_size,
                          const std::string& reduce_op) {
  CheckValueDataType(x, "x", "send_u_recv");
  auto out_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_size);
  paddle::dialect::SendURecvOp send_u_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendURecvOp>(
          x, src_index, dst_index, out_size_combine_op.out(), reduce_op);
  return send_u_recv_op.result(0);
}

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           const std::string& message_op,
                           const std::string& reduce_op,
                           const std::vector<int64_t>& out_size) {
  CheckValueDataType(x, "x", "send_ue_recv");
  paddle::dialect::SendUeRecvOp send_ue_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendUeRecvOp>(
          x, y, src_index, dst_index, message_op, reduce_op, out_size);
  return send_ue_recv_op.result(0);
}

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           pir::Value out_size,
                           const std::string& message_op,
                           const std::string& reduce_op) {
  CheckValueDataType(x, "x", "send_ue_recv");
  paddle::dialect::SendUeRecvOp send_ue_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendUeRecvOp>(
          x, y, src_index, dst_index, out_size, message_op, reduce_op);
  return send_ue_recv_op.result(0);
}

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           std::vector<pir::Value> out_size,
                           const std::string& message_op,
                           const std::string& reduce_op) {
  CheckValueDataType(x, "x", "send_ue_recv");
  auto out_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_size);
  paddle::dialect::SendUeRecvOp send_ue_recv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendUeRecvOp>(
          x,
          y,
          src_index,
          dst_index,
          out_size_combine_op.out(),
          message_op,
          reduce_op);
  return send_ue_recv_op.result(0);
}

pir::OpResult send_uv(const pir::Value& x,
                      const pir::Value& y,
                      const pir::Value& src_index,
                      const pir::Value& dst_index,
                      const std::string& message_op) {
  CheckValueDataType(x, "x", "send_uv");
  paddle::dialect::SendUvOp send_uv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendUvOp>(
          x, y, src_index, dst_index, message_op);
  return send_uv_op.result(0);
}

std::tuple<pir::OpResult, paddle::optional<pir::OpResult>> sgd_(
    const pir::Value& param,
    const pir::Value& learning_rate,
    const pir::Value& grad,
    const paddle::optional<pir::Value>& master_param,
    bool multi_precision) {
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::DenseTensorType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "sgd_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::Sgd_Op sgd__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Sgd_Op>(
            param,
            learning_rate,
            grad,
            optional_master_param.get(),
            multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(sgd__op.result(1))) {
      optional_master_param_out =
          paddle::make_optional<pir::OpResult>(sgd__op.result(1));
    }
    if (!master_param) {
      sgd__op.result(1).set_type(pir::Type());
    }
    return std::make_tuple(sgd__op.result(0), optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::DenseTensorType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::DenseTensorType>())) {
    CheckValueDataType(param, "param", "sgd_dense_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::SgdDenseParamSparseGrad_Op
        sgd_dense_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::SgdDenseParamSparseGrad_Op>(
                    param,
                    learning_rate,
                    grad,
                    optional_master_param.get(),
                    multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(sgd_dense_param_sparse_grad__op.result(1))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          sgd_dense_param_sparse_grad__op.result(1));
    }
    if (!master_param) {
      sgd_dense_param_sparse_grad__op.result(1).set_type(pir::Type());
    }
    return std::make_tuple(sgd_dense_param_sparse_grad__op.result(0),
                           optional_master_param_out);
  }
  if (param.type().isa<paddle::dialect::SelectedRowsType>() &&
      learning_rate.type().isa<paddle::dialect::DenseTensorType>() &&
      grad.type().isa<paddle::dialect::SelectedRowsType>() &&
      (!master_param ||
       master_param->type().isa<paddle::dialect::SelectedRowsType>())) {
    CheckValueDataType(param, "param", "sgd_sparse_param_sparse_grad_");
    paddle::optional<pir::Value> optional_master_param;
    if (!master_param) {
      optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
    } else {
      optional_master_param = master_param;
    }
    paddle::dialect::SgdSparseParamSparseGrad_Op
        sgd_sparse_param_sparse_grad__op =
            ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::SgdSparseParamSparseGrad_Op>(
                    param,
                    learning_rate,
                    grad,
                    optional_master_param.get(),
                    multi_precision);
    paddle::optional<pir::OpResult> optional_master_param_out;
    if (!IsEmptyValue(sgd_sparse_param_sparse_grad__op.result(1))) {
      optional_master_param_out = paddle::make_optional<pir::OpResult>(
          sgd_sparse_param_sparse_grad__op.result(1));
    }
    if (!master_param) {
      sgd_sparse_param_sparse_grad__op.result(1).set_type(pir::Type());
    }
    return std::make_tuple(sgd_sparse_param_sparse_grad__op.result(0),
                           optional_master_param_out);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (sgd_) for input Value is unimplemented, please check the "
      "type of input Value."));
}

pir::OpResult shape(const pir::Value& input) {
  if (input.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(input, "input", "shape");
    paddle::dialect::ShapeOp shape_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ShapeOp>(
            input);
    return shape_op.result(0);
  }
  if (input.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(input, "input", "shape_sr");
    paddle::dialect::ShapeSrOp shape_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ShapeSrOp>(
            input);
    return shape_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (shape) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult shard_index(const pir::Value& input,
                          int index_num,
                          int nshards,
                          int shard_id,
                          int ignore_value) {
  CheckValueDataType(input, "input", "shard_index");
  paddle::dialect::ShardIndexOp shard_index_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ShardIndexOp>(
          input, index_num, nshards, shard_id, ignore_value);
  return shard_index_op.result(0);
}

pir::OpResult sigmoid(const pir::Value& x) {
  CheckValueDataType(x, "x", "sigmoid");
  paddle::dialect::SigmoidOp sigmoid_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SigmoidOp>(x);
  return sigmoid_op.result(0);
}

pir::OpResult sigmoid_(const pir::Value& x) {
  CheckValueDataType(x, "x", "sigmoid_");
  paddle::dialect::Sigmoid_Op sigmoid__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Sigmoid_Op>(
          x);
  return sigmoid__op.result(0);
}

pir::OpResult sigmoid_cross_entropy_with_logits(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    bool normalize,
    int ignore_index) {
  if (pos_weight) {
    CheckValueDataType(
        pos_weight.get(), "pos_weight", "sigmoid_cross_entropy_with_logits");
  } else {
    CheckValueDataType(label, "label", "sigmoid_cross_entropy_with_logits");
  }
  paddle::optional<pir::Value> optional_pos_weight;
  if (!pos_weight) {
    optional_pos_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pos_weight = pos_weight;
  }
  paddle::dialect::SigmoidCrossEntropyWithLogitsOp
      sigmoid_cross_entropy_with_logits_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::SigmoidCrossEntropyWithLogitsOp>(
                  x, label, optional_pos_weight.get(), normalize, ignore_index);
  return sigmoid_cross_entropy_with_logits_op.result(0);
}

pir::OpResult sigmoid_cross_entropy_with_logits_(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    bool normalize,
    int ignore_index) {
  if (pos_weight) {
    CheckValueDataType(
        pos_weight.get(), "pos_weight", "sigmoid_cross_entropy_with_logits_");
  } else {
    CheckValueDataType(label, "label", "sigmoid_cross_entropy_with_logits_");
  }
  paddle::optional<pir::Value> optional_pos_weight;
  if (!pos_weight) {
    optional_pos_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pos_weight = pos_weight;
  }
  paddle::dialect::SigmoidCrossEntropyWithLogits_Op
      sigmoid_cross_entropy_with_logits__op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::SigmoidCrossEntropyWithLogits_Op>(
                  x, label, optional_pos_weight.get(), normalize, ignore_index);
  return sigmoid_cross_entropy_with_logits__op.result(0);
}

pir::OpResult sign(const pir::Value& x) {
  CheckValueDataType(x, "x", "sign");
  paddle::dialect::SignOp sign_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SignOp>(x);
  return sign_op.result(0);
}

pir::OpResult silu(const pir::Value& x) {
  CheckValueDataType(x, "x", "silu");
  paddle::dialect::SiluOp silu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SiluOp>(x);
  return silu_op.result(0);
}

pir::OpResult sin(const pir::Value& x) {
  CheckValueDataType(x, "x", "sin");
  paddle::dialect::SinOp sin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinOp>(x);
  return sin_op.result(0);
}

pir::OpResult sin_(const pir::Value& x) {
  CheckValueDataType(x, "x", "sin_");
  paddle::dialect::Sin_Op sin__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Sin_Op>(x);
  return sin__op.result(0);
}

pir::OpResult sinh(const pir::Value& x) {
  CheckValueDataType(x, "x", "sinh");
  paddle::dialect::SinhOp sinh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinhOp>(x);
  return sinh_op.result(0);
}

pir::OpResult sinh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "sinh_");
  paddle::dialect::Sinh_Op sinh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Sinh_Op>(x);
  return sinh__op.result(0);
}

pir::OpResult slogdet(const pir::Value& x) {
  CheckValueDataType(x, "x", "slogdet");
  paddle::dialect::SlogdetOp slogdet_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SlogdetOp>(x);
  return slogdet_op.result(0);
}

pir::OpResult softplus(const pir::Value& x, float beta, float threshold) {
  CheckValueDataType(x, "x", "softplus");
  paddle::dialect::SoftplusOp softplus_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SoftplusOp>(
          x, beta, threshold);
  return softplus_op.result(0);
}

pir::OpResult softshrink(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "softshrink");
  paddle::dialect::SoftshrinkOp softshrink_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SoftshrinkOp>(
          x, threshold);
  return softshrink_op.result(0);
}

pir::OpResult softsign(const pir::Value& x) {
  CheckValueDataType(x, "x", "softsign");
  paddle::dialect::SoftsignOp softsign_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SoftsignOp>(
          x);
  return softsign_op.result(0);
}

pir::OpResult solve(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "solve");
  paddle::dialect::SolveOp solve_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SolveOp>(x,
                                                                           y);
  return solve_op.result(0);
}

pir::OpResult spectral_norm(const pir::Value& weight,
                            const pir::Value& u,
                            const pir::Value& v,
                            int dim,
                            int power_iters,
                            float eps) {
  CheckValueDataType(weight, "weight", "spectral_norm");
  paddle::dialect::SpectralNormOp spectral_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SpectralNormOp>(
              weight, u, v, dim, power_iters, eps);
  return spectral_norm_op.result(0);
}

pir::OpResult sqrt(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "sqrt");
    paddle::dialect::SqrtOp sqrt_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqrtOp>(x);
    return sqrt_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "sqrt_sr");
    paddle::dialect::SqrtSrOp sqrt_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqrtSrOp>(
            x);
    return sqrt_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (sqrt) for input Value is unimplemented, please check the "
      "type of input Value."));
}

pir::OpResult sqrt_(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "sqrt_");
    paddle::dialect::Sqrt_Op sqrt__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Sqrt_Op>(x);
    return sqrt__op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "sqrt_sr_");
    paddle::dialect::SqrtSr_Op sqrt_sr__op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqrtSr_Op>(
            x);
    return sqrt_sr__op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (sqrt_) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult square(const pir::Value& x) {
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(x, "x", "square");
    paddle::dialect::SquareOp square_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SquareOp>(
            x);
    return square_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(x, "x", "square_sr");
    paddle::dialect::SquareSrOp square_sr_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SquareSrOp>(
            x);
    return square_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (square) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult squared_l2_norm(const pir::Value& x) {
  CheckValueDataType(x, "x", "squared_l2_norm");
  paddle::dialect::SquaredL2NormOp squared_l2_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SquaredL2NormOp>(x);
  return squared_l2_norm_op.result(0);
}

pir::OpResult squeeze(const pir::Value& x, const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "squeeze");
  paddle::dialect::SqueezeOp squeeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqueezeOp>(
          x, axis);
  return squeeze_op.result(0);
}

pir::OpResult squeeze(const pir::Value& x, pir::Value axis) {
  CheckValueDataType(x, "x", "squeeze");
  paddle::dialect::SqueezeOp squeeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqueezeOp>(
          x, axis);
  return squeeze_op.result(0);
}

pir::OpResult squeeze(const pir::Value& x, std::vector<pir::Value> axis) {
  CheckValueDataType(x, "x", "squeeze");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::SqueezeOp squeeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqueezeOp>(
          x, axis_combine_op.out());
  return squeeze_op.result(0);
}

pir::OpResult squeeze_(const pir::Value& x, const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "squeeze_");
  paddle::dialect::Squeeze_Op squeeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Squeeze_Op>(
          x, axis);
  return squeeze__op.result(0);
}

pir::OpResult squeeze_(const pir::Value& x, pir::Value axis) {
  CheckValueDataType(x, "x", "squeeze_");
  paddle::dialect::Squeeze_Op squeeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Squeeze_Op>(
          x, axis);
  return squeeze__op.result(0);
}

pir::OpResult squeeze_(const pir::Value& x, std::vector<pir::Value> axis) {
  CheckValueDataType(x, "x", "squeeze_");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::Squeeze_Op squeeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Squeeze_Op>(
          x, axis_combine_op.out());
  return squeeze__op.result(0);
}

pir::OpResult stack(const std::vector<pir::Value>& x, int axis) {
  CheckVectorOfValueDataType(x, "x", "stack");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::StackOp stack_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::StackOp>(
          x_combine_op.out(), axis);
  return stack_op.result(0);
}

pir::OpResult standard_gamma(const pir::Value& x) {
  CheckValueDataType(x, "x", "standard_gamma");
  paddle::dialect::StandardGammaOp standard_gamma_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StandardGammaOp>(x);
  return standard_gamma_op.result(0);
}

pir::OpResult stanh(const pir::Value& x, float scale_a, float scale_b) {
  CheckValueDataType(x, "x", "stanh");
  paddle::dialect::StanhOp stanh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::StanhOp>(
          x, scale_a, scale_b);
  return stanh_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> svd(
    const pir::Value& x, bool full_matrices) {
  CheckValueDataType(x, "x", "svd");
  paddle::dialect::SvdOp svd_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SvdOp>(
          x, full_matrices);
  return std::make_tuple(svd_op.result(0), svd_op.result(1), svd_op.result(2));
}

pir::OpResult take_along_axis(const pir::Value& arr,
                              const pir::Value& indices,
                              int axis) {
  CheckValueDataType(arr, "arr", "take_along_axis");
  paddle::dialect::TakeAlongAxisOp take_along_axis_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TakeAlongAxisOp>(arr, indices, axis);
  return take_along_axis_op.result(0);
}

pir::OpResult tan(const pir::Value& x) {
  CheckValueDataType(x, "x", "tan");
  paddle::dialect::TanOp tan_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanOp>(x);
  return tan_op.result(0);
}

pir::OpResult tan_(const pir::Value& x) {
  CheckValueDataType(x, "x", "tan_");
  paddle::dialect::Tan_Op tan__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Tan_Op>(x);
  return tan__op.result(0);
}

pir::OpResult tanh(const pir::Value& x) {
  CheckValueDataType(x, "x", "tanh");
  paddle::dialect::TanhOp tanh_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanhOp>(x);
  return tanh_op.result(0);
}

pir::OpResult tanh_(const pir::Value& x) {
  CheckValueDataType(x, "x", "tanh_");
  paddle::dialect::Tanh_Op tanh__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Tanh_Op>(x);
  return tanh__op.result(0);
}

pir::OpResult tanh_shrink(const pir::Value& x) {
  CheckValueDataType(x, "x", "tanh_shrink");
  paddle::dialect::TanhShrinkOp tanh_shrink_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanhShrinkOp>(
          x);
  return tanh_shrink_op.result(0);
}

pir::OpResult temporal_shift(const pir::Value& x,
                             int seg_num,
                             float shift_ratio,
                             const std::string& data_format) {
  CheckValueDataType(x, "x", "temporal_shift");
  paddle::dialect::TemporalShiftOp temporal_shift_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TemporalShiftOp>(
              x, seg_num, shift_ratio, data_format);
  return temporal_shift_op.result(0);
}

pir::OpResult tensor_unfold(const pir::Value& input,
                            int64_t axis,
                            int64_t size,
                            int64_t step) {
  CheckValueDataType(input, "input", "tensor_unfold");
  paddle::dialect::TensorUnfoldOp tensor_unfold_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TensorUnfoldOp>(input, axis, size, step);
  return tensor_unfold_op.result(0);
}

pir::OpResult thresholded_relu(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "thresholded_relu");
  paddle::dialect::ThresholdedReluOp thresholded_relu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ThresholdedReluOp>(x, threshold);
  return thresholded_relu_op.result(0);
}

pir::OpResult thresholded_relu_(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "thresholded_relu_");
  paddle::dialect::ThresholdedRelu_Op thresholded_relu__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ThresholdedRelu_Op>(x, threshold);
  return thresholded_relu__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> top_p_sampling(
    const pir::Value& x,
    const pir::Value& ps,
    const paddle::optional<pir::Value>& threshold,
    int seed) {
  CheckValueDataType(x, "x", "top_p_sampling");
  paddle::optional<pir::Value> optional_threshold;
  if (!threshold) {
    optional_threshold = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_threshold = threshold;
  }
  paddle::dialect::TopPSamplingOp top_p_sampling_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TopPSamplingOp>(
              x, ps, optional_threshold.get(), seed);
  return std::make_tuple(top_p_sampling_op.result(0),
                         top_p_sampling_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> topk(
    const pir::Value& x, int k, int axis, bool largest, bool sorted) {
  CheckValueDataType(x, "x", "topk");
  paddle::dialect::TopkOp topk_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TopkOp>(
          x, k, axis, largest, sorted);
  return std::make_tuple(topk_op.result(0), topk_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> topk(
    const pir::Value& x, pir::Value k, int axis, bool largest, bool sorted) {
  CheckValueDataType(x, "x", "topk");
  paddle::dialect::TopkOp topk_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TopkOp>(
          x, k, axis, largest, sorted);
  return std::make_tuple(topk_op.result(0), topk_op.result(1));
}

pir::OpResult trace(const pir::Value& x, int offset, int axis1, int axis2) {
  CheckValueDataType(x, "x", "trace");
  paddle::dialect::TraceOp trace_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TraceOp>(
          x, offset, axis1, axis2);
  return trace_op.result(0);
}

pir::OpResult triangular_solve(const pir::Value& x,
                               const pir::Value& y,
                               bool upper,
                               bool transpose,
                               bool unitriangular) {
  CheckValueDataType(x, "x", "triangular_solve");
  paddle::dialect::TriangularSolveOp triangular_solve_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TriangularSolveOp>(
              x, y, upper, transpose, unitriangular);
  return triangular_solve_op.result(0);
}

pir::OpResult trilinear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(x, "x", "trilinear_interp");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::TrilinearInterpOp trilinear_interp_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TrilinearInterpOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return trilinear_interp_op.result(0);
}

pir::OpResult trunc(const pir::Value& input) {
  CheckValueDataType(input, "input", "trunc");
  paddle::dialect::TruncOp trunc_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TruncOp>(
          input);
  return trunc_op.result(0);
}

pir::OpResult trunc_(const pir::Value& input) {
  CheckValueDataType(input, "input", "trunc_");
  paddle::dialect::Trunc_Op trunc__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Trunc_Op>(
          input);
  return trunc__op.result(0);
}

std::vector<pir::OpResult> unbind(const pir::Value& input, int axis) {
  CheckValueDataType(input, "input", "unbind");
  paddle::dialect::UnbindOp unbind_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnbindOp>(
          input, axis);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      unbind_op.result(0));
  return out_split_op.outputs();
}

pir::OpResult unfold(const pir::Value& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  CheckValueDataType(x, "x", "unfold");
  paddle::dialect::UnfoldOp unfold_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnfoldOp>(
          x, kernel_sizes, strides, paddings, dilations);
  return unfold_op.result(0);
}

pir::OpResult uniform_inplace(const pir::Value& x,
                              float min,
                              float max,
                              int seed,
                              int diag_num,
                              int diag_step,
                              float diag_val) {
  CheckValueDataType(x, "x", "uniform_inplace");
  paddle::dialect::UniformInplaceOp uniform_inplace_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UniformInplaceOp>(
              x, min, max, seed, diag_num, diag_step, diag_val);
  return uniform_inplace_op.result(0);
}

pir::OpResult uniform_inplace_(const pir::Value& x,
                               float min,
                               float max,
                               int seed,
                               int diag_num,
                               int diag_step,
                               float diag_val) {
  CheckValueDataType(x, "x", "uniform_inplace_");
  paddle::dialect::UniformInplace_Op uniform_inplace__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UniformInplace_Op>(
              x, min, max, seed, diag_num, diag_step, diag_val);
  return uniform_inplace__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> unique_consecutive(
    const pir::Value& x,
    bool return_inverse,
    bool return_counts,
    const std::vector<int>& axis,
    phi::DataType dtype) {
  CheckValueDataType(x, "x", "unique_consecutive");
  paddle::dialect::UniqueConsecutiveOp unique_consecutive_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UniqueConsecutiveOp>(
              x, return_inverse, return_counts, axis, dtype);
  return std::make_tuple(unique_consecutive_op.result(0),
                         unique_consecutive_op.result(1),
                         unique_consecutive_op.result(2));
}

pir::OpResult unpool3d(const pir::Value& x,
                       const pir::Value& indices,
                       const std::vector<int>& ksize,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& output_size,
                       const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool3d");
  paddle::dialect::Unpool3dOp unpool3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Unpool3dOp>(
          x, indices, ksize, strides, paddings, output_size, data_format);
  return unpool3d_op.result(0);
}

pir::OpResult unsqueeze(const pir::Value& x, const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "unsqueeze");
  paddle::dialect::UnsqueezeOp unsqueeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnsqueezeOp>(
          x, axis);
  return unsqueeze_op.result(0);
}

pir::OpResult unsqueeze(const pir::Value& x, pir::Value axis) {
  CheckValueDataType(x, "x", "unsqueeze");
  paddle::dialect::UnsqueezeOp unsqueeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnsqueezeOp>(
          x, axis);
  return unsqueeze_op.result(0);
}

pir::OpResult unsqueeze(const pir::Value& x, std::vector<pir::Value> axis) {
  CheckValueDataType(x, "x", "unsqueeze");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::UnsqueezeOp unsqueeze_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnsqueezeOp>(
          x, axis_combine_op.out());
  return unsqueeze_op.result(0);
}

pir::OpResult unsqueeze_(const pir::Value& x,
                         const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "unsqueeze_");
  paddle::dialect::Unsqueeze_Op unsqueeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Unsqueeze_Op>(
          x, axis);
  return unsqueeze__op.result(0);
}

pir::OpResult unsqueeze_(const pir::Value& x, pir::Value axis) {
  CheckValueDataType(x, "x", "unsqueeze_");
  paddle::dialect::Unsqueeze_Op unsqueeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Unsqueeze_Op>(
          x, axis);
  return unsqueeze__op.result(0);
}

pir::OpResult unsqueeze_(const pir::Value& x, std::vector<pir::Value> axis) {
  CheckValueDataType(x, "x", "unsqueeze_");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::Unsqueeze_Op unsqueeze__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Unsqueeze_Op>(
          x, axis_combine_op.out());
  return unsqueeze__op.result(0);
}

std::vector<pir::OpResult> unstack(const pir::Value& x, int axis, int num) {
  CheckValueDataType(x, "x", "unstack");
  paddle::dialect::UnstackOp unstack_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnstackOp>(
          x, axis, num);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      unstack_op.result(0));
  return out_split_op.outputs();
}

std::tuple<std::vector<pir::OpResult>,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
update_loss_scaling_(const std::vector<pir::Value>& x,
                     const pir::Value& found_infinite,
                     const pir::Value& prev_loss_scaling,
                     const pir::Value& in_good_steps,
                     const pir::Value& in_bad_steps,
                     int incr_every_n_steps,
                     int decr_every_n_nan_or_inf,
                     float incr_ratio,
                     float decr_ratio,
                     bool stop_update) {
  CheckVectorOfValueDataType(x, "x", "update_loss_scaling_");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::UpdateLossScaling_Op update_loss_scaling__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UpdateLossScaling_Op>(
              x_combine_op.out(),
              found_infinite,
              prev_loss_scaling,
              in_good_steps,
              in_bad_steps,
              incr_every_n_steps,
              decr_every_n_nan_or_inf,
              incr_ratio,
              decr_ratio,
              stop_update);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      update_loss_scaling__op.result(0));
  return std::make_tuple(out_split_op.outputs(),
                         update_loss_scaling__op.result(1),
                         update_loss_scaling__op.result(2),
                         update_loss_scaling__op.result(3));
}

std::tuple<std::vector<pir::OpResult>,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
update_loss_scaling_(const std::vector<pir::Value>& x,
                     const pir::Value& found_infinite,
                     const pir::Value& prev_loss_scaling,
                     const pir::Value& in_good_steps,
                     const pir::Value& in_bad_steps,
                     pir::Value stop_update,
                     int incr_every_n_steps,
                     int decr_every_n_nan_or_inf,
                     float incr_ratio,
                     float decr_ratio) {
  CheckVectorOfValueDataType(x, "x", "update_loss_scaling_");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::UpdateLossScaling_Op update_loss_scaling__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UpdateLossScaling_Op>(
              x_combine_op.out(),
              found_infinite,
              prev_loss_scaling,
              in_good_steps,
              in_bad_steps,
              stop_update,
              incr_every_n_steps,
              decr_every_n_nan_or_inf,
              incr_ratio,
              decr_ratio);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      update_loss_scaling__op.result(0));
  return std::make_tuple(out_split_op.outputs(),
                         update_loss_scaling__op.result(1),
                         update_loss_scaling__op.result(2),
                         update_loss_scaling__op.result(3));
}

pir::OpResult view_dtype(const pir::Value& input, phi::DataType dtype) {
  CheckValueDataType(input, "input", "view_dtype");
  paddle::dialect::ViewDtypeOp view_dtype_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ViewDtypeOp>(
          input, dtype);
  return view_dtype_op.result(0);
}

pir::OpResult view_shape(const pir::Value& input,
                         const std::vector<int64_t>& dims) {
  CheckValueDataType(input, "input", "view_shape");
  paddle::dialect::ViewShapeOp view_shape_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ViewShapeOp>(
          input, dims);
  return view_shape_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> viterbi_decode(
    const pir::Value& potentials,
    const pir::Value& transition_params,
    const pir::Value& lengths,
    bool include_bos_eos_tag) {
  CheckValueDataType(potentials, "potentials", "viterbi_decode");
  paddle::dialect::ViterbiDecodeOp viterbi_decode_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ViterbiDecodeOp>(
              potentials, transition_params, lengths, include_bos_eos_tag);
  return std::make_tuple(viterbi_decode_op.result(0),
                         viterbi_decode_op.result(1));
}

pir::OpResult warpctc(const pir::Value& logits,
                      const pir::Value& label,
                      const paddle::optional<pir::Value>& logits_length,
                      const paddle::optional<pir::Value>& labels_length,
                      int blank,
                      bool norm_by_times) {
  CheckValueDataType(logits, "logits", "warpctc");
  paddle::optional<pir::Value> optional_logits_length;
  if (!logits_length) {
    optional_logits_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_logits_length = logits_length;
  }
  paddle::optional<pir::Value> optional_labels_length;
  if (!labels_length) {
    optional_labels_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_labels_length = labels_length;
  }
  paddle::dialect::WarpctcOp warpctc_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::WarpctcOp>(
          logits,
          label,
          optional_logits_length.get(),
          optional_labels_length.get(),
          blank,
          norm_by_times);
  return warpctc_op.result(0);
}

pir::OpResult warprnnt(const pir::Value& input,
                       const pir::Value& label,
                       const pir::Value& input_lengths,
                       const pir::Value& label_lengths,
                       int blank,
                       float fastemit_lambda) {
  CheckValueDataType(input, "input", "warprnnt");
  paddle::dialect::WarprnntOp warprnnt_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::WarprnntOp>(
          input, label, input_lengths, label_lengths, blank, fastemit_lambda);
  return warprnnt_op.result(0);
}

pir::OpResult weight_dequantize(const pir::Value& x,
                                const pir::Value& scale,
                                const std::string& algo,
                                phi::DataType out_dtype,
                                int group_size) {
  CheckDataType(out_dtype, "out_dtype", "weight_dequantize");
  paddle::dialect::WeightDequantizeOp weight_dequantize_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WeightDequantizeOp>(
              x, scale, algo, out_dtype, group_size);
  return weight_dequantize_op.result(0);
}

pir::OpResult weight_only_linear(const pir::Value& x,
                                 const pir::Value& weight,
                                 const paddle::optional<pir::Value>& bias,
                                 const pir::Value& weight_scale,
                                 const std::string& weight_dtype,
                                 int arch,
                                 int group_size) {
  CheckValueDataType(x, "x", "weight_only_linear");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::WeightOnlyLinearOp weight_only_linear_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WeightOnlyLinearOp>(x,
                                                       weight,
                                                       optional_bias.get(),
                                                       weight_scale,
                                                       weight_dtype,
                                                       arch,
                                                       group_size);
  return weight_only_linear_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> weight_quantize(
    const pir::Value& x, const std::string& algo, int arch, int group_size) {
  CheckValueDataType(x, "x", "weight_quantize");
  paddle::dialect::WeightQuantizeOp weight_quantize_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WeightQuantizeOp>(x, algo, arch, group_size);
  return std::make_tuple(weight_quantize_op.result(0),
                         weight_quantize_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
weighted_sample_neighbors(const pir::Value& row,
                          const pir::Value& colptr,
                          const pir::Value& edge_weight,
                          const pir::Value& input_nodes,
                          const paddle::optional<pir::Value>& eids,
                          int sample_size,
                          bool return_eids) {
  if (eids) {
    CheckValueDataType(eids.get(), "eids", "weighted_sample_neighbors");
  } else {
    CheckValueDataType(input_nodes, "input_nodes", "weighted_sample_neighbors");
  }
  paddle::optional<pir::Value> optional_eids;
  if (!eids) {
    optional_eids = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_eids = eids;
  }
  paddle::dialect::WeightedSampleNeighborsOp weighted_sample_neighbors_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WeightedSampleNeighborsOp>(
              row,
              colptr,
              edge_weight,
              input_nodes,
              optional_eids.get(),
              sample_size,
              return_eids);
  return std::make_tuple(weighted_sample_neighbors_op.result(0),
                         weighted_sample_neighbors_op.result(1),
                         weighted_sample_neighbors_op.result(2));
}

pir::OpResult where(const pir::Value& condition,
                    const pir::Value& x,
                    const pir::Value& y) {
  CheckValueDataType(y, "y", "where");
  paddle::dialect::WhereOp where_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::WhereOp>(
          condition, x, y);
  return where_op.result(0);
}

pir::OpResult where_(const pir::Value& condition,
                     const pir::Value& x,
                     const pir::Value& y) {
  CheckValueDataType(y, "y", "where_");
  paddle::dialect::Where_Op where__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Where_Op>(
          condition, x, y);
  return where__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> yolo_box(
    const pir::Value& x,
    const pir::Value& img_size,
    const std::vector<int>& anchors,
    int class_num,
    float conf_thresh,
    int downsample_ratio,
    bool clip_bbox,
    float scale_x_y,
    bool iou_aware,
    float iou_aware_factor) {
  CheckValueDataType(x, "x", "yolo_box");
  paddle::dialect::YoloBoxOp yolo_box_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::YoloBoxOp>(
          x,
          img_size,
          anchors,
          class_num,
          conf_thresh,
          downsample_ratio,
          clip_bbox,
          scale_x_y,
          iou_aware,
          iou_aware_factor);
  return std::make_tuple(yolo_box_op.result(0), yolo_box_op.result(1));
}

pir::OpResult yolo_loss(const pir::Value& x,
                        const pir::Value& gt_box,
                        const pir::Value& gt_label,
                        const paddle::optional<pir::Value>& gt_score,
                        const std::vector<int>& anchors,
                        const std::vector<int>& anchor_mask,
                        int class_num,
                        float ignore_thresh,
                        int downsample_ratio,
                        bool use_label_smooth,
                        float scale_x_y) {
  CheckValueDataType(x, "x", "yolo_loss");
  paddle::optional<pir::Value> optional_gt_score;
  if (!gt_score) {
    optional_gt_score = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_gt_score = gt_score;
  }
  paddle::dialect::YoloLossOp yolo_loss_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::YoloLossOp>(
          x,
          gt_box,
          gt_label,
          optional_gt_score.get(),
          anchors,
          anchor_mask,
          class_num,
          ignore_thresh,
          downsample_ratio,
          use_label_smooth,
          scale_x_y);
  return yolo_loss_op.result(0);
}

pir::OpResult abs_double_grad(const pir::Value& x,
                              const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "abs_double_grad");
  paddle::dialect::AbsDoubleGradOp abs_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AbsDoubleGradOp>(x, grad_x_grad);
  return abs_double_grad_op.result(0);
}

pir::OpResult abs_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(x, "x", "abs_grad");
  paddle::dialect::AbsGradOp abs_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AbsGradOp>(
          x, out_grad);
  return abs_grad_op.result(0);
}

pir::OpResult acos_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "acos_grad");
  paddle::dialect::AcosGradOp acos_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcosGradOp>(
          x, out_grad);
  return acos_grad_op.result(0);
}

pir::OpResult acos_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "acos_grad_");
  paddle::dialect::AcosGrad_Op acos_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcosGrad_Op>(
          x, out_grad);
  return acos_grad__op.result(0);
}

pir::OpResult acosh_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "acosh_grad");
  paddle::dialect::AcoshGradOp acosh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcoshGradOp>(
          x, out_grad);
  return acosh_grad_op.result(0);
}

pir::OpResult acosh_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "acosh_grad_");
  paddle::dialect::AcoshGrad_Op acosh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AcoshGrad_Op>(
          x, out_grad);
  return acosh_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> addmm_grad(
    const pir::Value& input,
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    float alpha,
    float beta) {
  CheckValueDataType(out_grad, "out_grad", "addmm_grad");
  paddle::dialect::AddmmGradOp addmm_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddmmGradOp>(
          input, x, y, out_grad, alpha, beta);
  return std::make_tuple(addmm_grad_op.result(0),
                         addmm_grad_op.result(1),
                         addmm_grad_op.result(2));
}

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               const std::vector<int64_t>& output_shape,
                               bool align_corners) {
  CheckValueDataType(output_grad, "output_grad", "affine_grid_grad");
  paddle::dialect::AffineGridGradOp affine_grid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AffineGridGradOp>(
              input, output_grad, output_shape, align_corners);
  return affine_grid_grad_op.result(0);
}

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               pir::Value output_shape,
                               bool align_corners) {
  CheckValueDataType(output_grad, "output_grad", "affine_grid_grad");
  paddle::dialect::AffineGridGradOp affine_grid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AffineGridGradOp>(
              input, output_grad, output_shape, align_corners);
  return affine_grid_grad_op.result(0);
}

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               std::vector<pir::Value> output_shape,
                               bool align_corners) {
  CheckValueDataType(output_grad, "output_grad", "affine_grid_grad");
  auto output_shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_shape);
  paddle::dialect::AffineGridGradOp affine_grid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AffineGridGradOp>(
              input, output_grad, output_shape_combine_op.out(), align_corners);
  return affine_grid_grad_op.result(0);
}

pir::OpResult angle_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "angle_grad");
  paddle::dialect::AngleGradOp angle_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AngleGradOp>(
          x, out_grad);
  return angle_grad_op.result(0);
}

pir::OpResult argsort_grad(const pir::Value& indices,
                           const pir::Value& x,
                           const pir::Value& out_grad,
                           int axis,
                           bool descending) {
  CheckValueDataType(out_grad, "out_grad", "argsort_grad");
  paddle::dialect::ArgsortGradOp argsort_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ArgsortGradOp>(
              indices, x, out_grad, axis, descending);
  return argsort_grad_op.result(0);
}

pir::OpResult as_strided_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& stride,
                              int64_t offset) {
  CheckValueDataType(out_grad, "out_grad", "as_strided_grad");
  paddle::dialect::AsStridedGradOp as_strided_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AsStridedGradOp>(
              input, out_grad, dims, stride, offset);
  return as_strided_grad_op.result(0);
}

pir::OpResult asin_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "asin_grad");
  paddle::dialect::AsinGradOp asin_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinGradOp>(
          x, out_grad);
  return asin_grad_op.result(0);
}

pir::OpResult asin_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "asin_grad_");
  paddle::dialect::AsinGrad_Op asin_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinGrad_Op>(
          x, out_grad);
  return asin_grad__op.result(0);
}

pir::OpResult asinh_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "asinh_grad");
  paddle::dialect::AsinhGradOp asinh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinhGradOp>(
          x, out_grad);
  return asinh_grad_op.result(0);
}

pir::OpResult asinh_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "asinh_grad_");
  paddle::dialect::AsinhGrad_Op asinh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AsinhGrad_Op>(
          x, out_grad);
  return asinh_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> atan2_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "atan2_grad");
  paddle::dialect::Atan2GradOp atan2_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Atan2GradOp>(
          x, y, out_grad);
  return std::make_tuple(atan2_grad_op.result(0), atan2_grad_op.result(1));
}

pir::OpResult atan_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "atan_grad");
  paddle::dialect::AtanGradOp atan_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanGradOp>(
          x, out_grad);
  return atan_grad_op.result(0);
}

pir::OpResult atan_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "atan_grad_");
  paddle::dialect::AtanGrad_Op atan_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanGrad_Op>(
          x, out_grad);
  return atan_grad__op.result(0);
}

pir::OpResult atanh_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "atanh_grad");
  paddle::dialect::AtanhGradOp atanh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanhGradOp>(
          x, out_grad);
  return atanh_grad_op.result(0);
}

pir::OpResult atanh_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "atanh_grad_");
  paddle::dialect::AtanhGrad_Op atanh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AtanhGrad_Op>(
          x, out_grad);
  return atanh_grad__op.result(0);
}

pir::OpResult bce_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "bce_loss_grad");
  paddle::dialect::BceLossGradOp bce_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BceLossGradOp>(input, label, out_grad);
  return bce_loss_grad_op.result(0);
}

pir::OpResult bce_loss_grad_(const pir::Value& input,
                             const pir::Value& label,
                             const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "bce_loss_grad_");
  paddle::dialect::BceLossGrad_Op bce_loss_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BceLossGrad_Op>(input, label, out_grad);
  return bce_loss_grad__op.result(0);
}

pir::OpResult bicubic_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(output_grad, "output_grad", "bicubic_interp_grad");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::BicubicInterpGradOp bicubic_interp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BicubicInterpGradOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              output_grad,
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return bicubic_interp_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
bilinear_grad(const pir::Value& x,
              const pir::Value& y,
              const pir::Value& weight,
              const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "bilinear_grad");
  paddle::dialect::BilinearGradOp bilinear_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BilinearGradOp>(x, y, weight, out_grad);
  return std::make_tuple(bilinear_grad_op.result(0),
                         bilinear_grad_op.result(1),
                         bilinear_grad_op.result(2),
                         bilinear_grad_op.result(3));
}

pir::OpResult bilinear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(output_grad, "output_grad", "bilinear_interp_grad");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::BilinearInterpGradOp bilinear_interp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BilinearInterpGradOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              output_grad,
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return bilinear_interp_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> bmm_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "bmm_grad");
  paddle::dialect::BmmGradOp bmm_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BmmGradOp>(
          x, y, out_grad);
  return std::make_tuple(bmm_grad_op.result(0), bmm_grad_op.result(1));
}

std::vector<pir::OpResult> broadcast_tensors_grad(
    const std::vector<pir::Value>& input,
    const std::vector<pir::Value>& out_grad) {
  CheckVectorOfValueDataType(out_grad, "out_grad", "broadcast_tensors_grad");
  auto input_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(input);
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::BroadcastTensorsGradOp broadcast_tensors_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BroadcastTensorsGradOp>(
              input_combine_op.out(), out_grad_combine_op.out());
  auto input_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          broadcast_tensors_grad_op.result(0));
  return input_grad_split_op.outputs();
}

pir::OpResult ceil_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "ceil_grad");
  paddle::dialect::CeilGradOp ceil_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeilGradOp>(
          out_grad);
  return ceil_grad_op.result(0);
}

pir::OpResult ceil_grad_(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "ceil_grad_");
  paddle::dialect::CeilGrad_Op ceil_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeilGrad_Op>(
          out_grad);
  return ceil_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> celu_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "celu_double_grad");
  paddle::dialect::CeluDoubleGradOp celu_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CeluDoubleGradOp>(
              x, grad_out, grad_x_grad, alpha);
  return std::make_tuple(celu_double_grad_op.result(0),
                         celu_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> celu_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "celu_double_grad_");
  paddle::dialect::CeluDoubleGrad_Op celu_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CeluDoubleGrad_Op>(
              x, grad_out, grad_x_grad, alpha);
  return std::make_tuple(celu_double_grad__op.result(0),
                         celu_double_grad__op.result(1));
}

pir::OpResult celu_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float alpha) {
  CheckValueDataType(out_grad, "out_grad", "celu_grad");
  paddle::dialect::CeluGradOp celu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeluGradOp>(
          x, out_grad, alpha);
  return celu_grad_op.result(0);
}

pir::OpResult celu_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float alpha) {
  CheckValueDataType(out_grad, "out_grad", "celu_grad_");
  paddle::dialect::CeluGrad_Op celu_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CeluGrad_Op>(
          x, out_grad, alpha);
  return celu_grad__op.result(0);
}

pir::OpResult cholesky_grad(const pir::Value& out,
                            const pir::Value& out_grad,
                            bool upper) {
  CheckValueDataType(out_grad, "out_grad", "cholesky_grad");
  paddle::dialect::CholeskyGradOp cholesky_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CholeskyGradOp>(out, out_grad, upper);
  return cholesky_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> cholesky_solve_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& out_grad,
    bool upper) {
  CheckValueDataType(out_grad, "out_grad", "cholesky_solve_grad");
  paddle::dialect::CholeskySolveGradOp cholesky_solve_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CholeskySolveGradOp>(
              x, y, out, out_grad, upper);
  return std::make_tuple(cholesky_solve_grad_op.result(0),
                         cholesky_solve_grad_op.result(1));
}

pir::OpResult clip_double_grad(const pir::Value& x,
                               const pir::Value& grad_x_grad,
                               float min,
                               float max) {
  CheckValueDataType(x, "x", "clip_grad");
  paddle::dialect::ClipDoubleGradOp clip_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ClipDoubleGradOp>(x, grad_x_grad, min, max);
  return clip_double_grad_op.result(0);
}

pir::OpResult clip_double_grad(const pir::Value& x,
                               const pir::Value& grad_x_grad,
                               pir::Value min,
                               pir::Value max) {
  CheckValueDataType(x, "x", "clip_grad");
  paddle::dialect::ClipDoubleGradOp clip_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ClipDoubleGradOp>(x, grad_x_grad, min, max);
  return clip_double_grad_op.result(0);
}

pir::OpResult clip_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float min,
                        float max) {
  CheckValueDataType(out_grad, "out_grad", "clip_grad");
  paddle::dialect::ClipGradOp clip_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipGradOp>(
          x, out_grad, min, max);
  return clip_grad_op.result(0);
}

pir::OpResult clip_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value min,
                        pir::Value max) {
  CheckValueDataType(out_grad, "out_grad", "clip_grad");
  paddle::dialect::ClipGradOp clip_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipGradOp>(
          x, out_grad, min, max);
  return clip_grad_op.result(0);
}

pir::OpResult clip_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float min,
                         float max) {
  CheckValueDataType(out_grad, "out_grad", "clip_grad_");
  paddle::dialect::ClipGrad_Op clip_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipGrad_Op>(
          x, out_grad, min, max);
  return clip_grad__op.result(0);
}

pir::OpResult clip_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         pir::Value min,
                         pir::Value max) {
  CheckValueDataType(out_grad, "out_grad", "clip_grad_");
  paddle::dialect::ClipGrad_Op clip_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ClipGrad_Op>(
          x, out_grad, min, max);
  return clip_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> complex_grad(
    const pir::Value& real,
    const pir::Value& imag,
    const pir::Value& out_grad) {
  CheckValueDataType(real, "real", "complex_grad");
  paddle::dialect::ComplexGradOp complex_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ComplexGradOp>(real, imag, out_grad);
  return std::make_tuple(complex_grad_op.result(0), complex_grad_op.result(1));
}

std::vector<pir::OpResult> concat_grad(const std::vector<pir::Value>& x,
                                       const pir::Value& out_grad,
                                       int axis) {
  CheckValueDataType(out_grad, "out_grad", "concat_grad");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::ConcatGradOp concat_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ConcatGradOp>(
          x_combine_op.out(), out_grad, axis);
  auto x_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          concat_grad_op.result(0));
  return x_grad_split_op.outputs();
}

std::vector<pir::OpResult> concat_grad(const std::vector<pir::Value>& x,
                                       const pir::Value& out_grad,
                                       pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "concat_grad");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::ConcatGradOp concat_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ConcatGradOp>(
          x_combine_op.out(), out_grad, axis);
  auto x_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          concat_grad_op.result(0));
  return x_grad_split_op.outputs();
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "conv2d_grad");
  paddle::dialect::Conv2dGradOp conv2d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv2dGradOp>(
          input,
          filter,
          out_grad,
          strides,
          paddings,
          padding_algorithm,
          dilations,
          groups,
          data_format);
  return std::make_tuple(conv2d_grad_op.result(0), conv2d_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> conv2d_grad_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "conv2d_double_grad");
  paddle::optional<pir::Value> optional_grad_input_grad;
  if (!grad_input_grad) {
    optional_grad_input_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_input_grad = grad_input_grad;
  }
  paddle::optional<pir::Value> optional_grad_filter_grad;
  if (!grad_filter_grad) {
    optional_grad_filter_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_filter_grad = grad_filter_grad;
  }
  paddle::dialect::Conv2dGradGradOp conv2d_grad_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dGradGradOp>(
              input,
              filter,
              grad_out,
              optional_grad_input_grad.get(),
              optional_grad_filter_grad.get(),
              strides,
              paddings,
              padding_algorithm,
              dilations,
              groups,
              data_format);
  return std::make_tuple(conv2d_grad_grad_op.result(0),
                         conv2d_grad_grad_op.result(1),
                         conv2d_grad_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> conv3d_double_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "conv3d_double_grad");
  paddle::optional<pir::Value> optional_grad_input_grad;
  if (!grad_input_grad) {
    optional_grad_input_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_input_grad = grad_input_grad;
  }
  paddle::optional<pir::Value> optional_grad_filter_grad;
  if (!grad_filter_grad) {
    optional_grad_filter_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_filter_grad = grad_filter_grad;
  }
  paddle::dialect::Conv3dDoubleGradOp conv3d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv3dDoubleGradOp>(
              input,
              filter,
              grad_out,
              optional_grad_input_grad.get(),
              optional_grad_filter_grad.get(),
              strides,
              paddings,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(conv3d_double_grad_op.result(0),
                         conv3d_double_grad_op.result(1),
                         conv3d_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> conv3d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "conv3d_grad");
  paddle::dialect::Conv3dGradOp conv3d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv3dGradOp>(
          input,
          filter,
          out_grad,
          strides,
          paddings,
          padding_algorithm,
          groups,
          dilations,
          data_format);
  return std::make_tuple(conv3d_grad_op.result(0), conv3d_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> conv3d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "conv3d_transpose_grad");
  paddle::dialect::Conv3dTransposeGradOp conv3d_transpose_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv3dTransposeGradOp>(x,
                                                          filter,
                                                          out_grad,
                                                          strides,
                                                          paddings,
                                                          output_padding,
                                                          output_size,
                                                          padding_algorithm,
                                                          groups,
                                                          dilations,
                                                          data_format);
  return std::make_tuple(conv3d_transpose_grad_op.result(0),
                         conv3d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> cos_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "cos_double_grad");
  paddle::optional<pir::Value> optional_grad_out;
  if (!grad_out) {
    optional_grad_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out = grad_out;
  }
  paddle::dialect::CosDoubleGradOp cos_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CosDoubleGradOp>(
              x, optional_grad_out.get(), grad_x_grad);
  return std::make_tuple(cos_double_grad_op.result(0),
                         cos_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> cos_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "cos_double_grad_");
  paddle::optional<pir::Value> optional_grad_out;
  if (!grad_out) {
    optional_grad_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out = grad_out;
  }
  paddle::dialect::CosDoubleGrad_Op cos_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CosDoubleGrad_Op>(
              x, optional_grad_out.get(), grad_x_grad);
  return std::make_tuple(cos_double_grad__op.result(0),
                         cos_double_grad__op.result(1));
}

pir::OpResult cos_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "cos_grad");
  paddle::dialect::CosGradOp cos_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CosGradOp>(
          x, out_grad);
  return cos_grad_op.result(0);
}

pir::OpResult cos_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "cos_grad_");
  paddle::dialect::CosGrad_Op cos_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CosGrad_Op>(
          x, out_grad);
  return cos_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> cos_triple_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "cos_triple_grad");
  } else {
    CheckValueDataType(grad_x_grad, "grad_x_grad", "cos_triple_grad");
  }
  paddle::optional<pir::Value> optional_grad_out_forward;
  if (!grad_out_forward) {
    optional_grad_out_forward = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_forward = grad_out_forward;
  }
  paddle::optional<pir::Value> optional_grad_x_grad_forward;
  if (!grad_x_grad_forward) {
    optional_grad_x_grad_forward =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad_forward = grad_x_grad_forward;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::CosTripleGradOp cos_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CosTripleGradOp>(
              x,
              optional_grad_out_forward.get(),
              optional_grad_x_grad_forward.get(),
              grad_x_grad,
              optional_grad_out_grad_grad.get());
  return std::make_tuple(cos_triple_grad_op.result(0),
                         cos_triple_grad_op.result(1),
                         cos_triple_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> cos_triple_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "cos_triple_grad_");
  } else {
    CheckValueDataType(grad_x_grad, "grad_x_grad", "cos_triple_grad_");
  }
  paddle::optional<pir::Value> optional_grad_out_forward;
  if (!grad_out_forward) {
    optional_grad_out_forward = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_forward = grad_out_forward;
  }
  paddle::optional<pir::Value> optional_grad_x_grad_forward;
  if (!grad_x_grad_forward) {
    optional_grad_x_grad_forward =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad_forward = grad_x_grad_forward;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::CosTripleGrad_Op cos_triple_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CosTripleGrad_Op>(
              x,
              optional_grad_out_forward.get(),
              optional_grad_x_grad_forward.get(),
              grad_x_grad,
              optional_grad_out_grad_grad.get());
  return std::make_tuple(cos_triple_grad__op.result(0),
                         cos_triple_grad__op.result(1),
                         cos_triple_grad__op.result(2));
}

pir::OpResult cosh_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "cosh_grad");
  paddle::dialect::CoshGradOp cosh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CoshGradOp>(
          x, out_grad);
  return cosh_grad_op.result(0);
}

pir::OpResult cosh_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "cosh_grad_");
  paddle::dialect::CoshGrad_Op cosh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CoshGrad_Op>(
          x, out_grad);
  return cosh_grad__op.result(0);
}

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& offsets) {
  CheckValueDataType(x, "x", "crop_grad");
  paddle::dialect::CropGradOp crop_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropGradOp>(
          x, out_grad, offsets);
  return crop_grad_op.result(0);
}

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value offsets) {
  CheckValueDataType(x, "x", "crop_grad");
  paddle::dialect::CropGradOp crop_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropGradOp>(
          x, out_grad, offsets);
  return crop_grad_op.result(0);
}

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> offsets) {
  CheckValueDataType(x, "x", "crop_grad");
  auto offsets_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(offsets);
  paddle::dialect::CropGradOp crop_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CropGradOp>(
          x, out_grad, offsets_combine_op.out());
  return crop_grad_op.result(0);
}

pir::OpResult cross_entropy_with_softmax_grad(const pir::Value& label,
                                              const pir::Value& softmax,
                                              const pir::Value& loss_grad,
                                              bool soft_label,
                                              bool use_softmax,
                                              bool numeric_stable_mode,
                                              int ignore_index,
                                              int axis) {
  CheckValueDataType(loss_grad, "loss_grad", "cross_entropy_with_softmax_grad");
  paddle::dialect::CrossEntropyWithSoftmaxGradOp
      cross_entropy_with_softmax_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::CrossEntropyWithSoftmaxGradOp>(
                  label,
                  softmax,
                  loss_grad,
                  soft_label,
                  use_softmax,
                  numeric_stable_mode,
                  ignore_index,
                  axis);
  return cross_entropy_with_softmax_grad_op.result(0);
}

pir::OpResult cross_entropy_with_softmax_grad_(const pir::Value& label,
                                               const pir::Value& softmax,
                                               const pir::Value& loss_grad,
                                               bool soft_label,
                                               bool use_softmax,
                                               bool numeric_stable_mode,
                                               int ignore_index,
                                               int axis) {
  CheckValueDataType(
      loss_grad, "loss_grad", "cross_entropy_with_softmax_grad_");
  paddle::dialect::CrossEntropyWithSoftmaxGrad_Op
      cross_entropy_with_softmax_grad__op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::CrossEntropyWithSoftmaxGrad_Op>(
                  label,
                  softmax,
                  loss_grad,
                  soft_label,
                  use_softmax,
                  numeric_stable_mode,
                  ignore_index,
                  axis);
  return cross_entropy_with_softmax_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> cross_grad(const pir::Value& x,
                                                    const pir::Value& y,
                                                    const pir::Value& out_grad,
                                                    int axis) {
  CheckValueDataType(out_grad, "out_grad", "cross_grad");
  paddle::dialect::CrossGradOp cross_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CrossGradOp>(
          x, y, out_grad, axis);
  return std::make_tuple(cross_grad_op.result(0), cross_grad_op.result(1));
}

pir::OpResult cummax_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out_grad,
                          int axis,
                          phi::DataType dtype) {
  CheckValueDataType(out_grad, "out_grad", "cummax_grad");
  paddle::dialect::CummaxGradOp cummax_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CummaxGradOp>(
          x, indices, out_grad, axis, dtype);
  return cummax_grad_op.result(0);
}

pir::OpResult cummin_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out_grad,
                          int axis,
                          phi::DataType dtype) {
  CheckValueDataType(out_grad, "out_grad", "cummin_grad");
  paddle::dialect::CumminGradOp cummin_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumminGradOp>(
          x, indices, out_grad, axis, dtype);
  return cummin_grad_op.result(0);
}

pir::OpResult cumprod_grad(const pir::Value& x,
                           const pir::Value& out,
                           const pir::Value& out_grad,
                           int dim) {
  CheckValueDataType(out_grad, "out_grad", "cumprod_grad");
  paddle::dialect::CumprodGradOp cumprod_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CumprodGradOp>(x, out, out_grad, dim);
  return cumprod_grad_op.result(0);
}

pir::OpResult cumsum_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          int axis,
                          bool flatten,
                          bool exclusive,
                          bool reverse) {
  CheckValueDataType(x, "x", "cumsum_grad");
  paddle::dialect::CumsumGradOp cumsum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumsumGradOp>(
          x, out_grad, axis, flatten, exclusive, reverse);
  return cumsum_grad_op.result(0);
}

pir::OpResult cumsum_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          pir::Value axis,
                          bool flatten,
                          bool exclusive,
                          bool reverse) {
  CheckValueDataType(x, "x", "cumsum_grad");
  paddle::dialect::CumsumGradOp cumsum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CumsumGradOp>(
          x, out_grad, axis, flatten, exclusive, reverse);
  return cumsum_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
depthwise_conv2d_double_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "depthwise_conv2d_double_grad");
  paddle::optional<pir::Value> optional_grad_input_grad;
  if (!grad_input_grad) {
    optional_grad_input_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_input_grad = grad_input_grad;
  }
  paddle::optional<pir::Value> optional_grad_filter_grad;
  if (!grad_filter_grad) {
    optional_grad_filter_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_filter_grad = grad_filter_grad;
  }
  paddle::dialect::DepthwiseConv2dDoubleGradOp depthwise_conv2d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dDoubleGradOp>(
              input,
              filter,
              grad_out,
              optional_grad_input_grad.get(),
              optional_grad_filter_grad.get(),
              strides,
              paddings,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(depthwise_conv2d_double_grad_op.result(0),
                         depthwise_conv2d_double_grad_op.result(1),
                         depthwise_conv2d_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(input, "input", "depthwise_conv2d_grad");
  paddle::dialect::DepthwiseConv2dGradOp depthwise_conv2d_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dGradOp>(input,
                                                          filter,
                                                          out_grad,
                                                          strides,
                                                          paddings,
                                                          padding_algorithm,
                                                          groups,
                                                          dilations,
                                                          data_format);
  return std::make_tuple(depthwise_conv2d_grad_op.result(0),
                         depthwise_conv2d_grad_op.result(1));
}

pir::OpResult det_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "determinant_grad");
  paddle::dialect::DetGradOp det_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DetGradOp>(
          x, out, out_grad);
  return det_grad_op.result(0);
}

pir::OpResult diag_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        int offset) {
  CheckValueDataType(out_grad, "out_grad", "diag_grad");
  paddle::dialect::DiagGradOp diag_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DiagGradOp>(
          x, out_grad, offset);
  return diag_grad_op.result(0);
}

pir::OpResult diagonal_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            int offset,
                            int axis1,
                            int axis2) {
  CheckValueDataType(out_grad, "out_grad", "diagonal_grad");
  paddle::dialect::DiagonalGradOp diagonal_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DiagonalGradOp>(
              x, out_grad, offset, axis1, axis2);
  return diagonal_grad_op.result(0);
}

pir::OpResult digamma_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "digamma_grad");
  paddle::dialect::DigammaGradOp digamma_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DigammaGradOp>(x, out_grad);
  return digamma_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> dist_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out,
                                                   const pir::Value& out_grad,
                                                   float p) {
  CheckValueDataType(out_grad, "out_grad", "dist_grad");
  paddle::dialect::DistGradOp dist_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DistGradOp>(
          x, y, out, out_grad, p);
  return std::make_tuple(dist_grad_op.result(0), dist_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> dot_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "dot_grad");
  paddle::dialect::DotGradOp dot_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DotGradOp>(
          x, y, out_grad);
  return std::make_tuple(dot_grad_op.result(0), dot_grad_op.result(1));
}

pir::OpResult eig_grad(const pir::Value& out_w,
                       const pir::Value& out_v,
                       const pir::Value& out_w_grad,
                       const pir::Value& out_v_grad) {
  CheckValueDataType(out_v, "out_v", "eig_grad");
  paddle::dialect::EigGradOp eig_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EigGradOp>(
          out_w, out_v, out_w_grad, out_v_grad);
  return eig_grad_op.result(0);
}

pir::OpResult eigh_grad(const pir::Value& out_w,
                        const pir::Value& out_v,
                        const pir::Value& out_w_grad,
                        const pir::Value& out_v_grad) {
  CheckValueDataType(out_v, "out_v", "eigh_grad");
  paddle::dialect::EighGradOp eigh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EighGradOp>(
          out_w, out_v, out_w_grad, out_v_grad);
  return eigh_grad_op.result(0);
}

pir::OpResult eigvalsh_grad(const pir::Value& eigenvectors,
                            const pir::Value& eigenvalues_grad,
                            const std::string& uplo,
                            bool is_test) {
  CheckValueDataType(eigenvectors, "eigenvectors", "eigvalsh_grad");
  paddle::dialect::EigvalshGradOp eigvalsh_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::EigvalshGradOp>(
              eigenvectors, eigenvalues_grad, uplo, is_test);
  return eigvalsh_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> elu_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "elu_double_grad");
  paddle::dialect::EluDoubleGradOp elu_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::EluDoubleGradOp>(
              x, grad_out, grad_x_grad, alpha);
  return std::make_tuple(elu_double_grad_op.result(0),
                         elu_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> elu_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "elu_double_grad_");
  paddle::dialect::EluDoubleGrad_Op elu_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::EluDoubleGrad_Op>(
              x, grad_out, grad_x_grad, alpha);
  return std::make_tuple(elu_double_grad__op.result(0),
                         elu_double_grad__op.result(1));
}

pir::OpResult elu_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       float alpha) {
  CheckValueDataType(out_grad, "out_grad", "elu_grad");
  paddle::dialect::EluGradOp elu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EluGradOp>(
          x, out, out_grad, alpha);
  return elu_grad_op.result(0);
}

pir::OpResult elu_grad_(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        float alpha) {
  CheckValueDataType(out_grad, "out_grad", "elu_grad_");
  paddle::dialect::EluGrad_Op elu_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EluGrad_Op>(
          x, out, out_grad, alpha);
  return elu_grad__op.result(0);
}

pir::OpResult erf_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "erf_grad");
  paddle::dialect::ErfGradOp erf_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ErfGradOp>(
          x, out_grad);
  return erf_grad_op.result(0);
}

pir::OpResult erfinv_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "erfinv_grad");
  paddle::dialect::ErfinvGradOp erfinv_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ErfinvGradOp>(
          out, out_grad);
  return erfinv_grad_op.result(0);
}

pir::OpResult exp_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "exp_grad");
  paddle::dialect::ExpGradOp exp_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpGradOp>(
          out, out_grad);
  return exp_grad_op.result(0);
}

pir::OpResult exp_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "exp_grad_");
  paddle::dialect::ExpGrad_Op exp_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpGrad_Op>(
          out, out_grad);
  return exp_grad__op.result(0);
}

pir::OpResult expand_as_grad(const pir::Value& x,
                             const pir::Value& out_grad,
                             const std::vector<int>& target_shape) {
  CheckValueDataType(out_grad, "out_grad", "expand_as_grad");
  paddle::dialect::ExpandAsGradOp expand_as_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ExpandAsGradOp>(x, out_grad, target_shape);
  return expand_as_grad_op.result(0);
}

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          const std::vector<int64_t>& shape) {
  CheckValueDataType(out_grad, "out_grad", "expand_grad");
  paddle::dialect::ExpandGradOp expand_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandGradOp>(
          x, out_grad, shape);
  return expand_grad_op.result(0);
}

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          pir::Value shape) {
  CheckValueDataType(out_grad, "out_grad", "expand_grad");
  paddle::dialect::ExpandGradOp expand_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandGradOp>(
          x, out_grad, shape);
  return expand_grad_op.result(0);
}

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> shape) {
  CheckValueDataType(out_grad, "out_grad", "expand_grad");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::ExpandGradOp expand_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ExpandGradOp>(
          x, out_grad, shape_combine_op.out());
  return expand_grad_op.result(0);
}

pir::OpResult expm1_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "expm1_grad");
  paddle::dialect::Expm1GradOp expm1_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Expm1GradOp>(
          out, out_grad);
  return expm1_grad_op.result(0);
}

pir::OpResult expm1_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "expm1_grad_");
  paddle::dialect::Expm1Grad_Op expm1_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Expm1Grad_Op>(
          out, out_grad);
  return expm1_grad__op.result(0);
}

pir::OpResult fft_c2c_grad(const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward) {
  CheckValueDataType(out_grad, "out_grad", "fft_c2c_grad");
  paddle::dialect::FftC2cGradOp fft_c2c_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftC2cGradOp>(
          out_grad, axes, normalization, forward);
  return fft_c2c_grad_op.result(0);
}

pir::OpResult fft_c2r_grad(const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward,
                           int64_t last_dim_size) {
  CheckValueDataType(out_grad, "out_grad", "fft_c2r_grad");
  paddle::dialect::FftC2rGradOp fft_c2r_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftC2rGradOp>(
          out_grad, axes, normalization, forward, last_dim_size);
  return fft_c2r_grad_op.result(0);
}

pir::OpResult fft_r2c_grad(const pir::Value& x,
                           const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward,
                           bool onesided) {
  CheckValueDataType(out_grad, "out_grad", "fft_r2c_grad");
  paddle::dialect::FftR2cGradOp fft_r2c_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FftR2cGradOp>(
          x, out_grad, axes, normalization, forward, onesided);
  return fft_r2c_grad_op.result(0);
}

pir::OpResult fill_diagonal_grad(const pir::Value& out_grad,
                                 float value,
                                 int offset,
                                 bool wrap) {
  CheckValueDataType(out_grad, "out_grad", "fill_diagonal_grad");
  paddle::dialect::FillDiagonalGradOp fill_diagonal_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalGradOp>(
              out_grad, value, offset, wrap);
  return fill_diagonal_grad_op.result(0);
}

pir::OpResult fill_diagonal_tensor_grad(const pir::Value& out_grad,
                                        int64_t offset,
                                        int dim1,
                                        int dim2) {
  CheckValueDataType(out_grad, "out_grad", "fill_diagonal_tensor_grad");
  paddle::dialect::FillDiagonalTensorGradOp fill_diagonal_tensor_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalTensorGradOp>(
              out_grad, offset, dim1, dim2);
  return fill_diagonal_tensor_grad_op.result(0);
}

pir::OpResult fill_diagonal_tensor_grad_(const pir::Value& out_grad,
                                         int64_t offset,
                                         int dim1,
                                         int dim2) {
  CheckValueDataType(out_grad, "out_grad", "fill_diagonal_tensor_grad_");
  paddle::dialect::FillDiagonalTensorGrad_Op fill_diagonal_tensor_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FillDiagonalTensorGrad_Op>(
              out_grad, offset, dim1, dim2);
  return fill_diagonal_tensor_grad__op.result(0);
}

pir::OpResult fill_grad(const pir::Value& out_grad, float value) {
  CheckValueDataType(out_grad, "out_grad", "fill_grad");
  paddle::dialect::FillGradOp fill_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillGradOp>(
          out_grad, value);
  return fill_grad_op.result(0);
}

pir::OpResult fill_grad(const pir::Value& out_grad, pir::Value value) {
  CheckValueDataType(out_grad, "out_grad", "fill_grad");
  paddle::dialect::FillGradOp fill_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillGradOp>(
          out_grad, value);
  return fill_grad_op.result(0);
}

pir::OpResult fill_grad_(const pir::Value& out_grad, float value) {
  CheckValueDataType(out_grad, "out_grad", "fill_grad_");
  paddle::dialect::FillGrad_Op fill_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillGrad_Op>(
          out_grad, value);
  return fill_grad__op.result(0);
}

pir::OpResult fill_grad_(const pir::Value& out_grad, pir::Value value) {
  CheckValueDataType(out_grad, "out_grad", "fill_grad_");
  paddle::dialect::FillGrad_Op fill_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FillGrad_Op>(
          out_grad, value);
  return fill_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> flash_attn_grad(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const pir::Value& out,
    const pir::Value& softmax_lse,
    const pir::Value& seed_offset,
    const paddle::optional<pir::Value>& attn_mask,
    const pir::Value& out_grad,
    float dropout,
    bool causal) {
  CheckValueDataType(q, "q", "flash_attn_grad");
  paddle::optional<pir::Value> optional_attn_mask;
  if (!attn_mask) {
    optional_attn_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_attn_mask = attn_mask;
  }
  paddle::dialect::FlashAttnGradOp flash_attn_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FlashAttnGradOp>(q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse,
                                                    seed_offset,
                                                    optional_attn_mask.get(),
                                                    out_grad,
                                                    dropout,
                                                    causal);
  return std::make_tuple(flash_attn_grad_op.result(0),
                         flash_attn_grad_op.result(1),
                         flash_attn_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
flash_attn_unpadded_grad(const pir::Value& q,
                         const pir::Value& k,
                         const pir::Value& v,
                         const pir::Value& cu_seqlens_q,
                         const pir::Value& cu_seqlens_k,
                         const pir::Value& out,
                         const pir::Value& softmax_lse,
                         const pir::Value& seed_offset,
                         const paddle::optional<pir::Value>& attn_mask,
                         const pir::Value& out_grad,
                         int64_t max_seqlen_q,
                         int64_t max_seqlen_k,
                         float scale,
                         float dropout,
                         bool causal) {
  CheckValueDataType(q, "q", "flash_attn_unpadded_grad");
  paddle::optional<pir::Value> optional_attn_mask;
  if (!attn_mask) {
    optional_attn_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_attn_mask = attn_mask;
  }
  paddle::dialect::FlashAttnUnpaddedGradOp flash_attn_unpadded_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FlashAttnUnpaddedGradOp>(
              q,
              k,
              v,
              cu_seqlens_q,
              cu_seqlens_k,
              out,
              softmax_lse,
              seed_offset,
              optional_attn_mask.get(),
              out_grad,
              max_seqlen_q,
              max_seqlen_k,
              scale,
              dropout,
              causal);
  return std::make_tuple(flash_attn_unpadded_grad_op.result(0),
                         flash_attn_unpadded_grad_op.result(1),
                         flash_attn_unpadded_grad_op.result(2));
}

pir::OpResult flatten_grad(const pir::Value& xshape,
                           const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "flatten_grad");
  paddle::dialect::FlattenGradOp flatten_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FlattenGradOp>(xshape, out_grad);
  return flatten_grad_op.result(0);
}

pir::OpResult flatten_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "flatten_grad_");
  paddle::dialect::FlattenGrad_Op flatten_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FlattenGrad_Op>(xshape, out_grad);
  return flatten_grad__op.result(0);
}

pir::OpResult floor_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "floor_grad");
  paddle::dialect::FloorGradOp floor_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FloorGradOp>(
          out_grad);
  return floor_grad_op.result(0);
}

pir::OpResult floor_grad_(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "floor_grad_");
  paddle::dialect::FloorGrad_Op floor_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FloorGrad_Op>(
          out_grad);
  return floor_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> fmax_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "fmax_grad");
  paddle::dialect::FmaxGradOp fmax_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FmaxGradOp>(
          x, y, out_grad);
  return std::make_tuple(fmax_grad_op.result(0), fmax_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> fmin_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "fmin_grad");
  paddle::dialect::FminGradOp fmin_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FminGradOp>(
          x, y, out_grad);
  return std::make_tuple(fmin_grad_op.result(0), fmin_grad_op.result(1));
}

pir::OpResult fold_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int>& output_sizes,
                        const std::vector<int>& kernel_sizes,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations) {
  CheckValueDataType(out_grad, "out_grad", "fold_grad");
  paddle::dialect::FoldGradOp fold_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FoldGradOp>(
          x,
          out_grad,
          output_sizes,
          kernel_sizes,
          strides,
          paddings,
          dilations);
  return fold_grad_op.result(0);
}

pir::OpResult frame_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         int frame_length,
                         int hop_length,
                         int axis) {
  CheckValueDataType(out_grad, "out_grad", "frame_grad");
  paddle::dialect::FrameGradOp frame_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FrameGradOp>(
          x, out_grad, frame_length, hop_length, axis);
  return frame_grad_op.result(0);
}

pir::OpResult gammaln_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "gammaln_grad");
  paddle::dialect::GammalnGradOp gammaln_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GammalnGradOp>(x, out_grad);
  return gammaln_grad_op.result(0);
}

pir::OpResult gather_grad(const pir::Value& x,
                          const pir::Value& index,
                          const pir::Value& out_grad,
                          int axis) {
  CheckValueDataType(out_grad, "out_grad", "gather_grad");
  paddle::dialect::GatherGradOp gather_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherGradOp>(
          x, index, out_grad, axis);
  return gather_grad_op.result(0);
}

pir::OpResult gather_grad(const pir::Value& x,
                          const pir::Value& index,
                          const pir::Value& out_grad,
                          pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "gather_grad");
  paddle::dialect::GatherGradOp gather_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GatherGradOp>(
          x, index, out_grad, axis);
  return gather_grad_op.result(0);
}

pir::OpResult gather_nd_grad(const pir::Value& x,
                             const pir::Value& index,
                             const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "gather_nd_grad");
  paddle::dialect::GatherNdGradOp gather_nd_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GatherNdGradOp>(x, index, out_grad);
  return gather_nd_grad_op.result(0);
}

pir::OpResult gaussian_inplace_grad(const pir::Value& out_grad,
                                    float mean,
                                    float std,
                                    int seed) {
  CheckValueDataType(out_grad, "out_grad", "gaussian_inplace_grad");
  paddle::dialect::GaussianInplaceGradOp gaussian_inplace_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GaussianInplaceGradOp>(
              out_grad, mean, std, seed);
  return gaussian_inplace_grad_op.result(0);
}

pir::OpResult gaussian_inplace_grad_(const pir::Value& out_grad,
                                     float mean,
                                     float std,
                                     int seed) {
  CheckValueDataType(out_grad, "out_grad", "gaussian_inplace_grad_");
  paddle::dialect::GaussianInplaceGrad_Op gaussian_inplace_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GaussianInplaceGrad_Op>(
              out_grad, mean, std, seed);
  return gaussian_inplace_grad__op.result(0);
}

pir::OpResult gelu_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        bool approximate) {
  CheckValueDataType(out_grad, "out_grad", "gelu_grad");
  paddle::dialect::GeluGradOp gelu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GeluGradOp>(
          x, out_grad, approximate);
  return gelu_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> grid_sample_grad(
    const pir::Value& x,
    const pir::Value& grid,
    const pir::Value& out_grad,
    const std::string& mode,
    const std::string& padding_mode,
    bool align_corners) {
  CheckValueDataType(x, "x", "grid_sample_grad");
  paddle::dialect::GridSampleGradOp grid_sample_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GridSampleGradOp>(
              x, grid, out_grad, mode, padding_mode, align_corners);
  return std::make_tuple(grid_sample_grad_op.result(0),
                         grid_sample_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> group_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& y,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& y_grad,
    float epsilon,
    int groups,
    const std::string& data_layout) {
  CheckValueDataType(y_grad, "y_grad", "group_norm_grad");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::GroupNormGradOp group_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GroupNormGradOp>(x,
                                                    optional_scale.get(),
                                                    optional_bias.get(),
                                                    y,
                                                    mean,
                                                    variance,
                                                    y_grad,
                                                    epsilon,
                                                    groups,
                                                    data_layout);
  return std::make_tuple(group_norm_grad_op.result(0),
                         group_norm_grad_op.result(1),
                         group_norm_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> group_norm_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& y,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& y_grad,
    float epsilon,
    int groups,
    const std::string& data_layout) {
  CheckValueDataType(y_grad, "y_grad", "group_norm_grad_");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::GroupNormGrad_Op group_norm_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GroupNormGrad_Op>(x,
                                                     optional_scale.get(),
                                                     optional_bias.get(),
                                                     y,
                                                     mean,
                                                     variance,
                                                     y_grad,
                                                     epsilon,
                                                     groups,
                                                     data_layout);
  return std::make_tuple(group_norm_grad__op.result(0),
                         group_norm_grad__op.result(1),
                         group_norm_grad__op.result(2));
}

pir::OpResult gumbel_softmax_grad(const pir::Value& out,
                                  const pir::Value& out_grad,
                                  int axis) {
  CheckValueDataType(out_grad, "out_grad", "gumbel_softmax_grad");
  paddle::dialect::GumbelSoftmaxGradOp gumbel_softmax_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GumbelSoftmaxGradOp>(out, out_grad, axis);
  return gumbel_softmax_grad_op.result(0);
}

pir::OpResult hardshrink_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float threshold) {
  CheckValueDataType(out_grad, "out_grad", "hard_shrink_grad");
  paddle::dialect::HardshrinkGradOp hardshrink_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardshrinkGradOp>(x, out_grad, threshold);
  return hardshrink_grad_op.result(0);
}

pir::OpResult hardshrink_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float threshold) {
  CheckValueDataType(out_grad, "out_grad", "hard_shrink_grad_");
  paddle::dialect::HardshrinkGrad_Op hardshrink_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardshrinkGrad_Op>(x, out_grad, threshold);
  return hardshrink_grad__op.result(0);
}

pir::OpResult hardsigmoid_grad(const pir::Value& out,
                               const pir::Value& out_grad,
                               float slope,
                               float offset) {
  CheckValueDataType(out_grad, "out_grad", "hardsigmoid_grad");
  paddle::dialect::HardsigmoidGradOp hardsigmoid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardsigmoidGradOp>(
              out, out_grad, slope, offset);
  return hardsigmoid_grad_op.result(0);
}

pir::OpResult hardsigmoid_grad_(const pir::Value& out,
                                const pir::Value& out_grad,
                                float slope,
                                float offset) {
  CheckValueDataType(out_grad, "out_grad", "hardsigmoid_grad_");
  paddle::dialect::HardsigmoidGrad_Op hardsigmoid_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardsigmoidGrad_Op>(
              out, out_grad, slope, offset);
  return hardsigmoid_grad__op.result(0);
}

pir::OpResult hardtanh_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            float t_min,
                            float t_max) {
  CheckValueDataType(out_grad, "out_grad", "hardtanh_grad");
  paddle::dialect::HardtanhGradOp hardtanh_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardtanhGradOp>(x, out_grad, t_min, t_max);
  return hardtanh_grad_op.result(0);
}

pir::OpResult hardtanh_grad_(const pir::Value& x,
                             const pir::Value& out_grad,
                             float t_min,
                             float t_max) {
  CheckValueDataType(out_grad, "out_grad", "hardtanh_grad_");
  paddle::dialect::HardtanhGrad_Op hardtanh_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardtanhGrad_Op>(x, out_grad, t_min, t_max);
  return hardtanh_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> heaviside_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "heaviside_grad");
  paddle::dialect::HeavisideGradOp heaviside_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HeavisideGradOp>(x, y, out_grad);
  return std::make_tuple(heaviside_grad_op.result(0),
                         heaviside_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> huber_loss_grad(
    const pir::Value& residual, const pir::Value& out_grad, float delta) {
  CheckValueDataType(out_grad, "out_grad", "huber_loss_grad");
  paddle::dialect::HuberLossGradOp huber_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HuberLossGradOp>(residual, out_grad, delta);
  return std::make_tuple(huber_loss_grad_op.result(0),
                         huber_loss_grad_op.result(1));
}

pir::OpResult i0_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "i0_grad");
  paddle::dialect::I0GradOp i0_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I0GradOp>(
          x, out_grad);
  return i0_grad_op.result(0);
}

pir::OpResult i0e_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "i0e_grad");
  paddle::dialect::I0eGradOp i0e_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I0eGradOp>(
          x, out, out_grad);
  return i0e_grad_op.result(0);
}

pir::OpResult i1_grad(const pir::Value& x,
                      const pir::Value& out,
                      const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "i1_grad");
  paddle::dialect::I1GradOp i1_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I1GradOp>(
          x, out, out_grad);
  return i1_grad_op.result(0);
}

pir::OpResult i1e_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "i1e_grad");
  paddle::dialect::I1eGradOp i1e_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::I1eGradOp>(
          x, out, out_grad);
  return i1e_grad_op.result(0);
}

pir::OpResult identity_loss_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 int reduction) {
  CheckValueDataType(out_grad, "out_grad", "identity_loss_grad");
  paddle::dialect::IdentityLossGradOp identity_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IdentityLossGradOp>(x, out_grad, reduction);
  return identity_loss_grad_op.result(0);
}

pir::OpResult identity_loss_grad_(const pir::Value& x,
                                  const pir::Value& out_grad,
                                  int reduction) {
  CheckValueDataType(out_grad, "out_grad", "identity_loss_grad_");
  paddle::dialect::IdentityLossGrad_Op identity_loss_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IdentityLossGrad_Op>(x, out_grad, reduction);
  return identity_loss_grad__op.result(0);
}

pir::OpResult imag_grad(const pir::Value& out_grad) {
  paddle::dialect::ImagGradOp imag_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ImagGradOp>(
          out_grad);
  return imag_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> index_add_grad(
    const pir::Value& index,
    const pir::Value& add_value,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(out_grad, "out_grad", "index_add_grad");
  paddle::dialect::IndexAddGradOp index_add_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexAddGradOp>(
              index, add_value, out_grad, axis);
  return std::make_tuple(index_add_grad_op.result(0),
                         index_add_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> index_add_grad_(
    const pir::Value& index,
    const pir::Value& add_value,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(out_grad, "out_grad", "index_add_grad_");
  paddle::dialect::IndexAddGrad_Op index_add_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexAddGrad_Op>(
              index, add_value, out_grad, axis);
  return std::make_tuple(index_add_grad__op.result(0),
                         index_add_grad__op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> index_put_grad(
    const pir::Value& x,
    const std::vector<pir::Value>& indices,
    const pir::Value& value,
    const pir::Value& out_grad,
    bool accumulate) {
  CheckValueDataType(out_grad, "out_grad", "index_put_grad");
  auto indices_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(indices);
  paddle::dialect::IndexPutGradOp index_put_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexPutGradOp>(
              x, indices_combine_op.out(), value, out_grad, accumulate);
  return std::make_tuple(index_put_grad_op.result(0),
                         index_put_grad_op.result(1));
}

pir::OpResult index_sample_grad(const pir::Value& x,
                                const pir::Value& index,
                                const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "index_sample_grad");
  paddle::dialect::IndexSampleGradOp index_sample_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSampleGradOp>(x, index, out_grad);
  return index_sample_grad_op.result(0);
}

pir::OpResult index_select_grad(const pir::Value& x,
                                const pir::Value& index,
                                const pir::Value& out_grad,
                                int axis) {
  CheckValueDataType(out_grad, "out_grad", "index_select_grad");
  paddle::dialect::IndexSelectGradOp index_select_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSelectGradOp>(x, index, out_grad, axis);
  return index_select_grad_op.result(0);
}

pir::OpResult index_select_strided_grad(const pir::Value& x,
                                        const pir::Value& out_grad,
                                        int64_t index,
                                        int axis) {
  CheckValueDataType(out_grad, "out_grad", "index_select_strided_grad");
  paddle::dialect::IndexSelectStridedGradOp index_select_strided_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::IndexSelectStridedGradOp>(
              x, out_grad, index, axis);
  return index_select_strided_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
instance_norm_double_grad(const pir::Value& x,
                          const paddle::optional<pir::Value>& fwd_scale,
                          const pir::Value& saved_mean,
                          const pir::Value& saved_variance,
                          const pir::Value& grad_y,
                          const paddle::optional<pir::Value>& grad_x_grad,
                          const paddle::optional<pir::Value>& grad_scale_grad,
                          const paddle::optional<pir::Value>& grad_bias_grad,
                          float epsilon) {
  CheckValueDataType(x, "x", "instance_norm_double_grad");
  paddle::optional<pir::Value> optional_fwd_scale;
  if (!fwd_scale) {
    optional_fwd_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fwd_scale = fwd_scale;
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_scale_grad;
  if (!grad_scale_grad) {
    optional_grad_scale_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_scale_grad = grad_scale_grad;
  }
  paddle::optional<pir::Value> optional_grad_bias_grad;
  if (!grad_bias_grad) {
    optional_grad_bias_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_bias_grad = grad_bias_grad;
  }
  paddle::dialect::InstanceNormDoubleGradOp instance_norm_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::InstanceNormDoubleGradOp>(
              x,
              optional_fwd_scale.get(),
              saved_mean,
              saved_variance,
              grad_y,
              optional_grad_x_grad.get(),
              optional_grad_scale_grad.get(),
              optional_grad_bias_grad.get(),
              epsilon);
  return std::make_tuple(instance_norm_double_grad_op.result(0),
                         instance_norm_double_grad_op.result(1),
                         instance_norm_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> instance_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& y_grad,
    float epsilon) {
  CheckValueDataType(x, "x", "instance_norm_grad");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::dialect::InstanceNormGradOp instance_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::InstanceNormGradOp>(x,
                                                       optional_scale.get(),
                                                       saved_mean,
                                                       saved_variance,
                                                       y_grad,
                                                       epsilon);
  return std::make_tuple(instance_norm_grad_op.result(0),
                         instance_norm_grad_op.result(1),
                         instance_norm_grad_op.result(2));
}

pir::OpResult inverse_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "inverse_grad");
  paddle::dialect::InverseGradOp inverse_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::InverseGradOp>(out, out_grad);
  return inverse_grad_op.result(0);
}

pir::OpResult kldiv_loss_grad(const pir::Value& x,
                              const pir::Value& label,
                              const pir::Value& out_grad,
                              const std::string& reduction) {
  CheckValueDataType(out_grad, "out_grad", "kldiv_loss_grad");
  paddle::dialect::KldivLossGradOp kldiv_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::KldivLossGradOp>(
              x, label, out_grad, reduction);
  return kldiv_loss_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> kron_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "kron_grad");
  paddle::dialect::KronGradOp kron_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::KronGradOp>(
          x, y, out_grad);
  return std::make_tuple(kron_grad_op.result(0), kron_grad_op.result(1));
}

pir::OpResult kthvalue_grad(const pir::Value& x,
                            const pir::Value& indices,
                            const pir::Value& out_grad,
                            int k,
                            int axis,
                            bool keepdim) {
  CheckValueDataType(out_grad, "out_grad", "kthvalue_grad");
  paddle::dialect::KthvalueGradOp kthvalue_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::KthvalueGradOp>(
              x, indices, out_grad, k, axis, keepdim);
  return kthvalue_grad_op.result(0);
}

pir::OpResult label_smooth_grad(const pir::Value& out_grad, float epsilon) {
  CheckValueDataType(out_grad, "out_grad", "label_smooth_grad");
  paddle::dialect::LabelSmoothGradOp label_smooth_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LabelSmoothGradOp>(out_grad, epsilon);
  return label_smooth_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> layer_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& out_grad,
    float epsilon,
    int begin_norm_axis) {
  CheckValueDataType(x, "x", "layer_norm_grad");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::LayerNormGradOp layer_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LayerNormGradOp>(x,
                                                    optional_scale.get(),
                                                    optional_bias.get(),
                                                    mean,
                                                    variance,
                                                    out_grad,
                                                    epsilon,
                                                    begin_norm_axis);
  return std::make_tuple(layer_norm_grad_op.result(0),
                         layer_norm_grad_op.result(1),
                         layer_norm_grad_op.result(2));
}

pir::OpResult leaky_relu_double_grad(const pir::Value& x,
                                     const pir::Value& grad_x_grad,
                                     float negative_slope) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "leaky_relu_double_grad");
  paddle::dialect::LeakyReluDoubleGradOp leaky_relu_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LeakyReluDoubleGradOp>(
              x, grad_x_grad, negative_slope);
  return leaky_relu_double_grad_op.result(0);
}

pir::OpResult leaky_relu_double_grad_(const pir::Value& x,
                                      const pir::Value& grad_x_grad,
                                      float negative_slope) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "leaky_relu_double_grad_");
  paddle::dialect::LeakyReluDoubleGrad_Op leaky_relu_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LeakyReluDoubleGrad_Op>(
              x, grad_x_grad, negative_slope);
  return leaky_relu_double_grad__op.result(0);
}

pir::OpResult leaky_relu_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float negative_slope) {
  CheckValueDataType(out_grad, "out_grad", "leaky_relu_grad");
  paddle::dialect::LeakyReluGradOp leaky_relu_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LeakyReluGradOp>(
              x, out_grad, negative_slope);
  return leaky_relu_grad_op.result(0);
}

pir::OpResult leaky_relu_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float negative_slope) {
  CheckValueDataType(out_grad, "out_grad", "leaky_relu_grad_");
  paddle::dialect::LeakyReluGrad_Op leaky_relu_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LeakyReluGrad_Op>(
              x, out_grad, negative_slope);
  return leaky_relu_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> lerp_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& weight,
                                                   const pir::Value& out,
                                                   const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "lerp_grad");
  paddle::dialect::LerpGradOp lerp_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LerpGradOp>(
          x, y, weight, out, out_grad);
  return std::make_tuple(lerp_grad_op.result(0), lerp_grad_op.result(1));
}

pir::OpResult lgamma_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "lgamma_grad");
  paddle::dialect::LgammaGradOp lgamma_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LgammaGradOp>(
          x, out_grad);
  return lgamma_grad_op.result(0);
}

pir::OpResult linear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(output_grad, "output_grad", "linear_interp_grad");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::LinearInterpGradOp linear_interp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LinearInterpGradOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              output_grad,
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return linear_interp_grad_op.result(0);
}

pir::OpResult log10_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log10_grad");
  paddle::dialect::Log10GradOp log10_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log10GradOp>(
          x, out_grad);
  return log10_grad_op.result(0);
}

pir::OpResult log10_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log10_grad_");
  paddle::dialect::Log10Grad_Op log10_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log10Grad_Op>(
          x, out_grad);
  return log10_grad__op.result(0);
}

pir::OpResult log1p_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log1p_grad");
  paddle::dialect::Log1pGradOp log1p_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log1pGradOp>(
          x, out_grad);
  return log1p_grad_op.result(0);
}

pir::OpResult log1p_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log1p_grad_");
  paddle::dialect::Log1pGrad_Op log1p_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log1pGrad_Op>(
          x, out_grad);
  return log1p_grad__op.result(0);
}

pir::OpResult log2_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log2_grad");
  paddle::dialect::Log2GradOp log2_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log2GradOp>(
          x, out_grad);
  return log2_grad_op.result(0);
}

pir::OpResult log2_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log2_grad_");
  paddle::dialect::Log2Grad_Op log2_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Log2Grad_Op>(
          x, out_grad);
  return log2_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> log_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "log_double_grad");
  paddle::dialect::LogDoubleGradOp log_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogDoubleGradOp>(x, grad_out, grad_x_grad);
  return std::make_tuple(log_double_grad_op.result(0),
                         log_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> log_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "log_double_grad_");
  paddle::dialect::LogDoubleGrad_Op log_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogDoubleGrad_Op>(x, grad_out, grad_x_grad);
  return std::make_tuple(log_double_grad__op.result(0),
                         log_double_grad__op.result(1));
}

pir::OpResult log_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log_grad");
  paddle::dialect::LogGradOp log_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogGradOp>(
          x, out_grad);
  return log_grad_op.result(0);
}

pir::OpResult log_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "log_grad_");
  paddle::dialect::LogGrad_Op log_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogGrad_Op>(
          x, out_grad);
  return log_grad__op.result(0);
}

pir::OpResult log_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const pir::Value& out_grad,
                            float epsilon) {
  CheckValueDataType(out_grad, "out_grad", "log_loss_grad");
  paddle::dialect::LogLossGradOp log_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogLossGradOp>(
              input, label, out_grad, epsilon);
  return log_loss_grad_op.result(0);
}

pir::OpResult log_softmax_grad(const pir::Value& out,
                               const pir::Value& out_grad,
                               int axis) {
  CheckValueDataType(out_grad, "out_grad", "log_softmax_grad");
  paddle::dialect::LogSoftmaxGradOp log_softmax_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogSoftmaxGradOp>(out, out_grad, axis);
  return log_softmax_grad_op.result(0);
}

pir::OpResult logcumsumexp_grad(const pir::Value& x,
                                const pir::Value& out,
                                const pir::Value& out_grad,
                                int axis,
                                bool flatten,
                                bool exclusive,
                                bool reverse) {
  CheckValueDataType(out_grad, "out_grad", "logcumsumexp_grad");
  paddle::dialect::LogcumsumexpGradOp logcumsumexp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogcumsumexpGradOp>(
              x, out, out_grad, axis, flatten, exclusive, reverse);
  return logcumsumexp_grad_op.result(0);
}

pir::OpResult logit_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         float eps) {
  CheckValueDataType(out_grad, "out_grad", "logit_grad");
  paddle::dialect::LogitGradOp logit_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogitGradOp>(
          x, out_grad, eps);
  return logit_grad_op.result(0);
}

pir::OpResult logsigmoid_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "logsigmoid_grad");
  paddle::dialect::LogsigmoidGradOp logsigmoid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogsigmoidGradOp>(x, out_grad);
  return logsigmoid_grad_op.result(0);
}

pir::OpResult logsigmoid_grad_(const pir::Value& x,
                               const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "logsigmoid_grad_");
  paddle::dialect::LogsigmoidGrad_Op logsigmoid_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogsigmoidGrad_Op>(x, out_grad);
  return logsigmoid_grad__op.result(0);
}

pir::OpResult lu_grad(const pir::Value& x,
                      const pir::Value& out,
                      const pir::Value& pivots,
                      const pir::Value& out_grad,
                      bool pivot) {
  CheckValueDataType(out_grad, "out_grad", "lu_grad");
  paddle::dialect::LuGradOp lu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LuGradOp>(
          x, out, pivots, out_grad, pivot);
  return lu_grad_op.result(0);
}

pir::OpResult lu_grad_(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& pivots,
                       const pir::Value& out_grad,
                       bool pivot) {
  CheckValueDataType(out_grad, "out_grad", "lu_grad_");
  paddle::dialect::LuGrad_Op lu_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LuGrad_Op>(
          x, out, pivots, out_grad, pivot);
  return lu_grad__op.result(0);
}

pir::OpResult lu_unpack_grad(const pir::Value& x,
                             const pir::Value& y,
                             const pir::Value& l,
                             const pir::Value& u,
                             const pir::Value& pmat,
                             const pir::Value& l_grad,
                             const pir::Value& u_grad,
                             bool unpack_ludata,
                             bool unpack_pivots) {
  CheckValueDataType(u_grad, "u_grad", "lu_unpack_grad");
  paddle::dialect::LuUnpackGradOp lu_unpack_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LuUnpackGradOp>(
              x, y, l, u, pmat, l_grad, u_grad, unpack_ludata, unpack_pivots);
  return lu_unpack_grad_op.result(0);
}

pir::OpResult margin_cross_entropy_grad(const pir::Value& logits,
                                        const pir::Value& label,
                                        const pir::Value& softmax,
                                        const pir::Value& loss_grad,
                                        bool return_softmax,
                                        int ring_id,
                                        int rank,
                                        int nranks,
                                        float margin1,
                                        float margin2,
                                        float margin3,
                                        float scale) {
  CheckValueDataType(softmax, "softmax", "margin_cross_entropy_grad");
  paddle::dialect::MarginCrossEntropyGradOp margin_cross_entropy_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MarginCrossEntropyGradOp>(logits,
                                                             label,
                                                             softmax,
                                                             loss_grad,
                                                             return_softmax,
                                                             ring_id,
                                                             rank,
                                                             nranks,
                                                             margin1,
                                                             margin2,
                                                             margin3,
                                                             scale);
  return margin_cross_entropy_grad_op.result(0);
}

pir::OpResult margin_cross_entropy_grad_(const pir::Value& logits,
                                         const pir::Value& label,
                                         const pir::Value& softmax,
                                         const pir::Value& loss_grad,
                                         bool return_softmax,
                                         int ring_id,
                                         int rank,
                                         int nranks,
                                         float margin1,
                                         float margin2,
                                         float margin3,
                                         float scale) {
  CheckValueDataType(softmax, "softmax", "margin_cross_entropy_grad_");
  paddle::dialect::MarginCrossEntropyGrad_Op margin_cross_entropy_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MarginCrossEntropyGrad_Op>(logits,
                                                              label,
                                                              softmax,
                                                              loss_grad,
                                                              return_softmax,
                                                              ring_id,
                                                              rank,
                                                              nranks,
                                                              margin1,
                                                              margin2,
                                                              margin3,
                                                              scale);
  return margin_cross_entropy_grad__op.result(0);
}

pir::OpResult masked_select_grad(const pir::Value& x,
                                 const pir::Value& mask,
                                 const pir::Value& out_grad) {
  CheckValueDataType(x, "x", "masked_select_grad");
  paddle::dialect::MaskedSelectGradOp masked_select_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaskedSelectGradOp>(x, mask, out_grad);
  return masked_select_grad_op.result(0);
}

pir::OpResult matrix_power_grad(const pir::Value& x,
                                const pir::Value& out,
                                const pir::Value& out_grad,
                                int n) {
  CheckValueDataType(out_grad, "out_grad", "matrix_power_grad");
  paddle::dialect::MatrixPowerGradOp matrix_power_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatrixPowerGradOp>(x, out, out_grad, n);
  return matrix_power_grad_op.result(0);
}

pir::OpResult max_pool2d_with_index_grad(const pir::Value& x,
                                         const pir::Value& mask,
                                         const pir::Value& out_grad,
                                         const std::vector<int>& kernel_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         bool global_pooling,
                                         bool adaptive) {
  CheckValueDataType(out_grad, "out_grad", "max_pool2d_with_index_grad");
  paddle::dialect::MaxPool2dWithIndexGradOp max_pool2d_with_index_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool2dWithIndexGradOp>(x,
                                                             mask,
                                                             out_grad,
                                                             kernel_size,
                                                             strides,
                                                             paddings,
                                                             global_pooling,
                                                             adaptive);
  return max_pool2d_with_index_grad_op.result(0);
}

pir::OpResult max_pool3d_with_index_grad(const pir::Value& x,
                                         const pir::Value& mask,
                                         const pir::Value& out_grad,
                                         const std::vector<int>& kernel_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         bool global_pooling,
                                         bool adaptive) {
  CheckValueDataType(out_grad, "out_grad", "max_pool3d_with_index_grad");
  paddle::dialect::MaxPool3dWithIndexGradOp max_pool3d_with_index_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool3dWithIndexGradOp>(x,
                                                             mask,
                                                             out_grad,
                                                             kernel_size,
                                                             strides,
                                                             paddings,
                                                             global_pooling,
                                                             adaptive);
  return max_pool3d_with_index_grad_op.result(0);
}

pir::OpResult maxout_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          int groups,
                          int axis) {
  CheckValueDataType(out_grad, "out_grad", "maxout_grad");
  paddle::dialect::MaxoutGradOp maxout_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxoutGradOp>(
          x, out, out_grad, groups, axis);
  return maxout_grad_op.result(0);
}

pir::OpResult mean_all_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "mean_all_grad");
  paddle::dialect::MeanAllGradOp mean_all_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MeanAllGradOp>(x, out_grad);
  return mean_all_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
memory_efficient_attention_grad(
    const pir::Value& query,
    const pir::Value& key,
    const pir::Value& value,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& cu_seqlens_q,
    const paddle::optional<pir::Value>& cu_seqlens_k,
    const pir::Value& output,
    const pir::Value& logsumexp,
    const pir::Value& seed_and_offset,
    const pir::Value& output_grad,
    float max_seqlen_q,
    float max_seqlen_k,
    bool causal,
    double dropout_p,
    float scale) {
  CheckValueDataType(
      output_grad, "output_grad", "memory_efficient_attention_grad");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_cu_seqlens_q;
  if (!cu_seqlens_q) {
    optional_cu_seqlens_q = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cu_seqlens_q = cu_seqlens_q;
  }
  paddle::optional<pir::Value> optional_cu_seqlens_k;
  if (!cu_seqlens_k) {
    optional_cu_seqlens_k = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cu_seqlens_k = cu_seqlens_k;
  }
  paddle::dialect::MemoryEfficientAttentionGradOp
      memory_efficient_attention_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::MemoryEfficientAttentionGradOp>(
                  query,
                  key,
                  value,
                  optional_bias.get(),
                  optional_cu_seqlens_q.get(),
                  optional_cu_seqlens_k.get(),
                  output,
                  logsumexp,
                  seed_and_offset,
                  output_grad,
                  max_seqlen_q,
                  max_seqlen_k,
                  causal,
                  dropout_p,
                  scale);
  return std::make_tuple(memory_efficient_attention_grad_op.result(0),
                         memory_efficient_attention_grad_op.result(1),
                         memory_efficient_attention_grad_op.result(2),
                         memory_efficient_attention_grad_op.result(3));
}

std::vector<pir::OpResult> meshgrid_grad(
    const std::vector<pir::Value>& inputs,
    const std::vector<pir::Value>& outputs_grad) {
  CheckVectorOfValueDataType(outputs_grad, "outputs_grad", "meshgrid_grad");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  auto outputs_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(outputs_grad);
  paddle::dialect::MeshgridGradOp meshgrid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MeshgridGradOp>(
              inputs_combine_op.out(), outputs_grad_combine_op.out());
  auto inputs_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          meshgrid_grad_op.result(0));
  return inputs_grad_split_op.outputs();
}

pir::OpResult mode_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        int axis,
                        bool keepdim) {
  CheckValueDataType(out_grad, "out_grad", "mode_grad");
  paddle::dialect::ModeGradOp mode_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ModeGradOp>(
          x, indices, out_grad, axis, keepdim);
  return mode_grad_op.result(0);
}

std::vector<pir::OpResult> multi_dot_grad(const std::vector<pir::Value>& x,
                                          const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "multi_dot_grad");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::MultiDotGradOp multi_dot_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiDotGradOp>(x_combine_op.out(),
                                                   out_grad);
  auto x_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          multi_dot_grad_op.result(0));
  return x_grad_split_op.outputs();
}

std::vector<pir::OpResult> multiplex_grad(const std::vector<pir::Value>& inputs,
                                          const pir::Value& index,
                                          const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "multiplex_grad");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::MultiplexGradOp multiplex_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiplexGradOp>(
              inputs_combine_op.out(), index, out_grad);
  auto inputs_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          multiplex_grad_op.result(0));
  return inputs_grad_split_op.outputs();
}

std::tuple<pir::OpResult, pir::OpResult> mv_grad(const pir::Value& x,
                                                 const pir::Value& vec,
                                                 const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "mv_grad");
  paddle::dialect::MvGradOp mv_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MvGradOp>(
          x, vec, out_grad);
  return std::make_tuple(mv_grad_op.result(0), mv_grad_op.result(1));
}

pir::OpResult nanmedian_grad(const pir::Value& x,
                             const pir::Value& medians,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis,
                             bool keepdim) {
  CheckValueDataType(out_grad, "out_grad", "nanmedian_grad");
  paddle::dialect::NanmedianGradOp nanmedian_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NanmedianGradOp>(
              x, medians, out_grad, axis, keepdim);
  return nanmedian_grad_op.result(0);
}

pir::OpResult nearest_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(output_grad, "output_grad", "nearest_interp_grad");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::NearestInterpGradOp nearest_interp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NearestInterpGradOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              output_grad,
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return nearest_interp_grad_op.result(0);
}

pir::OpResult nll_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const paddle::optional<pir::Value>& weight,
                            const pir::Value& total_weight,
                            const pir::Value& out_grad,
                            int64_t ignore_index,
                            const std::string& reduction) {
  CheckValueDataType(input, "input", "nll_loss_grad");
  paddle::optional<pir::Value> optional_weight;
  if (!weight) {
    optional_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_weight = weight;
  }
  paddle::dialect::NllLossGradOp nll_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NllLossGradOp>(input,
                                                  label,
                                                  optional_weight.get(),
                                                  total_weight,
                                                  out_grad,
                                                  ignore_index,
                                                  reduction);
  return nll_loss_grad_op.result(0);
}

pir::OpResult overlap_add_grad(const pir::Value& x,
                               const pir::Value& out_grad,
                               int hop_length,
                               int axis) {
  CheckValueDataType(x, "x", "overlap_add_grad");
  paddle::dialect::OverlapAddGradOp overlap_add_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::OverlapAddGradOp>(
              x, out_grad, hop_length, axis);
  return overlap_add_grad_op.result(0);
}

pir::OpResult p_norm_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          float porder,
                          int axis,
                          float epsilon,
                          bool keepdim,
                          bool asvector) {
  CheckValueDataType(out_grad, "out_grad", "p_norm_grad");
  paddle::dialect::PNormGradOp p_norm_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PNormGradOp>(
          x, out, out_grad, porder, axis, epsilon, keepdim, asvector);
  return p_norm_grad_op.result(0);
}

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                const std::vector<int64_t>& paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pad3d");
  paddle::dialect::Pad3dDoubleGradOp pad3d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pad3dDoubleGradOp>(
              grad_x_grad, paddings, mode, pad_value, data_format);
  return pad3d_double_grad_op.result(0);
}

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                pir::Value paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pad3d");
  paddle::dialect::Pad3dDoubleGradOp pad3d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pad3dDoubleGradOp>(
              grad_x_grad, paddings, mode, pad_value, data_format);
  return pad3d_double_grad_op.result(0);
}

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                std::vector<pir::Value> paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pad3d");
  auto paddings_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(paddings);
  paddle::dialect::Pad3dDoubleGradOp pad3d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pad3dDoubleGradOp>(grad_x_grad,
                                                      paddings_combine_op.out(),
                                                      mode,
                                                      pad_value,
                                                      data_format);
  return pad3d_double_grad_op.result(0);
}

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         const std::vector<int64_t>& paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "pad3d_grad");
  paddle::dialect::Pad3dGradOp pad3d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dGradOp>(
          x, out_grad, paddings, mode, pad_value, data_format);
  return pad3d_grad_op.result(0);
}

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         pir::Value paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "pad3d_grad");
  paddle::dialect::Pad3dGradOp pad3d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dGradOp>(
          x, out_grad, paddings, mode, pad_value, data_format);
  return pad3d_grad_op.result(0);
}

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         std::vector<pir::Value> paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "pad3d_grad");
  auto paddings_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(paddings);
  paddle::dialect::Pad3dGradOp pad3d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pad3dGradOp>(
          x, out_grad, paddings_combine_op.out(), mode, pad_value, data_format);
  return pad3d_grad_op.result(0);
}

pir::OpResult pixel_shuffle_grad(const pir::Value& out_grad,
                                 int upscale_factor,
                                 const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "pixel_shuffle_grad");
  paddle::dialect::PixelShuffleGradOp pixel_shuffle_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PixelShuffleGradOp>(
              out_grad, upscale_factor, data_format);
  return pixel_shuffle_grad_op.result(0);
}

pir::OpResult pixel_unshuffle_grad(const pir::Value& out_grad,
                                   int downscale_factor,
                                   const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "pixel_unshuffle_grad");
  paddle::dialect::PixelUnshuffleGradOp pixel_unshuffle_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PixelUnshuffleGradOp>(
              out_grad, downscale_factor, data_format);
  return pixel_unshuffle_grad_op.result(0);
}

pir::OpResult poisson_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "poisson_grad");
  paddle::dialect::PoissonGradOp poisson_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PoissonGradOp>(out_grad);
  return poisson_grad_op.result(0);
}

pir::OpResult polygamma_grad(const pir::Value& x,
                             const pir::Value& out_grad,
                             int n) {
  CheckValueDataType(out_grad, "out_grad", "polygamma_grad");
  paddle::dialect::PolygammaGradOp polygamma_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PolygammaGradOp>(x, out_grad, n);
  return polygamma_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> pow_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float y) {
  CheckValueDataType(x, "x", "pow_double_grad");
  paddle::dialect::PowDoubleGradOp pow_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PowDoubleGradOp>(
              x, grad_out, grad_x_grad, y);
  return std::make_tuple(pow_double_grad_op.result(0),
                         pow_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> pow_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float y) {
  CheckValueDataType(x, "x", "pow_double_grad_");
  paddle::dialect::PowDoubleGrad_Op pow_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PowDoubleGrad_Op>(
              x, grad_out, grad_x_grad, y);
  return std::make_tuple(pow_double_grad__op.result(0),
                         pow_double_grad__op.result(1));
}

pir::OpResult pow_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       float y) {
  CheckValueDataType(out_grad, "out_grad", "pow_grad");
  paddle::dialect::PowGradOp pow_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PowGradOp>(
          x, out_grad, y);
  return pow_grad_op.result(0);
}

pir::OpResult pow_grad_(const pir::Value& x,
                        const pir::Value& out_grad,
                        float y) {
  CheckValueDataType(out_grad, "out_grad", "pow_grad_");
  paddle::dialect::PowGrad_Op pow_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PowGrad_Op>(
          x, out_grad, y);
  return pow_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> pow_triple_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad,
    float y) {
  CheckValueDataType(x, "x", "pow_triple_grad");
  paddle::optional<pir::Value> optional_grad_grad_out_grad;
  if (!grad_grad_out_grad) {
    optional_grad_grad_out_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_grad_out_grad = grad_grad_out_grad;
  }
  paddle::dialect::PowTripleGradOp pow_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PowTripleGradOp>(
              x,
              grad_out,
              grad_grad_x,
              grad_x_grad,
              optional_grad_grad_out_grad.get(),
              y);
  return std::make_tuple(pow_triple_grad_op.result(0),
                         pow_triple_grad_op.result(1),
                         pow_triple_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> prelu_grad(
    const pir::Value& x,
    const pir::Value& alpha,
    const pir::Value& out_grad,
    const std::string& data_format,
    const std::string& mode) {
  CheckValueDataType(x, "x", "prelu_grad");
  paddle::dialect::PreluGradOp prelu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PreluGradOp>(
          x, alpha, out_grad, data_format, mode);
  return std::make_tuple(prelu_grad_op.result(0), prelu_grad_op.result(1));
}

pir::OpResult psroi_pool_grad(const pir::Value& x,
                              const pir::Value& boxes,
                              const paddle::optional<pir::Value>& boxes_num,
                              const pir::Value& out_grad,
                              int pooled_height,
                              int pooled_width,
                              int output_channels,
                              float spatial_scale) {
  CheckValueDataType(x, "x", "psroi_pool_grad");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::PsroiPoolGradOp psroi_pool_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PsroiPoolGradOp>(x,
                                                    boxes,
                                                    optional_boxes_num.get(),
                                                    out_grad,
                                                    pooled_height,
                                                    pooled_width,
                                                    output_channels,
                                                    spatial_scale);
  return psroi_pool_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> put_along_axis_grad(
    const pir::Value& arr,
    const pir::Value& indices,
    const pir::Value& values,
    const pir::Value& out,
    const pir::Value& out_grad,
    int axis,
    const std::string& reduce,
    bool include_self) {
  CheckValueDataType(out_grad, "out_grad", "put_along_axis_grad");
  paddle::dialect::PutAlongAxisGradOp put_along_axis_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PutAlongAxisGradOp>(
              arr, indices, values, out, out_grad, axis, reduce, include_self);
  return std::make_tuple(put_along_axis_grad_op.result(0),
                         put_along_axis_grad_op.result(1));
}

pir::OpResult qr_grad(const pir::Value& x,
                      const pir::Value& q,
                      const pir::Value& r,
                      const pir::Value& q_grad,
                      const pir::Value& r_grad,
                      const std::string& mode) {
  CheckValueDataType(r_grad, "r_grad", "qr_grad");
  paddle::dialect::QrGradOp qr_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::QrGradOp>(
          x, q, r, q_grad, r_grad, mode);
  return qr_grad_op.result(0);
}

pir::OpResult real_grad(const pir::Value& out_grad) {
  paddle::dialect::RealGradOp real_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RealGradOp>(
          out_grad);
  return real_grad_op.result(0);
}

pir::OpResult reciprocal_grad(const pir::Value& out,
                              const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "reciprocal_grad");
  paddle::dialect::ReciprocalGradOp reciprocal_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReciprocalGradOp>(out, out_grad);
  return reciprocal_grad_op.result(0);
}

pir::OpResult reciprocal_grad_(const pir::Value& out,
                               const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "reciprocal_grad_");
  paddle::dialect::ReciprocalGrad_Op reciprocal_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReciprocalGrad_Op>(out, out_grad);
  return reciprocal_grad__op.result(0);
}

pir::OpResult relu6_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "relu6_grad");
  paddle::dialect::Relu6GradOp relu6_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Relu6GradOp>(
          out, out_grad);
  return relu6_grad_op.result(0);
}

pir::OpResult relu6_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "relu6_grad_");
  paddle::dialect::Relu6Grad_Op relu6_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Relu6Grad_Op>(
          out, out_grad);
  return relu6_grad__op.result(0);
}

pir::OpResult relu_double_grad(const pir::Value& out,
                               const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "relu_double_grad");
  paddle::dialect::ReluDoubleGradOp relu_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReluDoubleGradOp>(out, grad_x_grad);
  return relu_double_grad_op.result(0);
}

pir::OpResult relu_double_grad_(const pir::Value& out,
                                const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "relu_double_grad_");
  paddle::dialect::ReluDoubleGrad_Op relu_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReluDoubleGrad_Op>(out, grad_x_grad);
  return relu_double_grad__op.result(0);
}

pir::OpResult relu_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "relu_grad");
  paddle::dialect::ReluGradOp relu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReluGradOp>(
          out, out_grad);
  return relu_grad_op.result(0);
}

pir::OpResult relu_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "relu_grad_");
  paddle::dialect::ReluGrad_Op relu_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReluGrad_Op>(
          out, out_grad);
  return relu_grad__op.result(0);
}

pir::OpResult renorm_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          float p,
                          int axis,
                          float max_norm) {
  CheckValueDataType(out_grad, "out_grad", "renorm_grad");
  paddle::dialect::RenormGradOp renorm_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RenormGradOp>(
          x, out_grad, p, axis, max_norm);
  return renorm_grad_op.result(0);
}

pir::OpResult roi_align_grad(const pir::Value& x,
                             const pir::Value& boxes,
                             const paddle::optional<pir::Value>& boxes_num,
                             const pir::Value& out_grad,
                             int pooled_height,
                             int pooled_width,
                             float spatial_scale,
                             int sampling_ratio,
                             bool aligned) {
  CheckValueDataType(boxes, "boxes", "roi_align_grad");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::RoiAlignGradOp roi_align_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RoiAlignGradOp>(x,
                                                   boxes,
                                                   optional_boxes_num.get(),
                                                   out_grad,
                                                   pooled_height,
                                                   pooled_width,
                                                   spatial_scale,
                                                   sampling_ratio,
                                                   aligned);
  return roi_align_grad_op.result(0);
}

pir::OpResult roi_pool_grad(const pir::Value& x,
                            const pir::Value& boxes,
                            const paddle::optional<pir::Value>& boxes_num,
                            const pir::Value& arg_max,
                            const pir::Value& out_grad,
                            int pooled_height,
                            int pooled_width,
                            float spatial_scale) {
  CheckValueDataType(x, "x", "roi_pool_grad");
  paddle::optional<pir::Value> optional_boxes_num;
  if (!boxes_num) {
    optional_boxes_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_boxes_num = boxes_num;
  }
  paddle::dialect::RoiPoolGradOp roi_pool_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RoiPoolGradOp>(x,
                                                  boxes,
                                                  optional_boxes_num.get(),
                                                  arg_max,
                                                  out_grad,
                                                  pooled_height,
                                                  pooled_width,
                                                  spatial_scale);
  return roi_pool_grad_op.result(0);
}

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& shifts,
                        const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll_grad");
  paddle::dialect::RollGradOp roll_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollGradOp>(
          x, out_grad, shifts, axis);
  return roll_grad_op.result(0);
}

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value shifts,
                        const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll_grad");
  paddle::dialect::RollGradOp roll_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollGradOp>(
          x, out_grad, shifts, axis);
  return roll_grad_op.result(0);
}

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> shifts,
                        const std::vector<int64_t>& axis) {
  CheckValueDataType(x, "x", "roll_grad");
  auto shifts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shifts);
  paddle::dialect::RollGradOp roll_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RollGradOp>(
          x, out_grad, shifts_combine_op.out(), axis);
  return roll_grad_op.result(0);
}

pir::OpResult round_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "round_grad");
  paddle::dialect::RoundGradOp round_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RoundGradOp>(
          out_grad);
  return round_grad_op.result(0);
}

pir::OpResult round_grad_(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "round_grad_");
  paddle::dialect::RoundGrad_Op round_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RoundGrad_Op>(
          out_grad);
  return round_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> rsqrt_double_grad(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "rsqrt_double_grad");
  paddle::dialect::RsqrtDoubleGradOp rsqrt_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RsqrtDoubleGradOp>(out, grad_x, grad_x_grad);
  return std::make_tuple(rsqrt_double_grad_op.result(0),
                         rsqrt_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> rsqrt_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "rsqrt_double_grad_");
  paddle::dialect::RsqrtDoubleGrad_Op rsqrt_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RsqrtDoubleGrad_Op>(
              out, grad_x, grad_x_grad);
  return std::make_tuple(rsqrt_double_grad__op.result(0),
                         rsqrt_double_grad__op.result(1));
}

pir::OpResult rsqrt_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "rsqrt_grad");
  paddle::dialect::RsqrtGradOp rsqrt_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RsqrtGradOp>(
          out, out_grad);
  return rsqrt_grad_op.result(0);
}

pir::OpResult rsqrt_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "rsqrt_grad_");
  paddle::dialect::RsqrtGrad_Op rsqrt_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RsqrtGrad_Op>(
          out, out_grad);
  return rsqrt_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> scatter_grad(
    const pir::Value& index,
    const pir::Value& updates,
    const pir::Value& out_grad,
    bool overwrite) {
  CheckValueDataType(out_grad, "out_grad", "scatter_grad");
  paddle::dialect::ScatterGradOp scatter_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ScatterGradOp>(
              index, updates, out_grad, overwrite);
  return std::make_tuple(scatter_grad_op.result(0), scatter_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> scatter_nd_add_grad(
    const pir::Value& index,
    const pir::Value& updates,
    const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "scatter_nd_add_grad");
  paddle::dialect::ScatterNdAddGradOp scatter_nd_add_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ScatterNdAddGradOp>(
              index, updates, out_grad);
  return std::make_tuple(scatter_nd_add_grad_op.result(0),
                         scatter_nd_add_grad_op.result(1));
}

pir::OpResult segment_pool_grad(const pir::Value& x,
                                const pir::Value& segment_ids,
                                const pir::Value& out,
                                const paddle::optional<pir::Value>& summed_ids,
                                const pir::Value& out_grad,
                                const std::string& pooltype) {
  CheckValueDataType(out_grad, "out_grad", "segment_pool_grad");
  paddle::optional<pir::Value> optional_summed_ids;
  if (!summed_ids) {
    optional_summed_ids = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_summed_ids = summed_ids;
  }
  paddle::dialect::SegmentPoolGradOp segment_pool_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SegmentPoolGradOp>(x,
                                                      segment_ids,
                                                      out,
                                                      optional_summed_ids.get(),
                                                      out_grad,
                                                      pooltype);
  return segment_pool_grad_op.result(0);
}

pir::OpResult selu_grad(const pir::Value& out,
                        const pir::Value& out_grad,
                        float scale,
                        float alpha) {
  CheckValueDataType(out, "out", "selu_grad");
  paddle::dialect::SeluGradOp selu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SeluGradOp>(
          out, out_grad, scale, alpha);
  return selu_grad_op.result(0);
}

pir::OpResult send_u_recv_grad(const pir::Value& x,
                               const pir::Value& src_index,
                               const pir::Value& dst_index,
                               const paddle::optional<pir::Value>& out,
                               const paddle::optional<pir::Value>& dst_count,
                               const pir::Value& out_grad,
                               const std::string& reduce_op) {
  CheckValueDataType(out_grad, "out_grad", "send_u_recv_grad");
  paddle::optional<pir::Value> optional_out;
  if (!out) {
    optional_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out = out;
  }
  paddle::optional<pir::Value> optional_dst_count;
  if (!dst_count) {
    optional_dst_count = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dst_count = dst_count;
  }
  paddle::dialect::SendURecvGradOp send_u_recv_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SendURecvGradOp>(x,
                                                    src_index,
                                                    dst_index,
                                                    optional_out.get(),
                                                    optional_dst_count.get(),
                                                    out_grad,
                                                    reduce_op);
  return send_u_recv_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> send_ue_recv_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& src_index,
    const pir::Value& dst_index,
    const paddle::optional<pir::Value>& out,
    const paddle::optional<pir::Value>& dst_count,
    const pir::Value& out_grad,
    const std::string& message_op,
    const std::string& reduce_op) {
  CheckValueDataType(out_grad, "out_grad", "send_ue_recv_grad");
  paddle::optional<pir::Value> optional_out;
  if (!out) {
    optional_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out = out;
  }
  paddle::optional<pir::Value> optional_dst_count;
  if (!dst_count) {
    optional_dst_count = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dst_count = dst_count;
  }
  paddle::dialect::SendUeRecvGradOp send_ue_recv_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SendUeRecvGradOp>(x,
                                                     y,
                                                     src_index,
                                                     dst_index,
                                                     optional_out.get(),
                                                     optional_dst_count.get(),
                                                     out_grad,
                                                     message_op,
                                                     reduce_op);
  return std::make_tuple(send_ue_recv_grad_op.result(0),
                         send_ue_recv_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> send_uv_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& src_index,
    const pir::Value& dst_index,
    const pir::Value& out_grad,
    const std::string& message_op) {
  CheckValueDataType(x, "x", "send_uv_grad");
  paddle::dialect::SendUvGradOp send_uv_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendUvGradOp>(
          x, y, src_index, dst_index, out_grad, message_op);
  return std::make_tuple(send_uv_grad_op.result(0), send_uv_grad_op.result(1));
}

pir::OpResult sigmoid_cross_entropy_with_logits_grad(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    const pir::Value& out_grad,
    bool normalize,
    int ignore_index) {
  CheckValueDataType(
      out_grad, "out_grad", "sigmoid_cross_entropy_with_logits_grad");
  paddle::optional<pir::Value> optional_pos_weight;
  if (!pos_weight) {
    optional_pos_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pos_weight = pos_weight;
  }
  paddle::dialect::SigmoidCrossEntropyWithLogitsGradOp
      sigmoid_cross_entropy_with_logits_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::SigmoidCrossEntropyWithLogitsGradOp>(
                  x,
                  label,
                  optional_pos_weight.get(),
                  out_grad,
                  normalize,
                  ignore_index);
  return sigmoid_cross_entropy_with_logits_grad_op.result(0);
}

pir::OpResult sigmoid_cross_entropy_with_logits_grad_(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    const pir::Value& out_grad,
    bool normalize,
    int ignore_index) {
  CheckValueDataType(
      out_grad, "out_grad", "sigmoid_cross_entropy_with_logits_grad_");
  paddle::optional<pir::Value> optional_pos_weight;
  if (!pos_weight) {
    optional_pos_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pos_weight = pos_weight;
  }
  paddle::dialect::SigmoidCrossEntropyWithLogitsGrad_Op
      sigmoid_cross_entropy_with_logits_grad__op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::SigmoidCrossEntropyWithLogitsGrad_Op>(
                  x,
                  label,
                  optional_pos_weight.get(),
                  out_grad,
                  normalize,
                  ignore_index);
  return sigmoid_cross_entropy_with_logits_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> sigmoid_double_grad(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sigmoid_double_grad");
  paddle::dialect::SigmoidDoubleGradOp sigmoid_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidDoubleGradOp>(
              out, fwd_grad_out, grad_x_grad);
  return std::make_tuple(sigmoid_double_grad_op.result(0),
                         sigmoid_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> sigmoid_double_grad_(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sigmoid_double_grad_");
  paddle::dialect::SigmoidDoubleGrad_Op sigmoid_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidDoubleGrad_Op>(
              out, fwd_grad_out, grad_x_grad);
  return std::make_tuple(sigmoid_double_grad__op.result(0),
                         sigmoid_double_grad__op.result(1));
}

pir::OpResult sigmoid_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sigmoid_grad");
  paddle::dialect::SigmoidGradOp sigmoid_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidGradOp>(out, out_grad);
  return sigmoid_grad_op.result(0);
}

pir::OpResult sigmoid_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sigmoid_grad_");
  paddle::dialect::SigmoidGrad_Op sigmoid_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidGrad_Op>(out, out_grad);
  return sigmoid_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sigmoid_triple_grad(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_out_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad) {
  if (grad_grad_out_grad) {
    CheckValueDataType(
        grad_grad_out_grad.get(), "grad_grad_out_grad", "sigmoid_triple_grad");
  } else {
    CheckValueDataType(grad_out_grad, "grad_out_grad", "sigmoid_triple_grad");
  }
  paddle::optional<pir::Value> optional_grad_grad_out_grad;
  if (!grad_grad_out_grad) {
    optional_grad_grad_out_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_grad_out_grad = grad_grad_out_grad;
  }
  paddle::dialect::SigmoidTripleGradOp sigmoid_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidTripleGradOp>(
              out,
              fwd_grad_out,
              grad_grad_x,
              grad_out_grad,
              optional_grad_grad_out_grad.get());
  return std::make_tuple(sigmoid_triple_grad_op.result(0),
                         sigmoid_triple_grad_op.result(1),
                         sigmoid_triple_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sigmoid_triple_grad_(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_out_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad) {
  if (grad_grad_out_grad) {
    CheckValueDataType(
        grad_grad_out_grad.get(), "grad_grad_out_grad", "sigmoid_triple_grad_");
  } else {
    CheckValueDataType(grad_out_grad, "grad_out_grad", "sigmoid_triple_grad_");
  }
  paddle::optional<pir::Value> optional_grad_grad_out_grad;
  if (!grad_grad_out_grad) {
    optional_grad_grad_out_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_grad_out_grad = grad_grad_out_grad;
  }
  paddle::dialect::SigmoidTripleGrad_Op sigmoid_triple_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SigmoidTripleGrad_Op>(
              out,
              fwd_grad_out,
              grad_grad_x,
              grad_out_grad,
              optional_grad_grad_out_grad.get());
  return std::make_tuple(sigmoid_triple_grad__op.result(0),
                         sigmoid_triple_grad__op.result(1),
                         sigmoid_triple_grad__op.result(2));
}

pir::OpResult silu_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "silu_grad");
  paddle::dialect::SiluGradOp silu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SiluGradOp>(
          x, out, out_grad);
  return silu_grad_op.result(0);
}

pir::OpResult silu_grad_(const pir::Value& x,
                         const pir::Value& out,
                         const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "silu_grad_");
  paddle::dialect::SiluGrad_Op silu_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SiluGrad_Op>(
          x, out, out_grad);
  return silu_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> sin_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sin_double_grad");
  paddle::optional<pir::Value> optional_grad_out;
  if (!grad_out) {
    optional_grad_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out = grad_out;
  }
  paddle::dialect::SinDoubleGradOp sin_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SinDoubleGradOp>(
              x, optional_grad_out.get(), grad_x_grad);
  return std::make_tuple(sin_double_grad_op.result(0),
                         sin_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> sin_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sin_double_grad_");
  paddle::optional<pir::Value> optional_grad_out;
  if (!grad_out) {
    optional_grad_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out = grad_out;
  }
  paddle::dialect::SinDoubleGrad_Op sin_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SinDoubleGrad_Op>(
              x, optional_grad_out.get(), grad_x_grad);
  return std::make_tuple(sin_double_grad__op.result(0),
                         sin_double_grad__op.result(1));
}

pir::OpResult sin_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sin_grad");
  paddle::dialect::SinGradOp sin_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinGradOp>(
          x, out_grad);
  return sin_grad_op.result(0);
}

pir::OpResult sin_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sin_grad_");
  paddle::dialect::SinGrad_Op sin_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinGrad_Op>(
          x, out_grad);
  return sin_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sin_triple_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "sin_triple_grad");
  } else {
    CheckValueDataType(grad_x_grad, "grad_x_grad", "sin_triple_grad");
  }
  paddle::optional<pir::Value> optional_grad_out_forward;
  if (!grad_out_forward) {
    optional_grad_out_forward = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_forward = grad_out_forward;
  }
  paddle::optional<pir::Value> optional_grad_x_grad_forward;
  if (!grad_x_grad_forward) {
    optional_grad_x_grad_forward =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad_forward = grad_x_grad_forward;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::SinTripleGradOp sin_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SinTripleGradOp>(
              x,
              optional_grad_out_forward.get(),
              optional_grad_x_grad_forward.get(),
              grad_x_grad,
              optional_grad_out_grad_grad.get());
  return std::make_tuple(sin_triple_grad_op.result(0),
                         sin_triple_grad_op.result(1),
                         sin_triple_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sin_triple_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "sin_triple_grad_");
  } else {
    CheckValueDataType(grad_x_grad, "grad_x_grad", "sin_triple_grad_");
  }
  paddle::optional<pir::Value> optional_grad_out_forward;
  if (!grad_out_forward) {
    optional_grad_out_forward = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_forward = grad_out_forward;
  }
  paddle::optional<pir::Value> optional_grad_x_grad_forward;
  if (!grad_x_grad_forward) {
    optional_grad_x_grad_forward =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad_forward = grad_x_grad_forward;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::SinTripleGrad_Op sin_triple_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SinTripleGrad_Op>(
              x,
              optional_grad_out_forward.get(),
              optional_grad_x_grad_forward.get(),
              grad_x_grad,
              optional_grad_out_grad_grad.get());
  return std::make_tuple(sin_triple_grad__op.result(0),
                         sin_triple_grad__op.result(1),
                         sin_triple_grad__op.result(2));
}

pir::OpResult sinh_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sinh_grad");
  paddle::dialect::SinhGradOp sinh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinhGradOp>(
          x, out_grad);
  return sinh_grad_op.result(0);
}

pir::OpResult sinh_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sinh_grad_");
  paddle::dialect::SinhGrad_Op sinh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinhGrad_Op>(
          x, out_grad);
  return sinh_grad__op.result(0);
}

pir::OpResult slogdet_grad(const pir::Value& x,
                           const pir::Value& out,
                           const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "slogdet_grad");
  paddle::dialect::SlogdetGradOp slogdet_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SlogdetGradOp>(x, out, out_grad);
  return slogdet_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> softplus_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float beta,
    float threshold) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "softplus_double_grad");
  paddle::dialect::SoftplusDoubleGradOp softplus_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftplusDoubleGradOp>(
              x, grad_out, grad_x_grad, beta, threshold);
  return std::make_tuple(softplus_double_grad_op.result(0),
                         softplus_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> softplus_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float beta,
    float threshold) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "softplus_double_grad_");
  paddle::dialect::SoftplusDoubleGrad_Op softplus_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftplusDoubleGrad_Op>(
              x, grad_out, grad_x_grad, beta, threshold);
  return std::make_tuple(softplus_double_grad__op.result(0),
                         softplus_double_grad__op.result(1));
}

pir::OpResult softplus_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            float beta,
                            float threshold) {
  CheckValueDataType(out_grad, "out_grad", "softplus_grad");
  paddle::dialect::SoftplusGradOp softplus_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftplusGradOp>(
              x, out_grad, beta, threshold);
  return softplus_grad_op.result(0);
}

pir::OpResult softplus_grad_(const pir::Value& x,
                             const pir::Value& out_grad,
                             float beta,
                             float threshold) {
  CheckValueDataType(out_grad, "out_grad", "softplus_grad_");
  paddle::dialect::SoftplusGrad_Op softplus_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftplusGrad_Op>(
              x, out_grad, beta, threshold);
  return softplus_grad__op.result(0);
}

pir::OpResult softshrink_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float threshold) {
  CheckValueDataType(out_grad, "out_grad", "softshrink_grad");
  paddle::dialect::SoftshrinkGradOp softshrink_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftshrinkGradOp>(x, out_grad, threshold);
  return softshrink_grad_op.result(0);
}

pir::OpResult softshrink_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float threshold) {
  CheckValueDataType(out_grad, "out_grad", "softshrink_grad_");
  paddle::dialect::SoftshrinkGrad_Op softshrink_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftshrinkGrad_Op>(x, out_grad, threshold);
  return softshrink_grad__op.result(0);
}

pir::OpResult softsign_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "softsign_grad");
  paddle::dialect::SoftsignGradOp softsign_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftsignGradOp>(x, out_grad);
  return softsign_grad_op.result(0);
}

pir::OpResult softsign_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "softsign_grad_");
  paddle::dialect::SoftsignGrad_Op softsign_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftsignGrad_Op>(x, out_grad);
  return softsign_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> solve_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "solve_grad");
  paddle::dialect::SolveGradOp solve_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SolveGradOp>(
          x, y, out, out_grad);
  return std::make_tuple(solve_grad_op.result(0), solve_grad_op.result(1));
}

pir::OpResult spectral_norm_grad(const pir::Value& weight,
                                 const pir::Value& u,
                                 const pir::Value& v,
                                 const pir::Value& out_grad,
                                 int dim,
                                 int power_iters,
                                 float eps) {
  CheckValueDataType(weight, "weight", "spectral_norm_grad");
  paddle::dialect::SpectralNormGradOp spectral_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SpectralNormGradOp>(
              weight, u, v, out_grad, dim, power_iters, eps);
  return spectral_norm_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> sqrt_double_grad(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sqrt_double_grad");
  paddle::dialect::SqrtDoubleGradOp sqrt_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqrtDoubleGradOp>(out, grad_x, grad_x_grad);
  return std::make_tuple(sqrt_double_grad_op.result(0),
                         sqrt_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> sqrt_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "sqrt_double_grad_");
  paddle::dialect::SqrtDoubleGrad_Op sqrt_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqrtDoubleGrad_Op>(out, grad_x, grad_x_grad);
  return std::make_tuple(sqrt_double_grad__op.result(0),
                         sqrt_double_grad__op.result(1));
}

pir::OpResult sqrt_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sqrt_grad");
  paddle::dialect::SqrtGradOp sqrt_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqrtGradOp>(
          out, out_grad);
  return sqrt_grad_op.result(0);
}

pir::OpResult sqrt_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "sqrt_grad_");
  paddle::dialect::SqrtGrad_Op sqrt_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SqrtGrad_Op>(
          out, out_grad);
  return sqrt_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> square_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "square_double_grad");
  paddle::dialect::SquareDoubleGradOp square_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SquareDoubleGradOp>(
              x, grad_out, grad_x_grad);
  return std::make_tuple(square_double_grad_op.result(0),
                         square_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> square_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "square_double_grad_");
  paddle::dialect::SquareDoubleGrad_Op square_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SquareDoubleGrad_Op>(
              x, grad_out, grad_x_grad);
  return std::make_tuple(square_double_grad__op.result(0),
                         square_double_grad__op.result(1));
}

pir::OpResult square_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "square_grad");
  paddle::dialect::SquareGradOp square_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SquareGradOp>(
          x, out_grad);
  return square_grad_op.result(0);
}

pir::OpResult square_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "square_grad_");
  paddle::dialect::SquareGrad_Op square_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SquareGrad_Op>(x, out_grad);
  return square_grad__op.result(0);
}

pir::OpResult squared_l2_norm_grad(const pir::Value& x,
                                   const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "squared_l2_norm_grad");
  paddle::dialect::SquaredL2NormGradOp squared_l2_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SquaredL2NormGradOp>(x, out_grad);
  return squared_l2_norm_grad_op.result(0);
}

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           const std::vector<int64_t>& axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad");
  paddle::dialect::SqueezeGradOp squeeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGradOp>(xshape, out_grad, axis);
  return squeeze_grad_op.result(0);
}

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad");
  paddle::dialect::SqueezeGradOp squeeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGradOp>(xshape, out_grad, axis);
  return squeeze_grad_op.result(0);
}

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           std::vector<pir::Value> axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::SqueezeGradOp squeeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGradOp>(
              xshape, out_grad, axis_combine_op.out());
  return squeeze_grad_op.result(0);
}

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            const std::vector<int64_t>& axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad_");
  paddle::dialect::SqueezeGrad_Op squeeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGrad_Op>(xshape, out_grad, axis);
  return squeeze_grad__op.result(0);
}

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad_");
  paddle::dialect::SqueezeGrad_Op squeeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGrad_Op>(xshape, out_grad, axis);
  return squeeze_grad__op.result(0);
}

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            std::vector<pir::Value> axis) {
  CheckValueDataType(out_grad, "out_grad", "squeeze_grad_");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::SqueezeGrad_Op squeeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeGrad_Op>(
              xshape, out_grad, axis_combine_op.out());
  return squeeze_grad__op.result(0);
}

std::vector<pir::OpResult> stack_grad(const std::vector<pir::Value>& x,
                                      const pir::Value& out_grad,
                                      int axis) {
  CheckValueDataType(out_grad, "out_grad", "stack_grad");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::StackGradOp stack_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::StackGradOp>(
          x_combine_op.out(), out_grad, axis);
  auto x_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          stack_grad_op.result(0));
  return x_grad_split_op.outputs();
}

pir::OpResult stanh_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         float scale_a,
                         float scale_b) {
  CheckValueDataType(out_grad, "out_grad", "stanh_grad");
  paddle::dialect::StanhGradOp stanh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::StanhGradOp>(
          x, out_grad, scale_a, scale_b);
  return stanh_grad_op.result(0);
}

pir::OpResult svd_grad(const pir::Value& x,
                       const pir::Value& u,
                       const pir::Value& vh,
                       const pir::Value& s,
                       const paddle::optional<pir::Value>& u_grad,
                       const paddle::optional<pir::Value>& vh_grad,
                       const paddle::optional<pir::Value>& s_grad,
                       bool full_matrices) {
  if (s_grad) {
    CheckValueDataType(s_grad.get(), "s_grad", "svd_grad");
  } else if (vh_grad) {
    CheckValueDataType(vh_grad.get(), "vh_grad", "svd_grad");
  } else if (u_grad) {
    CheckValueDataType(u_grad.get(), "u_grad", "svd_grad");
  } else {
    CheckValueDataType(s, "s", "svd_grad");
  }
  paddle::optional<pir::Value> optional_u_grad;
  if (!u_grad) {
    optional_u_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_u_grad = u_grad;
  }
  paddle::optional<pir::Value> optional_vh_grad;
  if (!vh_grad) {
    optional_vh_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_vh_grad = vh_grad;
  }
  paddle::optional<pir::Value> optional_s_grad;
  if (!s_grad) {
    optional_s_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_s_grad = s_grad;
  }
  paddle::dialect::SvdGradOp svd_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SvdGradOp>(
          x,
          u,
          vh,
          s,
          optional_u_grad.get(),
          optional_vh_grad.get(),
          optional_s_grad.get(),
          full_matrices);
  return svd_grad_op.result(0);
}

pir::OpResult take_along_axis_grad(const pir::Value& arr,
                                   const pir::Value& indices,
                                   const pir::Value& out_grad,
                                   int axis) {
  CheckValueDataType(out_grad, "out_grad", "take_along_axis_grad");
  paddle::dialect::TakeAlongAxisGradOp take_along_axis_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TakeAlongAxisGradOp>(
              arr, indices, out_grad, axis);
  return take_along_axis_grad_op.result(0);
}

pir::OpResult tan_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tan_grad");
  paddle::dialect::TanGradOp tan_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanGradOp>(
          x, out_grad);
  return tan_grad_op.result(0);
}

pir::OpResult tan_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tan_grad_");
  paddle::dialect::TanGrad_Op tan_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanGrad_Op>(
          x, out_grad);
  return tan_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> tanh_double_grad(
    const pir::Value& out,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "tanh_double_grad");
  paddle::dialect::TanhDoubleGradOp tanh_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhDoubleGradOp>(
              out, grad_out, grad_x_grad);
  return std::make_tuple(tanh_double_grad_op.result(0),
                         tanh_double_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> tanh_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "tanh_double_grad_");
  paddle::dialect::TanhDoubleGrad_Op tanh_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhDoubleGrad_Op>(
              out, grad_out, grad_x_grad);
  return std::make_tuple(tanh_double_grad__op.result(0),
                         tanh_double_grad__op.result(1));
}

pir::OpResult tanh_grad(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tanh_grad");
  paddle::dialect::TanhGradOp tanh_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanhGradOp>(
          out, out_grad);
  return tanh_grad_op.result(0);
}

pir::OpResult tanh_grad_(const pir::Value& out, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tanh_grad_");
  paddle::dialect::TanhGrad_Op tanh_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanhGrad_Op>(
          out, out_grad);
  return tanh_grad__op.result(0);
}

pir::OpResult tanh_shrink_grad(const pir::Value& x,
                               const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tanh_shrink_grad");
  paddle::dialect::TanhShrinkGradOp tanh_shrink_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhShrinkGradOp>(x, out_grad);
  return tanh_shrink_grad_op.result(0);
}

pir::OpResult tanh_shrink_grad_(const pir::Value& x,
                                const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "tanh_shrink_grad_");
  paddle::dialect::TanhShrinkGrad_Op tanh_shrink_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhShrinkGrad_Op>(x, out_grad);
  return tanh_shrink_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> tanh_triple_grad(
    const pir::Value& out,
    const pir::Value& grad_out_forward,
    const pir::Value& grad_x_grad_forward,
    const paddle::optional<pir::Value>& grad_out_new_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "tanh_triple_grad");
  } else if (grad_out_new_grad) {
    CheckValueDataType(
        grad_out_new_grad.get(), "grad_out_new_grad", "tanh_triple_grad");
  } else {
    CheckValueDataType(
        grad_x_grad_forward, "grad_x_grad_forward", "tanh_triple_grad");
  }
  paddle::optional<pir::Value> optional_grad_out_new_grad;
  if (!grad_out_new_grad) {
    optional_grad_out_new_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_new_grad = grad_out_new_grad;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::TanhTripleGradOp tanh_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhTripleGradOp>(
              out,
              grad_out_forward,
              grad_x_grad_forward,
              optional_grad_out_new_grad.get(),
              optional_grad_out_grad_grad.get());
  return std::make_tuple(tanh_triple_grad_op.result(0),
                         tanh_triple_grad_op.result(1),
                         tanh_triple_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> tanh_triple_grad_(
    const pir::Value& out,
    const pir::Value& grad_out_forward,
    const pir::Value& grad_x_grad_forward,
    const paddle::optional<pir::Value>& grad_out_new_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad) {
  if (grad_out_grad_grad) {
    CheckValueDataType(
        grad_out_grad_grad.get(), "grad_out_grad_grad", "tanh_triple_grad_");
  } else if (grad_out_new_grad) {
    CheckValueDataType(
        grad_out_new_grad.get(), "grad_out_new_grad", "tanh_triple_grad_");
  } else {
    CheckValueDataType(
        grad_x_grad_forward, "grad_x_grad_forward", "tanh_triple_grad_");
  }
  paddle::optional<pir::Value> optional_grad_out_new_grad;
  if (!grad_out_new_grad) {
    optional_grad_out_new_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_new_grad = grad_out_new_grad;
  }
  paddle::optional<pir::Value> optional_grad_out_grad_grad;
  if (!grad_out_grad_grad) {
    optional_grad_out_grad_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_out_grad_grad = grad_out_grad_grad;
  }
  paddle::dialect::TanhTripleGrad_Op tanh_triple_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TanhTripleGrad_Op>(
              out,
              grad_out_forward,
              grad_x_grad_forward,
              optional_grad_out_new_grad.get(),
              optional_grad_out_grad_grad.get());
  return std::make_tuple(tanh_triple_grad__op.result(0),
                         tanh_triple_grad__op.result(1),
                         tanh_triple_grad__op.result(2));
}

pir::OpResult temporal_shift_grad(const pir::Value& out_grad,
                                  int seg_num,
                                  float shift_ratio,
                                  const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "temporal_shift_grad");
  paddle::dialect::TemporalShiftGradOp temporal_shift_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TemporalShiftGradOp>(
              out_grad, seg_num, shift_ratio, data_format);
  return temporal_shift_grad_op.result(0);
}

pir::OpResult tensor_unfold_grad(const pir::Value& input,
                                 const pir::Value& out_grad,
                                 int64_t axis,
                                 int64_t size,
                                 int64_t step) {
  CheckValueDataType(out_grad, "out_grad", "tensor_unfold_grad");
  paddle::dialect::TensorUnfoldGradOp tensor_unfold_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TensorUnfoldGradOp>(
              input, out_grad, axis, size, step);
  return tensor_unfold_grad_op.result(0);
}

pir::OpResult thresholded_relu_grad(const pir::Value& x,
                                    const pir::Value& out_grad,
                                    float threshold) {
  CheckValueDataType(out_grad, "out_grad", "thresholded_relu_grad");
  paddle::dialect::ThresholdedReluGradOp thresholded_relu_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ThresholdedReluGradOp>(
              x, out_grad, threshold);
  return thresholded_relu_grad_op.result(0);
}

pir::OpResult thresholded_relu_grad_(const pir::Value& x,
                                     const pir::Value& out_grad,
                                     float threshold) {
  CheckValueDataType(out_grad, "out_grad", "thresholded_relu_grad_");
  paddle::dialect::ThresholdedReluGrad_Op thresholded_relu_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ThresholdedReluGrad_Op>(
              x, out_grad, threshold);
  return thresholded_relu_grad__op.result(0);
}

pir::OpResult topk_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        int k,
                        int axis,
                        bool largest,
                        bool sorted) {
  CheckValueDataType(out_grad, "out_grad", "topk_grad");
  paddle::dialect::TopkGradOp topk_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TopkGradOp>(
          x, indices, out_grad, k, axis, largest, sorted);
  return topk_grad_op.result(0);
}

pir::OpResult topk_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        pir::Value k,
                        int axis,
                        bool largest,
                        bool sorted) {
  CheckValueDataType(out_grad, "out_grad", "topk_grad");
  paddle::dialect::TopkGradOp topk_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TopkGradOp>(
          x, indices, out_grad, k, axis, largest, sorted);
  return topk_grad_op.result(0);
}

pir::OpResult trace_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         int offset,
                         int axis1,
                         int axis2) {
  CheckValueDataType(out_grad, "out_grad", "trace_grad");
  paddle::dialect::TraceGradOp trace_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TraceGradOp>(
          x, out_grad, offset, axis1, axis2);
  return trace_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> triangular_solve_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& out_grad,
    bool upper,
    bool transpose,
    bool unitriangular) {
  CheckValueDataType(out_grad, "out_grad", "triangular_solve_grad");
  paddle::dialect::TriangularSolveGradOp triangular_solve_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TriangularSolveGradOp>(
              x, y, out, out_grad, upper, transpose, unitriangular);
  return std::make_tuple(triangular_solve_grad_op.result(0),
                         triangular_solve_grad_op.result(1));
}

pir::OpResult trilinear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode) {
  CheckValueDataType(output_grad, "output_grad", "trilinear_interp_grad");
  paddle::optional<pir::Value> optional_out_size;
  if (!out_size) {
    optional_out_size = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_size = out_size;
  }
  paddle::optional<pir::Value> optional_size_tensor;
  if (!size_tensor) {
    optional_size_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_size_tensor_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            size_tensor.get());
    optional_size_tensor = paddle::make_optional<pir::Value>(
        optional_size_tensor_combine_op.out());
  }
  paddle::optional<pir::Value> optional_scale_tensor;
  if (!scale_tensor) {
    optional_scale_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_tensor = scale_tensor;
  }
  paddle::dialect::TrilinearInterpGradOp trilinear_interp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TrilinearInterpGradOp>(
              x,
              optional_out_size.get(),
              optional_size_tensor.get(),
              optional_scale_tensor.get(),
              output_grad,
              data_layout,
              out_d,
              out_h,
              out_w,
              scale,
              interp_method,
              align_corners,
              align_mode);
  return trilinear_interp_grad_op.result(0);
}

pir::OpResult trunc_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "trunc_grad");
  paddle::dialect::TruncGradOp trunc_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TruncGradOp>(
          out_grad);
  return trunc_grad_op.result(0);
}

pir::OpResult unfold_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          const std::vector<int>& kernel_sizes,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations) {
  CheckValueDataType(out_grad, "out_grad", "unfold_grad");
  paddle::dialect::UnfoldGradOp unfold_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnfoldGradOp>(
          x, out_grad, kernel_sizes, strides, paddings, dilations);
  return unfold_grad_op.result(0);
}

pir::OpResult uniform_inplace_grad(const pir::Value& out_grad,
                                   float min,
                                   float max,
                                   int seed,
                                   int diag_num,
                                   int diag_step,
                                   float diag_val) {
  CheckValueDataType(out_grad, "out_grad", "uniform_inplace_grad");
  paddle::dialect::UniformInplaceGradOp uniform_inplace_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UniformInplaceGradOp>(
              out_grad, min, max, seed, diag_num, diag_step, diag_val);
  return uniform_inplace_grad_op.result(0);
}

pir::OpResult uniform_inplace_grad_(const pir::Value& out_grad,
                                    float min,
                                    float max,
                                    int seed,
                                    int diag_num,
                                    int diag_step,
                                    float diag_val) {
  CheckValueDataType(out_grad, "out_grad", "uniform_inplace_grad_");
  paddle::dialect::UniformInplaceGrad_Op uniform_inplace_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UniformInplaceGrad_Op>(
              out_grad, min, max, seed, diag_num, diag_step, diag_val);
  return uniform_inplace_grad__op.result(0);
}

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad");
  paddle::dialect::UnsqueezeGradOp unsqueeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGradOp>(xshape, out_grad, axis);
  return unsqueeze_grad_op.result(0);
}

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad");
  paddle::dialect::UnsqueezeGradOp unsqueeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGradOp>(xshape, out_grad, axis);
  return unsqueeze_grad_op.result(0);
}

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             std::vector<pir::Value> axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::UnsqueezeGradOp unsqueeze_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGradOp>(
              xshape, out_grad, axis_combine_op.out());
  return unsqueeze_grad_op.result(0);
}

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad_");
  paddle::dialect::UnsqueezeGrad_Op unsqueeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGrad_Op>(xshape, out_grad, axis);
  return unsqueeze_grad__op.result(0);
}

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              pir::Value axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad_");
  paddle::dialect::UnsqueezeGrad_Op unsqueeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGrad_Op>(xshape, out_grad, axis);
  return unsqueeze_grad__op.result(0);
}

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              std::vector<pir::Value> axis) {
  CheckValueDataType(out_grad, "out_grad", "unsqueeze_grad_");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::UnsqueezeGrad_Op unsqueeze_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnsqueezeGrad_Op>(
              xshape, out_grad, axis_combine_op.out());
  return unsqueeze_grad__op.result(0);
}

pir::OpResult unstack_grad(const std::vector<pir::Value>& out_grad, int axis) {
  CheckVectorOfValueDataType(out_grad, "out_grad", "unstack_grad");
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::UnstackGradOp unstack_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::UnstackGradOp>(out_grad_combine_op.out(),
                                                  axis);
  return unstack_grad_op.result(0);
}

pir::OpResult view_dtype_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              phi::DataType dtype) {
  CheckValueDataType(out_grad, "out_grad", "view_dtype_grad");
  paddle::dialect::ViewDtypeGradOp view_dtype_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ViewDtypeGradOp>(input, out_grad, dtype);
  return view_dtype_grad_op.result(0);
}

pir::OpResult view_shape_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& dims) {
  CheckValueDataType(out_grad, "out_grad", "view_shape_grad");
  paddle::dialect::ViewShapeGradOp view_shape_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ViewShapeGradOp>(input, out_grad, dims);
  return view_shape_grad_op.result(0);
}

pir::OpResult warpctc_grad(const pir::Value& logits,
                           const paddle::optional<pir::Value>& logits_length,
                           const pir::Value& warpctcgrad,
                           const pir::Value& loss_grad,
                           int blank,
                           bool norm_by_times) {
  CheckValueDataType(loss_grad, "loss_grad", "warpctc_grad");
  paddle::optional<pir::Value> optional_logits_length;
  if (!logits_length) {
    optional_logits_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_logits_length = logits_length;
  }
  paddle::dialect::WarpctcGradOp warpctc_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WarpctcGradOp>(logits,
                                                  optional_logits_length.get(),
                                                  warpctcgrad,
                                                  loss_grad,
                                                  blank,
                                                  norm_by_times);
  return warpctc_grad_op.result(0);
}

pir::OpResult warprnnt_grad(const pir::Value& input,
                            const pir::Value& input_lengths,
                            const pir::Value& warprnntgrad,
                            const pir::Value& loss_grad,
                            int blank,
                            float fastemit_lambda) {
  CheckValueDataType(loss_grad, "loss_grad", "warprnnt_grad");
  paddle::dialect::WarprnntGradOp warprnnt_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WarprnntGradOp>(input,
                                                   input_lengths,
                                                   warprnntgrad,
                                                   loss_grad,
                                                   blank,
                                                   fastemit_lambda);
  return warprnnt_grad_op.result(0);
}

pir::OpResult weight_only_linear_grad(const pir::Value& x,
                                      const pir::Value& weight,
                                      const paddle::optional<pir::Value>& bias,
                                      const pir::Value& weight_scale,
                                      const pir::Value& out_grad,
                                      const std::string& weight_dtype,
                                      int arch,
                                      int group_size) {
  CheckValueDataType(out_grad, "out_grad", "weight_only_linear_grad");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::WeightOnlyLinearGradOp weight_only_linear_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::WeightOnlyLinearGradOp>(x,
                                                           weight,
                                                           optional_bias.get(),
                                                           weight_scale,
                                                           out_grad,
                                                           weight_dtype,
                                                           arch,
                                                           group_size);
  return weight_only_linear_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> where_grad(
    const pir::Value& condition,
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "where_grad");
  paddle::dialect::WhereGradOp where_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::WhereGradOp>(
          condition, x, y, out_grad);
  return std::make_tuple(where_grad_op.result(0), where_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
yolo_loss_grad(const pir::Value& x,
               const pir::Value& gt_box,
               const pir::Value& gt_label,
               const paddle::optional<pir::Value>& gt_score,
               const pir::Value& objectness_mask,
               const pir::Value& gt_match_mask,
               const pir::Value& loss_grad,
               const std::vector<int>& anchors,
               const std::vector<int>& anchor_mask,
               int class_num,
               float ignore_thresh,
               int downsample_ratio,
               bool use_label_smooth,
               float scale_x_y) {
  CheckValueDataType(loss_grad, "loss_grad", "yolo_loss_grad");
  paddle::optional<pir::Value> optional_gt_score;
  if (!gt_score) {
    optional_gt_score = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_gt_score = gt_score;
  }
  paddle::dialect::YoloLossGradOp yolo_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::YoloLossGradOp>(x,
                                                   gt_box,
                                                   gt_label,
                                                   optional_gt_score.get(),
                                                   objectness_mask,
                                                   gt_match_mask,
                                                   loss_grad,
                                                   anchors,
                                                   anchor_mask,
                                                   class_num,
                                                   ignore_thresh,
                                                   downsample_ratio,
                                                   use_label_smooth,
                                                   scale_x_y);
  return std::make_tuple(yolo_loss_grad_op.result(0),
                         yolo_loss_grad_op.result(1),
                         yolo_loss_grad_op.result(2),
                         yolo_loss_grad_op.result(3));
}

pir::OpResult unpool3d_grad(const pir::Value& x,
                            const pir::Value& indices,
                            const pir::Value& out,
                            const pir::Value& out_grad,
                            const std::vector<int>& ksize,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_size,
                            const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool3d_grad");
  paddle::dialect::Unpool3dGradOp unpool3d_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Unpool3dGradOp>(x,
                                                   indices,
                                                   out,
                                                   out_grad,
                                                   ksize,
                                                   strides,
                                                   paddings,
                                                   output_size,
                                                   data_format);
  return unpool3d_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> add_act_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& y,
    const paddle::optional<pir::Value>& y_max,
    int act_type) {
  CheckValueDataType(x, "x", "add_act_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::optional<pir::Value> optional_y_max;
  if (!y_max) {
    optional_y_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_y_max = y_max;
  }
  paddle::dialect::AddActXpuOp add_act_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddActXpuOp>(
          x, optional_x_max.get(), y, optional_y_max.get(), act_type);
  return std::make_tuple(add_act_xpu_op.result(0), add_act_xpu_op.result(1));
}

pir::OpResult add_layernorm_xpu(const pir::Value& x,
                                const pir::Value& y,
                                const pir::Value& scale,
                                const pir::Value& bias,
                                int begin_norm_axis,
                                float epsilon) {
  CheckValueDataType(x, "x", "add_layernorm_xpu");
  paddle::dialect::AddLayernormXpuOp add_layernorm_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddLayernormXpuOp>(
              x, y, scale, bias, begin_norm_axis, epsilon);
  return add_layernorm_xpu_op.result(0);
}

pir::OpResult addcmul_xpu(const pir::Value& x,
                          const pir::Value& y,
                          const pir::Value& w) {
  CheckValueDataType(x, "x", "addcmul_xpu");
  paddle::dialect::AddcmulXpuOp addcmul_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddcmulXpuOp>(
          x, y, w);
  return addcmul_xpu_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
block_multihead_attention_(
    const pir::Value& qkv,
    const pir::Value& key_cache,
    const pir::Value& value_cache,
    const pir::Value& seq_lens_encoder,
    const pir::Value& seq_lens_decoder,
    const pir::Value& seq_lens_this_time,
    const pir::Value& padding_offsets,
    const pir::Value& cum_offsets,
    const pir::Value& cu_seqlens_q,
    const pir::Value& cu_seqlens_k,
    const pir::Value& block_tables,
    const paddle::optional<pir::Value>& pre_key_cache,
    const paddle::optional<pir::Value>& pre_value_cache,
    const paddle::optional<pir::Value>& rope_emb,
    const paddle::optional<pir::Value>& mask,
    const paddle::optional<pir::Value>& tgt_mask,
    const paddle::optional<pir::Value>& cache_k_quant_scales,
    const paddle::optional<pir::Value>& cache_v_quant_scales,
    const paddle::optional<pir::Value>& cache_k_dequant_scales,
    const paddle::optional<pir::Value>& cache_v_dequant_scales,
    const paddle::optional<pir::Value>& qkv_out_scale,
    const paddle::optional<pir::Value>& qkv_bias,
    const paddle::optional<pir::Value>& out_shift,
    const paddle::optional<pir::Value>& out_smooth,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    bool dynamic_cachekv_quant,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound,
    float out_scale,
    const std::string& compute_dtype) {
  CheckValueDataType(qkv, "qkv", "block_multihead_attention_");
  paddle::optional<pir::Value> optional_pre_key_cache;
  if (!pre_key_cache) {
    optional_pre_key_cache = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pre_key_cache = pre_key_cache;
  }
  paddle::optional<pir::Value> optional_pre_value_cache;
  if (!pre_value_cache) {
    optional_pre_value_cache = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_pre_value_cache = pre_value_cache;
  }
  paddle::optional<pir::Value> optional_rope_emb;
  if (!rope_emb) {
    optional_rope_emb = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rope_emb = rope_emb;
  }
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  paddle::optional<pir::Value> optional_tgt_mask;
  if (!tgt_mask) {
    optional_tgt_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_tgt_mask = tgt_mask;
  }
  paddle::optional<pir::Value> optional_cache_k_quant_scales;
  if (!cache_k_quant_scales) {
    optional_cache_k_quant_scales =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cache_k_quant_scales = cache_k_quant_scales;
  }
  paddle::optional<pir::Value> optional_cache_v_quant_scales;
  if (!cache_v_quant_scales) {
    optional_cache_v_quant_scales =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cache_v_quant_scales = cache_v_quant_scales;
  }
  paddle::optional<pir::Value> optional_cache_k_dequant_scales;
  if (!cache_k_dequant_scales) {
    optional_cache_k_dequant_scales =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cache_k_dequant_scales = cache_k_dequant_scales;
  }
  paddle::optional<pir::Value> optional_cache_v_dequant_scales;
  if (!cache_v_dequant_scales) {
    optional_cache_v_dequant_scales =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cache_v_dequant_scales = cache_v_dequant_scales;
  }
  paddle::optional<pir::Value> optional_qkv_out_scale;
  if (!qkv_out_scale) {
    optional_qkv_out_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_out_scale = qkv_out_scale;
  }
  paddle::optional<pir::Value> optional_qkv_bias;
  if (!qkv_bias) {
    optional_qkv_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_bias = qkv_bias;
  }
  paddle::optional<pir::Value> optional_out_shift;
  if (!out_shift) {
    optional_out_shift = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_shift = out_shift;
  }
  paddle::optional<pir::Value> optional_out_smooth;
  if (!out_smooth) {
    optional_out_smooth = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_smooth = out_smooth;
  }
  paddle::dialect::BlockMultiheadAttention_Op block_multihead_attention__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BlockMultiheadAttention_Op>(
              qkv,
              key_cache,
              value_cache,
              seq_lens_encoder,
              seq_lens_decoder,
              seq_lens_this_time,
              padding_offsets,
              cum_offsets,
              cu_seqlens_q,
              cu_seqlens_k,
              block_tables,
              optional_pre_key_cache.get(),
              optional_pre_value_cache.get(),
              optional_rope_emb.get(),
              optional_mask.get(),
              optional_tgt_mask.get(),
              optional_cache_k_quant_scales.get(),
              optional_cache_v_quant_scales.get(),
              optional_cache_k_dequant_scales.get(),
              optional_cache_v_dequant_scales.get(),
              optional_qkv_out_scale.get(),
              optional_qkv_bias.get(),
              optional_out_shift.get(),
              optional_out_smooth.get(),
              max_seq_len,
              block_size,
              use_neox_style,
              dynamic_cachekv_quant,
              quant_round_type,
              quant_max_bound,
              quant_min_bound,
              out_scale,
              compute_dtype);
  return std::make_tuple(block_multihead_attention__op.result(0),
                         block_multihead_attention__op.result(1),
                         block_multihead_attention__op.result(2),
                         block_multihead_attention__op.result(3));
}

pir::OpResult bn_act_xpu(const pir::Value& x,
                         const pir::Value& mean,
                         const pir::Value& variance,
                         const pir::Value& scale,
                         const pir::Value& bias,
                         float momentum,
                         float epsilon,
                         const std::string& data_layout,
                         int act_type) {
  CheckValueDataType(x, "x", "bn_act_xpu");
  paddle::dialect::BnActXpuOp bn_act_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BnActXpuOp>(
          x,
          mean,
          variance,
          scale,
          bias,
          momentum,
          epsilon,
          data_layout,
          act_type);
  return bn_act_xpu_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> conv1d_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const paddle::optional<pir::Value>& branch_max,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int dilations,
    int strides,
    int groups,
    int act_type,
    float act_param) {
  CheckValueDataType(x, "x", "conv1d_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_branch;
  if (!branch) {
    optional_branch = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_branch = branch;
  }
  paddle::optional<pir::Value> optional_branch_max;
  if (!branch_max) {
    optional_branch_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_branch_max = branch_max;
  }
  paddle::dialect::Conv1dXpuOp conv1d_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv1dXpuOp>(
          x,
          optional_x_max.get(),
          filter,
          filter_max,
          optional_bias.get(),
          optional_branch.get(),
          optional_branch_max.get(),
          paddings,
          padding_algorithm,
          dilations,
          strides,
          groups,
          act_type,
          act_param);
  return std::make_tuple(conv1d_xpu_op.result(0), conv1d_xpu_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    bool has_bias,
    bool with_act,
    const std::string& act_type) {
  CheckValueDataType(x, "x", "conv2d_transpose_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::Conv2dTransposeXpuOp conv2d_transpose_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeXpuOp>(x,
                                                         optional_x_max.get(),
                                                         filter,
                                                         filter_max,
                                                         optional_bias.get(),
                                                         strides,
                                                         paddings,
                                                         output_padding,
                                                         output_size,
                                                         padding_algorithm,
                                                         groups,
                                                         dilations,
                                                         data_format,
                                                         has_bias,
                                                         with_act,
                                                         act_type);
  return std::make_tuple(conv2d_transpose_xpu_op.result(0),
                         conv2d_transpose_xpu_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const paddle::optional<pir::Value>& branch_max,
    const paddle::optional<pir::Value>& scale_max,
    const paddle::optional<pir::Value>& out_max_in,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    int groups,
    int act_type,
    float act_param,
    phi::DataType out_dtype) {
  CheckValueDataType(x, "x", "conv2d_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_branch;
  if (!branch) {
    optional_branch = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_branch = branch;
  }
  paddle::optional<pir::Value> optional_branch_max;
  if (!branch_max) {
    optional_branch_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_branch_max = branch_max;
  }
  paddle::optional<pir::Value> optional_scale_max;
  if (!scale_max) {
    optional_scale_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_max = scale_max;
  }
  paddle::optional<pir::Value> optional_out_max_in;
  if (!out_max_in) {
    optional_out_max_in = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_max_in = out_max_in;
  }
  paddle::dialect::Conv2dXpuOp conv2d_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Conv2dXpuOp>(
          x,
          optional_x_max.get(),
          filter,
          filter_max,
          optional_bias.get(),
          optional_branch.get(),
          optional_branch_max.get(),
          optional_scale_max.get(),
          optional_out_max_in.get(),
          paddings,
          dilations,
          strides,
          padding_algorithm,
          groups,
          act_type,
          act_param,
          out_dtype);
  return std::make_tuple(conv2d_xpu_op.result(0), conv2d_xpu_op.result(1));
}

pir::OpResult dequantize_xpu(const pir::Value& x,
                             phi::DataType out_dtype,
                             float scale) {
  CheckValueDataType(x, "x", "dequantize_xpu");
  paddle::dialect::DequantizeXpuOp dequantize_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DequantizeXpuOp>(x, out_dtype, scale);
  return dequantize_xpu_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
embedding_with_eltwise_add_xpu(const std::vector<pir::Value>& ids,
                               const std::vector<pir::Value>& tables,
                               const paddle::optional<pir::Value>& mask,
                               int64_t padding_idx) {
  CheckVectorOfValueDataType(
      tables, "tables", "embedding_with_eltwise_add_xpu");
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  auto ids_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ids);
  auto tables_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(tables);
  paddle::dialect::EmbeddingWithEltwiseAddXpuOp
      embedding_with_eltwise_add_xpu_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::EmbeddingWithEltwiseAddXpuOp>(
                  ids_combine_op.out(),
                  tables_combine_op.out(),
                  optional_mask.get(),
                  padding_idx);
  return std::make_tuple(embedding_with_eltwise_add_xpu_op.result(0),
                         embedding_with_eltwise_add_xpu_op.result(1),
                         embedding_with_eltwise_add_xpu_op.result(2));
}

pir::OpResult fast_layernorm_xpu(const pir::Value& x,
                                 const pir::Value& scale,
                                 const pir::Value& bias,
                                 int begin_norm_axis,
                                 float epsilon) {
  CheckValueDataType(x, "x", "fast_layernorm_xpu");
  paddle::dialect::FastLayernormXpuOp fast_layernorm_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FastLayernormXpuOp>(
              x, scale, bias, begin_norm_axis, epsilon);
  return fast_layernorm_xpu_op.result(0);
}

pir::OpResult fast_where_xpu(const pir::Value& condition,
                             const pir::Value& x,
                             const pir::Value& y) {
  CheckValueDataType(x, "x", "fast_where_xpu");
  paddle::dialect::FastWhereXpuOp fast_where_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FastWhereXpuOp>(condition, x, y);
  return fast_where_xpu_op.result(0);
}

pir::OpResult fc(const pir::Value& input,
                 const pir::Value& w,
                 const paddle::optional<pir::Value>& bias,
                 int in_num_col_dims,
                 const std::string& activation_type,
                 bool padding_weights) {
  CheckValueDataType(input, "input", "fc");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::FcOp fc_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FcOp>(
          input,
          w,
          optional_bias.get(),
          in_num_col_dims,
          activation_type,
          padding_weights);
  return fc_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> fc_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& w,
    const pir::Value& w_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& scale_max,
    const paddle::optional<pir::Value>& out_max_in,
    int in_num_col_dims,
    bool transpose_x,
    float alpha,
    float beta,
    int act_type,
    float act_alpha,
    phi::DataType out_dtype) {
  CheckValueDataType(x, "x", "fc_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_scale_max;
  if (!scale_max) {
    optional_scale_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale_max = scale_max;
  }
  paddle::optional<pir::Value> optional_out_max_in;
  if (!out_max_in) {
    optional_out_max_in = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_max_in = out_max_in;
  }
  paddle::dialect::FcXpuOp fc_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FcXpuOp>(
          x,
          optional_x_max.get(),
          w,
          w_max,
          optional_bias.get(),
          optional_scale_max.get(),
          optional_out_max_in.get(),
          in_num_col_dims,
          transpose_x,
          alpha,
          beta,
          act_type,
          act_alpha,
          out_dtype);
  return std::make_tuple(fc_xpu_op.result(0), fc_xpu_op.result(1));
}

pir::OpResult fused_bias_act(const pir::Value& x,
                             const paddle::optional<pir::Value>& bias,
                             const paddle::optional<pir::Value>& dequant_scales,
                             const paddle::optional<pir::Value>& shift,
                             const paddle::optional<pir::Value>& smooth,
                             const std::string& act_method,
                             const std::string& compute_dtype,
                             float quant_scale,
                             int quant_round_type,
                             float quant_max_bound,
                             float quant_min_bound) {
  CheckValueDataType(x, "x", "fused_bias_act");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_dequant_scales;
  if (!dequant_scales) {
    optional_dequant_scales = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dequant_scales = dequant_scales;
  }
  paddle::optional<pir::Value> optional_shift;
  if (!shift) {
    optional_shift = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_shift = shift;
  }
  paddle::optional<pir::Value> optional_smooth;
  if (!smooth) {
    optional_smooth = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_smooth = smooth;
  }
  paddle::dialect::FusedBiasActOp fused_bias_act_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBiasActOp>(
              x,
              optional_bias.get(),
              optional_dequant_scales.get(),
              optional_shift.get(),
              optional_smooth.get(),
              act_method,
              compute_dtype,
              quant_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
  return fused_bias_act_op.result(0);
}

pir::OpResult fused_bias_dropout_residual_layer_norm(
    const pir::Value& x,
    const pir::Value& residual,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    float dropout_rate,
    bool is_test,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string& dropout_implementation,
    float ln_epsilon) {
  CheckValueDataType(x, "x", "fused_bias_dropout_residual_layer_norm");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_ln_scale;
  if (!ln_scale) {
    optional_ln_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale = ln_scale;
  }
  paddle::optional<pir::Value> optional_ln_bias;
  if (!ln_bias) {
    optional_ln_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias = ln_bias;
  }
  paddle::dialect::FusedBiasDropoutResidualLayerNormOp
      fused_bias_dropout_residual_layer_norm_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedBiasDropoutResidualLayerNormOp>(
                  x,
                  residual,
                  optional_bias.get(),
                  optional_ln_scale.get(),
                  optional_ln_bias.get(),
                  dropout_rate,
                  is_test,
                  dropout_fix_seed,
                  dropout_seed,
                  dropout_implementation,
                  ln_epsilon);
  return fused_bias_dropout_residual_layer_norm_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
fused_bias_residual_layernorm(const pir::Value& x,
                              const paddle::optional<pir::Value>& bias,
                              const paddle::optional<pir::Value>& residual,
                              const paddle::optional<pir::Value>& norm_weight,
                              const paddle::optional<pir::Value>& norm_bias,
                              float epsilon,
                              float residual_alpha,
                              int begin_norm_axis,
                              float quant_scale,
                              int quant_round_type,
                              float quant_max_bound,
                              float quant_min_bound) {
  CheckValueDataType(x, "x", "fused_bias_residual_layernorm");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_residual;
  if (!residual) {
    optional_residual = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_residual = residual;
  }
  paddle::optional<pir::Value> optional_norm_weight;
  if (!norm_weight) {
    optional_norm_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_norm_weight = norm_weight;
  }
  paddle::optional<pir::Value> optional_norm_bias;
  if (!norm_bias) {
    optional_norm_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_norm_bias = norm_bias;
  }
  paddle::dialect::FusedBiasResidualLayernormOp
      fused_bias_residual_layernorm_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedBiasResidualLayernormOp>(
                  x,
                  optional_bias.get(),
                  optional_residual.get(),
                  optional_norm_weight.get(),
                  optional_norm_bias.get(),
                  epsilon,
                  residual_alpha,
                  begin_norm_axis,
                  quant_scale,
                  quant_round_type,
                  quant_max_bound,
                  quant_min_bound);
  return std::make_tuple(fused_bias_residual_layernorm_op.result(0),
                         fused_bias_residual_layernorm_op.result(1),
                         fused_bias_residual_layernorm_op.result(2),
                         fused_bias_residual_layernorm_op.result(3));
}

std::tuple<pir::OpResult, std::vector<pir::OpResult>> fused_conv2d_add_act(
    const pir::Value& input,
    const pir::Value& filter,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& residual_data,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha) {
  CheckValueDataType(input, "input", "fused_conv2d_add_act");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_residual_data;
  if (!residual_data) {
    optional_residual_data = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_residual_data = residual_data;
  }
  paddle::dialect::FusedConv2dAddActOp fused_conv2d_add_act_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedConv2dAddActOp>(
              input,
              filter,
              optional_bias.get(),
              optional_residual_data.get(),
              strides,
              paddings,
              padding_algorithm,
              dilations,
              groups,
              data_format,
              activation,
              split_channels,
              exhaustive_search,
              workspace_size_MB,
              fuse_alpha);
  auto outputs_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_conv2d_add_act_op.result(1));
  return std::make_tuple(fused_conv2d_add_act_op.result(0),
                         outputs_split_op.outputs());
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_dconv_drelu_dbn(const pir::Value& grad_output,
                      const pir::Value& weight,
                      const paddle::optional<pir::Value>& grad_output_add,
                      const paddle::optional<pir::Value>& residual_input,
                      const paddle::optional<pir::Value>& bn1_eqscale,
                      const paddle::optional<pir::Value>& bn1_eqbias,
                      const paddle::optional<pir::Value>& conv_input,
                      const pir::Value& bn1_mean,
                      const pir::Value& bn1_inv_std,
                      const pir::Value& bn1_gamma,
                      const pir::Value& bn1_beta,
                      const pir::Value& bn1_input,
                      const paddle::optional<pir::Value>& bn2_mean,
                      const paddle::optional<pir::Value>& bn2_inv_std,
                      const paddle::optional<pir::Value>& bn2_gamma,
                      const paddle::optional<pir::Value>& bn2_beta,
                      const paddle::optional<pir::Value>& bn2_input,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::string& data_format,
                      bool fuse_shortcut,
                      bool fuse_dual,
                      bool fuse_add,
                      bool exhaustive_search) {
  CheckValueDataType(grad_output, "grad_output", "fused_dconv_drelu_dbn");
  paddle::optional<pir::Value> optional_grad_output_add;
  if (!grad_output_add) {
    optional_grad_output_add = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_output_add = grad_output_add;
  }
  paddle::optional<pir::Value> optional_residual_input;
  if (!residual_input) {
    optional_residual_input = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_residual_input = residual_input;
  }
  paddle::optional<pir::Value> optional_bn1_eqscale;
  if (!bn1_eqscale) {
    optional_bn1_eqscale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn1_eqscale = bn1_eqscale;
  }
  paddle::optional<pir::Value> optional_bn1_eqbias;
  if (!bn1_eqbias) {
    optional_bn1_eqbias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn1_eqbias = bn1_eqbias;
  }
  paddle::optional<pir::Value> optional_conv_input;
  if (!conv_input) {
    optional_conv_input = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_conv_input = conv_input;
  }
  paddle::optional<pir::Value> optional_bn2_mean;
  if (!bn2_mean) {
    optional_bn2_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn2_mean = bn2_mean;
  }
  paddle::optional<pir::Value> optional_bn2_inv_std;
  if (!bn2_inv_std) {
    optional_bn2_inv_std = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn2_inv_std = bn2_inv_std;
  }
  paddle::optional<pir::Value> optional_bn2_gamma;
  if (!bn2_gamma) {
    optional_bn2_gamma = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn2_gamma = bn2_gamma;
  }
  paddle::optional<pir::Value> optional_bn2_beta;
  if (!bn2_beta) {
    optional_bn2_beta = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn2_beta = bn2_beta;
  }
  paddle::optional<pir::Value> optional_bn2_input;
  if (!bn2_input) {
    optional_bn2_input = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bn2_input = bn2_input;
  }
  paddle::dialect::FusedDconvDreluDbnOp fused_dconv_drelu_dbn_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedDconvDreluDbnOp>(
              grad_output,
              weight,
              optional_grad_output_add.get(),
              optional_residual_input.get(),
              optional_bn1_eqscale.get(),
              optional_bn1_eqbias.get(),
              optional_conv_input.get(),
              bn1_mean,
              bn1_inv_std,
              bn1_gamma,
              bn1_beta,
              bn1_input,
              optional_bn2_mean.get(),
              optional_bn2_inv_std.get(),
              optional_bn2_gamma.get(),
              optional_bn2_beta.get(),
              optional_bn2_input.get(),
              paddings,
              dilations,
              strides,
              padding_algorithm,
              groups,
              data_format,
              fuse_shortcut,
              fuse_dual,
              fuse_add,
              exhaustive_search);
  return std::make_tuple(fused_dconv_drelu_dbn_op.result(0),
                         fused_dconv_drelu_dbn_op.result(1),
                         fused_dconv_drelu_dbn_op.result(2),
                         fused_dconv_drelu_dbn_op.result(3),
                         fused_dconv_drelu_dbn_op.result(4),
                         fused_dconv_drelu_dbn_op.result(5),
                         fused_dconv_drelu_dbn_op.result(6));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_dot_product_attention(const pir::Value& q,
                            const pir::Value& k,
                            const pir::Value& v,
                            const pir::Value& mask,
                            float scaling_factor,
                            float dropout_probability,
                            bool is_training,
                            bool is_causal_masking) {
  CheckValueDataType(q, "q", "fused_dot_product_attention");
  paddle::dialect::FusedDotProductAttentionOp fused_dot_product_attention_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedDotProductAttentionOp>(
              q,
              k,
              v,
              mask,
              scaling_factor,
              dropout_probability,
              is_training,
              is_causal_masking);
  return std::make_tuple(fused_dot_product_attention_op.result(0),
                         fused_dot_product_attention_op.result(1),
                         fused_dot_product_attention_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> fused_dropout_add(
    const pir::Value& x,
    const pir::Value& y,
    const paddle::optional<pir::Value>& seed_tensor,
    float p,
    bool is_test,
    const std::string& mode,
    int seed,
    bool fix_seed) {
  CheckValueDataType(x, "x", "fused_dropout_add");
  paddle::optional<pir::Value> optional_seed_tensor;
  if (!seed_tensor) {
    optional_seed_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seed_tensor = seed_tensor;
  }
  paddle::dialect::FusedDropoutAddOp fused_dropout_add_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedDropoutAddOp>(
              x,
              y,
              optional_seed_tensor.get(),
              p,
              is_test,
              mode,
              seed,
              fix_seed);
  return std::make_tuple(fused_dropout_add_op.result(0),
                         fused_dropout_add_op.result(1));
}

pir::OpResult fused_embedding_eltwise_layernorm(
    const std::vector<pir::Value>& ids,
    const std::vector<pir::Value>& embs,
    const pir::Value& bias,
    const pir::Value& scale,
    float epsilon) {
  CheckVectorOfValueDataType(embs, "embs", "fused_embedding_eltwise_layernorm");
  auto ids_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ids);
  auto embs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(embs);
  paddle::dialect::FusedEmbeddingEltwiseLayernormOp
      fused_embedding_eltwise_layernorm_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedEmbeddingEltwiseLayernormOp>(
                  ids_combine_op.out(),
                  embs_combine_op.out(),
                  bias,
                  scale,
                  epsilon);
  return fused_embedding_eltwise_layernorm_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_fc_elementwise_layernorm(const pir::Value& x,
                               const pir::Value& w,
                               const pir::Value& y,
                               const paddle::optional<pir::Value>& bias0,
                               const paddle::optional<pir::Value>& scale,
                               const paddle::optional<pir::Value>& bias1,
                               int x_num_col_dims,
                               const std::string& activation_type,
                               float epsilon,
                               int begin_norm_axis) {
  CheckValueDataType(x, "x", "fused_fc_elementwise_layernorm");
  paddle::optional<pir::Value> optional_bias0;
  if (!bias0) {
    optional_bias0 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias0 = bias0;
  }
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias1;
  if (!bias1) {
    optional_bias1 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias1 = bias1;
  }
  paddle::dialect::FusedFcElementwiseLayernormOp
      fused_fc_elementwise_layernorm_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedFcElementwiseLayernormOp>(
                  x,
                  w,
                  y,
                  optional_bias0.get(),
                  optional_scale.get(),
                  optional_bias1.get(),
                  x_num_col_dims,
                  activation_type,
                  epsilon,
                  begin_norm_axis);
  return std::make_tuple(fused_fc_elementwise_layernorm_op.result(0),
                         fused_fc_elementwise_layernorm_op.result(1),
                         fused_fc_elementwise_layernorm_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> fused_linear_param_grad_add(
    const pir::Value& x,
    const pir::Value& dout,
    const paddle::optional<pir::Value>& dweight,
    const paddle::optional<pir::Value>& dbias,
    bool multi_precision,
    bool has_bias) {
  CheckValueDataType(dout, "dout", "fused_linear_param_grad_add");
  paddle::optional<pir::Value> optional_dweight;
  if (!dweight) {
    optional_dweight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dweight = dweight;
  }
  paddle::optional<pir::Value> optional_dbias;
  if (!dbias) {
    optional_dbias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dbias = dbias;
  }
  paddle::dialect::FusedLinearParamGradAddOp fused_linear_param_grad_add_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedLinearParamGradAddOp>(
              x,
              dout,
              optional_dweight.get(),
              optional_dbias.get(),
              multi_precision,
              has_bias);
  return std::make_tuple(fused_linear_param_grad_add_op.result(0),
                         fused_linear_param_grad_add_op.result(1));
}

std::tuple<pir::OpResult, std::vector<pir::OpResult>>
fused_multi_transformer_int8_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const std::vector<pir::Value>& qkv_in_max,
    const std::vector<pir::Value>& qkvw,
    const std::vector<pir::Value>& qkv_bias,
    const std::vector<pir::Value>& qkv_scales,
    const std::vector<pir::Value>& out_linear_in_max,
    const std::vector<pir::Value>& out_linear_w,
    const std::vector<pir::Value>& out_linear_bias,
    const std::vector<pir::Value>& out_linear_scales,
    const std::vector<pir::Value>& ffn_ln_scale,
    const std::vector<pir::Value>& ffn_ln_bias,
    const std::vector<pir::Value>& ffn1_in_max,
    const std::vector<pir::Value>& ffn1_weight,
    const std::vector<pir::Value>& ffn1_bias,
    const std::vector<pir::Value>& ffn1_scales,
    const std::vector<pir::Value>& ffn2_in_max,
    const std::vector<pir::Value>& ffn2_weight,
    const std::vector<pir::Value>& ffn2_bias,
    const std::vector<pir::Value>& ffn2_scales,
    const paddle::optional<std::vector<pir::Value>>& cache_kv,
    const paddle::optional<std::vector<pir::Value>>& pre_caches,
    const paddle::optional<pir::Value>& rotary_pos_emb,
    const paddle::optional<pir::Value>& time_step,
    const paddle::optional<pir::Value>& seq_lengths,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& gather_index,
    const pir::Value& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis) {
  CheckValueDataType(x, "x", "fused_multi_transformer_int8_xpu");
  paddle::optional<pir::Value> optional_cache_kv;
  if (!cache_kv) {
    optional_cache_kv = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_cache_kv_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            cache_kv.get());
    optional_cache_kv =
        paddle::make_optional<pir::Value>(optional_cache_kv_combine_op.out());
  }
  paddle::optional<pir::Value> optional_pre_caches;
  if (!pre_caches) {
    optional_pre_caches = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_pre_caches_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            pre_caches.get());
    optional_pre_caches =
        paddle::make_optional<pir::Value>(optional_pre_caches_combine_op.out());
  }
  paddle::optional<pir::Value> optional_rotary_pos_emb;
  if (!rotary_pos_emb) {
    optional_rotary_pos_emb = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rotary_pos_emb = rotary_pos_emb;
  }
  paddle::optional<pir::Value> optional_time_step;
  if (!time_step) {
    optional_time_step = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_time_step = time_step;
  }
  paddle::optional<pir::Value> optional_seq_lengths;
  if (!seq_lengths) {
    optional_seq_lengths = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seq_lengths = seq_lengths;
  }
  paddle::optional<pir::Value> optional_src_mask;
  if (!src_mask) {
    optional_src_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask = src_mask;
  }
  paddle::optional<pir::Value> optional_gather_index;
  if (!gather_index) {
    optional_gather_index = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_gather_index = gather_index;
  }
  auto ln_scale_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_scale);
  auto ln_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_bias);
  auto qkv_in_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkv_in_max);
  auto qkvw_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkvw);
  auto qkv_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkv_bias);
  auto qkv_scales_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkv_scales);
  auto out_linear_in_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          out_linear_in_max);
  auto out_linear_w_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_linear_w);
  auto out_linear_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          out_linear_bias);
  auto out_linear_scales_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          out_linear_scales);
  auto ffn_ln_scale_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn_ln_scale);
  auto ffn_ln_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn_ln_bias);
  auto ffn1_in_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_in_max);
  auto ffn1_weight_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_weight);
  auto ffn1_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_bias);
  auto ffn1_scales_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_scales);
  auto ffn2_in_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_in_max);
  auto ffn2_weight_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_weight);
  auto ffn2_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_bias);
  auto ffn2_scales_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_scales);
  paddle::dialect::FusedMultiTransformerInt8XpuOp
      fused_multi_transformer_int8_xpu_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedMultiTransformerInt8XpuOp>(
                  x,
                  ln_scale_combine_op.out(),
                  ln_bias_combine_op.out(),
                  qkv_in_max_combine_op.out(),
                  qkvw_combine_op.out(),
                  qkv_bias_combine_op.out(),
                  qkv_scales_combine_op.out(),
                  out_linear_in_max_combine_op.out(),
                  out_linear_w_combine_op.out(),
                  out_linear_bias_combine_op.out(),
                  out_linear_scales_combine_op.out(),
                  ffn_ln_scale_combine_op.out(),
                  ffn_ln_bias_combine_op.out(),
                  ffn1_in_max_combine_op.out(),
                  ffn1_weight_combine_op.out(),
                  ffn1_bias_combine_op.out(),
                  ffn1_scales_combine_op.out(),
                  ffn2_in_max_combine_op.out(),
                  ffn2_weight_combine_op.out(),
                  ffn2_bias_combine_op.out(),
                  ffn2_scales_combine_op.out(),
                  optional_cache_kv.get(),
                  optional_pre_caches.get(),
                  optional_rotary_pos_emb.get(),
                  optional_time_step.get(),
                  optional_seq_lengths.get(),
                  optional_src_mask.get(),
                  optional_gather_index.get(),
                  max_buffer,
                  pre_layer_norm,
                  rotary_emb_dims,
                  epsilon,
                  dropout_rate,
                  is_test,
                  dropout_implementation,
                  act_method,
                  trans_qkvw,
                  ring_id,
                  gather_axis);
  auto cache_kv_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_multi_transformer_int8_xpu_op.result(1));
  return std::make_tuple(fused_multi_transformer_int8_xpu_op.result(0),
                         cache_kv_out_split_op.outputs());
}

std::tuple<pir::OpResult, std::vector<pir::OpResult>>
fused_multi_transformer_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const std::vector<pir::Value>& qkvw,
    const std::vector<pir::Value>& qkvw_max,
    const std::vector<pir::Value>& qkv_bias,
    const std::vector<pir::Value>& out_linear_w,
    const std::vector<pir::Value>& out_linear_wmax,
    const std::vector<pir::Value>& out_linear_bias,
    const std::vector<pir::Value>& ffn_ln_scale,
    const std::vector<pir::Value>& ffn_ln_bias,
    const std::vector<pir::Value>& ffn1_weight,
    const std::vector<pir::Value>& ffn1_weight_max,
    const std::vector<pir::Value>& ffn1_bias,
    const std::vector<pir::Value>& ffn2_weight,
    const std::vector<pir::Value>& ffn2_weight_max,
    const std::vector<pir::Value>& ffn2_bias,
    const paddle::optional<std::vector<pir::Value>>& cache_kv,
    const paddle::optional<std::vector<pir::Value>>& pre_caches,
    const paddle::optional<pir::Value>& rotary_pos_emb,
    const paddle::optional<pir::Value>& time_step,
    const paddle::optional<pir::Value>& seq_lengths,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& gather_index,
    const pir::Value& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis) {
  CheckValueDataType(x, "x", "fused_multi_transformer_xpu");
  paddle::optional<pir::Value> optional_cache_kv;
  if (!cache_kv) {
    optional_cache_kv = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_cache_kv_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            cache_kv.get());
    optional_cache_kv =
        paddle::make_optional<pir::Value>(optional_cache_kv_combine_op.out());
  }
  paddle::optional<pir::Value> optional_pre_caches;
  if (!pre_caches) {
    optional_pre_caches = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_pre_caches_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            pre_caches.get());
    optional_pre_caches =
        paddle::make_optional<pir::Value>(optional_pre_caches_combine_op.out());
  }
  paddle::optional<pir::Value> optional_rotary_pos_emb;
  if (!rotary_pos_emb) {
    optional_rotary_pos_emb = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rotary_pos_emb = rotary_pos_emb;
  }
  paddle::optional<pir::Value> optional_time_step;
  if (!time_step) {
    optional_time_step = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_time_step = time_step;
  }
  paddle::optional<pir::Value> optional_seq_lengths;
  if (!seq_lengths) {
    optional_seq_lengths = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seq_lengths = seq_lengths;
  }
  paddle::optional<pir::Value> optional_src_mask;
  if (!src_mask) {
    optional_src_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask = src_mask;
  }
  paddle::optional<pir::Value> optional_gather_index;
  if (!gather_index) {
    optional_gather_index = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_gather_index = gather_index;
  }
  auto ln_scale_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_scale);
  auto ln_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_bias);
  auto qkvw_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkvw);
  auto qkvw_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkvw_max);
  auto qkv_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(qkv_bias);
  auto out_linear_w_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_linear_w);
  auto out_linear_wmax_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          out_linear_wmax);
  auto out_linear_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          out_linear_bias);
  auto ffn_ln_scale_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn_ln_scale);
  auto ffn_ln_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn_ln_bias);
  auto ffn1_weight_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_weight);
  auto ffn1_weight_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          ffn1_weight_max);
  auto ffn1_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn1_bias);
  auto ffn2_weight_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_weight);
  auto ffn2_weight_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
          ffn2_weight_max);
  auto ffn2_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ffn2_bias);
  paddle::dialect::FusedMultiTransformerXpuOp fused_multi_transformer_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedMultiTransformerXpuOp>(
              x,
              ln_scale_combine_op.out(),
              ln_bias_combine_op.out(),
              qkvw_combine_op.out(),
              qkvw_max_combine_op.out(),
              qkv_bias_combine_op.out(),
              out_linear_w_combine_op.out(),
              out_linear_wmax_combine_op.out(),
              out_linear_bias_combine_op.out(),
              ffn_ln_scale_combine_op.out(),
              ffn_ln_bias_combine_op.out(),
              ffn1_weight_combine_op.out(),
              ffn1_weight_max_combine_op.out(),
              ffn1_bias_combine_op.out(),
              ffn2_weight_combine_op.out(),
              ffn2_weight_max_combine_op.out(),
              ffn2_bias_combine_op.out(),
              optional_cache_kv.get(),
              optional_pre_caches.get(),
              optional_rotary_pos_emb.get(),
              optional_time_step.get(),
              optional_seq_lengths.get(),
              optional_src_mask.get(),
              optional_gather_index.get(),
              max_buffer,
              pre_layer_norm,
              rotary_emb_dims,
              epsilon,
              dropout_rate,
              is_test,
              dropout_implementation,
              act_method,
              trans_qkvw,
              ring_id,
              gather_axis);
  auto cache_kv_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_multi_transformer_xpu_op.result(1));
  return std::make_tuple(fused_multi_transformer_xpu_op.result(0),
                         cache_kv_out_split_op.outputs());
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_rotary_position_embedding(
    const pir::Value& q,
    const paddle::optional<pir::Value>& k,
    const paddle::optional<pir::Value>& v,
    const paddle::optional<pir::Value>& sin,
    const paddle::optional<pir::Value>& cos,
    const paddle::optional<pir::Value>& position_ids,
    bool use_neox_rotary_style) {
  CheckValueDataType(q, "q", "fused_rotary_position_embedding");
  paddle::optional<pir::Value> optional_k;
  if (!k) {
    optional_k = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_k = k;
  }
  paddle::optional<pir::Value> optional_v;
  if (!v) {
    optional_v = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_v = v;
  }
  paddle::optional<pir::Value> optional_sin;
  if (!sin) {
    optional_sin = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sin = sin;
  }
  paddle::optional<pir::Value> optional_cos;
  if (!cos) {
    optional_cos = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cos = cos;
  }
  paddle::optional<pir::Value> optional_position_ids;
  if (!position_ids) {
    optional_position_ids = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_position_ids = position_ids;
  }
  paddle::dialect::FusedRotaryPositionEmbeddingOp
      fused_rotary_position_embedding_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedRotaryPositionEmbeddingOp>(
                  q,
                  optional_k.get(),
                  optional_v.get(),
                  optional_sin.get(),
                  optional_cos.get(),
                  optional_position_ids.get(),
                  use_neox_rotary_style);
  return std::make_tuple(fused_rotary_position_embedding_op.result(0),
                         fused_rotary_position_embedding_op.result(1),
                         fused_rotary_position_embedding_op.result(2));
}

pir::OpResult fused_scale_bias_add_relu(
    const pir::Value& x1,
    const pir::Value& scale1,
    const pir::Value& bias1,
    const pir::Value& x2,
    const paddle::optional<pir::Value>& scale2,
    const paddle::optional<pir::Value>& bias2,
    bool fuse_dual,
    bool exhaustive_search) {
  CheckValueDataType(x1, "x1", "fused_scale_bias_add_relu");
  paddle::optional<pir::Value> optional_scale2;
  if (!scale2) {
    optional_scale2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale2 = scale2;
  }
  paddle::optional<pir::Value> optional_bias2;
  if (!bias2) {
    optional_bias2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias2 = bias2;
  }
  paddle::dialect::FusedScaleBiasAddReluOp fused_scale_bias_add_relu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedScaleBiasAddReluOp>(
              x1,
              scale1,
              bias1,
              x2,
              optional_scale2.get(),
              optional_bias2.get(),
              fuse_dual,
              exhaustive_search);
  return fused_scale_bias_add_relu_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_scale_bias_relu_conv_bn(const pir::Value& x,
                              const pir::Value& w,
                              const paddle::optional<pir::Value>& scale,
                              const paddle::optional<pir::Value>& bias,
                              const pir::Value& bn_scale,
                              const pir::Value& bn_bias,
                              const pir::Value& input_running_mean,
                              const pir::Value& input_running_var,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              const std::vector<int>& strides,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::string& data_format,
                              float momentum,
                              float epsilon,
                              bool fuse_prologue,
                              bool exhaustive_search,
                              int64_t accumulation_count) {
  CheckValueDataType(x, "x", "fused_scale_bias_relu_conv_bn");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::FusedScaleBiasReluConvBnOp fused_scale_bias_relu_conv_bn_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedScaleBiasReluConvBnOp>(
              x,
              w,
              optional_scale.get(),
              optional_bias.get(),
              bn_scale,
              bn_bias,
              input_running_mean,
              input_running_var,
              paddings,
              dilations,
              strides,
              padding_algorithm,
              groups,
              data_format,
              momentum,
              epsilon,
              fuse_prologue,
              exhaustive_search,
              accumulation_count);
  return std::make_tuple(fused_scale_bias_relu_conv_bn_op.result(0),
                         fused_scale_bias_relu_conv_bn_op.result(1),
                         fused_scale_bias_relu_conv_bn_op.result(2),
                         fused_scale_bias_relu_conv_bn_op.result(3),
                         fused_scale_bias_relu_conv_bn_op.result(4),
                         fused_scale_bias_relu_conv_bn_op.result(5),
                         fused_scale_bias_relu_conv_bn_op.result(6));
}

pir::OpResult fusion_gru(const pir::Value& x,
                         const paddle::optional<pir::Value>& h0,
                         const pir::Value& weight_x,
                         const pir::Value& weight_h,
                         const paddle::optional<pir::Value>& bias,
                         const std::string& activation,
                         const std::string& gate_activation,
                         bool is_reverse,
                         bool use_seq,
                         bool origin_mode,
                         bool use_mkldnn,
                         const std::string& mkldnn_data_type,
                         float scale_data,
                         float shift_data,
                         const std::vector<float>& scale_weights,
                         bool force_fp32_output) {
  CheckValueDataType(x, "x", "fusion_gru");
  paddle::optional<pir::Value> optional_h0;
  if (!h0) {
    optional_h0 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_h0 = h0;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::FusionGruOp fusion_gru_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FusionGruOp>(
          x,
          optional_h0.get(),
          weight_x,
          weight_h,
          optional_bias.get(),
          activation,
          gate_activation,
          is_reverse,
          use_seq,
          origin_mode,
          use_mkldnn,
          mkldnn_data_type,
          scale_data,
          shift_data,
          scale_weights,
          force_fp32_output);
  return fusion_gru_op.result(4);
}

pir::OpResult fusion_repeated_fc_relu(const pir::Value& x,
                                      const std::vector<pir::Value>& w,
                                      const std::vector<pir::Value>& bias) {
  CheckValueDataType(x, "x", "fusion_repeated_fc_relu");
  auto w_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(w);
  auto bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(bias);
  paddle::dialect::FusionRepeatedFcReluOp fusion_repeated_fc_relu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusionRepeatedFcReluOp>(
              x, w_combine_op.out(), bias_combine_op.out());
  return fusion_repeated_fc_relu_op.result(1);
}

pir::OpResult fusion_seqconv_eltadd_relu(const pir::Value& x,
                                         const pir::Value& filter,
                                         const pir::Value& bias,
                                         int context_length,
                                         int context_start,
                                         int context_stride) {
  CheckValueDataType(x, "x", "fusion_seqconv_eltadd_relu");
  paddle::dialect::FusionSeqconvEltaddReluOp fusion_seqconv_eltadd_relu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusionSeqconvEltaddReluOp>(
              x, filter, bias, context_length, context_start, context_stride);
  return fusion_seqconv_eltadd_relu_op.result(0);
}

pir::OpResult fusion_seqexpand_concat_fc(
    const std::vector<pir::Value>& x,
    const pir::Value& fc_weight,
    const paddle::optional<pir::Value>& fc_bias,
    const std::string& fc_activation) {
  CheckVectorOfValueDataType(x, "x", "fusion_seqexpand_concat_fc");
  paddle::optional<pir::Value> optional_fc_bias;
  if (!fc_bias) {
    optional_fc_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fc_bias = fc_bias;
  }
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::FusionSeqexpandConcatFcOp fusion_seqexpand_concat_fc_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusionSeqexpandConcatFcOp>(
              x_combine_op.out(),
              fc_weight,
              optional_fc_bias.get(),
              fc_activation);
  return fusion_seqexpand_concat_fc_op.result(0);
}

pir::OpResult fusion_squared_mat_sub(const pir::Value& x,
                                     const pir::Value& y,
                                     float scalar) {
  CheckValueDataType(x, "x", "fusion_squared_mat_sub");
  paddle::dialect::FusionSquaredMatSubOp fusion_squared_mat_sub_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusionSquaredMatSubOp>(x, y, scalar);
  return fusion_squared_mat_sub_op.result(3);
}

pir::OpResult fusion_transpose_flatten_concat(
    const std::vector<pir::Value>& x,
    const std::vector<int>& trans_axis,
    int flatten_axis,
    int concat_axis) {
  CheckVectorOfValueDataType(x, "x", "fusion_transpose_flatten_concat");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::FusionTransposeFlattenConcatOp
      fusion_transpose_flatten_concat_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusionTransposeFlattenConcatOp>(
                  x_combine_op.out(), trans_axis, flatten_axis, concat_axis);
  return fusion_transpose_flatten_concat_op.result(0);
}

pir::OpResult generate_sequence_xpu(const pir::Value& x, phi::DataType dtype) {
  CheckDataType(dtype, "dtype", "generate_sequence_xpu");
  paddle::dialect::GenerateSequenceXpuOp generate_sequence_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GenerateSequenceXpuOp>(x, dtype);
  return generate_sequence_xpu_op.result(0);
}

pir::OpResult layer_norm_act_xpu(const pir::Value& x,
                                 const pir::Value& scale,
                                 const pir::Value& bias,
                                 int begin_norm_axis,
                                 float epsilon,
                                 int act_type,
                                 float act_param) {
  CheckValueDataType(x, "x", "layer_norm_act_xpu");
  paddle::dialect::LayerNormActXpuOp layer_norm_act_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LayerNormActXpuOp>(
              x, scale, bias, begin_norm_axis, epsilon, act_type, act_param);
  return layer_norm_act_xpu_op.result(0);
}

pir::OpResult max_pool2d_v2(const pir::Value& x,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::string& data_format,
                            bool global_pooling,
                            bool adaptive) {
  CheckValueDataType(x, "x", "max_pool2d_v2");
  paddle::dialect::MaxPool2dV2Op max_pool2d_v2_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool2dV2Op>(x,
                                                  kernel_size,
                                                  strides,
                                                  paddings,
                                                  data_format,
                                                  global_pooling,
                                                  adaptive);
  return max_pool2d_v2_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multi_encoder_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& fc_weight,
    const std::vector<pir::Value>& fc_weight_max,
    const std::vector<pir::Value>& fc_bias,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const paddle::optional<pir::Value>& mask,
    const paddle::optional<pir::Value>& seq_lod,
    const paddle::optional<pir::Value>& max_seq_len,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx) {
  CheckValueDataType(x, "x", "multi_encoder_xpu");
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  paddle::optional<pir::Value> optional_seq_lod;
  if (!seq_lod) {
    optional_seq_lod = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seq_lod = seq_lod;
  }
  paddle::optional<pir::Value> optional_max_seq_len;
  if (!max_seq_len) {
    optional_max_seq_len = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_max_seq_len = max_seq_len;
  }
  auto fc_weight_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(fc_weight);
  auto fc_weight_max_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(fc_weight_max);
  auto fc_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(fc_bias);
  auto ln_scale_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_scale);
  auto ln_bias_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ln_bias);
  paddle::dialect::MultiEncoderXpuOp multi_encoder_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiEncoderXpuOp>(
              x,
              fc_weight_combine_op.out(),
              fc_weight_max_combine_op.out(),
              fc_bias_combine_op.out(),
              ln_scale_combine_op.out(),
              ln_bias_combine_op.out(),
              optional_mask.get(),
              optional_seq_lod.get(),
              optional_max_seq_len.get(),
              layer_num,
              norm_before,
              hidden_dim,
              head_num,
              size_per_head,
              ffn_hidden_dim_scale,
              act_type,
              relative_type,
              slice_idx);
  return std::make_tuple(multi_encoder_xpu_op.result(0),
                         multi_encoder_xpu_op.result(1),
                         multi_encoder_xpu_op.result(2));
}

pir::OpResult multihead_matmul(const pir::Value& input,
                               const pir::Value& w,
                               const pir::Value& bias,
                               const paddle::optional<pir::Value>& bias_qk,
                               bool transpose_q,
                               bool transpose_k,
                               bool transpose_v,
                               float alpha,
                               int head_number) {
  CheckValueDataType(input, "input", "multihead_matmul");
  paddle::optional<pir::Value> optional_bias_qk;
  if (!bias_qk) {
    optional_bias_qk = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias_qk = bias_qk;
  }
  paddle::dialect::MultiheadMatmulOp multihead_matmul_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiheadMatmulOp>(input,
                                                      w,
                                                      bias,
                                                      optional_bias_qk.get(),
                                                      transpose_q,
                                                      transpose_k,
                                                      transpose_v,
                                                      alpha,
                                                      head_number);
  return multihead_matmul_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> qkv_attention_xpu(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const paddle::optional<pir::Value>& q_max,
    const paddle::optional<pir::Value>& k_max,
    const paddle::optional<pir::Value>& v_max,
    float alpha,
    int head_num,
    int head_dim,
    bool qkv_fc_fusion,
    phi::DataType out_dtype) {
  CheckValueDataType(q, "q", "qkv_attention_xpu");
  paddle::optional<pir::Value> optional_q_max;
  if (!q_max) {
    optional_q_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_q_max = q_max;
  }
  paddle::optional<pir::Value> optional_k_max;
  if (!k_max) {
    optional_k_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_k_max = k_max;
  }
  paddle::optional<pir::Value> optional_v_max;
  if (!v_max) {
    optional_v_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_v_max = v_max;
  }
  paddle::dialect::QkvAttentionXpuOp qkv_attention_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::QkvAttentionXpuOp>(q,
                                                      k,
                                                      v,
                                                      optional_q_max.get(),
                                                      optional_k_max.get(),
                                                      optional_v_max.get(),
                                                      alpha,
                                                      head_num,
                                                      head_dim,
                                                      qkv_fc_fusion,
                                                      out_dtype);
  return std::make_tuple(qkv_attention_xpu_op.result(0),
                         qkv_attention_xpu_op.result(1));
}

pir::OpResult quantize_xpu(const pir::Value& x,
                           phi::DataType out_dtype,
                           float scale) {
  CheckValueDataType(x, "x", "quantize_xpu");
  paddle::dialect::QuantizeXpuOp quantize_xpu_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::QuantizeXpuOp>(x, out_dtype, scale);
  return quantize_xpu_op.result(0);
}

pir::OpResult self_dp_attention(const pir::Value& x,
                                float alpha,
                                int head_number) {
  CheckValueDataType(x, "x", "self_dp_attention");
  paddle::dialect::SelfDpAttentionOp self_dp_attention_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SelfDpAttentionOp>(x, alpha, head_number);
  return self_dp_attention_op.result(0);
}

pir::OpResult sine_pos_xpu(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(x, "x", "sine_pos_xpu");
  paddle::dialect::SinePosXpuOp sine_pos_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SinePosXpuOp>(
          x, y);
  return sine_pos_xpu_op.result(0);
}

pir::OpResult skip_layernorm(const pir::Value& x,
                             const pir::Value& y,
                             const pir::Value& scale,
                             const pir::Value& bias,
                             float epsilon,
                             int begin_norm_axis) {
  CheckValueDataType(x, "x", "skip_layernorm");
  paddle::dialect::SkipLayernormOp skip_layernorm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SkipLayernormOp>(
              x, y, scale, bias, epsilon, begin_norm_axis);
  return skip_layernorm_op.result(0);
}

pir::OpResult squeeze_excitation_block(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const std::vector<int>& act_type,
    const std::vector<float>& act_param,
    const std::vector<int>& filter_dims) {
  CheckValueDataType(x, "x", "squeeze_excitation_block");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_branch;
  if (!branch) {
    optional_branch = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_branch = branch;
  }
  paddle::dialect::SqueezeExcitationBlockOp squeeze_excitation_block_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SqueezeExcitationBlockOp>(
              x,
              filter,
              filter_max,
              optional_bias.get(),
              optional_branch.get(),
              act_type,
              act_param,
              filter_dims);
  return squeeze_excitation_block_op.result(0);
}

pir::OpResult variable_length_memory_efficient_attention(
    const pir::Value& query,
    const pir::Value& key,
    const pir::Value& value,
    const pir::Value& seq_lens,
    const pir::Value& kv_seq_lens,
    const paddle::optional<pir::Value>& mask,
    float scale,
    bool causal,
    int pre_cache_length) {
  CheckValueDataType(
      query, "query", "variable_length_memory_efficient_attention");
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  paddle::dialect::VariableLengthMemoryEfficientAttentionOp
      variable_length_memory_efficient_attention_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<
                  paddle::dialect::VariableLengthMemoryEfficientAttentionOp>(
                  query,
                  key,
                  value,
                  seq_lens,
                  kv_seq_lens,
                  optional_mask.get(),
                  scale,
                  causal,
                  pre_cache_length);
  return variable_length_memory_efficient_attention_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> yolo_box_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& grid,
    const pir::Value& stride,
    const pir::Value& anchor_grid,
    float offset) {
  CheckValueDataType(x, "x", "yolo_box_xpu");
  paddle::optional<pir::Value> optional_x_max;
  if (!x_max) {
    optional_x_max = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x_max = x_max;
  }
  paddle::dialect::YoloBoxXpuOp yolo_box_xpu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::YoloBoxXpuOp>(
          x, optional_x_max.get(), grid, stride, anchor_grid, offset);
  return std::make_tuple(yolo_box_xpu_op.result(0), yolo_box_xpu_op.result(1));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bias_dropout_residual_layer_norm_grad(
    const pir::Value& y_grad,
    const pir::Value& x,
    const pir::Value& residual,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    const pir::Value& ln_mean,
    const pir::Value& ln_variance,
    const pir::Value& bias_dropout_residual_out,
    const pir::Value& dropout_mask_out,
    float dropout_rate,
    bool is_test,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string& dropout_implementation,
    float ln_epsilon) {
  CheckValueDataType(
      y_grad, "y_grad", "fused_bias_dropout_residual_layer_norm_grad");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_ln_scale;
  if (!ln_scale) {
    optional_ln_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale = ln_scale;
  }
  paddle::optional<pir::Value> optional_ln_bias;
  if (!ln_bias) {
    optional_ln_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias = ln_bias;
  }
  paddle::dialect::FusedBiasDropoutResidualLayerNormGradOp
      fused_bias_dropout_residual_layer_norm_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedBiasDropoutResidualLayerNormGradOp>(
                  y_grad,
                  x,
                  residual,
                  optional_bias.get(),
                  optional_ln_scale.get(),
                  optional_ln_bias.get(),
                  ln_mean,
                  ln_variance,
                  bias_dropout_residual_out,
                  dropout_mask_out,
                  dropout_rate,
                  is_test,
                  dropout_fix_seed,
                  dropout_seed,
                  dropout_implementation,
                  ln_epsilon);
  return std::make_tuple(
      fused_bias_dropout_residual_layer_norm_grad_op.result(0),
      fused_bias_dropout_residual_layer_norm_grad_op.result(1),
      fused_bias_dropout_residual_layer_norm_grad_op.result(2),
      fused_bias_dropout_residual_layer_norm_grad_op.result(3),
      fused_bias_dropout_residual_layer_norm_grad_op.result(4));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_dot_product_attention_grad(const pir::Value& q,
                                 const pir::Value& k,
                                 const pir::Value& v,
                                 const pir::Value& out,
                                 const pir::Value& softmax_out,
                                 const pir::Value& rng_state,
                                 const pir::Value& mask,
                                 const pir::Value& out_grad,
                                 float scaling_factor,
                                 float dropout_probability,
                                 bool is_causal_masking) {
  CheckValueDataType(q, "q", "fused_dot_product_attention_grad");
  paddle::dialect::FusedDotProductAttentionGradOp
      fused_dot_product_attention_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedDotProductAttentionGradOp>(
                  q,
                  k,
                  v,
                  out,
                  softmax_out,
                  rng_state,
                  mask,
                  out_grad,
                  scaling_factor,
                  dropout_probability,
                  is_causal_masking);
  return std::make_tuple(fused_dot_product_attention_grad_op.result(0),
                         fused_dot_product_attention_grad_op.result(1),
                         fused_dot_product_attention_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> fused_dropout_add_grad(
    const pir::Value& seed_offset,
    const pir::Value& out_grad,
    float p,
    bool is_test,
    const std::string& mode,
    bool fix_seed) {
  CheckValueDataType(out_grad, "out_grad", "fused_dropout_add_grad");
  paddle::dialect::FusedDropoutAddGradOp fused_dropout_add_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedDropoutAddGradOp>(
              seed_offset, out_grad, p, is_test, mode, fix_seed);
  return std::make_tuple(fused_dropout_add_grad_op.result(0),
                         fused_dropout_add_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_rotary_position_embedding_grad(
    const paddle::optional<pir::Value>& sin,
    const paddle::optional<pir::Value>& cos,
    const paddle::optional<pir::Value>& position_ids,
    const pir::Value& out_q_grad,
    const paddle::optional<pir::Value>& out_k_grad,
    const paddle::optional<pir::Value>& out_v_grad,
    bool use_neox_rotary_style) {
  CheckValueDataType(
      out_q_grad, "out_q_grad", "fused_rotary_position_embedding_grad");
  paddle::optional<pir::Value> optional_sin;
  if (!sin) {
    optional_sin = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sin = sin;
  }
  paddle::optional<pir::Value> optional_cos;
  if (!cos) {
    optional_cos = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cos = cos;
  }
  paddle::optional<pir::Value> optional_position_ids;
  if (!position_ids) {
    optional_position_ids = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_position_ids = position_ids;
  }
  paddle::optional<pir::Value> optional_out_k_grad;
  if (!out_k_grad) {
    optional_out_k_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_k_grad = out_k_grad;
  }
  paddle::optional<pir::Value> optional_out_v_grad;
  if (!out_v_grad) {
    optional_out_v_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_v_grad = out_v_grad;
  }
  paddle::dialect::FusedRotaryPositionEmbeddingGradOp
      fused_rotary_position_embedding_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedRotaryPositionEmbeddingGradOp>(
                  optional_sin.get(),
                  optional_cos.get(),
                  optional_position_ids.get(),
                  out_q_grad,
                  optional_out_k_grad.get(),
                  optional_out_v_grad.get(),
                  use_neox_rotary_style);
  return std::make_tuple(fused_rotary_position_embedding_grad_op.result(0),
                         fused_rotary_position_embedding_grad_op.result(1),
                         fused_rotary_position_embedding_grad_op.result(2));
}

pir::OpResult max_pool2d_v2_grad(const pir::Value& x,
                                 const pir::Value& out,
                                 const pir::Value& saved_idx,
                                 const pir::Value& out_grad,
                                 const std::vector<int>& kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::string& data_format,
                                 bool global_pooling,
                                 bool adaptive) {
  CheckValueDataType(out_grad, "out_grad", "max_pool2d_v2_grad");
  paddle::dialect::MaxPool2dV2GradOp max_pool2d_v2_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaxPool2dV2GradOp>(x,
                                                      out,
                                                      saved_idx,
                                                      out_grad,
                                                      kernel_size,
                                                      strides,
                                                      paddings,
                                                      data_format,
                                                      global_pooling,
                                                      adaptive);
  return max_pool2d_v2_grad_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adadelta_(const pir::Value& param,
          const pir::Value& grad,
          const pir::Value& avg_squared_grad,
          const pir::Value& avg_squared_update,
          const pir::Value& learning_rate,
          const paddle::optional<pir::Value>& master_param,
          float rho,
          float epsilon,
          bool multi_precision) {
  CheckValueDataType(param, "param", "adadelta_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::dialect::Adadelta_Op adadelta__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Adadelta_Op>(
          param,
          grad,
          avg_squared_grad,
          avg_squared_update,
          learning_rate,
          optional_master_param.get(),
          rho,
          epsilon,
          multi_precision);
  paddle::optional<pir::OpResult> optional_master_param_out;
  if (!IsEmptyValue(adadelta__op.result(3))) {
    optional_master_param_out =
        paddle::make_optional<pir::OpResult>(adadelta__op.result(3));
  }
  if (!master_param) {
    adadelta__op.result(3).set_type(pir::Type());
  }
  return std::make_tuple(adadelta__op.result(0),
                         adadelta__op.result(1),
                         adadelta__op.result(2),
                         optional_master_param_out);
}

pir::OpResult add(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "add");
  paddle::dialect::AddOp add_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddOp>(x, y);
  return add_op.result(0);
}

pir::OpResult add_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "add_");
  paddle::dialect::Add_Op add__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Add_Op>(x, y);
  return add__op.result(0);
}

pir::OpResult add_n(const std::vector<pir::Value>& inputs) {
  CheckVectorOfValueDataType(inputs, "inputs", "add_n");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::AddNOp add_n_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddNOp>(
          inputs_combine_op.out());
  return add_n_op.result(0);
}

pir::OpResult add_n_(const std::vector<pir::Value>& inputs) {
  CheckVectorOfValueDataType(inputs, "inputs", "add_n_");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::AddN_Op add_n__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddN_Op>(
          inputs_combine_op.out());
  return add_n__op.result(0);
}

pir::OpResult add_n_with_kernel(const std::vector<pir::Value>& inputs) {
  CheckVectorOfValueDataType(inputs, "inputs", "add_n");
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::AddNWithKernelOp add_n_with_kernel_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddNWithKernelOp>(inputs_combine_op.out());
  return add_n_with_kernel_op.result(0);
}

pir::OpResult all(const pir::Value& x,
                  const std::vector<int64_t>& axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "all");
  paddle::dialect::AllOp all_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AllOp>(
          x, axis, keepdim);
  return all_op.result(0);
}

pir::OpResult amax(const pir::Value& x,
                   const std::vector<int64_t>& axis,
                   bool keepdim) {
  CheckValueDataType(x, "x", "amax");
  paddle::dialect::AmaxOp amax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AmaxOp>(
          x, axis, keepdim);
  return amax_op.result(0);
}

pir::OpResult amin(const pir::Value& x,
                   const std::vector<int64_t>& axis,
                   bool keepdim) {
  CheckValueDataType(x, "x", "amin");
  paddle::dialect::AminOp amin_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AminOp>(
          x, axis, keepdim);
  return amin_op.result(0);
}

pir::OpResult any(const pir::Value& x,
                  const std::vector<int64_t>& axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "any");
  paddle::dialect::AnyOp any_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AnyOp>(
          x, axis, keepdim);
  return any_op.result(0);
}

pir::OpResult assign(const pir::Value& x) {
  CheckValueDataType(x, "x", "assign");
  paddle::dialect::AssignOp assign_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AssignOp>(x);
  return assign_op.result(0);
}

pir::OpResult assign_(const pir::Value& x) {
  CheckValueDataType(x, "x", "assign_");
  paddle::dialect::Assign_Op assign__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Assign_Op>(x);
  return assign__op.result(0);
}

pir::OpResult assign_out_(const pir::Value& x, const pir::Value& output) {
  CheckValueDataType(output, "output", "assign_");
  paddle::dialect::AssignOut_Op assign_out__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AssignOut_Op>(
          x, output);
  return assign_out__op.result(0);
}

pir::OpResult assign_value(const std::vector<int>& shape,
                           phi::DataType dtype,
                           std::vector<phi::Scalar> values,
                           const Place& place) {
  CheckDataType(dtype, "dtype", "assign_value");
  paddle::dialect::AssignValueOp assign_value_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AssignValueOp>(shape, dtype, values, place);
  return assign_value_op.result(0);
}

pir::OpResult assign_value_(const pir::Value& output,
                            const std::vector<int>& shape,
                            phi::DataType dtype,
                            std::vector<phi::Scalar> values,
                            const Place& place) {
  CheckDataType(dtype, "dtype", "assign_value_");
  paddle::dialect::AssignValue_Op assign_value__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AssignValue_Op>(
              output, shape, dtype, values, place);
  return assign_value__op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
batch_norm(const pir::Value& x,
           const pir::Value& mean,
           const pir::Value& variance,
           const paddle::optional<pir::Value>& scale,
           const paddle::optional<pir::Value>& bias,
           bool is_test,
           float momentum,
           float epsilon,
           const std::string& data_layout,
           bool use_global_stats,
           bool trainable_statistics) {
  CheckValueDataType(x, "x", "batch_norm");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::BatchNormOp batch_norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BatchNormOp>(
          x,
          mean,
          variance,
          optional_scale.get(),
          optional_bias.get(),
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics);
  return std::make_tuple(batch_norm_op.result(0),
                         batch_norm_op.result(1),
                         batch_norm_op.result(2),
                         batch_norm_op.result(3),
                         batch_norm_op.result(4),
                         batch_norm_op.result(5));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
batch_norm_(const pir::Value& x,
            const pir::Value& mean,
            const pir::Value& variance,
            const paddle::optional<pir::Value>& scale,
            const paddle::optional<pir::Value>& bias,
            bool is_test,
            float momentum,
            float epsilon,
            const std::string& data_layout,
            bool use_global_stats,
            bool trainable_statistics) {
  CheckValueDataType(x, "x", "batch_norm_");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::BatchNorm_Op batch_norm__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::BatchNorm_Op>(
          x,
          mean,
          variance,
          optional_scale.get(),
          optional_bias.get(),
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics);
  return std::make_tuple(batch_norm__op.result(0),
                         batch_norm__op.result(1),
                         batch_norm__op.result(2),
                         batch_norm__op.result(3),
                         batch_norm__op.result(4),
                         batch_norm__op.result(5));
}

pir::OpResult c_allgather(const pir::Value& x,
                          int ring_id,
                          int nranks,
                          bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_allgather");
  paddle::dialect::CAllgatherOp c_allgather_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CAllgatherOp>(
          x, ring_id, nranks, use_calc_stream);
  return c_allgather_op.result(0);
}

pir::OpResult c_allreduce_max(const pir::Value& x,
                              int ring_id,
                              bool use_calc_stream,
                              bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_allreduce_max");
  paddle::dialect::CAllreduceMaxOp c_allreduce_max_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CAllreduceMaxOp>(
              x, ring_id, use_calc_stream, use_model_parallel);
  return c_allreduce_max_op.result(0);
}

pir::OpResult c_allreduce_max_(const pir::Value& x,
                               int ring_id,
                               bool use_calc_stream,
                               bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_allreduce_max_");
  paddle::dialect::CAllreduceMax_Op c_allreduce_max__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CAllreduceMax_Op>(
              x, ring_id, use_calc_stream, use_model_parallel);
  return c_allreduce_max__op.result(0);
}

pir::OpResult c_allreduce_sum(const pir::Value& x,
                              int ring_id,
                              bool use_calc_stream,
                              bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_allreduce_sum");
  paddle::dialect::CAllreduceSumOp c_allreduce_sum_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CAllreduceSumOp>(
              x, ring_id, use_calc_stream, use_model_parallel);
  return c_allreduce_sum_op.result(0);
}

pir::OpResult c_allreduce_sum_(const pir::Value& x,
                               int ring_id,
                               bool use_calc_stream,
                               bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_allreduce_sum_");
  paddle::dialect::CAllreduceSum_Op c_allreduce_sum__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CAllreduceSum_Op>(
              x, ring_id, use_calc_stream, use_model_parallel);
  return c_allreduce_sum__op.result(0);
}

pir::OpResult c_broadcast(const pir::Value& x,
                          int ring_id,
                          int root,
                          bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_broadcast");
  paddle::dialect::CBroadcastOp c_broadcast_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CBroadcastOp>(
          x, ring_id, root, use_calc_stream);
  return c_broadcast_op.result(0);
}

pir::OpResult c_broadcast_(const pir::Value& x,
                           int ring_id,
                           int root,
                           bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_broadcast_");
  paddle::dialect::CBroadcast_Op c_broadcast__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CBroadcast_Op>(
              x, ring_id, root, use_calc_stream);
  return c_broadcast__op.result(0);
}

pir::OpResult c_concat(const pir::Value& x,
                       int rank,
                       int nranks,
                       int ring_id,
                       bool use_calc_stream,
                       bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_concat");
  paddle::dialect::CConcatOp c_concat_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CConcatOp>(
          x, rank, nranks, ring_id, use_calc_stream, use_model_parallel);
  return c_concat_op.result(0);
}

pir::OpResult c_embedding(const pir::Value& weight,
                          const pir::Value& x,
                          int64_t start_index,
                          int64_t vocab_size) {
  CheckValueDataType(weight, "weight", "c_embedding");
  paddle::dialect::CEmbeddingOp c_embedding_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CEmbeddingOp>(
          weight, x, start_index, vocab_size);
  return c_embedding_op.result(0);
}

pir::OpResult c_identity(const pir::Value& x,
                         int ring_id,
                         bool use_calc_stream,
                         bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_identity");
  paddle::dialect::CIdentityOp c_identity_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CIdentityOp>(
          x, ring_id, use_calc_stream, use_model_parallel);
  return c_identity_op.result(0);
}

pir::OpResult c_identity_(const pir::Value& x,
                          int ring_id,
                          bool use_calc_stream,
                          bool use_model_parallel) {
  CheckValueDataType(x, "x", "c_identity_");
  paddle::dialect::CIdentity_Op c_identity__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CIdentity_Op>(
          x, ring_id, use_calc_stream, use_model_parallel);
  return c_identity__op.result(0);
}

pir::OpResult c_reduce_min(const pir::Value& x,
                           int ring_id,
                           int root_id,
                           bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_reduce_min");
  paddle::dialect::CReduceMinOp c_reduce_min_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CReduceMinOp>(
          x, ring_id, root_id, use_calc_stream);
  return c_reduce_min_op.result(0);
}

pir::OpResult c_reduce_min_(const pir::Value& x,
                            int ring_id,
                            int root_id,
                            bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_reduce_min_");
  paddle::dialect::CReduceMin_Op c_reduce_min__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CReduceMin_Op>(
              x, ring_id, root_id, use_calc_stream);
  return c_reduce_min__op.result(0);
}

pir::OpResult c_reduce_sum(const pir::Value& x,
                           int ring_id,
                           int root_id,
                           bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_reduce_sum");
  paddle::dialect::CReduceSumOp c_reduce_sum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CReduceSumOp>(
          x, ring_id, root_id, use_calc_stream);
  return c_reduce_sum_op.result(0);
}

pir::OpResult c_reduce_sum_(const pir::Value& x,
                            int ring_id,
                            int root_id,
                            bool use_calc_stream) {
  CheckValueDataType(x, "x", "c_reduce_sum_");
  paddle::dialect::CReduceSum_Op c_reduce_sum__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CReduceSum_Op>(
              x, ring_id, root_id, use_calc_stream);
  return c_reduce_sum__op.result(0);
}

pir::OpResult c_reducescatter(const pir::Value& x,
                              int ring_id,
                              int nranks,
                              bool use_calc_stream) {
  CheckValueDataType(x, "x", "reduce_scatter");
  paddle::dialect::CReducescatterOp c_reducescatter_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CReducescatterOp>(
              x, ring_id, nranks, use_calc_stream);
  return c_reducescatter_op.result(0);
}

pir::OpResult c_sync_calc_stream(const pir::Value& x) {
  CheckValueDataType(x, "x", "c_sync_calc_stream");
  paddle::dialect::CSyncCalcStreamOp c_sync_calc_stream_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CSyncCalcStreamOp>(x);
  return c_sync_calc_stream_op.result(0);
}

pir::OpResult c_sync_calc_stream_(const pir::Value& x) {
  CheckValueDataType(x, "x", "c_sync_calc_stream_");
  paddle::dialect::CSyncCalcStream_Op c_sync_calc_stream__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CSyncCalcStream_Op>(x);
  return c_sync_calc_stream__op.result(0);
}

pir::OpResult c_sync_comm_stream(const pir::Value& x, int ring_id) {
  CheckValueDataType(x, "x", "c_sync_comm_stream");
  paddle::dialect::CSyncCommStreamOp c_sync_comm_stream_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CSyncCommStreamOp>(x, ring_id);
  return c_sync_comm_stream_op.result(0);
}

pir::OpResult c_sync_comm_stream_(const pir::Value& x, int ring_id) {
  CheckValueDataType(x, "x", "c_sync_comm_stream_");
  paddle::dialect::CSyncCommStream_Op c_sync_comm_stream__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CSyncCommStream_Op>(x, ring_id);
  return c_sync_comm_stream__op.result(0);
}

pir::OpResult cast(const pir::Value& x, phi::DataType dtype) {
  CheckValueDataType(x, "x", "cast");
  paddle::dialect::CastOp cast_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::CastOp>(
          x, dtype);
  return cast_op.result(0);
}

pir::OpResult cast_(const pir::Value& x, phi::DataType dtype) {
  CheckValueDataType(x, "x", "cast_");
  paddle::dialect::Cast_Op cast__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Cast_Op>(
          x, dtype);
  return cast__op.result(0);
}

pir::OpResult channel_shuffle(const pir::Value& x,
                              int groups,
                              const std::string& data_format) {
  CheckValueDataType(x, "x", "channel_shuffle");
  paddle::dialect::ChannelShuffleOp channel_shuffle_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ChannelShuffleOp>(x, groups, data_format);
  return channel_shuffle_op.result(0);
}

pir::OpResult conv2d_transpose(const pir::Value& x,
                               const pir::Value& filter,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int64_t>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose");
  paddle::dialect::Conv2dTransposeOp conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeOp>(x,
                                                      filter,
                                                      strides,
                                                      paddings,
                                                      output_padding,
                                                      output_size,
                                                      padding_algorithm,
                                                      groups,
                                                      dilations,
                                                      data_format);
  return conv2d_transpose_op.result(0);
}

pir::OpResult conv2d_transpose(const pir::Value& x,
                               const pir::Value& filter,
                               pir::Value output_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose");
  paddle::dialect::Conv2dTransposeOp conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeOp>(x,
                                                      filter,
                                                      output_size,
                                                      strides,
                                                      paddings,
                                                      output_padding,
                                                      padding_algorithm,
                                                      groups,
                                                      dilations,
                                                      data_format);
  return conv2d_transpose_op.result(0);
}

pir::OpResult conv2d_transpose(const pir::Value& x,
                               const pir::Value& filter,
                               std::vector<pir::Value> output_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::Conv2dTransposeOp conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeOp>(
              x,
              filter,
              output_size_combine_op.out(),
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return conv2d_transpose_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> decayed_adagrad(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& moment,
    const pir::Value& learning_rate,
    float decay,
    float epsilon) {
  CheckValueDataType(param, "param", "decayed_adagrad");
  paddle::dialect::DecayedAdagradOp decayed_adagrad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DecayedAdagradOp>(
              param, grad, moment, learning_rate, decay, epsilon);
  return std::make_tuple(decayed_adagrad_op.result(0),
                         decayed_adagrad_op.result(1));
}

pir::OpResult decode_jpeg(const pir::Value& x,
                          const std::string& mode,
                          const Place& place) {
  CheckValueDataType(x, "x", "decode_jpeg");
  paddle::dialect::DecodeJpegOp decode_jpeg_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DecodeJpegOp>(
          x, mode, place);
  return decode_jpeg_op.result(0);
}

pir::OpResult deformable_conv(const pir::Value& x,
                              const pir::Value& offset,
                              const pir::Value& filter,
                              const paddle::optional<pir::Value>& mask,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              int deformable_groups,
                              int groups,
                              int im2col_step) {
  CheckValueDataType(x, "x", "deformable_conv");
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  paddle::dialect::DeformableConvOp deformable_conv_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DeformableConvOp>(x,
                                                     offset,
                                                     filter,
                                                     optional_mask.get(),
                                                     strides,
                                                     paddings,
                                                     dilations,
                                                     deformable_groups,
                                                     groups,
                                                     im2col_step);
  return deformable_conv_op.result(0);
}

pir::OpResult depthwise_conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose");
  paddle::dialect::DepthwiseConv2dTransposeOp depthwise_conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dTransposeOp>(
              x,
              filter,
              strides,
              paddings,
              output_padding,
              output_size,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return depthwise_conv2d_transpose_op.result(0);
}

pir::OpResult depthwise_conv2d_transpose(const pir::Value& x,
                                         const pir::Value& filter,
                                         pir::Value output_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         const std::vector<int>& output_padding,
                                         const std::string& padding_algorithm,
                                         int groups,
                                         const std::vector<int>& dilations,
                                         const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose");
  paddle::dialect::DepthwiseConv2dTransposeOp depthwise_conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dTransposeOp>(
              x,
              filter,
              output_size,
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return depthwise_conv2d_transpose_op.result(0);
}

pir::OpResult depthwise_conv2d_transpose(const pir::Value& x,
                                         const pir::Value& filter,
                                         std::vector<pir::Value> output_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         const std::vector<int>& output_padding,
                                         const std::string& padding_algorithm,
                                         int groups,
                                         const std::vector<int>& dilations,
                                         const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::DepthwiseConv2dTransposeOp depthwise_conv2d_transpose_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DepthwiseConv2dTransposeOp>(
              x,
              filter,
              output_size_combine_op.out(),
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return depthwise_conv2d_transpose_op.result(0);
}

pir::OpResult disable_check_model_nan_inf(const pir::Value& x, int flag) {
  CheckValueDataType(x, "x", "check_model_nan_inf");
  paddle::dialect::DisableCheckModelNanInfOp disable_check_model_nan_inf_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DisableCheckModelNanInfOp>(x, flag);
  return disable_check_model_nan_inf_op.result(0);
}

std::
    tuple<std::vector<pir::OpResult>, std::vector<pir::OpResult>, pir::OpResult>
    distribute_fpn_proposals(const pir::Value& fpn_rois,
                             const paddle::optional<pir::Value>& rois_num,
                             int min_level,
                             int max_level,
                             int refer_level,
                             int refer_scale,
                             bool pixel_offset) {
  CheckValueDataType(fpn_rois, "fpn_rois", "distribute_fpn_proposals");
  paddle::optional<pir::Value> optional_rois_num;
  if (!rois_num) {
    optional_rois_num = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_rois_num = rois_num;
  }
  paddle::dialect::DistributeFpnProposalsOp distribute_fpn_proposals_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DistributeFpnProposalsOp>(
              fpn_rois,
              optional_rois_num.get(),
              min_level,
              max_level,
              refer_level,
              refer_scale,
              pixel_offset);
  auto multi_fpn_rois_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          distribute_fpn_proposals_op.result(0));
  auto multi_level_rois_num_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          distribute_fpn_proposals_op.result(1));
  return std::make_tuple(multi_fpn_rois_split_op.outputs(),
                         multi_level_rois_num_split_op.outputs(),
                         distribute_fpn_proposals_op.result(2));
}

pir::OpResult divide(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "divide");
  paddle::dialect::DivideOp divide_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DivideOp>(x,
                                                                            y);
  return divide_op.result(0);
}

pir::OpResult divide_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "divide_");
  paddle::dialect::Divide_Op divide__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Divide_Op>(x,
                                                                             y);
  return divide__op.result(0);
}

pir::OpResult dropout(const pir::Value& x,
                      const paddle::optional<pir::Value>& seed_tensor,
                      float p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed) {
  CheckValueDataType(x, "x", "dropout");
  paddle::optional<pir::Value> optional_seed_tensor;
  if (!seed_tensor) {
    optional_seed_tensor = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_seed_tensor = seed_tensor;
  }
  paddle::dialect::DropoutOp dropout_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DropoutOp>(
          x, optional_seed_tensor.get(), p, is_test, mode, seed, fix_seed);
  return dropout_op.result(0);
}

std::
    tuple<pir::OpResult, std::vector<pir::OpResult>, std::vector<pir::OpResult>>
    einsum(const std::vector<pir::Value>& x, const std::string& equation) {
  CheckVectorOfValueDataType(x, "x", "einsum");
  auto x_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  paddle::dialect::EinsumOp einsum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EinsumOp>(
          x_combine_op.out(), equation);
  auto inner_cache_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          einsum_op.result(1));
  auto xshape_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          einsum_op.result(2));
  return std::make_tuple(einsum_op.result(0),
                         inner_cache_split_op.outputs(),
                         xshape_split_op.outputs());
}

pir::OpResult elementwise_pow(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "elementwise_pow");
  paddle::dialect::ElementwisePowOp elementwise_pow_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ElementwisePowOp>(x, y);
  return elementwise_pow_op.result(0);
}

pir::OpResult embedding(const pir::Value& x,
                        const pir::Value& weight,
                        int64_t padding_idx,
                        bool sparse) {
  if (x.type().isa<paddle::dialect::DenseTensorType>() &&
      weight.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(weight, "weight", "embedding");
    paddle::dialect::EmbeddingOp embedding_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::EmbeddingOp>(
                x, weight, padding_idx, sparse);
    return embedding_op.result(0);
  }
  if (x.type().isa<paddle::dialect::DenseTensorType>() &&
      weight.type().isa<paddle::dialect::SelectedRowsType>()) {
    CheckValueDataType(weight, "weight", "sparse_weight_embedding");
    paddle::dialect::SparseWeightEmbeddingOp sparse_weight_embedding_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::SparseWeightEmbeddingOp>(
                x, weight, padding_idx, sparse);
    return sparse_weight_embedding_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (embedding) for input Value is unimplemented, please "
      "check the type of input Value."));
}

pir::OpResult empty(const std::vector<int64_t>& shape,
                    phi::DataType dtype,
                    const Place& place) {
  CheckDataType(dtype, "dtype", "empty");
  paddle::dialect::EmptyOp empty_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EmptyOp>(
          shape, dtype, place);
  return empty_op.result(0);
}

pir::OpResult empty(pir::Value shape, phi::DataType dtype, const Place& place) {
  CheckDataType(dtype, "dtype", "empty");
  paddle::dialect::EmptyOp empty_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EmptyOp>(
          shape, dtype, place);
  return empty_op.result(0);
}

pir::OpResult empty(std::vector<pir::Value> shape,
                    phi::DataType dtype,
                    const Place& place) {
  CheckDataType(dtype, "dtype", "empty");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::EmptyOp empty_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EmptyOp>(
          shape_combine_op.out(), dtype, place);
  return empty_op.result(0);
}

pir::OpResult empty_like(const pir::Value& x,
                         phi::DataType dtype,
                         const Place& place) {
  CheckDataTypeOrValue(dtype, "dtype", x, "x", "empty_like");
  paddle::dialect::EmptyLikeOp empty_like_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EmptyLikeOp>(
          x, dtype, place);
  return empty_like_op.result(0);
}

pir::OpResult enable_check_model_nan_inf(const pir::Value& x, int flag) {
  CheckValueDataType(x, "x", "check_model_nan_inf");
  paddle::dialect::EnableCheckModelNanInfOp enable_check_model_nan_inf_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::EnableCheckModelNanInfOp>(x, flag);
  return enable_check_model_nan_inf_op.result(0);
}

pir::OpResult equal(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "equal");
  paddle::dialect::EqualOp equal_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EqualOp>(x,
                                                                           y);
  return equal_op.result(0);
}

pir::OpResult equal_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "equal_");
  paddle::dialect::Equal_Op equal__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Equal_Op>(x,
                                                                            y);
  return equal__op.result(0);
}

pir::OpResult exponential_(const pir::Value& x, float lam) {
  CheckValueDataType(x, "x", "exponential_");
  paddle::dialect::Exponential_Op exponential__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Exponential_Op>(x, lam);
  return exponential__op.result(0);
}

pir::OpResult eye(float num_rows,
                  float num_columns,
                  phi::DataType dtype,
                  const Place& place) {
  CheckDataType(dtype, "dtype", "eye");
  paddle::dialect::EyeOp eye_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EyeOp>(
          num_rows, num_columns, dtype, place);
  return eye_op.result(0);
}

pir::OpResult eye(pir::Value num_rows,
                  pir::Value num_columns,
                  phi::DataType dtype,
                  const Place& place) {
  CheckDataType(dtype, "dtype", "eye");
  paddle::dialect::EyeOp eye_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EyeOp>(
          num_rows, num_columns, dtype, place);
  return eye_op.result(0);
}

pir::OpResult fetch(const pir::Value& x, const std::string& name, int col) {
  CheckValueDataType(x, "x", "fetch");
  paddle::dialect::FetchOp fetch_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FetchOp>(
          x, name, col);
  return fetch_op.result(0);
}

pir::OpResult floor_divide(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "floor_divide");
  paddle::dialect::FloorDivideOp floor_divide_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FloorDivideOp>(x, y);
  return floor_divide_op.result(0);
}

pir::OpResult floor_divide_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "floor_divide_");
  paddle::dialect::FloorDivide_Op floor_divide__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FloorDivide_Op>(x, y);
  return floor_divide__op.result(0);
}

pir::OpResult frobenius_norm(const pir::Value& x,
                             const std::vector<int64_t>& axis,
                             bool keep_dim,
                             bool reduce_all) {
  CheckValueDataType(x, "x", "frobenius_norm");
  paddle::dialect::FrobeniusNormOp frobenius_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormOp>(
              x, axis, keep_dim, reduce_all);
  return frobenius_norm_op.result(0);
}

pir::OpResult frobenius_norm(const pir::Value& x,
                             pir::Value axis,
                             bool keep_dim,
                             bool reduce_all) {
  CheckValueDataType(x, "x", "frobenius_norm");
  paddle::dialect::FrobeniusNormOp frobenius_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormOp>(
              x, axis, keep_dim, reduce_all);
  return frobenius_norm_op.result(0);
}

pir::OpResult frobenius_norm(const pir::Value& x,
                             std::vector<pir::Value> axis,
                             bool keep_dim,
                             bool reduce_all) {
  CheckValueDataType(x, "x", "frobenius_norm");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::FrobeniusNormOp frobenius_norm_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormOp>(
              x, axis_combine_op.out(), keep_dim, reduce_all);
  return frobenius_norm_op.result(0);
}

pir::OpResult full(const std::vector<int64_t>& shape,
                   float value,
                   phi::DataType dtype,
                   const Place& place) {
  CheckDataType(dtype, "dtype", "full");
  paddle::dialect::FullOp full_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FullOp>(
          shape, value, dtype, place);
  return full_op.result(0);
}

pir::OpResult full_(const pir::Value& output,
                    const std::vector<int64_t>& shape,
                    float value,
                    phi::DataType dtype,
                    const Place& place) {
  CheckDataType(dtype, "dtype", "full_");
  paddle::dialect::Full_Op full__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Full_Op>(
          output, shape, value, dtype, place);
  return full__op.result(0);
}

pir::OpResult full_batch_size_like(const pir::Value& input,
                                   const std::vector<int>& shape,
                                   phi::DataType dtype,
                                   float value,
                                   int input_dim_idx,
                                   int output_dim_idx,
                                   const Place& place) {
  CheckDataType(dtype, "dtype", "full_batch_size_like");
  paddle::dialect::FullBatchSizeLikeOp full_batch_size_like_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FullBatchSizeLikeOp>(
              input, shape, dtype, value, input_dim_idx, output_dim_idx, place);
  return full_batch_size_like_op.result(0);
}

pir::OpResult full_like(const pir::Value& x,
                        float value,
                        phi::DataType dtype,
                        const Place& place) {
  CheckDataTypeOrValue(dtype, "dtype", x, "x", "full_like");
  paddle::dialect::FullLikeOp full_like_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FullLikeOp>(
          x, value, dtype, place);
  return full_like_op.result(0);
}

pir::OpResult full_like(const pir::Value& x,
                        pir::Value value,
                        phi::DataType dtype,
                        const Place& place) {
  CheckDataTypeOrValue(dtype, "dtype", x, "x", "full_like");
  paddle::dialect::FullLikeOp full_like_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FullLikeOp>(
          x, value, dtype, place);
  return full_like_op.result(0);
}

pir::OpResult full_with_tensor(const pir::Value& shape,
                               const pir::Value& value,
                               phi::DataType dtype) {
  CheckDataType(dtype, "dtype", "full_with_tensor");
  paddle::dialect::FullWithTensorOp full_with_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FullWithTensorOp>(shape, value, dtype);
  return full_with_tensor_op.result(0);
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
fused_adam_(const std::vector<pir::Value>& params,
            const std::vector<pir::Value>& grads,
            const pir::Value& learning_rate,
            const std::vector<pir::Value>& moments1,
            const std::vector<pir::Value>& moments2,
            const std::vector<pir::Value>& beta1_pows,
            const std::vector<pir::Value>& beta2_pows,
            const paddle::optional<std::vector<pir::Value>>& master_params,
            const paddle::optional<pir::Value>& skip_update,
            float beta1,
            float beta2,
            float epsilon,
            int chunk_size,
            float weight_decay,
            bool use_adamw,
            bool multi_precision,
            bool use_global_beta_pow) {
  CheckVectorOfValueDataType(params, "params", "fused_adam_");
  paddle::optional<pir::Value> optional_master_params;
  if (!master_params) {
    optional_master_params = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_params_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_params.get());
    optional_master_params = paddle::make_optional<pir::Value>(
        optional_master_params_combine_op.out());
  }
  paddle::optional<pir::Value> optional_skip_update;
  if (!skip_update) {
    optional_skip_update = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_skip_update = skip_update;
  }
  auto params_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(params);
  auto grads_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grads);
  auto moments1_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moments1);
  auto moments2_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(moments2);
  auto beta1_pows_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta1_pows);
  auto beta2_pows_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(beta2_pows);
  paddle::dialect::FusedAdam_Op fused_adam__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FusedAdam_Op>(
          params_combine_op.out(),
          grads_combine_op.out(),
          learning_rate,
          moments1_combine_op.out(),
          moments2_combine_op.out(),
          beta1_pows_combine_op.out(),
          beta2_pows_combine_op.out(),
          optional_master_params.get(),
          optional_skip_update.get(),
          beta1,
          beta2,
          epsilon,
          chunk_size,
          weight_decay,
          use_adamw,
          multi_precision,
          use_global_beta_pow);
  paddle::optional<std::vector<pir::OpResult>> optional_master_params_out;
  if (!IsEmptyValue(fused_adam__op.result(5))) {
    auto optional_master_params_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            fused_adam__op.result(5));
    optional_master_params_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_params_out_slice_op.outputs());
  }
  if (!master_params) {
    fused_adam__op.result(5).set_type(pir::Type());
  }
  auto params_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_adam__op.result(0));
  auto moments1_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_adam__op.result(1));
  auto moments2_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_adam__op.result(2));
  auto beta1_pows_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_adam__op.result(3));
  auto beta2_pows_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          fused_adam__op.result(4));
  return std::make_tuple(params_out_split_op.outputs(),
                         moments1_out_split_op.outputs(),
                         moments2_out_split_op.outputs(),
                         beta1_pows_out_split_op.outputs(),
                         beta2_pows_out_split_op.outputs(),
                         optional_master_params_out);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_batch_norm_act(const pir::Value& x,
                     const pir::Value& scale,
                     const pir::Value& bias,
                     const pir::Value& mean,
                     const pir::Value& variance,
                     float momentum,
                     float epsilon,
                     const std::string& act_type) {
  CheckValueDataType(x, "x", "fused_batch_norm_act");
  paddle::dialect::FusedBatchNormActOp fused_batch_norm_act_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBatchNormActOp>(
              x, scale, bias, mean, variance, momentum, epsilon, act_type);
  return std::make_tuple(fused_batch_norm_act_op.result(0),
                         fused_batch_norm_act_op.result(1),
                         fused_batch_norm_act_op.result(2),
                         fused_batch_norm_act_op.result(3),
                         fused_batch_norm_act_op.result(4),
                         fused_batch_norm_act_op.result(5));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_batch_norm_act_(const pir::Value& x,
                      const pir::Value& scale,
                      const pir::Value& bias,
                      const pir::Value& mean,
                      const pir::Value& variance,
                      float momentum,
                      float epsilon,
                      const std::string& act_type) {
  CheckValueDataType(x, "x", "fused_batch_norm_act_");
  paddle::dialect::FusedBatchNormAct_Op fused_batch_norm_act__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBatchNormAct_Op>(
              x, scale, bias, mean, variance, momentum, epsilon, act_type);
  return std::make_tuple(fused_batch_norm_act__op.result(0),
                         fused_batch_norm_act__op.result(1),
                         fused_batch_norm_act__op.result(2),
                         fused_batch_norm_act__op.result(3),
                         fused_batch_norm_act__op.result(4),
                         fused_batch_norm_act__op.result(5));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bn_add_activation(const pir::Value& x,
                        const pir::Value& z,
                        const pir::Value& scale,
                        const pir::Value& bias,
                        const pir::Value& mean,
                        const pir::Value& variance,
                        float momentum,
                        float epsilon,
                        const std::string& act_type) {
  CheckValueDataType(x, "x", "fused_bn_add_activation");
  paddle::dialect::FusedBnAddActivationOp fused_bn_add_activation_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBnAddActivationOp>(
              x, z, scale, bias, mean, variance, momentum, epsilon, act_type);
  return std::make_tuple(fused_bn_add_activation_op.result(0),
                         fused_bn_add_activation_op.result(1),
                         fused_bn_add_activation_op.result(2),
                         fused_bn_add_activation_op.result(3),
                         fused_bn_add_activation_op.result(4),
                         fused_bn_add_activation_op.result(5));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bn_add_activation_(const pir::Value& x,
                         const pir::Value& z,
                         const pir::Value& scale,
                         const pir::Value& bias,
                         const pir::Value& mean,
                         const pir::Value& variance,
                         float momentum,
                         float epsilon,
                         const std::string& act_type) {
  CheckValueDataType(x, "x", "fused_bn_add_activation_");
  paddle::dialect::FusedBnAddActivation_Op fused_bn_add_activation__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBnAddActivation_Op>(
              x, z, scale, bias, mean, variance, momentum, epsilon, act_type);
  return std::make_tuple(fused_bn_add_activation__op.result(0),
                         fused_bn_add_activation__op.result(1),
                         fused_bn_add_activation__op.result(2),
                         fused_bn_add_activation__op.result(3),
                         fused_bn_add_activation__op.result(4),
                         fused_bn_add_activation__op.result(5));
}

pir::OpResult fused_softmax_mask_upper_triangle(const pir::Value& X) {
  CheckValueDataType(X, "X", "fused_softmax_mask_upper_triangle");
  paddle::dialect::FusedSoftmaxMaskUpperTriangleOp
      fused_softmax_mask_upper_triangle_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedSoftmaxMaskUpperTriangleOp>(X);
  return fused_softmax_mask_upper_triangle_op.result(0);
}

pir::OpResult gaussian(const std::vector<int64_t>& shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place) {
  CheckDataType(dtype, "dtype", "gaussian");
  paddle::dialect::GaussianOp gaussian_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GaussianOp>(
          shape, mean, std, seed, dtype, place);
  return gaussian_op.result(0);
}

pir::OpResult gaussian(pir::Value shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place) {
  CheckDataType(dtype, "dtype", "gaussian");
  paddle::dialect::GaussianOp gaussian_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GaussianOp>(
          shape, mean, std, seed, dtype, place);
  return gaussian_op.result(0);
}

pir::OpResult gaussian(std::vector<pir::Value> shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place) {
  CheckDataType(dtype, "dtype", "gaussian");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::GaussianOp gaussian_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::GaussianOp>(
          shape_combine_op.out(), mean, std, seed, dtype, place);
  return gaussian_op.result(0);
}

pir::OpResult get_tensor_from_selected_rows(const pir::Value& x) {
  CheckValueDataType(x, "x", "get_tensor_from_selected_rows");
  paddle::dialect::GetTensorFromSelectedRowsOp
      get_tensor_from_selected_rows_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::GetTensorFromSelectedRowsOp>(x);
  return get_tensor_from_selected_rows_op.result(0);
}

pir::OpResult greater_equal(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "greater_equal");
  paddle::dialect::GreaterEqualOp greater_equal_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GreaterEqualOp>(x, y);
  return greater_equal_op.result(0);
}

pir::OpResult greater_equal_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "greater_equal_");
  paddle::dialect::GreaterEqual_Op greater_equal__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GreaterEqual_Op>(x, y);
  return greater_equal__op.result(0);
}

pir::OpResult greater_than(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "greater_than");
  paddle::dialect::GreaterThanOp greater_than_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GreaterThanOp>(x, y);
  return greater_than_op.result(0);
}

pir::OpResult greater_than_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "greater_than_");
  paddle::dialect::GreaterThan_Op greater_than__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::GreaterThan_Op>(x, y);
  return greater_than__op.result(0);
}

pir::OpResult hardswish(const pir::Value& x) {
  CheckValueDataType(x, "x", "hardswish");
  paddle::dialect::HardswishOp hardswish_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::HardswishOp>(
          x);
  return hardswish_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> hsigmoid_loss(
    const pir::Value& x,
    const pir::Value& label,
    const pir::Value& w,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& path,
    const paddle::optional<pir::Value>& code,
    int num_classes,
    bool is_sparse) {
  CheckValueDataType(x, "x", "hsigmoid_loss");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_path;
  if (!path) {
    optional_path = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_path = path;
  }
  paddle::optional<pir::Value> optional_code;
  if (!code) {
    optional_code = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_code = code;
  }
  paddle::dialect::HsigmoidLossOp hsigmoid_loss_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HsigmoidLossOp>(x,
                                                   label,
                                                   w,
                                                   optional_bias.get(),
                                                   optional_path.get(),
                                                   optional_code.get(),
                                                   num_classes,
                                                   is_sparse);
  return std::make_tuple(hsigmoid_loss_op.result(0),
                         hsigmoid_loss_op.result(1),
                         hsigmoid_loss_op.result(2));
}

pir::OpResult increment(const pir::Value& x, float value) {
  CheckValueDataType(x, "x", "increment");
  paddle::dialect::IncrementOp increment_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::IncrementOp>(
          x, value);
  return increment_op.result(0);
}

pir::OpResult increment_(const pir::Value& x, float value) {
  CheckValueDataType(x, "x", "increment_");
  paddle::dialect::Increment_Op increment__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Increment_Op>(
          x, value);
  return increment__op.result(0);
}

pir::OpResult less_equal(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "less_equal");
  paddle::dialect::LessEqualOp less_equal_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LessEqualOp>(
          x, y);
  return less_equal_op.result(0);
}

pir::OpResult less_equal_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "less_equal_");
  paddle::dialect::LessEqual_Op less_equal__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LessEqual_Op>(
          x, y);
  return less_equal__op.result(0);
}

pir::OpResult less_than(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "less_than");
  paddle::dialect::LessThanOp less_than_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LessThanOp>(
          x, y);
  return less_than_op.result(0);
}

pir::OpResult less_than_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "less_than_");
  paddle::dialect::LessThan_Op less_than__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LessThan_Op>(
          x, y);
  return less_than__op.result(0);
}

pir::OpResult linspace(const pir::Value& start,
                       const pir::Value& stop,
                       const pir::Value& number,
                       phi::DataType dtype,
                       const Place& place) {
  CheckDataType(dtype, "dtype", "linspace");
  paddle::dialect::LinspaceOp linspace_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LinspaceOp>(
          start, stop, number, dtype, place);
  return linspace_op.result(0);
}

pir::OpResult logspace(const pir::Value& start,
                       const pir::Value& stop,
                       const pir::Value& num,
                       const pir::Value& base,
                       phi::DataType dtype,
                       const Place& place) {
  CheckDataType(dtype, "dtype", "logspace");
  paddle::dialect::LogspaceOp logspace_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogspaceOp>(
          start, stop, num, base, dtype, place);
  return logspace_op.result(0);
}

pir::OpResult logsumexp(const pir::Value& x,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all) {
  CheckValueDataType(x, "x", "logsumexp");
  paddle::dialect::LogsumexpOp logsumexp_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::LogsumexpOp>(
          x, axis, keepdim, reduce_all);
  return logsumexp_op.result(0);
}

pir::OpResult matmul(const pir::Value& x,
                     const pir::Value& y,
                     bool transpose_x,
                     bool transpose_y) {
  CheckValueDataType(y, "y", "matmul");
  paddle::dialect::MatmulOp matmul_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MatmulOp>(
          x, y, transpose_x, transpose_y);
  return matmul_op.result(0);
}

pir::OpResult matrix_rank(const pir::Value& x,
                          float tol,
                          bool use_default_tol,
                          bool hermitian) {
  CheckValueDataType(x, "x", "matrix_rank");
  paddle::dialect::MatrixRankOp matrix_rank_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MatrixRankOp>(
          x, tol, use_default_tol, hermitian);
  return matrix_rank_op.result(0);
}

pir::OpResult matrix_rank_tol(const pir::Value& x,
                              const pir::Value& atol_tensor,
                              bool use_default_tol,
                              bool hermitian) {
  CheckValueDataType(atol_tensor, "atol_tensor", "matrix_rank_tol");
  paddle::dialect::MatrixRankTolOp matrix_rank_tol_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatrixRankTolOp>(
              x, atol_tensor, use_default_tol, hermitian);
  return matrix_rank_tol_op.result(0);
}

pir::OpResult max(const pir::Value& x,
                  const std::vector<int64_t>& axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "max");
  paddle::dialect::MaxOp max_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxOp>(
          x, axis, keepdim);
  return max_op.result(0);
}

pir::OpResult max(const pir::Value& x, pir::Value axis, bool keepdim) {
  CheckValueDataType(x, "x", "max");
  paddle::dialect::MaxOp max_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxOp>(
          x, axis, keepdim);
  return max_op.result(0);
}

pir::OpResult max(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "max");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::MaxOp max_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxOp>(
          x, axis_combine_op.out(), keepdim);
  return max_op.result(0);
}

pir::OpResult maximum(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "maximum");
  paddle::dialect::MaximumOp maximum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaximumOp>(x,
                                                                             y);
  return maximum_op.result(0);
}

pir::OpResult mean(const pir::Value& x,
                   const std::vector<int64_t>& axis,
                   bool keepdim) {
  CheckValueDataType(x, "x", "mean");
  paddle::dialect::MeanOp mean_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeanOp>(
          x, axis, keepdim);
  return mean_op.result(0);
}

pir::OpResult memcpy(const pir::Value& x, int dst_place_type) {
  CheckValueDataType(x, "x", "memcpy");
  paddle::dialect::MemcpyOp memcpy_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MemcpyOp>(
          x, dst_place_type);
  return memcpy_op.result(0);
}

pir::OpResult memcpy_d2h(const pir::Value& x, int dst_place_type) {
  CheckValueDataType(x, "x", "memcpy_d2h");
  paddle::dialect::MemcpyD2hOp memcpy_d2h_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MemcpyD2hOp>(
          x, dst_place_type);
  return memcpy_d2h_op.result(0);
}

pir::OpResult memcpy_h2d(const pir::Value& x, int dst_place_type) {
  CheckValueDataType(x, "x", "memcpy_h2d");
  paddle::dialect::MemcpyH2dOp memcpy_h2d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MemcpyH2dOp>(
          x, dst_place_type);
  return memcpy_h2d_op.result(0);
}

pir::OpResult min(const pir::Value& x,
                  const std::vector<int64_t>& axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "min");
  paddle::dialect::MinOp min_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinOp>(
          x, axis, keepdim);
  return min_op.result(0);
}

pir::OpResult min(const pir::Value& x, pir::Value axis, bool keepdim) {
  CheckValueDataType(x, "x", "min");
  paddle::dialect::MinOp min_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinOp>(
          x, axis, keepdim);
  return min_op.result(0);
}

pir::OpResult min(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  bool keepdim) {
  CheckValueDataType(x, "x", "min");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::MinOp min_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinOp>(
          x, axis_combine_op.out(), keepdim);
  return min_op.result(0);
}

pir::OpResult minimum(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "minimum");
  paddle::dialect::MinimumOp minimum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinimumOp>(x,
                                                                             y);
  return minimum_op.result(0);
}

pir::OpResult mish(const pir::Value& x, float lambda) {
  CheckValueDataType(x, "x", "mish");
  paddle::dialect::MishOp mish_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MishOp>(
          x, lambda);
  return mish_op.result(0);
}

pir::OpResult multiply(const pir::Value& x, const pir::Value& y) {
  if (x.type().isa<paddle::dialect::DenseTensorType>() &&
      y.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(y, "y", "multiply");
    paddle::dialect::MultiplyOp multiply_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MultiplyOp>(
            x, y);
    return multiply_op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>() &&
      y.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(y, "y", "multiply_sr");
    paddle::dialect::MultiplySrOp multiply_sr_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::MultiplySrOp>(x, y);
    return multiply_sr_op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (multiply) for input Value is unimplemented, please check "
      "the type of input Value."));
}

pir::OpResult multiply_(const pir::Value& x, const pir::Value& y) {
  if (x.type().isa<paddle::dialect::DenseTensorType>() &&
      y.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(y, "y", "multiply_");
    paddle::dialect::Multiply_Op multiply__op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::Multiply_Op>(x, y);
    return multiply__op.result(0);
  }
  if (x.type().isa<paddle::dialect::SelectedRowsType>() &&
      y.type().isa<paddle::dialect::DenseTensorType>()) {
    CheckValueDataType(y, "y", "multiply_sr_");
    paddle::dialect::MultiplySr_Op multiply_sr__op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::MultiplySr_Op>(x, y);
    return multiply_sr__op.result(0);
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "The kernel of (multiply_) for input Value is unimplemented, please "
      "check the type of input Value."));
}

std::tuple<pir::OpResult, pir::OpResult> norm(const pir::Value& x,
                                              int axis,
                                              float epsilon,
                                              bool is_test) {
  CheckValueDataType(x, "x", "norm");
  paddle::dialect::NormOp norm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NormOp>(
          x, axis, epsilon, is_test);
  return std::make_tuple(norm_op.result(0), norm_op.result(1));
}

pir::OpResult not_equal(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "not_equal");
  paddle::dialect::NotEqualOp not_equal_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NotEqualOp>(
          x, y);
  return not_equal_op.result(0);
}

pir::OpResult not_equal_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "not_equal_");
  paddle::dialect::NotEqual_Op not_equal__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NotEqual_Op>(
          x, y);
  return not_equal__op.result(0);
}

pir::OpResult one_hot(const pir::Value& x, int num_classes) {
  CheckValueDataType(x, "x", "one_hot");
  paddle::dialect::OneHotOp one_hot_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::OneHotOp>(
          x, num_classes);
  return one_hot_op.result(0);
}

pir::OpResult one_hot(const pir::Value& x, pir::Value num_classes) {
  CheckValueDataType(x, "x", "one_hot");
  paddle::dialect::OneHotOp one_hot_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::OneHotOp>(
          x, num_classes);
  return one_hot_op.result(0);
}

pir::OpResult pad(const pir::Value& x,
                  const std::vector<int>& paddings,
                  float pad_value) {
  CheckValueDataType(x, "x", "pad");
  paddle::dialect::PadOp pad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PadOp>(
          x, paddings, pad_value);
  return pad_op.result(0);
}

pir::OpResult pad(const pir::Value& x,
                  pir::Value pad_value,
                  const std::vector<int>& paddings) {
  CheckValueDataType(x, "x", "pad");
  paddle::dialect::PadOp pad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PadOp>(
          x, pad_value, paddings);
  return pad_op.result(0);
}

pir::OpResult pool2d(const pir::Value& x,
                     const std::vector<int64_t>& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm) {
  CheckValueDataType(x, "x", "pool2d");
  paddle::dialect::Pool2dOp pool2d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dOp>(
          x,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_op.result(0);
}

pir::OpResult pool2d(const pir::Value& x,
                     pir::Value kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm) {
  CheckValueDataType(x, "x", "pool2d");
  paddle::dialect::Pool2dOp pool2d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dOp>(
          x,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_op.result(0);
}

pir::OpResult pool2d(const pir::Value& x,
                     std::vector<pir::Value> kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm) {
  CheckValueDataType(x, "x", "pool2d");
  auto kernel_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(kernel_size);
  paddle::dialect::Pool2dOp pool2d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dOp>(
          x,
          kernel_size_combine_op.out(),
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_op.result(0);
}

pir::OpResult pool3d(const pir::Value& x,
                     const std::vector<int>& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm) {
  CheckValueDataType(x, "x", "pool3d");
  paddle::dialect::Pool3dOp pool3d_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool3dOp>(
          x,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool3d_op.result(0);
}

pir::OpResult print(const pir::Value& in,
                    int first_n,
                    const std::string& message,
                    int summarize,
                    bool print_tensor_name,
                    bool print_tensor_type,
                    bool print_tensor_shape,
                    bool print_tensor_layout,
                    bool print_tensor_lod,
                    const std::string& print_phase,
                    bool is_forward) {
  CheckValueDataType(in, "in", "print_kernel");
  paddle::dialect::PrintOp print_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PrintOp>(
          in,
          first_n,
          message,
          summarize,
          print_tensor_name,
          print_tensor_type,
          print_tensor_shape,
          print_tensor_layout,
          print_tensor_lod,
          print_phase,
          is_forward);
  return print_op.result(0);
}

pir::OpResult prod(const pir::Value& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all) {
  CheckValueDataType(x, "x", "prod");
  paddle::dialect::ProdOp prod_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdOp>(
          x, dims, keep_dim, reduce_all);
  return prod_op.result(0);
}

pir::OpResult prod(const pir::Value& x,
                   pir::Value dims,
                   bool keep_dim,
                   bool reduce_all) {
  CheckValueDataType(x, "x", "prod");
  paddle::dialect::ProdOp prod_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdOp>(
          x, dims, keep_dim, reduce_all);
  return prod_op.result(0);
}

pir::OpResult prod(const pir::Value& x,
                   std::vector<pir::Value> dims,
                   bool keep_dim,
                   bool reduce_all) {
  CheckValueDataType(x, "x", "prod");
  auto dims_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(dims);
  paddle::dialect::ProdOp prod_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdOp>(
          x, dims_combine_op.out(), keep_dim, reduce_all);
  return prod_op.result(0);
}

pir::OpResult randint(int low,
                      int high,
                      const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "randint");
  paddle::dialect::RandintOp randint_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RandintOp>(
          low, high, shape, dtype, place);
  return randint_op.result(0);
}

pir::OpResult randint(pir::Value shape,
                      int low,
                      int high,
                      phi::DataType dtype,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "randint");
  paddle::dialect::RandintOp randint_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RandintOp>(
          shape, low, high, dtype, place);
  return randint_op.result(0);
}

pir::OpResult randint(std::vector<pir::Value> shape,
                      int low,
                      int high,
                      phi::DataType dtype,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "randint");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::RandintOp randint_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RandintOp>(
          shape_combine_op.out(), low, high, dtype, place);
  return randint_op.result(0);
}

pir::OpResult randperm(int n, phi::DataType dtype, const Place& place) {
  CheckDataType(dtype, "dtype", "randperm");
  paddle::dialect::RandpermOp randperm_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RandpermOp>(
          n, dtype, place);
  return randperm_op.result(0);
}

pir::OpResult read_file(const std::string& filename,
                        phi::DataType dtype,
                        const Place& place) {
  CheckDataType(dtype, "dtype", "read_file");
  paddle::dialect::ReadFileOp read_file_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReadFileOp>(
          filename, dtype, place);
  return read_file_op.result(0);
}

pir::OpResult recv_v2(const std::vector<int>& out_shape,
                      phi::DataType dtype,
                      int peer,
                      int ring_id,
                      bool use_calc_stream,
                      bool dynamic_shape) {
  CheckDataType(dtype, "dtype", "recv_v2");
  paddle::dialect::RecvV2Op recv_v2_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RecvV2Op>(
          out_shape, dtype, peer, ring_id, use_calc_stream, dynamic_shape);
  return recv_v2_op.result(0);
}

pir::OpResult remainder(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "remainder");
  paddle::dialect::RemainderOp remainder_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RemainderOp>(
          x, y);
  return remainder_op.result(0);
}

pir::OpResult remainder_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "remainder_");
  paddle::dialect::Remainder_Op remainder__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Remainder_Op>(
          x, y);
  return remainder__op.result(0);
}

pir::OpResult repeat_interleave(const pir::Value& x, int repeats, int axis) {
  CheckValueDataType(x, "x", "repeat_interleave");
  paddle::dialect::RepeatInterleaveOp repeat_interleave_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RepeatInterleaveOp>(x, repeats, axis);
  return repeat_interleave_op.result(0);
}

pir::OpResult repeat_interleave_with_tensor_index(const pir::Value& x,
                                                  const pir::Value& repeats,
                                                  int axis) {
  CheckValueDataType(x, "x", "repeat_interleave_with_tensor_index");
  paddle::dialect::RepeatInterleaveWithTensorIndexOp
      repeat_interleave_with_tensor_index_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::RepeatInterleaveWithTensorIndexOp>(
                  x, repeats, axis);
  return repeat_interleave_with_tensor_index_op.result(0);
}

pir::OpResult reshape(const pir::Value& x, const std::vector<int64_t>& shape) {
  CheckValueDataType(x, "x", "reshape");
  paddle::dialect::ReshapeOp reshape_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReshapeOp>(
          x, shape);
  return reshape_op.result(0);
}

pir::OpResult reshape(const pir::Value& x, pir::Value shape) {
  CheckValueDataType(x, "x", "reshape");
  paddle::dialect::ReshapeOp reshape_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReshapeOp>(
          x, shape);
  return reshape_op.result(0);
}

pir::OpResult reshape(const pir::Value& x, std::vector<pir::Value> shape) {
  CheckValueDataType(x, "x", "reshape");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::ReshapeOp reshape_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ReshapeOp>(
          x, shape_combine_op.out());
  return reshape_op.result(0);
}

pir::OpResult reshape_(const pir::Value& x, const std::vector<int64_t>& shape) {
  CheckValueDataType(x, "x", "reshape_");
  paddle::dialect::Reshape_Op reshape__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Reshape_Op>(
          x, shape);
  return reshape__op.result(0);
}

pir::OpResult reshape_(const pir::Value& x, pir::Value shape) {
  CheckValueDataType(x, "x", "reshape_");
  paddle::dialect::Reshape_Op reshape__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Reshape_Op>(
          x, shape);
  return reshape__op.result(0);
}

pir::OpResult reshape_(const pir::Value& x, std::vector<pir::Value> shape) {
  CheckValueDataType(x, "x", "reshape_");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::Reshape_Op reshape__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Reshape_Op>(
          x, shape_combine_op.out());
  return reshape__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, std::vector<pir::OpResult>> rnn(
    const pir::Value& x,
    const std::vector<pir::Value>& pre_state,
    const std::vector<pir::Value>& weight_list,
    const paddle::optional<pir::Value>& sequence_length,
    const pir::Value& dropout_state_in,
    float dropout_prob,
    bool is_bidirec,
    int input_size,
    int hidden_size,
    int num_layers,
    const std::string& mode,
    int seed,
    bool is_test) {
  CheckValueDataType(x, "x", "rnn");
  paddle::optional<pir::Value> optional_sequence_length;
  if (!sequence_length) {
    optional_sequence_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sequence_length = sequence_length;
  }
  auto pre_state_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(pre_state);
  auto weight_list_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(weight_list);
  paddle::dialect::RnnOp rnn_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RnnOp>(
          x,
          pre_state_combine_op.out(),
          weight_list_combine_op.out(),
          optional_sequence_length.get(),
          dropout_state_in,
          dropout_prob,
          is_bidirec,
          input_size,
          hidden_size,
          num_layers,
          mode,
          seed,
          is_test);
  auto state_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          rnn_op.result(2));
  return std::make_tuple(
      rnn_op.result(0), rnn_op.result(1), state_split_op.outputs());
}

std::tuple<pir::OpResult, pir::OpResult, std::vector<pir::OpResult>> rnn_(
    const pir::Value& x,
    const std::vector<pir::Value>& pre_state,
    const std::vector<pir::Value>& weight_list,
    const paddle::optional<pir::Value>& sequence_length,
    const pir::Value& dropout_state_in,
    float dropout_prob,
    bool is_bidirec,
    int input_size,
    int hidden_size,
    int num_layers,
    const std::string& mode,
    int seed,
    bool is_test) {
  CheckValueDataType(x, "x", "rnn_");
  paddle::optional<pir::Value> optional_sequence_length;
  if (!sequence_length) {
    optional_sequence_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sequence_length = sequence_length;
  }
  auto pre_state_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(pre_state);
  auto weight_list_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(weight_list);
  paddle::dialect::Rnn_Op rnn__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Rnn_Op>(
          x,
          pre_state_combine_op.out(),
          weight_list_combine_op.out(),
          optional_sequence_length.get(),
          dropout_state_in,
          dropout_prob,
          is_bidirec,
          input_size,
          hidden_size,
          num_layers,
          mode,
          seed,
          is_test);
  auto state_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          rnn__op.result(2));
  return std::make_tuple(
      rnn__op.result(0), rnn__op.result(1), state_split_op.outputs());
}

pir::OpResult row_conv(const pir::Value& x, const pir::Value& filter) {
  CheckValueDataType(filter, "filter", "row_conv");
  paddle::dialect::RowConvOp row_conv_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RowConvOp>(
          x, filter);
  return row_conv_op.result(0);
}

pir::OpResult rrelu(const pir::Value& x,
                    float lower,
                    float upper,
                    bool is_test) {
  CheckValueDataType(x, "x", "rrelu");
  paddle::dialect::RreluOp rrelu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RreluOp>(
          x, lower, upper, is_test);
  return rrelu_op.result(0);
}

pir::OpResult seed(int seed,
                   bool deterministic,
                   const std::string& rng_name,
                   bool force_cpu) {
  paddle::dialect::SeedOp seed_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SeedOp>(
          seed, deterministic, rng_name, force_cpu);
  return seed_op.result(0);
}

void send_v2(const pir::Value& x,
             int ring_id,
             int peer,
             bool use_calc_stream,
             bool dynamic_shape) {
  CheckValueDataType(x, "x", "send_v2");
  paddle::dialect::SendV2Op send_v2_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SendV2Op>(
          x, ring_id, peer, use_calc_stream, dynamic_shape);
  (void)send_v2_op;
  return;
}

pir::OpResult set_value(const pir::Value& x,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        const std::vector<int64_t>& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value");
  paddle::dialect::SetValueOp set_value_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValueOp>(
          x,
          starts,
          ends,
          steps,
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value_op.result(0);
}

pir::OpResult set_value(const pir::Value& x,
                        pir::Value starts,
                        pir::Value ends,
                        pir::Value steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value");
  paddle::dialect::SetValueOp set_value_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValueOp>(
          x,
          starts,
          ends,
          steps,
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value_op.result(0);
}

pir::OpResult set_value(const pir::Value& x,
                        std::vector<pir::Value> starts,
                        std::vector<pir::Value> ends,
                        std::vector<pir::Value> steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto steps_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(steps);
  paddle::dialect::SetValueOp set_value_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValueOp>(
          x,
          starts_combine_op.out(),
          ends_combine_op.out(),
          steps_combine_op.out(),
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value_op.result(0);
}

pir::OpResult set_value_(const pir::Value& x,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value_");
  paddle::dialect::SetValue_Op set_value__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValue_Op>(
          x,
          starts,
          ends,
          steps,
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value__op.result(0);
}

pir::OpResult set_value_(const pir::Value& x,
                         pir::Value starts,
                         pir::Value ends,
                         pir::Value steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value_");
  paddle::dialect::SetValue_Op set_value__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValue_Op>(
          x,
          starts,
          ends,
          steps,
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value__op.result(0);
}

pir::OpResult set_value_(const pir::Value& x,
                         std::vector<pir::Value> starts,
                         std::vector<pir::Value> ends,
                         std::vector<pir::Value> steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values) {
  CheckValueDataType(x, "x", "set_value_");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto steps_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(steps);
  paddle::dialect::SetValue_Op set_value__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SetValue_Op>(
          x,
          starts_combine_op.out(),
          ends_combine_op.out(),
          steps_combine_op.out(),
          axes,
          decrease_axes,
          none_axes,
          shape,
          values);
  return set_value__op.result(0);
}

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    const std::vector<int64_t>& steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor");
  paddle::dialect::SetValueWithTensorOp set_value_with_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorOp>(
              x, values, starts, ends, steps, axes, decrease_axes, none_axes);
  return set_value_with_tensor_op.result(0);
}

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    pir::Value starts,
                                    pir::Value ends,
                                    pir::Value steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor");
  paddle::dialect::SetValueWithTensorOp set_value_with_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorOp>(
              x, values, starts, ends, steps, axes, decrease_axes, none_axes);
  return set_value_with_tensor_op.result(0);
}

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    std::vector<pir::Value> starts,
                                    std::vector<pir::Value> ends,
                                    std::vector<pir::Value> steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto steps_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(steps);
  paddle::dialect::SetValueWithTensorOp set_value_with_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorOp>(
              x,
              values,
              starts_combine_op.out(),
              ends_combine_op.out(),
              steps_combine_op.out(),
              axes,
              decrease_axes,
              none_axes);
  return set_value_with_tensor_op.result(0);
}

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t>& ends,
                                     const std::vector<int64_t>& steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor_");
  paddle::dialect::SetValueWithTensor_Op set_value_with_tensor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensor_Op>(
              x, values, starts, ends, steps, axes, decrease_axes, none_axes);
  return set_value_with_tensor__op.result(0);
}

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     pir::Value starts,
                                     pir::Value ends,
                                     pir::Value steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor_");
  paddle::dialect::SetValueWithTensor_Op set_value_with_tensor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensor_Op>(
              x, values, starts, ends, steps, axes, decrease_axes, none_axes);
  return set_value_with_tensor__op.result(0);
}

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     std::vector<pir::Value> starts,
                                     std::vector<pir::Value> ends,
                                     std::vector<pir::Value> steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes) {
  CheckValueDataType(values, "values", "set_value_with_tensor_");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto steps_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(steps);
  paddle::dialect::SetValueWithTensor_Op set_value_with_tensor__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensor_Op>(
              x,
              values,
              starts_combine_op.out(),
              ends_combine_op.out(),
              steps_combine_op.out(),
              axes,
              decrease_axes,
              none_axes);
  return set_value_with_tensor__op.result(0);
}

pir::OpResult shadow_feed(const pir::Value& x) {
  CheckValueDataType(x, "x", "shadow_feed");
  paddle::dialect::ShadowFeedOp shadow_feed_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ShadowFeedOp>(
          x);
  return shadow_feed_op.result(0);
}

pir::OpResult share_data(const pir::Value& x) {
  CheckValueDataType(x, "x", "share_data");
  paddle::dialect::ShareDataOp share_data_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ShareDataOp>(
          x);
  return share_data_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> shuffle_batch(
    const pir::Value& x, const pir::Value& seed, int startup_seed) {
  CheckValueDataType(x, "x", "shuffle_batch");
  paddle::dialect::ShuffleBatchOp shuffle_batch_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ShuffleBatchOp>(x, seed, startup_seed);
  return std::make_tuple(shuffle_batch_op.result(0),
                         shuffle_batch_op.result(1),
                         shuffle_batch_op.result(2));
}

pir::OpResult slice(const pir::Value& input,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(input, "input", "slice");
  paddle::dialect::SliceOp slice_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceOp>(
          input, axes, starts, ends, infer_flags, decrease_axis);
  return slice_op.result(0);
}

pir::OpResult slice(const pir::Value& input,
                    pir::Value starts,
                    pir::Value ends,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(input, "input", "slice");
  paddle::dialect::SliceOp slice_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceOp>(
          input, starts, ends, axes, infer_flags, decrease_axis);
  return slice_op.result(0);
}

pir::OpResult slice(const pir::Value& input,
                    std::vector<pir::Value> starts,
                    std::vector<pir::Value> ends,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(input, "input", "slice");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  paddle::dialect::SliceOp slice_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceOp>(
          input,
          starts_combine_op.out(),
          ends_combine_op.out(),
          axes,
          infer_flags,
          decrease_axis);
  return slice_op.result(0);
}

pir::OpResult soft_relu(const pir::Value& x, float threshold) {
  CheckValueDataType(x, "x", "soft_relu");
  paddle::dialect::SoftReluOp soft_relu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SoftReluOp>(
          x, threshold);
  return soft_relu_op.result(0);
}

pir::OpResult softmax(const pir::Value& x, int axis) {
  CheckValueDataType(x, "x", "softmax");
  paddle::dialect::SoftmaxOp softmax_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SoftmaxOp>(
          x, axis);
  return softmax_op.result(0);
}

pir::OpResult softmax_(const pir::Value& x, int axis) {
  CheckValueDataType(x, "x", "softmax_");
  paddle::dialect::Softmax_Op softmax__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Softmax_Op>(
          x, axis);
  return softmax__op.result(0);
}

std::vector<pir::OpResult> split(const pir::Value& x,
                                 const std::vector<int64_t>& sections,
                                 int axis) {
  CheckValueDataType(x, "x", "split");
  paddle::dialect::SplitOp split_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitOp>(
          x, sections, axis);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      split_op.result(0));
  return out_split_op.outputs();
}

std::vector<pir::OpResult> split(const pir::Value& x,
                                 pir::Value sections,
                                 pir::Value axis) {
  CheckValueDataType(x, "x", "split");
  paddle::dialect::SplitOp split_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitOp>(
          x, sections, axis);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      split_op.result(0));
  return out_split_op.outputs();
}

std::vector<pir::OpResult> split(const pir::Value& x,
                                 std::vector<pir::Value> sections,
                                 pir::Value axis) {
  CheckValueDataType(x, "x", "split");
  auto sections_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(sections);
  paddle::dialect::SplitOp split_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitOp>(
          x, sections_combine_op.out(), axis);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      split_op.result(0));
  return out_split_op.outputs();
}

std::vector<pir::OpResult> split_with_num(const pir::Value& x,
                                          int num,
                                          int axis) {
  CheckValueDataType(x, "x", "split_with_num");
  paddle::dialect::SplitWithNumOp split_with_num_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SplitWithNumOp>(x, num, axis);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      split_with_num_op.result(0));
  return out_split_op.outputs();
}

std::vector<pir::OpResult> split_with_num(const pir::Value& x,
                                          pir::Value axis,
                                          int num) {
  CheckValueDataType(x, "x", "split_with_num");
  paddle::dialect::SplitWithNumOp split_with_num_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SplitWithNumOp>(x, axis, num);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      split_with_num_op.result(0));
  return out_split_op.outputs();
}

pir::OpResult strided_slice(const pir::Value& x,
                            const std::vector<int>& axes,
                            const std::vector<int64_t>& starts,
                            const std::vector<int64_t>& ends,
                            const std::vector<int64_t>& strides) {
  CheckValueDataType(x, "x", "strided_slice");
  paddle::dialect::StridedSliceOp strided_slice_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceOp>(
              x, axes, starts, ends, strides);
  return strided_slice_op.result(0);
}

pir::OpResult strided_slice(const pir::Value& x,
                            pir::Value starts,
                            pir::Value ends,
                            pir::Value strides,
                            const std::vector<int>& axes) {
  CheckValueDataType(x, "x", "strided_slice");
  paddle::dialect::StridedSliceOp strided_slice_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceOp>(
              x, starts, ends, strides, axes);
  return strided_slice_op.result(0);
}

pir::OpResult strided_slice(const pir::Value& x,
                            std::vector<pir::Value> starts,
                            std::vector<pir::Value> ends,
                            std::vector<pir::Value> strides,
                            const std::vector<int>& axes) {
  CheckValueDataType(x, "x", "strided_slice");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto strides_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(strides);
  paddle::dialect::StridedSliceOp strided_slice_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceOp>(x,
                                                   starts_combine_op.out(),
                                                   ends_combine_op.out(),
                                                   strides_combine_op.out(),
                                                   axes);
  return strided_slice_op.result(0);
}

pir::OpResult subtract(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "subtract");
  paddle::dialect::SubtractOp subtract_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SubtractOp>(
          x, y);
  return subtract_op.result(0);
}

pir::OpResult subtract_(const pir::Value& x, const pir::Value& y) {
  CheckValueDataType(y, "y", "subtract_");
  paddle::dialect::Subtract_Op subtract__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Subtract_Op>(
          x, y);
  return subtract__op.result(0);
}

pir::OpResult sum(const pir::Value& x,
                  const std::vector<int64_t>& axis,
                  phi::DataType dtype,
                  bool keepdim) {
  CheckValueDataType(x, "x", "sum");
  paddle::dialect::SumOp sum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumOp>(
          x, axis, dtype, keepdim);
  return sum_op.result(0);
}

pir::OpResult sum(const pir::Value& x,
                  pir::Value axis,
                  phi::DataType dtype,
                  bool keepdim) {
  CheckValueDataType(x, "x", "sum");
  paddle::dialect::SumOp sum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumOp>(
          x, axis, dtype, keepdim);
  return sum_op.result(0);
}

pir::OpResult sum(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  phi::DataType dtype,
                  bool keepdim) {
  CheckValueDataType(x, "x", "sum");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::SumOp sum_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumOp>(
          x, axis_combine_op.out(), dtype, keepdim);
  return sum_op.result(0);
}

pir::OpResult swish(const pir::Value& x) {
  CheckValueDataType(x, "x", "swish");
  paddle::dialect::SwishOp swish_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SwishOp>(x);
  return swish_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
sync_batch_norm_(const pir::Value& x,
                 const pir::Value& mean,
                 const pir::Value& variance,
                 const pir::Value& scale,
                 const pir::Value& bias,
                 bool is_test,
                 float momentum,
                 float epsilon,
                 const std::string& data_layout,
                 bool use_global_stats,
                 bool trainable_statistics) {
  CheckValueDataType(x, "x", "sync_batch_norm_");
  paddle::dialect::SyncBatchNorm_Op sync_batch_norm__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SyncBatchNorm_Op>(x,
                                                     mean,
                                                     variance,
                                                     scale,
                                                     bias,
                                                     is_test,
                                                     momentum,
                                                     epsilon,
                                                     data_layout,
                                                     use_global_stats,
                                                     trainable_statistics);
  return std::make_tuple(sync_batch_norm__op.result(0),
                         sync_batch_norm__op.result(1),
                         sync_batch_norm__op.result(2),
                         sync_batch_norm__op.result(3),
                         sync_batch_norm__op.result(4),
                         sync_batch_norm__op.result(5));
}

pir::OpResult tile(const pir::Value& x,
                   const std::vector<int64_t>& repeat_times) {
  CheckValueDataType(x, "x", "tile");
  paddle::dialect::TileOp tile_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileOp>(
          x, repeat_times);
  return tile_op.result(0);
}

pir::OpResult tile(const pir::Value& x, pir::Value repeat_times) {
  CheckValueDataType(x, "x", "tile");
  paddle::dialect::TileOp tile_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileOp>(
          x, repeat_times);
  return tile_op.result(0);
}

pir::OpResult tile(const pir::Value& x, std::vector<pir::Value> repeat_times) {
  CheckValueDataType(x, "x", "tile");
  auto repeat_times_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(repeat_times);
  paddle::dialect::TileOp tile_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileOp>(
          x, repeat_times_combine_op.out());
  return tile_op.result(0);
}

pir::OpResult trans_layout(const pir::Value& x, const std::vector<int>& perm) {
  CheckValueDataType(x, "x", "transpose");
  paddle::dialect::TransLayoutOp trans_layout_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TransLayoutOp>(x, perm);
  return trans_layout_op.result(0);
}

pir::OpResult transpose(const pir::Value& x, const std::vector<int>& perm) {
  CheckValueDataType(x, "x", "transpose");
  paddle::dialect::TransposeOp transpose_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TransposeOp>(
          x, perm);
  return transpose_op.result(0);
}

pir::OpResult transpose_(const pir::Value& x, const std::vector<int>& perm) {
  CheckValueDataType(x, "x", "transpose_");
  paddle::dialect::Transpose_Op transpose__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Transpose_Op>(
          x, perm);
  return transpose__op.result(0);
}

pir::OpResult tril(const pir::Value& x, int diagonal) {
  CheckValueDataType(x, "x", "tril");
  paddle::dialect::TrilOp tril_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TrilOp>(
          x, diagonal);
  return tril_op.result(0);
}

pir::OpResult tril_(const pir::Value& x, int diagonal) {
  CheckValueDataType(x, "x", "tril_");
  paddle::dialect::Tril_Op tril__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Tril_Op>(
          x, diagonal);
  return tril__op.result(0);
}

pir::OpResult tril_indices(
    int rows, int cols, int offset, phi::DataType dtype, const Place& place) {
  CheckDataType(dtype, "dtype", "tril_indices");
  paddle::dialect::TrilIndicesOp tril_indices_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TrilIndicesOp>(
              rows, cols, offset, dtype, place);
  return tril_indices_op.result(0);
}

pir::OpResult triu(const pir::Value& x, int diagonal) {
  CheckValueDataType(x, "x", "triu");
  paddle::dialect::TriuOp triu_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TriuOp>(
          x, diagonal);
  return triu_op.result(0);
}

pir::OpResult triu_(const pir::Value& x, int diagonal) {
  CheckValueDataType(x, "x", "triu_");
  paddle::dialect::Triu_Op triu__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Triu_Op>(
          x, diagonal);
  return triu__op.result(0);
}

pir::OpResult triu_indices(
    int row, int col, int offset, phi::DataType dtype, const Place& place) {
  CheckDataType(dtype, "dtype", "triu_indices");
  paddle::dialect::TriuIndicesOp triu_indices_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TriuIndicesOp>(
              row, col, offset, dtype, place);
  return triu_indices_op.result(0);
}

pir::OpResult truncated_gaussian_random(const std::vector<int>& shape,
                                        float mean,
                                        float std,
                                        int seed,
                                        phi::DataType dtype,
                                        const Place& place) {
  CheckDataType(dtype, "dtype", "truncated_gaussian_random");
  paddle::dialect::TruncatedGaussianRandomOp truncated_gaussian_random_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TruncatedGaussianRandomOp>(
              shape, mean, std, seed, dtype, place);
  return truncated_gaussian_random_op.result(0);
}

pir::OpResult uniform(const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      float min,
                      float max,
                      int seed,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "uniform");
  paddle::dialect::UniformOp uniform_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UniformOp>(
          shape, dtype, min, max, seed, place);
  return uniform_op.result(0);
}

pir::OpResult uniform(pir::Value shape,
                      pir::Value min,
                      pir::Value max,
                      phi::DataType dtype,
                      int seed,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "uniform");
  paddle::dialect::UniformOp uniform_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UniformOp>(
          shape, min, max, dtype, seed, place);
  return uniform_op.result(0);
}

pir::OpResult uniform(std::vector<pir::Value> shape,
                      pir::Value min,
                      pir::Value max,
                      phi::DataType dtype,
                      int seed,
                      const Place& place) {
  CheckDataType(dtype, "dtype", "uniform");
  auto shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(shape);
  paddle::dialect::UniformOp uniform_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UniformOp>(
          shape_combine_op.out(), min, max, dtype, seed, place);
  return uniform_op.result(0);
}

pir::OpResult uniform_random_batch_size_like(const pir::Value& input,
                                             const std::vector<int>& shape,
                                             int input_dim_idx,
                                             int output_dim_idx,
                                             float min,
                                             float max,
                                             int seed,
                                             int diag_num,
                                             int diag_step,
                                             float diag_val,
                                             phi::DataType dtype) {
  CheckDataType(dtype, "dtype", "uniform_random_batch_size_like");
  paddle::dialect::UniformRandomBatchSizeLikeOp
      uniform_random_batch_size_like_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::UniformRandomBatchSizeLikeOp>(
                  input,
                  shape,
                  input_dim_idx,
                  output_dim_idx,
                  min,
                  max,
                  seed,
                  diag_num,
                  diag_step,
                  diag_val,
                  dtype);
  return uniform_random_batch_size_like_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> unique(
    const pir::Value& x,
    bool return_index,
    bool return_inverse,
    bool return_counts,
    const std::vector<int>& axis,
    phi::DataType dtype,
    bool is_sorted) {
  CheckValueDataType(x, "x", "unique");
  paddle::dialect::UniqueOp unique_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UniqueOp>(
          x,
          return_index,
          return_inverse,
          return_counts,
          axis,
          dtype,
          is_sorted);
  return std::make_tuple(unique_op.result(0),
                         unique_op.result(1),
                         unique_op.result(2),
                         unique_op.result(3));
}

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::vector<int64_t>& output_size,
                     const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool");
  paddle::dialect::UnpoolOp unpool_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolOp>(
          x, indices, ksize, strides, padding, output_size, data_format);
  return unpool_op.result(0);
}

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     pir::Value output_size,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool");
  paddle::dialect::UnpoolOp unpool_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolOp>(
          x, indices, output_size, ksize, strides, padding, data_format);
  return unpool_op.result(0);
}

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     std::vector<pir::Value> output_size,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::UnpoolOp unpool_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolOp>(
          x,
          indices,
          output_size_combine_op.out(),
          ksize,
          strides,
          padding,
          data_format);
  return unpool_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> c_softmax_with_cross_entropy(
    const pir::Value& logits,
    const pir::Value& label,
    int64_t ignore_index,
    int ring_id,
    int rank,
    int nranks) {
  CheckValueDataType(logits, "logits", "c_softmax_with_cross_entropy");
  paddle::dialect::CSoftmaxWithCrossEntropyOp c_softmax_with_cross_entropy_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CSoftmaxWithCrossEntropyOp>(
              logits, label, ignore_index, ring_id, rank, nranks);
  return std::make_tuple(c_softmax_with_cross_entropy_op.result(0),
                         c_softmax_with_cross_entropy_op.result(1));
}

pir::OpResult dpsgd(const pir::Value& param,
                    const pir::Value& grad,
                    const pir::Value& learning_rate,
                    float clip,
                    float batch_size,
                    float sigma,
                    int seed) {
  CheckValueDataType(param, "param", "dpsgd");
  paddle::dialect::DpsgdOp dpsgd_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DpsgdOp>(
          param, grad, learning_rate, clip, batch_size, sigma, seed);
  return dpsgd_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> ftrl(
    const pir::Value& param,
    const pir::Value& squared_accumulator,
    const pir::Value& linear_accumulator,
    const pir::Value& grad,
    const pir::Value& learning_rate,
    float l1,
    float l2,
    float lr_power) {
  CheckValueDataType(param, "param", "ftrl");
  paddle::dialect::FtrlOp ftrl_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FtrlOp>(
          param,
          squared_accumulator,
          linear_accumulator,
          grad,
          learning_rate,
          l1,
          l2,
          lr_power);
  return std::make_tuple(
      ftrl_op.result(0), ftrl_op.result(1), ftrl_op.result(2));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_attention(const pir::Value& x,
                const paddle::optional<pir::Value>& ln_scale,
                const paddle::optional<pir::Value>& ln_bias,
                const pir::Value& qkv_weight,
                const paddle::optional<pir::Value>& qkv_bias,
                const paddle::optional<pir::Value>& cache_kv,
                const paddle::optional<pir::Value>& src_mask,
                const pir::Value& out_linear_weight,
                const paddle::optional<pir::Value>& out_linear_bias,
                const paddle::optional<pir::Value>& ln_scale_2,
                const paddle::optional<pir::Value>& ln_bias_2,
                int num_heads,
                bool transpose_qkv_wb,
                bool pre_layer_norm,
                float epsilon,
                float attn_dropout_rate,
                bool is_test,
                bool attn_dropout_fix_seed,
                int attn_dropout_seed,
                const std::string& attn_dropout_implementation,
                float dropout_rate,
                bool dropout_fix_seed,
                int dropout_seed,
                const std::string& dropout_implementation,
                float ln_epsilon,
                bool add_residual,
                int ring_id) {
  CheckValueDataType(x, "x", "fused_attention");
  paddle::optional<pir::Value> optional_ln_scale;
  if (!ln_scale) {
    optional_ln_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale = ln_scale;
  }
  paddle::optional<pir::Value> optional_ln_bias;
  if (!ln_bias) {
    optional_ln_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias = ln_bias;
  }
  paddle::optional<pir::Value> optional_qkv_bias;
  if (!qkv_bias) {
    optional_qkv_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_bias = qkv_bias;
  }
  paddle::optional<pir::Value> optional_cache_kv;
  if (!cache_kv) {
    optional_cache_kv = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_cache_kv = cache_kv;
  }
  paddle::optional<pir::Value> optional_src_mask;
  if (!src_mask) {
    optional_src_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask = src_mask;
  }
  paddle::optional<pir::Value> optional_out_linear_bias;
  if (!out_linear_bias) {
    optional_out_linear_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_linear_bias = out_linear_bias;
  }
  paddle::optional<pir::Value> optional_ln_scale_2;
  if (!ln_scale_2) {
    optional_ln_scale_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale_2 = ln_scale_2;
  }
  paddle::optional<pir::Value> optional_ln_bias_2;
  if (!ln_bias_2) {
    optional_ln_bias_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias_2 = ln_bias_2;
  }
  paddle::dialect::FusedAttentionOp fused_attention_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedAttentionOp>(
              x,
              optional_ln_scale.get(),
              optional_ln_bias.get(),
              qkv_weight,
              optional_qkv_bias.get(),
              optional_cache_kv.get(),
              optional_src_mask.get(),
              out_linear_weight,
              optional_out_linear_bias.get(),
              optional_ln_scale_2.get(),
              optional_ln_bias_2.get(),
              num_heads,
              transpose_qkv_wb,
              pre_layer_norm,
              epsilon,
              attn_dropout_rate,
              is_test,
              attn_dropout_fix_seed,
              attn_dropout_seed,
              attn_dropout_implementation,
              dropout_rate,
              dropout_fix_seed,
              dropout_seed,
              dropout_implementation,
              ln_epsilon,
              add_residual,
              ring_id);
  return std::make_tuple(fused_attention_op.result(0),
                         fused_attention_op.result(1),
                         fused_attention_op.result(2),
                         fused_attention_op.result(3),
                         fused_attention_op.result(4),
                         fused_attention_op.result(5),
                         fused_attention_op.result(6),
                         fused_attention_op.result(7),
                         fused_attention_op.result(8),
                         fused_attention_op.result(9),
                         fused_attention_op.result(10),
                         fused_attention_op.result(11),
                         fused_attention_op.result(12),
                         fused_attention_op.result(13),
                         fused_attention_op.result(14),
                         fused_attention_op.result(15),
                         fused_attention_op.result(16),
                         fused_attention_op.result(17),
                         fused_attention_op.result(18),
                         fused_attention_op.result(19));
}

pir::OpResult fused_elemwise_add_activation(
    const pir::Value& x,
    const pir::Value& y,
    const std::vector<std::string>& functor_list,
    float scale,
    int axis,
    bool save_intermediate_out) {
  CheckValueDataType(y, "y", "fused_elemwise_add_activation");
  paddle::dialect::FusedElemwiseAddActivationOp
      fused_elemwise_add_activation_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedElemwiseAddActivationOp>(
                  x, y, functor_list, scale, axis, save_intermediate_out);
  return fused_elemwise_add_activation_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_feedforward(const pir::Value& x,
                  const paddle::optional<pir::Value>& dropout1_seed,
                  const paddle::optional<pir::Value>& dropout2_seed,
                  const pir::Value& linear1_weight,
                  const paddle::optional<pir::Value>& linear1_bias,
                  const pir::Value& linear2_weight,
                  const paddle::optional<pir::Value>& linear2_bias,
                  const paddle::optional<pir::Value>& ln1_scale,
                  const paddle::optional<pir::Value>& ln1_bias,
                  const paddle::optional<pir::Value>& ln2_scale,
                  const paddle::optional<pir::Value>& ln2_bias,
                  bool pre_layer_norm,
                  float ln1_epsilon,
                  float ln2_epsilon,
                  const std::string& act_method,
                  float dropout1_prob,
                  float dropout2_prob,
                  const std::string& dropout1_implementation,
                  const std::string& dropout2_implementation,
                  bool is_test,
                  bool dropout1_fix_seed,
                  bool dropout2_fix_seed,
                  int dropout1_seed_val,
                  int dropout2_seed_val,
                  bool add_residual,
                  int ring_id) {
  CheckValueDataType(x, "x", "fused_feedforward");
  paddle::optional<pir::Value> optional_dropout1_seed;
  if (!dropout1_seed) {
    optional_dropout1_seed = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dropout1_seed = dropout1_seed;
  }
  paddle::optional<pir::Value> optional_dropout2_seed;
  if (!dropout2_seed) {
    optional_dropout2_seed = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dropout2_seed = dropout2_seed;
  }
  paddle::optional<pir::Value> optional_linear1_bias;
  if (!linear1_bias) {
    optional_linear1_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_linear1_bias = linear1_bias;
  }
  paddle::optional<pir::Value> optional_linear2_bias;
  if (!linear2_bias) {
    optional_linear2_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_linear2_bias = linear2_bias;
  }
  paddle::optional<pir::Value> optional_ln1_scale;
  if (!ln1_scale) {
    optional_ln1_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_scale = ln1_scale;
  }
  paddle::optional<pir::Value> optional_ln1_bias;
  if (!ln1_bias) {
    optional_ln1_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_bias = ln1_bias;
  }
  paddle::optional<pir::Value> optional_ln2_scale;
  if (!ln2_scale) {
    optional_ln2_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_scale = ln2_scale;
  }
  paddle::optional<pir::Value> optional_ln2_bias;
  if (!ln2_bias) {
    optional_ln2_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_bias = ln2_bias;
  }
  paddle::dialect::FusedFeedforwardOp fused_feedforward_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedFeedforwardOp>(
              x,
              optional_dropout1_seed.get(),
              optional_dropout2_seed.get(),
              linear1_weight,
              optional_linear1_bias.get(),
              linear2_weight,
              optional_linear2_bias.get(),
              optional_ln1_scale.get(),
              optional_ln1_bias.get(),
              optional_ln2_scale.get(),
              optional_ln2_bias.get(),
              pre_layer_norm,
              ln1_epsilon,
              ln2_epsilon,
              act_method,
              dropout1_prob,
              dropout2_prob,
              dropout1_implementation,
              dropout2_implementation,
              is_test,
              dropout1_fix_seed,
              dropout2_fix_seed,
              dropout1_seed_val,
              dropout2_seed_val,
              add_residual,
              ring_id);
  return std::make_tuple(fused_feedforward_op.result(0),
                         fused_feedforward_op.result(1),
                         fused_feedforward_op.result(2),
                         fused_feedforward_op.result(3),
                         fused_feedforward_op.result(4),
                         fused_feedforward_op.result(5),
                         fused_feedforward_op.result(6),
                         fused_feedforward_op.result(7),
                         fused_feedforward_op.result(8),
                         fused_feedforward_op.result(9),
                         fused_feedforward_op.result(10));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> nce(
    const pir::Value& input,
    const pir::Value& label,
    const pir::Value& weight,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& sample_weight,
    const paddle::optional<pir::Value>& custom_dist_probs,
    const paddle::optional<pir::Value>& custom_dist_alias,
    const paddle::optional<pir::Value>& custom_dist_alias_probs,
    int num_total_classes,
    const std::vector<int>& custom_neg_classes,
    int num_neg_samples,
    int sampler,
    int seed,
    bool is_sparse,
    bool remote_prefetch,
    bool is_test) {
  CheckValueDataType(input, "input", "nce");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_sample_weight;
  if (!sample_weight) {
    optional_sample_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sample_weight = sample_weight;
  }
  paddle::optional<pir::Value> optional_custom_dist_probs;
  if (!custom_dist_probs) {
    optional_custom_dist_probs =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_probs = custom_dist_probs;
  }
  paddle::optional<pir::Value> optional_custom_dist_alias;
  if (!custom_dist_alias) {
    optional_custom_dist_alias =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_alias = custom_dist_alias;
  }
  paddle::optional<pir::Value> optional_custom_dist_alias_probs;
  if (!custom_dist_alias_probs) {
    optional_custom_dist_alias_probs =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_alias_probs = custom_dist_alias_probs;
  }
  paddle::dialect::NceOp nce_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NceOp>(
          input,
          label,
          weight,
          optional_bias.get(),
          optional_sample_weight.get(),
          optional_custom_dist_probs.get(),
          optional_custom_dist_alias.get(),
          optional_custom_dist_alias_probs.get(),
          num_total_classes,
          custom_neg_classes,
          num_neg_samples,
          sampler,
          seed,
          is_sparse,
          remote_prefetch,
          is_test);
  return std::make_tuple(nce_op.result(0), nce_op.result(1), nce_op.result(2));
}

pir::OpResult number_count(const pir::Value& numbers, int upper_range) {
  CheckValueDataType(numbers, "numbers", "number_count");
  paddle::dialect::NumberCountOp number_count_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::NumberCountOp>(numbers, upper_range);
  return number_count_op.result(0);
}

pir::OpResult onednn_to_paddle_layout(const pir::Value& x, int dst_layout) {
  CheckValueDataType(x, "x", "onednn_to_paddle_layout");
  paddle::dialect::OnednnToPaddleLayoutOp onednn_to_paddle_layout_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::OnednnToPaddleLayoutOp>(x, dst_layout);
  return onednn_to_paddle_layout_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sparse_momentum(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& velocity,
    const pir::Value& index,
    const pir::Value& learning_rate,
    const paddle::optional<pir::Value>& master_param,
    float mu,
    float axis,
    bool use_nesterov,
    const std::string& regularization_method,
    float regularization_coeff,
    bool multi_precision,
    float rescale_grad) {
  CheckValueDataType(param, "param", "sparse_momentum");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::dialect::SparseMomentumOp sparse_momentum_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SparseMomentumOp>(
              param,
              grad,
              velocity,
              index,
              learning_rate,
              optional_master_param.get(),
              mu,
              axis,
              use_nesterov,
              regularization_method,
              regularization_coeff,
              multi_precision,
              rescale_grad);
  return std::make_tuple(sparse_momentum_op.result(0),
                         sparse_momentum_op.result(1),
                         sparse_momentum_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sparse_momentum(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& velocity,
    const pir::Value& index,
    const pir::Value& learning_rate,
    const paddle::optional<pir::Value>& master_param,
    pir::Value axis,
    float mu,
    bool use_nesterov,
    const std::string& regularization_method,
    float regularization_coeff,
    bool multi_precision,
    float rescale_grad) {
  CheckValueDataType(param, "param", "sparse_momentum");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_master_param = master_param;
  }
  paddle::dialect::SparseMomentumOp sparse_momentum_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SparseMomentumOp>(
              param,
              grad,
              velocity,
              index,
              learning_rate,
              optional_master_param.get(),
              axis,
              mu,
              use_nesterov,
              regularization_method,
              regularization_coeff,
              multi_precision,
              rescale_grad);
  return std::make_tuple(sparse_momentum_op.result(0),
                         sparse_momentum_op.result(1),
                         sparse_momentum_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> match_matrix_tensor(
    const pir::Value& x, const pir::Value& y, const pir::Value& w, int dim_t) {
  CheckValueDataType(w, "w", "match_matrix_tensor");
  paddle::dialect::MatchMatrixTensorOp match_matrix_tensor_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatchMatrixTensorOp>(x, y, w, dim_t);
  return std::make_tuple(match_matrix_tensor_op.result(0),
                         match_matrix_tensor_op.result(1));
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
lars_momentum(const std::vector<pir::Value>& param,
              const std::vector<pir::Value>& grad,
              const std::vector<pir::Value>& velocity,
              const std::vector<pir::Value>& learning_rate,
              const paddle::optional<std::vector<pir::Value>>& master_param,
              float mu,
              float lars_coeff,
              const std::vector<float>& lars_weight_decay,
              float epsilon,
              bool multi_precision,
              float rescale_grad) {
  CheckVectorOfValueDataType(param, "param", "lars_momentum");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_param_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_param.get());
    optional_master_param = paddle::make_optional<pir::Value>(
        optional_master_param_combine_op.out());
  }
  auto param_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(param);
  auto grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grad);
  auto velocity_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(velocity);
  auto learning_rate_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(learning_rate);
  paddle::dialect::LarsMomentumOp lars_momentum_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LarsMomentumOp>(
              param_combine_op.out(),
              grad_combine_op.out(),
              velocity_combine_op.out(),
              learning_rate_combine_op.out(),
              optional_master_param.get(),
              mu,
              lars_coeff,
              lars_weight_decay,
              epsilon,
              multi_precision,
              rescale_grad);
  paddle::optional<std::vector<pir::OpResult>> optional_master_param_out;
  if (!IsEmptyValue(lars_momentum_op.result(2))) {
    auto optional_master_param_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            lars_momentum_op.result(2));
    optional_master_param_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_param_out_slice_op.outputs());
  }
  if (!master_param) {
    lars_momentum_op.result(2).set_type(pir::Type());
  }
  auto param_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          lars_momentum_op.result(0));
  auto velocity_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          lars_momentum_op.result(1));
  return std::make_tuple(param_out_split_op.outputs(),
                         velocity_out_split_op.outputs(),
                         optional_master_param_out);
}

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
lars_momentum_(const std::vector<pir::Value>& param,
               const std::vector<pir::Value>& grad,
               const std::vector<pir::Value>& velocity,
               const std::vector<pir::Value>& learning_rate,
               const paddle::optional<std::vector<pir::Value>>& master_param,
               float mu,
               float lars_coeff,
               const std::vector<float>& lars_weight_decay,
               float epsilon,
               bool multi_precision,
               float rescale_grad) {
  CheckVectorOfValueDataType(param, "param", "lars_momentum_");
  paddle::optional<pir::Value> optional_master_param;
  if (!master_param) {
    optional_master_param = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    auto optional_master_param_combine_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(
            master_param.get());
    optional_master_param = paddle::make_optional<pir::Value>(
        optional_master_param_combine_op.out());
  }
  auto param_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(param);
  auto grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(grad);
  auto velocity_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(velocity);
  auto learning_rate_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(learning_rate);
  paddle::dialect::LarsMomentum_Op lars_momentum__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LarsMomentum_Op>(
              param_combine_op.out(),
              grad_combine_op.out(),
              velocity_combine_op.out(),
              learning_rate_combine_op.out(),
              optional_master_param.get(),
              mu,
              lars_coeff,
              lars_weight_decay,
              epsilon,
              multi_precision,
              rescale_grad);
  paddle::optional<std::vector<pir::OpResult>> optional_master_param_out;
  if (!IsEmptyValue(lars_momentum__op.result(2))) {
    auto optional_master_param_out_slice_op =
        ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
            lars_momentum__op.result(2));
    optional_master_param_out =
        paddle::make_optional<std::vector<pir::OpResult>>(
            optional_master_param_out_slice_op.outputs());
  }
  if (!master_param) {
    lars_momentum__op.result(2).set_type(pir::Type());
  }
  auto param_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          lars_momentum__op.result(0));
  auto velocity_out_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          lars_momentum__op.result(1));
  return std::make_tuple(param_out_split_op.outputs(),
                         velocity_out_split_op.outputs(),
                         optional_master_param_out);
}

pir::OpResult add_double_grad(const pir::Value& y,
                              const pir::Value& grad_out,
                              const paddle::optional<pir::Value>& grad_x_grad,
                              const paddle::optional<pir::Value>& grad_y_grad,
                              int axis) {
  if (grad_y_grad) {
    CheckValueDataType(grad_y_grad.get(), "grad_y_grad", "add_double_grad");
  } else if (grad_x_grad) {
    CheckValueDataType(grad_x_grad.get(), "grad_x_grad", "add_double_grad");
  } else {
    CheckValueDataType(grad_out, "grad_out", "add_double_grad");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::AddDoubleGradOp add_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddDoubleGradOp>(y,
                                                    grad_out,
                                                    optional_grad_x_grad.get(),
                                                    optional_grad_y_grad.get(),
                                                    axis);
  return add_double_grad_op.result(0);
}

pir::OpResult add_double_grad_(const pir::Value& y,
                               const pir::Value& grad_out,
                               const paddle::optional<pir::Value>& grad_x_grad,
                               const paddle::optional<pir::Value>& grad_y_grad,
                               int axis) {
  if (grad_y_grad) {
    CheckValueDataType(grad_y_grad.get(), "grad_y_grad", "add_double_grad_");
  } else if (grad_x_grad) {
    CheckValueDataType(grad_x_grad.get(), "grad_x_grad", "add_double_grad_");
  } else {
    CheckValueDataType(grad_out, "grad_out", "add_double_grad_");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::AddDoubleGrad_Op add_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddDoubleGrad_Op>(y,
                                                     grad_out,
                                                     optional_grad_x_grad.get(),
                                                     optional_grad_y_grad.get(),
                                                     axis);
  return add_double_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> add_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad,
                                                  int axis) {
  CheckValueDataType(out_grad, "out_grad", "add_grad");
  paddle::dialect::AddGradOp add_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddGradOp>(
          x, y, out_grad, axis);
  return std::make_tuple(add_grad_op.result(0), add_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> add_grad_(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad,
                                                   int axis) {
  CheckValueDataType(out_grad, "out_grad", "add_grad_");
  paddle::dialect::AddGrad_Op add_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddGrad_Op>(
          x, y, out_grad, axis);
  return std::make_tuple(add_grad__op.result(0), add_grad__op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> add_triple_grad(
    const pir::Value& grad_grad_x,
    const pir::Value& grad_grad_y,
    const pir::Value& grad_grad_out_grad,
    int axis) {
  CheckValueDataType(
      grad_grad_out_grad, "grad_grad_out_grad", "add_triple_grad");
  paddle::dialect::AddTripleGradOp add_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddTripleGradOp>(
              grad_grad_x, grad_grad_y, grad_grad_out_grad, axis);
  return std::make_tuple(add_triple_grad_op.result(0),
                         add_triple_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> add_triple_grad_(
    const pir::Value& grad_grad_x,
    const pir::Value& grad_grad_y,
    const pir::Value& grad_grad_out_grad,
    int axis) {
  CheckValueDataType(
      grad_grad_out_grad, "grad_grad_out_grad", "add_triple_grad_");
  paddle::dialect::AddTripleGrad_Op add_triple_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AddTripleGrad_Op>(
              grad_grad_x, grad_grad_y, grad_grad_out_grad, axis);
  return std::make_tuple(add_triple_grad__op.result(0),
                         add_triple_grad__op.result(1));
}

pir::OpResult amax_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "amax_grad");
  paddle::dialect::AmaxGradOp amax_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AmaxGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return amax_grad_op.result(0);
}

pir::OpResult amin_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "amin_grad");
  paddle::dialect::AminGradOp amin_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AminGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return amin_grad_op.result(0);
}

pir::OpResult assign_out__grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "assign");
  paddle::dialect::AssignOutGradOp assign_out__grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AssignOutGradOp>(out_grad);
  return assign_out__grad_op.result(0);
}

pir::OpResult assign_out__grad_(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "assign_");
  paddle::dialect::AssignOutGrad_Op assign_out__grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::AssignOutGrad_Op>(out_grad);
  return assign_out__grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& out_mean,
    const paddle::optional<pir::Value>& out_variance,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_scale_grad,
    const paddle::optional<pir::Value>& grad_bias_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics) {
  CheckValueDataType(x, "x", "batch_norm_double_grad");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_out_mean;
  if (!out_mean) {
    optional_out_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_mean = out_mean;
  }
  paddle::optional<pir::Value> optional_out_variance;
  if (!out_variance) {
    optional_out_variance = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_variance = out_variance;
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_scale_grad;
  if (!grad_scale_grad) {
    optional_grad_scale_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_scale_grad = grad_scale_grad;
  }
  paddle::optional<pir::Value> optional_grad_bias_grad;
  if (!grad_bias_grad) {
    optional_grad_bias_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_bias_grad = grad_bias_grad;
  }
  paddle::dialect::BatchNormDoubleGradOp batch_norm_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BatchNormDoubleGradOp>(
              x,
              optional_scale.get(),
              optional_out_mean.get(),
              optional_out_variance.get(),
              saved_mean,
              saved_variance,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_scale_grad.get(),
              optional_grad_bias_grad.get(),
              momentum,
              epsilon,
              data_layout,
              is_test,
              use_global_stats,
              trainable_statistics);
  return std::make_tuple(batch_norm_double_grad_op.result(0),
                         batch_norm_double_grad_op.result(1),
                         batch_norm_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& out_mean,
    const paddle::optional<pir::Value>& out_variance,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_scale_grad,
    const paddle::optional<pir::Value>& grad_bias_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics) {
  CheckValueDataType(x, "x", "batch_norm_double_grad_");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_out_mean;
  if (!out_mean) {
    optional_out_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_mean = out_mean;
  }
  paddle::optional<pir::Value> optional_out_variance;
  if (!out_variance) {
    optional_out_variance = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_variance = out_variance;
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_scale_grad;
  if (!grad_scale_grad) {
    optional_grad_scale_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_scale_grad = grad_scale_grad;
  }
  paddle::optional<pir::Value> optional_grad_bias_grad;
  if (!grad_bias_grad) {
    optional_grad_bias_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_bias_grad = grad_bias_grad;
  }
  paddle::dialect::BatchNormDoubleGrad_Op batch_norm_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BatchNormDoubleGrad_Op>(
              x,
              optional_scale.get(),
              optional_out_mean.get(),
              optional_out_variance.get(),
              saved_mean,
              saved_variance,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_scale_grad.get(),
              optional_grad_bias_grad.get(),
              momentum,
              epsilon,
              data_layout,
              is_test,
              use_global_stats,
              trainable_statistics);
  return std::make_tuple(batch_norm_double_grad__op.result(0),
                         batch_norm_double_grad__op.result(1),
                         batch_norm_double_grad__op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& mean_out,
    const paddle::optional<pir::Value>& variance_out,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const paddle::optional<pir::Value>& reserve_space,
    const pir::Value& out_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics) {
  CheckValueDataType(out_grad, "out_grad", "batch_norm_grad");
  paddle::optional<pir::Value> optional_scale;
  if (!scale) {
    optional_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_scale = scale;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_mean_out;
  if (!mean_out) {
    optional_mean_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mean_out = mean_out;
  }
  paddle::optional<pir::Value> optional_variance_out;
  if (!variance_out) {
    optional_variance_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_variance_out = variance_out;
  }
  paddle::optional<pir::Value> optional_reserve_space;
  if (!reserve_space) {
    optional_reserve_space = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_reserve_space = reserve_space;
  }
  paddle::dialect::BatchNormGradOp batch_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::BatchNormGradOp>(
              x,
              optional_scale.get(),
              optional_bias.get(),
              optional_mean_out.get(),
              optional_variance_out.get(),
              saved_mean,
              saved_variance,
              optional_reserve_space.get(),
              out_grad,
              momentum,
              epsilon,
              data_layout,
              is_test,
              use_global_stats,
              trainable_statistics);
  return std::make_tuple(batch_norm_grad_op.result(0),
                         batch_norm_grad_op.result(1),
                         batch_norm_grad_op.result(2));
}

pir::OpResult c_embedding_grad(const pir::Value& weight,
                               const pir::Value& x,
                               const pir::Value& out_grad,
                               int64_t start_index) {
  CheckValueDataType(out_grad, "out_grad", "c_embedding_grad");
  paddle::dialect::CEmbeddingGradOp c_embedding_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CEmbeddingGradOp>(
              weight, x, out_grad, start_index);
  return c_embedding_grad_op.result(0);
}

pir::OpResult c_softmax_with_cross_entropy_grad(const pir::Value& softmax,
                                                const pir::Value& label,
                                                const pir::Value& loss_grad,
                                                int64_t ignore_index,
                                                int ring_id,
                                                int rank,
                                                int nranks) {
  CheckValueDataType(
      loss_grad, "loss_grad", "c_softmax_with_cross_entropy_grad");
  paddle::dialect::CSoftmaxWithCrossEntropyGradOp
      c_softmax_with_cross_entropy_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::CSoftmaxWithCrossEntropyGradOp>(
                  softmax,
                  label,
                  loss_grad,
                  ignore_index,
                  ring_id,
                  rank,
                  nranks);
  return c_softmax_with_cross_entropy_grad_op.result(0);
}

pir::OpResult channel_shuffle_grad(const pir::Value& out_grad,
                                   int groups,
                                   const std::string& data_format) {
  CheckValueDataType(out_grad, "out_grad", "channel_shuffle_grad");
  paddle::dialect::ChannelShuffleGradOp channel_shuffle_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ChannelShuffleGradOp>(
              out_grad, groups, data_format);
  return channel_shuffle_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::vector<int64_t>& output_size,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_double_grad");
  paddle::dialect::Conv2dTransposeDoubleGradOp conv2d_transpose_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeDoubleGradOp>(
              x,
              filter,
              grad_out,
              grad_x_grad,
              grad_filter_grad,
              strides,
              paddings,
              output_padding,
              output_size,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(conv2d_transpose_double_grad_op.result(0),
                         conv2d_transpose_double_grad_op.result(1),
                         conv2d_transpose_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             pir::Value output_size,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_double_grad");
  paddle::dialect::Conv2dTransposeDoubleGradOp conv2d_transpose_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeDoubleGradOp>(
              x,
              filter,
              grad_out,
              grad_x_grad,
              grad_filter_grad,
              output_size,
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(conv2d_transpose_double_grad_op.result(0),
                         conv2d_transpose_double_grad_op.result(1),
                         conv2d_transpose_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             std::vector<pir::Value> output_size,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_double_grad");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::Conv2dTransposeDoubleGradOp conv2d_transpose_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeDoubleGradOp>(
              x,
              filter,
              grad_out,
              grad_x_grad,
              grad_filter_grad,
              output_size_combine_op.out(),
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(conv2d_transpose_double_grad_op.result(0),
                         conv2d_transpose_double_grad_op.result(1),
                         conv2d_transpose_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_grad");
  paddle::dialect::Conv2dTransposeGradOp conv2d_transpose_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeGradOp>(x,
                                                          filter,
                                                          out_grad,
                                                          strides,
                                                          paddings,
                                                          output_padding,
                                                          output_size,
                                                          padding_algorithm,
                                                          groups,
                                                          dilations,
                                                          data_format);
  return std::make_tuple(conv2d_transpose_grad_op.result(0),
                         conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    pir::Value output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_grad");
  paddle::dialect::Conv2dTransposeGradOp conv2d_transpose_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeGradOp>(x,
                                                          filter,
                                                          out_grad,
                                                          output_size,
                                                          strides,
                                                          paddings,
                                                          output_padding,
                                                          padding_algorithm,
                                                          groups,
                                                          dilations,
                                                          data_format);
  return std::make_tuple(conv2d_transpose_grad_op.result(0),
                         conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "conv2d_transpose_grad");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::Conv2dTransposeGradOp conv2d_transpose_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Conv2dTransposeGradOp>(
              x,
              filter,
              out_grad,
              output_size_combine_op.out(),
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format);
  return std::make_tuple(conv2d_transpose_grad_op.result(0),
                         conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
deformable_conv_grad(const pir::Value& x,
                     const pir::Value& offset,
                     const pir::Value& filter,
                     const paddle::optional<pir::Value>& mask,
                     const pir::Value& out_grad,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     int deformable_groups,
                     int groups,
                     int im2col_step) {
  CheckValueDataType(x, "x", "deformable_conv_grad");
  paddle::optional<pir::Value> optional_mask;
  if (!mask) {
    optional_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_mask = mask;
  }
  paddle::dialect::DeformableConvGradOp deformable_conv_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DeformableConvGradOp>(x,
                                                         offset,
                                                         filter,
                                                         optional_mask.get(),
                                                         out_grad,
                                                         strides,
                                                         paddings,
                                                         dilations,
                                                         deformable_groups,
                                                         groups,
                                                         im2col_step);
  return std::make_tuple(deformable_conv_grad_op.result(0),
                         deformable_conv_grad_op.result(1),
                         deformable_conv_grad_op.result(2),
                         deformable_conv_grad_op.result(3));
}

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose_grad");
  paddle::dialect::DepthwiseConv2dTransposeGradOp
      depthwise_conv2d_transpose_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::DepthwiseConv2dTransposeGradOp>(
                  x,
                  filter,
                  out_grad,
                  strides,
                  paddings,
                  output_padding,
                  output_size,
                  padding_algorithm,
                  groups,
                  dilations,
                  data_format);
  return std::make_tuple(depthwise_conv2d_transpose_grad_op.result(0),
                         depthwise_conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    pir::Value output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose_grad");
  paddle::dialect::DepthwiseConv2dTransposeGradOp
      depthwise_conv2d_transpose_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::DepthwiseConv2dTransposeGradOp>(
                  x,
                  filter,
                  out_grad,
                  output_size,
                  strides,
                  paddings,
                  output_padding,
                  padding_algorithm,
                  groups,
                  dilations,
                  data_format);
  return std::make_tuple(depthwise_conv2d_transpose_grad_op.result(0),
                         depthwise_conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format) {
  CheckValueDataType(x, "x", "depthwise_conv2d_transpose_grad");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::DepthwiseConv2dTransposeGradOp
      depthwise_conv2d_transpose_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::DepthwiseConv2dTransposeGradOp>(
                  x,
                  filter,
                  out_grad,
                  output_size_combine_op.out(),
                  strides,
                  paddings,
                  output_padding,
                  padding_algorithm,
                  groups,
                  dilations,
                  data_format);
  return std::make_tuple(depthwise_conv2d_transpose_grad_op.result(0),
                         depthwise_conv2d_transpose_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> divide_double_grad(
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& grad_x,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  CheckValueDataType(out, "out", "divide_double_grad");
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::DivideDoubleGradOp divide_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DivideDoubleGradOp>(
              y,
              out,
              grad_x,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return std::make_tuple(divide_double_grad_op.result(0),
                         divide_double_grad_op.result(1),
                         divide_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> divide_double_grad_(
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& grad_x,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  CheckValueDataType(out, "out", "divide_double_grad_");
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::DivideDoubleGrad_Op divide_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DivideDoubleGrad_Op>(
              y,
              out,
              grad_x,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return std::make_tuple(divide_double_grad__op.result(0),
                         divide_double_grad__op.result(1),
                         divide_double_grad__op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> divide_grad(const pir::Value& x,
                                                     const pir::Value& y,
                                                     const pir::Value& out,
                                                     const pir::Value& out_grad,
                                                     int axis) {
  CheckValueDataType(out_grad, "out_grad", "divide_grad");
  paddle::dialect::DivideGradOp divide_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::DivideGradOp>(
          x, y, out, out_grad, axis);
  return std::make_tuple(divide_grad_op.result(0), divide_grad_op.result(1));
}

pir::OpResult dropout_grad(const pir::Value& mask,
                           const pir::Value& out_grad,
                           float p,
                           bool is_test,
                           const std::string& mode) {
  CheckValueDataType(out_grad, "out_grad", "dropout_grad");
  paddle::dialect::DropoutGradOp dropout_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::DropoutGradOp>(
              mask, out_grad, p, is_test, mode);
  return dropout_grad_op.result(0);
}

std::vector<pir::OpResult> einsum_grad(
    const std::vector<pir::Value>& x_shape,
    const std::vector<pir::Value>& inner_cache,
    const pir::Value& out_grad,
    const std::string& equation) {
  CheckValueDataType(out_grad, "out_grad", "einsum_grad");
  auto x_shape_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x_shape);
  auto inner_cache_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inner_cache);
  paddle::dialect::EinsumGradOp einsum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::EinsumGradOp>(
          x_shape_combine_op.out(),
          inner_cache_combine_op.out(),
          out_grad,
          equation);
  auto x_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          einsum_grad_op.result(0));
  return x_grad_split_op.outputs();
}

std::tuple<pir::OpResult, pir::OpResult> elementwise_pow_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "elementwise_pow_grad");
  paddle::dialect::ElementwisePowGradOp elementwise_pow_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ElementwisePowGradOp>(x, y, out_grad);
  return std::make_tuple(elementwise_pow_grad_op.result(0),
                         elementwise_pow_grad_op.result(1));
}

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  const std::vector<int64_t>& axis,
                                  bool keep_dim,
                                  bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "frobenius_norm_grad");
  paddle::dialect::FrobeniusNormGradOp frobenius_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormGradOp>(
              x, out, out_grad, axis, keep_dim, reduce_all);
  return frobenius_norm_grad_op.result(0);
}

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  pir::Value axis,
                                  bool keep_dim,
                                  bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "frobenius_norm_grad");
  paddle::dialect::FrobeniusNormGradOp frobenius_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormGradOp>(
              x, out, out_grad, axis, keep_dim, reduce_all);
  return frobenius_norm_grad_op.result(0);
}

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  std::vector<pir::Value> axis,
                                  bool keep_dim,
                                  bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "frobenius_norm_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::FrobeniusNormGradOp frobenius_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FrobeniusNormGradOp>(
              x, out, out_grad, axis_combine_op.out(), keep_dim, reduce_all);
  return frobenius_norm_grad_op.result(0);
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_attention_grad(
    const pir::Value& out_grad,
    const pir::Value& x,
    const pir::Value& qkv_weight,
    const paddle::optional<pir::Value>& qkv_bias,
    const paddle::optional<pir::Value>& qkv_bias_out,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& src_mask_out,
    const pir::Value& out_linear_weight,
    const paddle::optional<pir::Value>& out_linear_bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    const paddle::optional<pir::Value>& ln_scale_2,
    const paddle::optional<pir::Value>& ln_bias_2,
    const paddle::optional<pir::Value>& ln_out,
    const paddle::optional<pir::Value>& ln_mean,
    const paddle::optional<pir::Value>& ln_var,
    const paddle::optional<pir::Value>& ln_mean_2,
    const paddle::optional<pir::Value>& ln_var_2,
    const paddle::optional<pir::Value>& bias_dropout_residual_out,
    const pir::Value& qkv_out,
    const pir::Value& transpose_out_2,
    const pir::Value& qk_out,
    const pir::Value& qktv_out,
    const pir::Value& softmax_out,
    const pir::Value& attn_dropout_mask_out,
    const pir::Value& attn_dropout_out,
    const pir::Value& fmha_out,
    const pir::Value& out_linear_out,
    const pir::Value& dropout_mask_out,
    int num_heads,
    bool transpose_qkv_wb,
    bool pre_layer_norm,
    float epsilon,
    float attn_dropout_rate,
    bool is_test,
    bool attn_dropout_fix_seed,
    int attn_dropout_seed,
    const std::string& attn_dropout_implementation,
    float dropout_rate,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string& dropout_implementation,
    float ln_epsilon,
    bool add_residual,
    int ring_id) {
  CheckValueDataType(x, "x", "fused_attention_grad");
  paddle::optional<pir::Value> optional_qkv_bias;
  if (!qkv_bias) {
    optional_qkv_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_bias = qkv_bias;
  }
  paddle::optional<pir::Value> optional_qkv_bias_out;
  if (!qkv_bias_out) {
    optional_qkv_bias_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_qkv_bias_out = qkv_bias_out;
  }
  paddle::optional<pir::Value> optional_src_mask;
  if (!src_mask) {
    optional_src_mask = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask = src_mask;
  }
  paddle::optional<pir::Value> optional_src_mask_out;
  if (!src_mask_out) {
    optional_src_mask_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_src_mask_out = src_mask_out;
  }
  paddle::optional<pir::Value> optional_out_linear_bias;
  if (!out_linear_bias) {
    optional_out_linear_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_out_linear_bias = out_linear_bias;
  }
  paddle::optional<pir::Value> optional_ln_scale;
  if (!ln_scale) {
    optional_ln_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale = ln_scale;
  }
  paddle::optional<pir::Value> optional_ln_bias;
  if (!ln_bias) {
    optional_ln_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias = ln_bias;
  }
  paddle::optional<pir::Value> optional_ln_scale_2;
  if (!ln_scale_2) {
    optional_ln_scale_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_scale_2 = ln_scale_2;
  }
  paddle::optional<pir::Value> optional_ln_bias_2;
  if (!ln_bias_2) {
    optional_ln_bias_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_bias_2 = ln_bias_2;
  }
  paddle::optional<pir::Value> optional_ln_out;
  if (!ln_out) {
    optional_ln_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_out = ln_out;
  }
  paddle::optional<pir::Value> optional_ln_mean;
  if (!ln_mean) {
    optional_ln_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_mean = ln_mean;
  }
  paddle::optional<pir::Value> optional_ln_var;
  if (!ln_var) {
    optional_ln_var = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_var = ln_var;
  }
  paddle::optional<pir::Value> optional_ln_mean_2;
  if (!ln_mean_2) {
    optional_ln_mean_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_mean_2 = ln_mean_2;
  }
  paddle::optional<pir::Value> optional_ln_var_2;
  if (!ln_var_2) {
    optional_ln_var_2 = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln_var_2 = ln_var_2;
  }
  paddle::optional<pir::Value> optional_bias_dropout_residual_out;
  if (!bias_dropout_residual_out) {
    optional_bias_dropout_residual_out =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias_dropout_residual_out = bias_dropout_residual_out;
  }
  paddle::dialect::FusedAttentionGradOp fused_attention_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedAttentionGradOp>(
              out_grad,
              x,
              qkv_weight,
              optional_qkv_bias.get(),
              optional_qkv_bias_out.get(),
              optional_src_mask.get(),
              optional_src_mask_out.get(),
              out_linear_weight,
              optional_out_linear_bias.get(),
              optional_ln_scale.get(),
              optional_ln_bias.get(),
              optional_ln_scale_2.get(),
              optional_ln_bias_2.get(),
              optional_ln_out.get(),
              optional_ln_mean.get(),
              optional_ln_var.get(),
              optional_ln_mean_2.get(),
              optional_ln_var_2.get(),
              optional_bias_dropout_residual_out.get(),
              qkv_out,
              transpose_out_2,
              qk_out,
              qktv_out,
              softmax_out,
              attn_dropout_mask_out,
              attn_dropout_out,
              fmha_out,
              out_linear_out,
              dropout_mask_out,
              num_heads,
              transpose_qkv_wb,
              pre_layer_norm,
              epsilon,
              attn_dropout_rate,
              is_test,
              attn_dropout_fix_seed,
              attn_dropout_seed,
              attn_dropout_implementation,
              dropout_rate,
              dropout_fix_seed,
              dropout_seed,
              dropout_implementation,
              ln_epsilon,
              add_residual,
              ring_id);
  return std::make_tuple(fused_attention_grad_op.result(0),
                         fused_attention_grad_op.result(1),
                         fused_attention_grad_op.result(2),
                         fused_attention_grad_op.result(3),
                         fused_attention_grad_op.result(4),
                         fused_attention_grad_op.result(5),
                         fused_attention_grad_op.result(6),
                         fused_attention_grad_op.result(7),
                         fused_attention_grad_op.result(8),
                         fused_attention_grad_op.result(9),
                         fused_attention_grad_op.result(10),
                         fused_attention_grad_op.result(11),
                         fused_attention_grad_op.result(12),
                         fused_attention_grad_op.result(13),
                         fused_attention_grad_op.result(14),
                         fused_attention_grad_op.result(15),
                         fused_attention_grad_op.result(16),
                         fused_attention_grad_op.result(17),
                         fused_attention_grad_op.result(18),
                         fused_attention_grad_op.result(19),
                         fused_attention_grad_op.result(20));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_batch_norm_act_grad(const pir::Value& x,
                          const pir::Value& scale,
                          const pir::Value& bias,
                          const pir::Value& out,
                          const pir::Value& saved_mean,
                          const pir::Value& saved_variance,
                          const paddle::optional<pir::Value>& reserve_space,
                          const pir::Value& out_grad,
                          float momentum,
                          float epsilon,
                          const std::string& act_type) {
  CheckValueDataType(out_grad, "out_grad", "fused_batch_norm_act_grad");
  paddle::optional<pir::Value> optional_reserve_space;
  if (!reserve_space) {
    optional_reserve_space = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_reserve_space = reserve_space;
  }
  paddle::dialect::FusedBatchNormActGradOp fused_batch_norm_act_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBatchNormActGradOp>(
              x,
              scale,
              bias,
              out,
              saved_mean,
              saved_variance,
              optional_reserve_space.get(),
              out_grad,
              momentum,
              epsilon,
              act_type);
  return std::make_tuple(fused_batch_norm_act_grad_op.result(0),
                         fused_batch_norm_act_grad_op.result(1),
                         fused_batch_norm_act_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
fused_bn_add_activation_grad(const pir::Value& x,
                             const pir::Value& scale,
                             const pir::Value& bias,
                             const pir::Value& out,
                             const pir::Value& saved_mean,
                             const pir::Value& saved_variance,
                             const paddle::optional<pir::Value>& reserve_space,
                             const pir::Value& out_grad,
                             float momentum,
                             float epsilon,
                             const std::string& act_type) {
  CheckValueDataType(out_grad, "out_grad", "fused_bn_add_activation_grad");
  paddle::optional<pir::Value> optional_reserve_space;
  if (!reserve_space) {
    optional_reserve_space = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_reserve_space = reserve_space;
  }
  paddle::dialect::FusedBnAddActivationGradOp fused_bn_add_activation_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedBnAddActivationGradOp>(
              x,
              scale,
              bias,
              out,
              saved_mean,
              saved_variance,
              optional_reserve_space.get(),
              out_grad,
              momentum,
              epsilon,
              act_type);
  return std::make_tuple(fused_bn_add_activation_grad_op.result(0),
                         fused_bn_add_activation_grad_op.result(1),
                         fused_bn_add_activation_grad_op.result(2),
                         fused_bn_add_activation_grad_op.result(3));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_feedforward_grad(const pir::Value& out_grad,
                       const pir::Value& x,
                       const pir::Value& linear1_weight,
                       const paddle::optional<pir::Value>& linear1_bias,
                       const pir::Value& linear2_weight,
                       const pir::Value& dropout1_mask,
                       const pir::Value& dropout2_mask,
                       const pir::Value& linear1_out,
                       const pir::Value& dropout1_out,
                       const paddle::optional<pir::Value>& dropout2_out,
                       const paddle::optional<pir::Value>& ln1_scale,
                       const paddle::optional<pir::Value>& ln1_bias,
                       const paddle::optional<pir::Value>& ln1_out,
                       const paddle::optional<pir::Value>& ln1_mean,
                       const paddle::optional<pir::Value>& ln1_variance,
                       const paddle::optional<pir::Value>& ln2_scale,
                       const paddle::optional<pir::Value>& ln2_bias,
                       const paddle::optional<pir::Value>& ln2_mean,
                       const paddle::optional<pir::Value>& ln2_variance,
                       const paddle::optional<pir::Value>& linear2_bias,
                       bool pre_layer_norm,
                       float ln1_epsilon,
                       float ln2_epsilon,
                       const std::string& act_method,
                       float dropout1_prob,
                       float dropout2_prob,
                       const std::string& dropout1_implementation,
                       const std::string& dropout2_implementation,
                       bool is_test,
                       bool dropout1_fix_seed,
                       bool dropout2_fix_seed,
                       int dropout1_seed_val,
                       int dropout2_seed_val,
                       bool add_residual,
                       int ring_id) {
  if (linear2_bias) {
    CheckValueDataType(
        linear2_bias.get(), "linear2_bias", "fused_feedforward_grad");
  } else if (ln2_variance) {
    CheckValueDataType(
        ln2_variance.get(), "ln2_variance", "fused_feedforward_grad");
  } else if (ln2_mean) {
    CheckValueDataType(ln2_mean.get(), "ln2_mean", "fused_feedforward_grad");
  } else if (ln2_bias) {
    CheckValueDataType(ln2_bias.get(), "ln2_bias", "fused_feedforward_grad");
  } else if (ln2_scale) {
    CheckValueDataType(ln2_scale.get(), "ln2_scale", "fused_feedforward_grad");
  } else if (ln1_variance) {
    CheckValueDataType(
        ln1_variance.get(), "ln1_variance", "fused_feedforward_grad");
  } else if (ln1_mean) {
    CheckValueDataType(ln1_mean.get(), "ln1_mean", "fused_feedforward_grad");
  } else if (ln1_out) {
    CheckValueDataType(ln1_out.get(), "ln1_out", "fused_feedforward_grad");
  } else if (ln1_bias) {
    CheckValueDataType(ln1_bias.get(), "ln1_bias", "fused_feedforward_grad");
  } else if (ln1_scale) {
    CheckValueDataType(ln1_scale.get(), "ln1_scale", "fused_feedforward_grad");
  } else if (dropout2_out) {
    CheckValueDataType(
        dropout2_out.get(), "dropout2_out", "fused_feedforward_grad");
  } else {
    CheckValueDataType(dropout1_out, "dropout1_out", "fused_feedforward_grad");
  }
  paddle::optional<pir::Value> optional_linear1_bias;
  if (!linear1_bias) {
    optional_linear1_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_linear1_bias = linear1_bias;
  }
  paddle::optional<pir::Value> optional_dropout2_out;
  if (!dropout2_out) {
    optional_dropout2_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_dropout2_out = dropout2_out;
  }
  paddle::optional<pir::Value> optional_ln1_scale;
  if (!ln1_scale) {
    optional_ln1_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_scale = ln1_scale;
  }
  paddle::optional<pir::Value> optional_ln1_bias;
  if (!ln1_bias) {
    optional_ln1_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_bias = ln1_bias;
  }
  paddle::optional<pir::Value> optional_ln1_out;
  if (!ln1_out) {
    optional_ln1_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_out = ln1_out;
  }
  paddle::optional<pir::Value> optional_ln1_mean;
  if (!ln1_mean) {
    optional_ln1_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_mean = ln1_mean;
  }
  paddle::optional<pir::Value> optional_ln1_variance;
  if (!ln1_variance) {
    optional_ln1_variance = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln1_variance = ln1_variance;
  }
  paddle::optional<pir::Value> optional_ln2_scale;
  if (!ln2_scale) {
    optional_ln2_scale = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_scale = ln2_scale;
  }
  paddle::optional<pir::Value> optional_ln2_bias;
  if (!ln2_bias) {
    optional_ln2_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_bias = ln2_bias;
  }
  paddle::optional<pir::Value> optional_ln2_mean;
  if (!ln2_mean) {
    optional_ln2_mean = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_mean = ln2_mean;
  }
  paddle::optional<pir::Value> optional_ln2_variance;
  if (!ln2_variance) {
    optional_ln2_variance = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_ln2_variance = ln2_variance;
  }
  paddle::optional<pir::Value> optional_linear2_bias;
  if (!linear2_bias) {
    optional_linear2_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_linear2_bias = linear2_bias;
  }
  paddle::dialect::FusedFeedforwardGradOp fused_feedforward_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedFeedforwardGradOp>(
              out_grad,
              x,
              linear1_weight,
              optional_linear1_bias.get(),
              linear2_weight,
              dropout1_mask,
              dropout2_mask,
              linear1_out,
              dropout1_out,
              optional_dropout2_out.get(),
              optional_ln1_scale.get(),
              optional_ln1_bias.get(),
              optional_ln1_out.get(),
              optional_ln1_mean.get(),
              optional_ln1_variance.get(),
              optional_ln2_scale.get(),
              optional_ln2_bias.get(),
              optional_ln2_mean.get(),
              optional_ln2_variance.get(),
              optional_linear2_bias.get(),
              pre_layer_norm,
              ln1_epsilon,
              ln2_epsilon,
              act_method,
              dropout1_prob,
              dropout2_prob,
              dropout1_implementation,
              dropout2_implementation,
              is_test,
              dropout1_fix_seed,
              dropout2_fix_seed,
              dropout1_seed_val,
              dropout2_seed_val,
              add_residual,
              ring_id);
  return std::make_tuple(fused_feedforward_grad_op.result(0),
                         fused_feedforward_grad_op.result(1),
                         fused_feedforward_grad_op.result(2),
                         fused_feedforward_grad_op.result(3),
                         fused_feedforward_grad_op.result(4),
                         fused_feedforward_grad_op.result(5),
                         fused_feedforward_grad_op.result(6),
                         fused_feedforward_grad_op.result(7),
                         fused_feedforward_grad_op.result(8));
}

pir::OpResult fused_softmax_mask_upper_triangle_grad(
    const pir::Value& Out, const pir::Value& Out_grad) {
  CheckValueDataType(
      Out_grad, "Out_grad", "fused_softmax_mask_upper_triangle_grad");
  paddle::dialect::FusedSoftmaxMaskUpperTriangleGradOp
      fused_softmax_mask_upper_triangle_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedSoftmaxMaskUpperTriangleGradOp>(
                  Out, Out_grad);
  return fused_softmax_mask_upper_triangle_grad_op.result(0);
}

pir::OpResult hardswish_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "hardswish_grad");
  paddle::dialect::HardswishGradOp hardswish_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardswishGradOp>(x, out_grad);
  return hardswish_grad_op.result(0);
}

pir::OpResult hardswish_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "hardswish_grad_");
  paddle::dialect::HardswishGrad_Op hardswish_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HardswishGrad_Op>(x, out_grad);
  return hardswish_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> hsigmoid_loss_grad(
    const pir::Value& x,
    const pir::Value& w,
    const pir::Value& label,
    const paddle::optional<pir::Value>& path,
    const paddle::optional<pir::Value>& code,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& pre_out,
    const pir::Value& out_grad,
    int num_classes,
    bool is_sparse) {
  CheckValueDataType(out_grad, "out_grad", "hsigmoid_loss_grad");
  paddle::optional<pir::Value> optional_path;
  if (!path) {
    optional_path = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_path = path;
  }
  paddle::optional<pir::Value> optional_code;
  if (!code) {
    optional_code = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_code = code;
  }
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::dialect::HsigmoidLossGradOp hsigmoid_loss_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::HsigmoidLossGradOp>(x,
                                                       w,
                                                       label,
                                                       optional_path.get(),
                                                       optional_code.get(),
                                                       optional_bias.get(),
                                                       pre_out,
                                                       out_grad,
                                                       num_classes,
                                                       is_sparse);
  return std::make_tuple(hsigmoid_loss_grad_op.result(0),
                         hsigmoid_loss_grad_op.result(1),
                         hsigmoid_loss_grad_op.result(2));
}

pir::OpResult logsumexp_grad(const pir::Value& x,
                             const pir::Value& out,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis,
                             bool keepdim,
                             bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "logsumexp_grad");
  paddle::dialect::LogsumexpGradOp logsumexp_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::LogsumexpGradOp>(
              x, out, out_grad, axis, keepdim, reduce_all);
  return logsumexp_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> matmul_double_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    bool transpose_x,
    bool transpose_y) {
  if (grad_y_grad) {
    CheckValueDataType(grad_y_grad.get(), "grad_y_grad", "matmul_double_grad");
  } else if (grad_x_grad) {
    CheckValueDataType(grad_x_grad.get(), "grad_x_grad", "matmul_double_grad");
  } else {
    CheckValueDataType(grad_out, "grad_out", "matmul_double_grad");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::MatmulDoubleGradOp matmul_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatmulDoubleGradOp>(
              x,
              y,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              transpose_x,
              transpose_y);
  return std::make_tuple(matmul_double_grad_op.result(0),
                         matmul_double_grad_op.result(1),
                         matmul_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> matmul_grad(const pir::Value& x,
                                                     const pir::Value& y,
                                                     const pir::Value& out_grad,
                                                     bool transpose_x,
                                                     bool transpose_y) {
  CheckValueDataType(out_grad, "out_grad", "matmul_grad");
  paddle::dialect::MatmulGradOp matmul_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MatmulGradOp>(
          x, y, out_grad, transpose_x, transpose_y);
  return std::make_tuple(matmul_grad_op.result(0), matmul_grad_op.result(1));
}

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "max_grad");
  paddle::dialect::MaxGradOp max_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return max_grad_op.result(0);
}

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "max_grad");
  paddle::dialect::MaxGradOp max_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return max_grad_op.result(0);
}

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "max_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::MaxGradOp max_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MaxGradOp>(
          x, out, out_grad, axis_combine_op.out(), keepdim, reduce_all);
  return max_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> maximum_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "maximum_grad");
  paddle::dialect::MaximumGradOp maximum_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MaximumGradOp>(x, y, out_grad);
  return std::make_tuple(maximum_grad_op.result(0), maximum_grad_op.result(1));
}

pir::OpResult mean_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "mean_grad");
  paddle::dialect::MeanGradOp mean_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeanGradOp>(
          x, out_grad, axis, keepdim, reduce_all);
  return mean_grad_op.result(0);
}

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "min_grad");
  paddle::dialect::MinGradOp min_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return min_grad_op.result(0);
}

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "min_grad");
  paddle::dialect::MinGradOp min_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinGradOp>(
          x, out, out_grad, axis, keepdim, reduce_all);
  return min_grad_op.result(0);
}

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "min_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::MinGradOp min_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MinGradOp>(
          x, out, out_grad, axis_combine_op.out(), keepdim, reduce_all);
  return min_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> minimum_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "minimum_grad");
  paddle::dialect::MinimumGradOp minimum_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MinimumGradOp>(x, y, out_grad);
  return std::make_tuple(minimum_grad_op.result(0), minimum_grad_op.result(1));
}

pir::OpResult mish_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float lambda) {
  CheckValueDataType(out_grad, "out_grad", "mish_grad");
  paddle::dialect::MishGradOp mish_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MishGradOp>(
          x, out_grad, lambda);
  return mish_grad_op.result(0);
}

pir::OpResult mish_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float lambda) {
  CheckValueDataType(out_grad, "out_grad", "mish_grad_");
  paddle::dialect::MishGrad_Op mish_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::MishGrad_Op>(
          x, out_grad, lambda);
  return mish_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiply_double_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  if (grad_y_grad) {
    CheckValueDataType(
        grad_y_grad.get(), "grad_y_grad", "multiply_double_grad");
  } else if (grad_x_grad) {
    CheckValueDataType(
        grad_x_grad.get(), "grad_x_grad", "multiply_double_grad");
  } else {
    CheckValueDataType(grad_out, "grad_out", "multiply_double_grad");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::MultiplyDoubleGradOp multiply_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiplyDoubleGradOp>(
              x,
              y,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return std::make_tuple(multiply_double_grad_op.result(0),
                         multiply_double_grad_op.result(1),
                         multiply_double_grad_op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiply_double_grad_(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  if (grad_y_grad) {
    CheckValueDataType(
        grad_y_grad.get(), "grad_y_grad", "multiply_double_grad_");
  } else if (grad_x_grad) {
    CheckValueDataType(
        grad_x_grad.get(), "grad_x_grad", "multiply_double_grad_");
  } else {
    CheckValueDataType(grad_out, "grad_out", "multiply_double_grad_");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::MultiplyDoubleGrad_Op multiply_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiplyDoubleGrad_Op>(
              x,
              y,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return std::make_tuple(multiply_double_grad__op.result(0),
                         multiply_double_grad__op.result(1),
                         multiply_double_grad__op.result(2));
}

std::tuple<pir::OpResult, pir::OpResult> multiply_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(out_grad, "out_grad", "multiply_grad");
  paddle::dialect::MultiplyGradOp multiply_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiplyGradOp>(x, y, out_grad, axis);
  return std::make_tuple(multiply_grad_op.result(0),
                         multiply_grad_op.result(1));
}

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
multiply_triple_grad(const pir::Value& x,
                     const pir::Value& y,
                     const pir::Value& fwd_grad_out,
                     const paddle::optional<pir::Value>& fwd_grad_grad_x,
                     const paddle::optional<pir::Value>& fwd_grad_grad_y,
                     const paddle::optional<pir::Value>& grad_x_grad,
                     const paddle::optional<pir::Value>& grad_y_grad,
                     const paddle::optional<pir::Value>& grad_grad_out_grad,
                     int axis) {
  if (grad_grad_out_grad) {
    CheckValueDataType(
        grad_grad_out_grad.get(), "grad_grad_out_grad", "multiply_triple_grad");
  } else if (grad_y_grad) {
    CheckValueDataType(
        grad_y_grad.get(), "grad_y_grad", "multiply_triple_grad");
  } else if (grad_x_grad) {
    CheckValueDataType(
        grad_x_grad.get(), "grad_x_grad", "multiply_triple_grad");
  } else if (fwd_grad_grad_y) {
    CheckValueDataType(
        fwd_grad_grad_y.get(), "fwd_grad_grad_y", "multiply_triple_grad");
  } else if (fwd_grad_grad_x) {
    CheckValueDataType(
        fwd_grad_grad_x.get(), "fwd_grad_grad_x", "multiply_triple_grad");
  } else {
    CheckValueDataType(fwd_grad_out, "fwd_grad_out", "multiply_triple_grad");
  }
  paddle::optional<pir::Value> optional_fwd_grad_grad_x;
  if (!fwd_grad_grad_x) {
    optional_fwd_grad_grad_x = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fwd_grad_grad_x = fwd_grad_grad_x;
  }
  paddle::optional<pir::Value> optional_fwd_grad_grad_y;
  if (!fwd_grad_grad_y) {
    optional_fwd_grad_grad_y = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_fwd_grad_grad_y = fwd_grad_grad_y;
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::optional<pir::Value> optional_grad_grad_out_grad;
  if (!grad_grad_out_grad) {
    optional_grad_grad_out_grad =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_grad_out_grad = grad_grad_out_grad;
  }
  paddle::dialect::MultiplyTripleGradOp multiply_triple_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MultiplyTripleGradOp>(
              x,
              y,
              fwd_grad_out,
              optional_fwd_grad_grad_x.get(),
              optional_fwd_grad_grad_y.get(),
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              optional_grad_grad_out_grad.get(),
              axis);
  return std::make_tuple(multiply_triple_grad_op.result(0),
                         multiply_triple_grad_op.result(1),
                         multiply_triple_grad_op.result(2),
                         multiply_triple_grad_op.result(3),
                         multiply_triple_grad_op.result(4));
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> nce_grad(
    const pir::Value& input,
    const pir::Value& label,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& weight,
    const pir::Value& sample_logits,
    const pir::Value& sample_labels,
    const paddle::optional<pir::Value>& sample_weight,
    const paddle::optional<pir::Value>& custom_dist_probs,
    const paddle::optional<pir::Value>& custom_dist_alias,
    const paddle::optional<pir::Value>& custom_dist_alias_probs,
    const pir::Value& cost_grad,
    int num_total_classes,
    const std::vector<int>& custom_neg_classes,
    int num_neg_samples,
    int sampler,
    int seed,
    bool is_sparse,
    bool remote_prefetch,
    bool is_test) {
  CheckValueDataType(input, "input", "nce_grad");
  paddle::optional<pir::Value> optional_bias;
  if (!bias) {
    optional_bias = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_bias = bias;
  }
  paddle::optional<pir::Value> optional_sample_weight;
  if (!sample_weight) {
    optional_sample_weight = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sample_weight = sample_weight;
  }
  paddle::optional<pir::Value> optional_custom_dist_probs;
  if (!custom_dist_probs) {
    optional_custom_dist_probs =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_probs = custom_dist_probs;
  }
  paddle::optional<pir::Value> optional_custom_dist_alias;
  if (!custom_dist_alias) {
    optional_custom_dist_alias =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_alias = custom_dist_alias;
  }
  paddle::optional<pir::Value> optional_custom_dist_alias_probs;
  if (!custom_dist_alias_probs) {
    optional_custom_dist_alias_probs =
        paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_custom_dist_alias_probs = custom_dist_alias_probs;
  }
  paddle::dialect::NceGradOp nce_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NceGradOp>(
          input,
          label,
          optional_bias.get(),
          weight,
          sample_logits,
          sample_labels,
          optional_sample_weight.get(),
          optional_custom_dist_probs.get(),
          optional_custom_dist_alias.get(),
          optional_custom_dist_alias_probs.get(),
          cost_grad,
          num_total_classes,
          custom_neg_classes,
          num_neg_samples,
          sampler,
          seed,
          is_sparse,
          remote_prefetch,
          is_test);
  return std::make_tuple(
      nce_grad_op.result(0), nce_grad_op.result(1), nce_grad_op.result(2));
}

pir::OpResult norm_grad(const pir::Value& x,
                        const pir::Value& norm,
                        const pir::Value& out_grad,
                        int axis,
                        float epsilon,
                        bool is_test) {
  CheckValueDataType(out_grad, "out_grad", "norm_grad");
  paddle::dialect::NormGradOp norm_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::NormGradOp>(
          x, norm, out_grad, axis, epsilon, is_test);
  return norm_grad_op.result(0);
}

pir::OpResult pad_double_grad(const pir::Value& grad_x_grad,
                              const std::vector<int>& paddings,
                              float pad_value) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pad");
  paddle::dialect::PadDoubleGradOp pad_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PadDoubleGradOp>(
              grad_x_grad, paddings, pad_value);
  return pad_double_grad_op.result(0);
}

pir::OpResult pad_double_grad(const pir::Value& grad_x_grad,
                              pir::Value pad_value,
                              const std::vector<int>& paddings) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pad");
  paddle::dialect::PadDoubleGradOp pad_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::PadDoubleGradOp>(
              grad_x_grad, pad_value, paddings);
  return pad_double_grad_op.result(0);
}

pir::OpResult pad_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       const std::vector<int>& paddings,
                       float pad_value) {
  CheckValueDataType(out_grad, "out_grad", "pad_grad");
  paddle::dialect::PadGradOp pad_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PadGradOp>(
          x, out_grad, paddings, pad_value);
  return pad_grad_op.result(0);
}

pir::OpResult pad_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       pir::Value pad_value,
                       const std::vector<int>& paddings) {
  CheckValueDataType(out_grad, "out_grad", "pad_grad");
  paddle::dialect::PadGradOp pad_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::PadGradOp>(
          x, out_grad, pad_value, paddings);
  return pad_grad_op.result(0);
}

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 const std::vector<int64_t>& kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pool2d_double_grad");
  paddle::dialect::Pool2dDoubleGradOp pool2d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pool2dDoubleGradOp>(x,
                                                       grad_x_grad,
                                                       kernel_size,
                                                       strides,
                                                       paddings,
                                                       ceil_mode,
                                                       exclusive,
                                                       data_format,
                                                       pooling_type,
                                                       global_pooling,
                                                       adaptive,
                                                       padding_algorithm);
  return pool2d_double_grad_op.result(0);
}

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 pir::Value kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pool2d_double_grad");
  paddle::dialect::Pool2dDoubleGradOp pool2d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pool2dDoubleGradOp>(x,
                                                       grad_x_grad,
                                                       kernel_size,
                                                       strides,
                                                       paddings,
                                                       ceil_mode,
                                                       exclusive,
                                                       data_format,
                                                       pooling_type,
                                                       global_pooling,
                                                       adaptive,
                                                       padding_algorithm);
  return pool2d_double_grad_op.result(0);
}

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 std::vector<pir::Value> kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "pool2d_double_grad");
  auto kernel_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(kernel_size);
  paddle::dialect::Pool2dDoubleGradOp pool2d_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::Pool2dDoubleGradOp>(
              x,
              grad_x_grad,
              kernel_size_combine_op.out(),
              strides,
              paddings,
              ceil_mode,
              exclusive,
              data_format,
              pooling_type,
              global_pooling,
              adaptive,
              padding_algorithm);
  return pool2d_double_grad_op.result(0);
}

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int64_t>& kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm) {
  CheckValueDataType(out_grad, "out_grad", "pool2d_grad");
  paddle::dialect::Pool2dGradOp pool2d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dGradOp>(
          x,
          out,
          out_grad,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_grad_op.result(0);
}

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          pir::Value kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm) {
  CheckValueDataType(out_grad, "out_grad", "pool2d_grad");
  paddle::dialect::Pool2dGradOp pool2d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dGradOp>(
          x,
          out,
          out_grad,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_grad_op.result(0);
}

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm) {
  CheckValueDataType(out_grad, "out_grad", "pool2d_grad");
  auto kernel_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(kernel_size);
  paddle::dialect::Pool2dGradOp pool2d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool2dGradOp>(
          x,
          out,
          out_grad,
          kernel_size_combine_op.out(),
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool2d_grad_op.result(0);
}

pir::OpResult pool3d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int>& kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm) {
  CheckValueDataType(out_grad, "out_grad", "pool3d_grad");
  paddle::dialect::Pool3dGradOp pool3d_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::Pool3dGradOp>(
          x,
          out,
          out_grad,
          kernel_size,
          strides,
          paddings,
          ceil_mode,
          exclusive,
          data_format,
          pooling_type,
          global_pooling,
          adaptive,
          padding_algorithm);
  return pool3d_grad_op.result(0);
}

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& dims,
                        bool keep_dim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "prod_grad");
  paddle::dialect::ProdGradOp prod_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdGradOp>(
          x, out, out_grad, dims, keep_dim, reduce_all);
  return prod_grad_op.result(0);
}

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        pir::Value dims,
                        bool keep_dim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "prod_grad");
  paddle::dialect::ProdGradOp prod_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdGradOp>(
          x, out, out_grad, dims, keep_dim, reduce_all);
  return prod_grad_op.result(0);
}

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> dims,
                        bool keep_dim,
                        bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "prod_grad");
  auto dims_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(dims);
  paddle::dialect::ProdGradOp prod_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ProdGradOp>(
          x, out, out_grad, dims_combine_op.out(), keep_dim, reduce_all);
  return prod_grad_op.result(0);
}

pir::OpResult repeat_interleave_grad(const pir::Value& x,
                                     const pir::Value& out_grad,
                                     int repeats,
                                     int axis) {
  CheckValueDataType(out_grad, "out_grad", "repeat_interleave_grad");
  paddle::dialect::RepeatInterleaveGradOp repeat_interleave_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RepeatInterleaveGradOp>(
              x, out_grad, repeats, axis);
  return repeat_interleave_grad_op.result(0);
}

pir::OpResult repeat_interleave_with_tensor_index_grad(
    const pir::Value& x,
    const pir::Value& repeats,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(x, "x", "repeat_interleave_with_tensor_index_grad");
  paddle::dialect::RepeatInterleaveWithTensorIndexGradOp
      repeat_interleave_with_tensor_index_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::RepeatInterleaveWithTensorIndexGradOp>(
                  x, repeats, out_grad, axis);
  return repeat_interleave_with_tensor_index_grad_op.result(0);
}

pir::OpResult reshape_double_grad(const pir::Value& grad_out,
                                  const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "reshape_double_grad");
  paddle::dialect::ReshapeDoubleGradOp reshape_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReshapeDoubleGradOp>(grad_out, grad_x_grad);
  return reshape_double_grad_op.result(0);
}

pir::OpResult reshape_double_grad_(const pir::Value& grad_out,
                                   const pir::Value& grad_x_grad) {
  CheckValueDataType(grad_x_grad, "grad_x_grad", "reshape_double_grad_");
  paddle::dialect::ReshapeDoubleGrad_Op reshape_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReshapeDoubleGrad_Op>(grad_out, grad_x_grad);
  return reshape_double_grad__op.result(0);
}

pir::OpResult reshape_grad(const pir::Value& xshape,
                           const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "reshape_grad");
  paddle::dialect::ReshapeGradOp reshape_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReshapeGradOp>(xshape, out_grad);
  return reshape_grad_op.result(0);
}

pir::OpResult reshape_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "reshape_grad_");
  paddle::dialect::ReshapeGrad_Op reshape_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ReshapeGrad_Op>(xshape, out_grad);
  return reshape_grad__op.result(0);
}

std::
    tuple<pir::OpResult, std::vector<pir::OpResult>, std::vector<pir::OpResult>>
    rnn_grad(const pir::Value& x,
             const std::vector<pir::Value>& pre_state,
             const std::vector<pir::Value>& weight_list,
             const paddle::optional<pir::Value>& sequence_length,
             const pir::Value& out,
             const pir::Value& dropout_state_out,
             const pir::Value& reserve,
             const pir::Value& out_grad,
             const std::vector<pir::Value>& state_grad,
             float dropout_prob,
             bool is_bidirec,
             int input_size,
             int hidden_size,
             int num_layers,
             const std::string& mode,
             int seed,
             bool is_test) {
  CheckValueDataType(out_grad, "out_grad", "rnn_grad");
  paddle::optional<pir::Value> optional_sequence_length;
  if (!sequence_length) {
    optional_sequence_length = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_sequence_length = sequence_length;
  }
  auto pre_state_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(pre_state);
  auto weight_list_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(weight_list);
  auto state_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(state_grad);
  paddle::dialect::RnnGradOp rnn_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RnnGradOp>(
          x,
          pre_state_combine_op.out(),
          weight_list_combine_op.out(),
          optional_sequence_length.get(),
          out,
          dropout_state_out,
          reserve,
          out_grad,
          state_grad_combine_op.out(),
          dropout_prob,
          is_bidirec,
          input_size,
          hidden_size,
          num_layers,
          mode,
          seed,
          is_test);
  auto pre_state_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          rnn_grad_op.result(1));
  auto weight_list_grad_split_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
          rnn_grad_op.result(2));
  return std::make_tuple(rnn_grad_op.result(0),
                         pre_state_grad_split_op.outputs(),
                         weight_list_grad_split_op.outputs());
}

std::tuple<pir::OpResult, pir::OpResult> row_conv_grad(
    const pir::Value& x, const pir::Value& filter, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "row_conv_grad");
  paddle::dialect::RowConvGradOp row_conv_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::RowConvGradOp>(x, filter, out_grad);
  return std::make_tuple(row_conv_grad_op.result(0),
                         row_conv_grad_op.result(1));
}

pir::OpResult rrelu_grad(const pir::Value& x,
                         const pir::Value& noise,
                         const pir::Value& out_grad) {
  CheckValueDataType(x, "x", "rrelu_grad");
  paddle::dialect::RreluGradOp rrelu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::RreluGradOp>(
          x, noise, out_grad);
  return rrelu_grad_op.result(0);
}

pir::OpResult set_value_grad(const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "assign");
  paddle::dialect::SetValueGradOp set_value_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueGradOp>(out_grad);
  return set_value_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(out_grad, "out_grad", "set_value_grad");
  paddle::dialect::SetValueWithTensorGradOp set_value_with_tensor_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorGradOp>(values,
                                                             out_grad,
                                                             starts,
                                                             ends,
                                                             steps,
                                                             axes,
                                                             decrease_axes,
                                                             none_axes);
  return std::make_tuple(set_value_with_tensor_grad_op.result(0),
                         set_value_with_tensor_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    pir::Value starts,
    pir::Value ends,
    pir::Value steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(out_grad, "out_grad", "set_value_grad");
  paddle::dialect::SetValueWithTensorGradOp set_value_with_tensor_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorGradOp>(values,
                                                             out_grad,
                                                             starts,
                                                             ends,
                                                             steps,
                                                             axes,
                                                             decrease_axes,
                                                             none_axes);
  return std::make_tuple(set_value_with_tensor_grad_op.result(0),
                         set_value_with_tensor_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    std::vector<pir::Value> starts,
    std::vector<pir::Value> ends,
    std::vector<pir::Value> steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes) {
  CheckValueDataType(out_grad, "out_grad", "set_value_grad");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto steps_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(steps);
  paddle::dialect::SetValueWithTensorGradOp set_value_with_tensor_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SetValueWithTensorGradOp>(
              values,
              out_grad,
              starts_combine_op.out(),
              ends_combine_op.out(),
              steps_combine_op.out(),
              axes,
              decrease_axes,
              none_axes);
  return std::make_tuple(set_value_with_tensor_grad_op.result(0),
                         set_value_with_tensor_grad_op.result(1));
}

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(out_grad, "out_grad", "slice_grad");
  paddle::dialect::SliceGradOp slice_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceGradOp>(
          input, out_grad, axes, starts, ends, infer_flags, decrease_axis);
  return slice_grad_op.result(0);
}

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         pir::Value starts,
                         pir::Value ends,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(out_grad, "out_grad", "slice_grad");
  paddle::dialect::SliceGradOp slice_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceGradOp>(
          input, out_grad, starts, ends, axes, infer_flags, decrease_axis);
  return slice_grad_op.result(0);
}

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         std::vector<pir::Value> starts,
                         std::vector<pir::Value> ends,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis) {
  CheckValueDataType(out_grad, "out_grad", "slice_grad");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  paddle::dialect::SliceGradOp slice_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceGradOp>(
          input,
          out_grad,
          starts_combine_op.out(),
          ends_combine_op.out(),
          axes,
          infer_flags,
          decrease_axis);
  return slice_grad_op.result(0);
}

pir::OpResult soft_relu_grad(const pir::Value& out,
                             const pir::Value& out_grad,
                             float threshold) {
  CheckValueDataType(out_grad, "out_grad", "soft_relu_grad");
  paddle::dialect::SoftReluGradOp soft_relu_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftReluGradOp>(out, out_grad, threshold);
  return soft_relu_grad_op.result(0);
}

pir::OpResult softmax_grad(const pir::Value& out,
                           const pir::Value& out_grad,
                           int axis) {
  CheckValueDataType(out_grad, "out_grad", "softmax_grad");
  paddle::dialect::SoftmaxGradOp softmax_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SoftmaxGradOp>(out, out_grad, axis);
  return softmax_grad_op.result(0);
}

pir::OpResult split_grad(const std::vector<pir::Value>& out_grad, int axis) {
  CheckVectorOfValueDataType(out_grad, "out_grad", "split_grad");
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::OpResult split_grad(const std::vector<pir::Value>& out_grad,
                         pir::Value axis) {
  CheckVectorOfValueDataType(out_grad, "out_grad", "split_grad");
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 const std::vector<int>& axes,
                                 const std::vector<int64_t>& starts,
                                 const std::vector<int64_t>& ends,
                                 const std::vector<int64_t>& strides) {
  CheckValueDataType(out_grad, "out_grad", "strided_slice_grad");
  paddle::dialect::StridedSliceGradOp strided_slice_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceGradOp>(
              x, out_grad, axes, starts, ends, strides);
  return strided_slice_grad_op.result(0);
}

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 pir::Value starts,
                                 pir::Value ends,
                                 pir::Value strides,
                                 const std::vector<int>& axes) {
  CheckValueDataType(out_grad, "out_grad", "strided_slice_grad");
  paddle::dialect::StridedSliceGradOp strided_slice_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceGradOp>(
              x, out_grad, starts, ends, strides, axes);
  return strided_slice_grad_op.result(0);
}

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 std::vector<pir::Value> starts,
                                 std::vector<pir::Value> ends,
                                 std::vector<pir::Value> strides,
                                 const std::vector<int>& axes) {
  CheckValueDataType(out_grad, "out_grad", "strided_slice_grad");
  auto starts_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(starts);
  auto ends_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(ends);
  auto strides_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(strides);
  paddle::dialect::StridedSliceGradOp strided_slice_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::StridedSliceGradOp>(x,
                                                       out_grad,
                                                       starts_combine_op.out(),
                                                       ends_combine_op.out(),
                                                       strides_combine_op.out(),
                                                       axes);
  return strided_slice_grad_op.result(0);
}

pir::OpResult subtract_double_grad(
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  if (grad_y_grad) {
    CheckValueDataType(
        grad_y_grad.get(), "grad_y_grad", "subtract_double_grad");
  } else if (grad_x_grad) {
    CheckValueDataType(
        grad_x_grad.get(), "grad_x_grad", "subtract_double_grad");
  } else {
    CheckValueDataType(grad_out, "grad_out", "subtract_double_grad");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::SubtractDoubleGradOp subtract_double_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SubtractDoubleGradOp>(
              y,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return subtract_double_grad_op.result(0);
}

pir::OpResult subtract_double_grad_(
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis) {
  if (grad_y_grad) {
    CheckValueDataType(
        grad_y_grad.get(), "grad_y_grad", "subtract_double_grad_");
  } else if (grad_x_grad) {
    CheckValueDataType(
        grad_x_grad.get(), "grad_x_grad", "subtract_double_grad_");
  } else {
    CheckValueDataType(grad_out, "grad_out", "subtract_double_grad_");
  }
  paddle::optional<pir::Value> optional_grad_x_grad;
  if (!grad_x_grad) {
    optional_grad_x_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_x_grad = grad_x_grad;
  }
  paddle::optional<pir::Value> optional_grad_y_grad;
  if (!grad_y_grad) {
    optional_grad_y_grad = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_grad_y_grad = grad_y_grad;
  }
  paddle::dialect::SubtractDoubleGrad_Op subtract_double_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SubtractDoubleGrad_Op>(
              y,
              grad_out,
              optional_grad_x_grad.get(),
              optional_grad_y_grad.get(),
              axis);
  return subtract_double_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> subtract_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(out_grad, "out_grad", "subtract_grad");
  paddle::dialect::SubtractGradOp subtract_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SubtractGradOp>(x, y, out_grad, axis);
  return std::make_tuple(subtract_grad_op.result(0),
                         subtract_grad_op.result(1));
}

std::tuple<pir::OpResult, pir::OpResult> subtract_grad_(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis) {
  CheckValueDataType(out_grad, "out_grad", "subtract_grad_");
  paddle::dialect::SubtractGrad_Op subtract_grad__op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SubtractGrad_Op>(x, y, out_grad, axis);
  return std::make_tuple(subtract_grad__op.result(0),
                         subtract_grad__op.result(1));
}

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "sum_grad");
  paddle::dialect::SumGradOp sum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumGradOp>(
          x, out_grad, axis, keepdim, reduce_all);
  return sum_grad_op.result(0);
}

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "sum_grad");
  paddle::dialect::SumGradOp sum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumGradOp>(
          x, out_grad, axis, keepdim, reduce_all);
  return sum_grad_op.result(0);
}

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim,
                       bool reduce_all) {
  CheckValueDataType(out_grad, "out_grad", "sum_grad");
  auto axis_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(axis);
  paddle::dialect::SumGradOp sum_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumGradOp>(
          x, out_grad, axis_combine_op.out(), keepdim, reduce_all);
  return sum_grad_op.result(0);
}

pir::OpResult swish_grad(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "swish_grad");
  paddle::dialect::SwishGradOp swish_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SwishGradOp>(
          x, out_grad);
  return swish_grad_op.result(0);
}

pir::OpResult swish_grad_(const pir::Value& x, const pir::Value& out_grad) {
  CheckValueDataType(out_grad, "out_grad", "swish_grad_");
  paddle::dialect::SwishGrad_Op swish_grad__op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SwishGrad_Op>(
          x, out_grad);
  return swish_grad__op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sync_batch_norm_grad(
    const pir::Value& x,
    const pir::Value& scale,
    const pir::Value& bias,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const paddle::optional<pir::Value>& reserve_space,
    const pir::Value& out_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics) {
  CheckValueDataType(out_grad, "out_grad", "sync_batch_norm_grad");
  paddle::optional<pir::Value> optional_reserve_space;
  if (!reserve_space) {
    optional_reserve_space = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_reserve_space = reserve_space;
  }
  paddle::dialect::SyncBatchNormGradOp sync_batch_norm_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SyncBatchNormGradOp>(
              x,
              scale,
              bias,
              saved_mean,
              saved_variance,
              optional_reserve_space.get(),
              out_grad,
              momentum,
              epsilon,
              data_layout,
              is_test,
              use_global_stats,
              trainable_statistics);
  return std::make_tuple(sync_batch_norm_grad_op.result(0),
                         sync_batch_norm_grad_op.result(1),
                         sync_batch_norm_grad_op.result(2));
}

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& repeat_times) {
  CheckValueDataType(out_grad, "out_grad", "tile_grad");
  paddle::dialect::TileGradOp tile_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileGradOp>(
          x, out_grad, repeat_times);
  return tile_grad_op.result(0);
}

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value repeat_times) {
  CheckValueDataType(out_grad, "out_grad", "tile_grad");
  paddle::dialect::TileGradOp tile_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileGradOp>(
          x, out_grad, repeat_times);
  return tile_grad_op.result(0);
}

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> repeat_times) {
  CheckValueDataType(out_grad, "out_grad", "tile_grad");
  auto repeat_times_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(repeat_times);
  paddle::dialect::TileGradOp tile_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TileGradOp>(
          x, out_grad, repeat_times_combine_op.out());
  return tile_grad_op.result(0);
}

pir::OpResult trans_layout_grad(const pir::Value& x,
                                const pir::Value& out_grad,
                                const std::vector<int>& perm) {
  CheckValueDataType(out_grad, "out_grad", "trans_layout_grad");
  paddle::dialect::TransLayoutGradOp trans_layout_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TransLayoutGradOp>(x, out_grad, perm);
  return trans_layout_grad_op.result(0);
}

pir::OpResult transpose_grad(const pir::Value& out_grad,
                             const std::vector<int>& perm) {
  CheckValueDataType(out_grad, "out_grad", "transpose_grad");
  paddle::dialect::TransposeGradOp transpose_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TransposeGradOp>(out_grad, perm);
  return transpose_grad_op.result(0);
}

pir::OpResult tril_grad(const pir::Value& out_grad, int diagonal) {
  CheckValueDataType(out_grad, "out_grad", "tril_grad");
  paddle::dialect::TrilGradOp tril_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TrilGradOp>(
          out_grad, diagonal);
  return tril_grad_op.result(0);
}

pir::OpResult triu_grad(const pir::Value& out_grad, int diagonal) {
  CheckValueDataType(out_grad, "out_grad", "triu_grad");
  paddle::dialect::TriuGradOp triu_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::TriuGradOp>(
          out_grad, diagonal);
  return triu_grad_op.result(0);
}

pir::OpResult disable_check_model_nan_inf_grad(const pir::Value& out_grad,
                                               int unsetflag) {
  CheckValueDataType(out_grad, "out_grad", "check_model_nan_inf");
  paddle::dialect::DisableCheckModelNanInfGradOp
      disable_check_model_nan_inf_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::DisableCheckModelNanInfGradOp>(
                  out_grad, unsetflag);
  return disable_check_model_nan_inf_grad_op.result(0);
}

pir::OpResult enable_check_model_nan_inf_grad(const pir::Value& out_grad,
                                              int unsetflag) {
  CheckValueDataType(out_grad, "out_grad", "check_model_nan_inf");
  paddle::dialect::EnableCheckModelNanInfGradOp
      enable_check_model_nan_inf_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::EnableCheckModelNanInfGradOp>(out_grad,
                                                                     unsetflag);
  return enable_check_model_nan_inf_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult> fused_elemwise_add_activation_grad(
    const paddle::optional<pir::Value>& x,
    const pir::Value& y,
    const pir::Value& out,
    const paddle::optional<pir::Value>& intermediate_out,
    const pir::Value& out_grad,
    const std::vector<std::string>& functor_list,
    float scale,
    int axis,
    bool save_intermediate_out) {
  CheckValueDataType(
      out_grad, "out_grad", "fused_elemwise_add_activation_grad");
  paddle::optional<pir::Value> optional_x;
  if (!x) {
    optional_x = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_x = x;
  }
  paddle::optional<pir::Value> optional_intermediate_out;
  if (!intermediate_out) {
    optional_intermediate_out = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_intermediate_out = intermediate_out;
  }
  paddle::dialect::FusedElemwiseAddActivationGradOp
      fused_elemwise_add_activation_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::FusedElemwiseAddActivationGradOp>(
                  optional_x.get(),
                  y,
                  out,
                  optional_intermediate_out.get(),
                  out_grad,
                  functor_list,
                  scale,
                  axis,
                  save_intermediate_out);
  return std::make_tuple(fused_elemwise_add_activation_grad_op.result(0),
                         fused_elemwise_add_activation_grad_op.result(1));
}

pir::OpResult shuffle_batch_grad(const pir::Value& shuffle_idx,
                                 const pir::Value& out_grad,
                                 int startup_seed) {
  CheckValueDataType(out_grad, "out_grad", "shuffle_batch_grad");
  paddle::dialect::ShuffleBatchGradOp shuffle_batch_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ShuffleBatchGradOp>(
              shuffle_idx, out_grad, startup_seed);
  return shuffle_batch_grad_op.result(0);
}

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::vector<int64_t>& output_size,
                          const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool_grad");
  paddle::dialect::UnpoolGradOp unpool_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolGradOp>(
          x,
          indices,
          out,
          out_grad,
          ksize,
          strides,
          padding,
          output_size,
          data_format);
  return unpool_grad_op.result(0);
}

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          pir::Value output_size,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool_grad");
  paddle::dialect::UnpoolGradOp unpool_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolGradOp>(
          x,
          indices,
          out,
          out_grad,
          output_size,
          ksize,
          strides,
          padding,
          data_format);
  return unpool_grad_op.result(0);
}

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> output_size,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::string& data_format) {
  CheckValueDataType(x, "x", "unpool_grad");
  auto output_size_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_size);
  paddle::dialect::UnpoolGradOp unpool_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::UnpoolGradOp>(
          x,
          indices,
          out,
          out_grad,
          output_size_combine_op.out(),
          ksize,
          strides,
          padding,
          data_format);
  return unpool_grad_op.result(0);
}

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
match_matrix_tensor_grad(const pir::Value& x,
                         const pir::Value& y,
                         const pir::Value& w,
                         const pir::Value& tmp,
                         const pir::Value& out_grad,
                         int dim_t) {
  CheckValueDataType(out_grad, "out_grad", "match_matrix_tensor_grad");
  paddle::dialect::MatchMatrixTensorGradOp match_matrix_tensor_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::MatchMatrixTensorGradOp>(
              x, y, w, tmp, out_grad, dim_t);
  return std::make_tuple(match_matrix_tensor_grad_op.result(0),
                         match_matrix_tensor_grad_op.result(1),
                         match_matrix_tensor_grad_op.result(2));
}

pir::OpResult arange(float start,
                     float end,
                     float step,
                     phi::DataType dtype,
                     const Place& place) {
  CheckDataType(dtype, "dtype", "arange");
  paddle::dialect::ArangeOp arange_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArangeOp>(
          start, end, step, dtype, place);
  return arange_op.result(0);
}

pir::OpResult arange(pir::Value start,
                     pir::Value end,
                     pir::Value step,
                     phi::DataType dtype,
                     const Place& place) {
  CheckDataType(dtype, "dtype", "arange");
  paddle::dialect::ArangeOp arange_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArangeOp>(
          start, end, step, dtype, place);
  return arange_op.result(0);
}

pir::OpResult sequence_mask(const pir::Value& x, int max_len, int out_dtype) {
  CheckValueDataType(x, "x", "sequence_mask_scalar");
  paddle::dialect::SequenceMaskOp sequence_mask_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SequenceMaskOp>(x, max_len, out_dtype);
  return sequence_mask_op.result(0);
}

pir::OpResult sequence_mask(const pir::Value& x,
                            pir::Value max_len,
                            int out_dtype) {
  CheckValueDataType(x, "x", "sequence_mask_scalar");
  paddle::dialect::SequenceMaskOp sequence_mask_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::SequenceMaskOp>(x, max_len, out_dtype);
  return sequence_mask_op.result(0);
}

}  // namespace dialect

}  // namespace paddle
