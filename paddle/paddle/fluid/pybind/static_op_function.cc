

#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_callstack_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {

namespace pybind {

PyObject *static_api_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add abs op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "abs", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("abs");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::abs(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_abs_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add abs_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "abs_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("abs_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::abs_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_accuracy(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add accuracy op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "accuracy", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "accuracy", 1);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 2);
    auto label = CastPyArg2Value(label_obj, "accuracy", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("accuracy");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::accuracy(x, indices, label);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_acos(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add acos op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "acos", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("acos");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::acos(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_acos_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add acos_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "acos_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("acos_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::acos_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_acosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add acosh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "acosh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("acosh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::acosh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_acosh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add acosh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "acosh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("acosh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::acosh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_adagrad_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add adagrad_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "adagrad_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "adagrad_", 1);
    PyObject *moment_obj = PyTuple_GET_ITEM(args, 2);
    auto moment = CastPyArg2Value(moment_obj, "adagrad_", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "adagrad_", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "adagrad_", 4);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon = CastPyArg2Float(epsilon_obj, "adagrad_", 5);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 6);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "adagrad_", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("adagrad_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::adagrad_(param, grad, moment, learning_rate,
                                  master_param, epsilon, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_adam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add adam_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "adam_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "adam_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "adam_", 2);
    PyObject *moment1_obj = PyTuple_GET_ITEM(args, 3);
    auto moment1 = CastPyArg2Value(moment1_obj, "adam_", 3);
    PyObject *moment2_obj = PyTuple_GET_ITEM(args, 4);
    auto moment2 = CastPyArg2Value(moment2_obj, "adam_", 4);
    PyObject *beta1_pow_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pow = CastPyArg2Value(beta1_pow_obj, "adam_", 5);
    PyObject *beta2_pow_obj = PyTuple_GET_ITEM(args, 6);
    auto beta2_pow = CastPyArg2Value(beta2_pow_obj, "adam_", 6);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 7);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "adam_", 7);
    PyObject *skip_update_obj = PyTuple_GET_ITEM(args, 8);
    auto skip_update = CastPyArg2OptionalValue(skip_update_obj, "adam_", 8);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 10);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 11);
    PyObject *lazy_mode_obj = PyTuple_GET_ITEM(args, 12);
    PyObject *min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 13);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 14);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 15);

    // Check for mutable attrs
    pir::Value beta1;

    pir::Value beta2;

    pir::Value epsilon;

    if (PyObject_CheckIRValue(beta1_obj)) {
      beta1 = CastPyArg2Value(beta1_obj, "adam_", 9);
    } else {
      float beta1_tmp = CastPyArg2Float(beta1_obj, "adam_", 9);
      beta1 = paddle::dialect::full(std::vector<int64_t>{1}, beta1_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(beta2_obj)) {
      beta2 = CastPyArg2Value(beta2_obj, "adam_", 10);
    } else {
      float beta2_tmp = CastPyArg2Float(beta2_obj, "adam_", 10);
      beta2 = paddle::dialect::full(std::vector<int64_t>{1}, beta2_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(epsilon_obj)) {
      epsilon = CastPyArg2Value(epsilon_obj, "adam_", 11);
    } else {
      float epsilon_tmp = CastPyArg2Float(epsilon_obj, "adam_", 11);
      epsilon = paddle::dialect::full(std::vector<int64_t>{1}, epsilon_tmp,
                                      phi::DataType::FLOAT32, phi::CPUPlace());
    }
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adam_", 12);
    int64_t min_row_size_to_use_multithread =
        CastPyArg2Long(min_row_size_to_use_multithread_obj, "adam_", 13);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adam_", 14);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "adam_", 15);

    // Call ir static api
    CallStackRecorder callstack_recorder("adam_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::adam_(
        param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow,
        master_param, skip_update, beta1, beta2, epsilon, lazy_mode,
        min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_adamax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add adamax_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "adamax_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "adamax_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "adamax_", 2);
    PyObject *moment_obj = PyTuple_GET_ITEM(args, 3);
    auto moment = CastPyArg2Value(moment_obj, "adamax_", 3);
    PyObject *inf_norm_obj = PyTuple_GET_ITEM(args, 4);
    auto inf_norm = CastPyArg2Value(inf_norm_obj, "adamax_", 4);
    PyObject *beta1_pow_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pow = CastPyArg2Value(beta1_pow_obj, "adamax_", 5);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 6);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "adamax_", 6);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 7);
    float beta1 = CastPyArg2Float(beta1_obj, "adamax_", 7);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 8);
    float beta2 = CastPyArg2Float(beta2_obj, "adamax_", 8);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 9);
    float epsilon = CastPyArg2Float(epsilon_obj, "adamax_", 9);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 10);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "adamax_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("adamax_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::adamax_(
        param, grad, learning_rate, moment, inf_norm, beta1_pow, master_param,
        beta1, beta2, epsilon, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_adamw_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add adamw_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "adamw_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "adamw_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "adamw_", 2);
    PyObject *moment1_obj = PyTuple_GET_ITEM(args, 3);
    auto moment1 = CastPyArg2Value(moment1_obj, "adamw_", 3);
    PyObject *moment2_obj = PyTuple_GET_ITEM(args, 4);
    auto moment2 = CastPyArg2Value(moment2_obj, "adamw_", 4);
    PyObject *beta1_pow_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pow = CastPyArg2Value(beta1_pow_obj, "adamw_", 5);
    PyObject *beta2_pow_obj = PyTuple_GET_ITEM(args, 6);
    auto beta2_pow = CastPyArg2Value(beta2_pow_obj, "adamw_", 6);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 7);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "adamw_", 7);
    PyObject *skip_update_obj = PyTuple_GET_ITEM(args, 8);
    auto skip_update = CastPyArg2OptionalValue(skip_update_obj, "adamw_", 8);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 10);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 11);
    PyObject *lr_ratio_obj = PyTuple_GET_ITEM(args, 12);
    PyObject *coeff_obj = PyTuple_GET_ITEM(args, 13);
    PyObject *with_decay_obj = PyTuple_GET_ITEM(args, 14);
    PyObject *lazy_mode_obj = PyTuple_GET_ITEM(args, 15);
    PyObject *min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 16);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 17);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 18);

    // Check for mutable attrs
    pir::Value beta1;

    pir::Value beta2;

    pir::Value epsilon;

    if (PyObject_CheckIRValue(beta1_obj)) {
      beta1 = CastPyArg2Value(beta1_obj, "adamw_", 9);
    } else {
      float beta1_tmp = CastPyArg2Float(beta1_obj, "adamw_", 9);
      beta1 = paddle::dialect::full(std::vector<int64_t>{1}, beta1_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(beta2_obj)) {
      beta2 = CastPyArg2Value(beta2_obj, "adamw_", 10);
    } else {
      float beta2_tmp = CastPyArg2Float(beta2_obj, "adamw_", 10);
      beta2 = paddle::dialect::full(std::vector<int64_t>{1}, beta2_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(epsilon_obj)) {
      epsilon = CastPyArg2Value(epsilon_obj, "adamw_", 11);
    } else {
      float epsilon_tmp = CastPyArg2Float(epsilon_obj, "adamw_", 11);
      epsilon = paddle::dialect::full(std::vector<int64_t>{1}, epsilon_tmp,
                                      phi::DataType::FLOAT32, phi::CPUPlace());
    }
    float lr_ratio = CastPyArg2Float(lr_ratio_obj, "adamw_", 12);
    float coeff = CastPyArg2Float(coeff_obj, "adamw_", 13);
    bool with_decay = CastPyArg2Boolean(with_decay_obj, "adamw_", 14);
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adamw_", 15);
    int64_t min_row_size_to_use_multithread =
        CastPyArg2Long(min_row_size_to_use_multithread_obj, "adamw_", 16);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adamw_", 17);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "adamw_", 18);

    // Call ir static api
    CallStackRecorder callstack_recorder("adamw_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::adamw_(
        param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow,
        master_param, skip_update, beta1, beta2, epsilon, lr_ratio, coeff,
        with_decay, lazy_mode, min_row_size_to_use_multithread, multi_precision,
        use_global_beta_pow);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_addmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add addmm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "addmm", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "addmm", 1);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 2);
    auto y = CastPyArg2Value(y_obj, "addmm", 2);

    // Parse Attributes
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 3);
    float beta = CastPyArg2Float(beta_obj, "addmm", 3);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 4);
    float alpha = CastPyArg2Float(alpha_obj, "addmm", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("addmm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::addmm(input, x, y, beta, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_addmm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add addmm_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "addmm_", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "addmm_", 1);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 2);
    auto y = CastPyArg2Value(y_obj, "addmm_", 2);

    // Parse Attributes
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 3);
    float beta = CastPyArg2Float(beta_obj, "addmm_", 3);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 4);
    float alpha = CastPyArg2Float(alpha_obj, "addmm_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("addmm_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::addmm_(input, x, y, beta, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_affine_grid(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add affine_grid op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "affine_grid", 0);

    // Parse Attributes
    PyObject *output_shape_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value output_shape;

    if (PyObject_CheckIRValue(output_shape_obj)) {
      output_shape = CastPyArg2Value(output_shape_obj, "affine_grid", 1);
    } else if (PyObject_CheckIRVectorOfValue(output_shape_obj)) {
      std::vector<pir::Value> output_shape_tmp =
          CastPyArg2VectorOfValue(output_shape_obj, "affine_grid", 1);
      output_shape = paddle::dialect::stack(output_shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> output_shape_tmp =
          CastPyArg2Longs(output_shape_obj, "affine_grid", 1);
      output_shape = paddle::dialect::full_int_array(
          output_shape_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "affine_grid", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("affine_grid");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::affine_grid(input, output_shape, align_corners);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_allclose(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add allclose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "allclose", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "allclose", 1);

    // Parse Attributes
    PyObject *rtol_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *atol_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *equal_nan_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value rtol;

    pir::Value atol;

    if (PyObject_CheckIRValue(rtol_obj)) {
      rtol = CastPyArg2Value(rtol_obj, "allclose", 2);
    } else {
      float rtol_tmp = CastPyArg2Float(rtol_obj, "allclose", 2);
      rtol = paddle::dialect::full(std::vector<int64_t>{1}, rtol_tmp,
                                   phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(atol_obj)) {
      atol = CastPyArg2Value(atol_obj, "allclose", 3);
    } else {
      float atol_tmp = CastPyArg2Float(atol_obj, "allclose", 3);
      atol = paddle::dialect::full(std::vector<int64_t>{1}, atol_tmp,
                                   phi::DataType::FLOAT32, phi::CPUPlace());
    }
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "allclose", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("allclose");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::allclose(x, y, rtol, atol, equal_nan);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_angle(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add angle op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "angle", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("angle");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::angle(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_apply_per_channel_scale(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add apply_per_channel_scale op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "apply_per_channel_scale", 0);
    PyObject *scales_obj = PyTuple_GET_ITEM(args, 1);
    auto scales = CastPyArg2Value(scales_obj, "apply_per_channel_scale", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("apply_per_channel_scale");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::apply_per_channel_scale(x, scales);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_argmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add argmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "argmax", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keepdims_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "argmax", 1);
    } else {
      int64_t axis_tmp = CastPyArg2Long(axis_obj, "argmax", 1);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT64, phi::CPUPlace());
    }
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmax", 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmax", 3);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "argmax", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("argmax");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::argmax(x, axis, keepdims, flatten, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_argmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add argmin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "argmin", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keepdims_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "argmin", 1);
    } else {
      int64_t axis_tmp = CastPyArg2Long(axis_obj, "argmin", 1);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT64, phi::CPUPlace());
    }
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmin", 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmin", 3);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "argmin", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("argmin");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::argmin(x, axis, keepdims, flatten, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_argsort(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add argsort op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "argsort", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "argsort", 1);
    PyObject *descending_obj = PyTuple_GET_ITEM(args, 2);
    bool descending = CastPyArg2Boolean(descending_obj, "argsort", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("argsort");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::argsort(x, axis, descending);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_as_complex(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add as_complex op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "as_complex", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("as_complex");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::as_complex(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_as_real(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add as_real op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "as_real", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("as_real");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::as_real(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_as_strided(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add as_strided op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "as_strided", 0);

    // Parse Attributes
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "as_strided", 1);
    PyObject *stride_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int64_t> stride = CastPyArg2Longs(stride_obj, "as_strided", 2);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 3);
    int64_t offset = CastPyArg2Long(offset_obj, "as_strided", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("as_strided");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::as_strided(input, dims, stride, offset);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_asgd_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add asgd_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "asgd_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "asgd_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "asgd_", 2);
    PyObject *d_obj = PyTuple_GET_ITEM(args, 3);
    auto d = CastPyArg2Value(d_obj, "asgd_", 3);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 4);
    auto y = CastPyArg2Value(y_obj, "asgd_", 4);
    PyObject *n_obj = PyTuple_GET_ITEM(args, 5);
    auto n = CastPyArg2Value(n_obj, "asgd_", 5);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 6);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "asgd_", 6);

    // Parse Attributes
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 7);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "asgd_", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("asgd_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::asgd_(
        param, grad, learning_rate, d, y, n, master_param, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_asin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add asin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "asin", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("asin");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::asin(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_asin_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add asin_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "asin_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("asin_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::asin_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_asinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add asinh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "asinh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("asinh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::asinh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_asinh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add asinh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "asinh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("asinh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::asinh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_atan(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add atan op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "atan", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("atan");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::atan(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_atan_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add atan_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "atan_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("atan_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::atan_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_atan2(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add atan2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "atan2", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "atan2", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("atan2");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::atan2(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_atanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add atanh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "atanh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("atanh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::atanh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_atanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add atanh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "atanh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("atanh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::atanh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_auc(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add auc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "auc", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "auc", 1);
    PyObject *stat_pos_obj = PyTuple_GET_ITEM(args, 2);
    auto stat_pos = CastPyArg2Value(stat_pos_obj, "auc", 2);
    PyObject *stat_neg_obj = PyTuple_GET_ITEM(args, 3);
    auto stat_neg = CastPyArg2Value(stat_neg_obj, "auc", 3);
    PyObject *ins_tag_weight_obj = PyTuple_GET_ITEM(args, 4);
    auto ins_tag_weight = CastPyArg2OptionalValue(ins_tag_weight_obj, "auc", 4);

    // Parse Attributes
    PyObject *curve_obj = PyTuple_GET_ITEM(args, 5);
    std::string curve = CastPyArg2String(curve_obj, "auc", 5);
    PyObject *num_thresholds_obj = PyTuple_GET_ITEM(args, 6);
    int num_thresholds = CastPyArg2Int(num_thresholds_obj, "auc", 6);
    PyObject *slide_steps_obj = PyTuple_GET_ITEM(args, 7);
    int slide_steps = CastPyArg2Int(slide_steps_obj, "auc", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("auc");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::auc(x, label, stat_pos, stat_neg, ins_tag_weight,
                             curve, num_thresholds, slide_steps);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_average_accumulates_(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add average_accumulates_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "average_accumulates_", 0);
    PyObject *in_sum_1_obj = PyTuple_GET_ITEM(args, 1);
    auto in_sum_1 = CastPyArg2Value(in_sum_1_obj, "average_accumulates_", 1);
    PyObject *in_sum_2_obj = PyTuple_GET_ITEM(args, 2);
    auto in_sum_2 = CastPyArg2Value(in_sum_2_obj, "average_accumulates_", 2);
    PyObject *in_sum_3_obj = PyTuple_GET_ITEM(args, 3);
    auto in_sum_3 = CastPyArg2Value(in_sum_3_obj, "average_accumulates_", 3);
    PyObject *in_num_accumulates_obj = PyTuple_GET_ITEM(args, 4);
    auto in_num_accumulates =
        CastPyArg2Value(in_num_accumulates_obj, "average_accumulates_", 4);
    PyObject *in_old_num_accumulates_obj = PyTuple_GET_ITEM(args, 5);
    auto in_old_num_accumulates =
        CastPyArg2Value(in_old_num_accumulates_obj, "average_accumulates_", 5);
    PyObject *in_num_updates_obj = PyTuple_GET_ITEM(args, 6);
    auto in_num_updates =
        CastPyArg2Value(in_num_updates_obj, "average_accumulates_", 6);

    // Parse Attributes
    PyObject *average_window_obj = PyTuple_GET_ITEM(args, 7);
    float average_window =
        CastPyArg2Float(average_window_obj, "average_accumulates_", 7);
    PyObject *max_average_window_obj = PyTuple_GET_ITEM(args, 8);
    int64_t max_average_window =
        CastPyArg2Long(max_average_window_obj, "average_accumulates_", 8);
    PyObject *min_average_window_obj = PyTuple_GET_ITEM(args, 9);
    int64_t min_average_window =
        CastPyArg2Long(min_average_window_obj, "average_accumulates_", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("average_accumulates_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::average_accumulates_(
        param, in_sum_1, in_sum_2, in_sum_3, in_num_accumulates,
        in_old_num_accumulates, in_num_updates, average_window,
        max_average_window, min_average_window);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bce_loss(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add bce_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "bce_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "bce_loss", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bce_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bce_loss(input, label);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bce_loss_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add bce_loss_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "bce_loss_", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "bce_loss_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bce_loss_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bce_loss_(input, label);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bernoulli(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add bernoulli op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bernoulli", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bernoulli");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bernoulli(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bicubic_interp(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add bicubic_interp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bicubic_interp", 0);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 1);
    auto out_size = CastPyArg2OptionalValue(out_size_obj, "bicubic_interp", 1);
    PyObject *size_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto size_tensor =
        CastPyArg2OptionalVectorOfValue(size_tensor_obj, "bicubic_interp", 2);
    PyObject *scale_tensor_obj = PyTuple_GET_ITEM(args, 3);
    auto scale_tensor =
        CastPyArg2OptionalValue(scale_tensor_obj, "bicubic_interp", 3);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "bicubic_interp", 4);
    PyObject *out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "bicubic_interp", 5);
    PyObject *out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "bicubic_interp", 6);
    PyObject *out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "bicubic_interp", 7);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "bicubic_interp", 8);
    PyObject *interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method =
        CastPyArg2String(interp_method_obj, "bicubic_interp", 9);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners =
        CastPyArg2Boolean(align_corners_obj, "bicubic_interp", 10);
    PyObject *align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "bicubic_interp", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("bicubic_interp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bicubic_interp(
        x, out_size, size_tensor, scale_tensor, data_format, out_d, out_h,
        out_w, scale, interp_method, align_corners, align_mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bilinear(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add bilinear op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bilinear", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bilinear", 1);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 2);
    auto weight = CastPyArg2Value(weight_obj, "bilinear", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2OptionalValue(bias_obj, "bilinear", 3);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bilinear");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bilinear(x, y, weight, bias);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bilinear_interp(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add bilinear_interp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bilinear_interp", 0);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 1);
    auto out_size = CastPyArg2OptionalValue(out_size_obj, "bilinear_interp", 1);
    PyObject *size_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto size_tensor =
        CastPyArg2OptionalVectorOfValue(size_tensor_obj, "bilinear_interp", 2);
    PyObject *scale_tensor_obj = PyTuple_GET_ITEM(args, 3);
    auto scale_tensor =
        CastPyArg2OptionalValue(scale_tensor_obj, "bilinear_interp", 3);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "bilinear_interp", 4);
    PyObject *out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "bilinear_interp", 5);
    PyObject *out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "bilinear_interp", 6);
    PyObject *out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "bilinear_interp", 7);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale =
        CastPyArg2Floats(scale_obj, "bilinear_interp", 8);
    PyObject *interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method =
        CastPyArg2String(interp_method_obj, "bilinear_interp", 9);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners =
        CastPyArg2Boolean(align_corners_obj, "bilinear_interp", 10);
    PyObject *align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "bilinear_interp", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("bilinear_interp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bilinear_interp(
        x, out_size, size_tensor, scale_tensor, data_format, out_d, out_h,
        out_w, scale, interp_method, align_corners, align_mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bincount(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add bincount op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bincount", 0);
    PyObject *weights_obj = PyTuple_GET_ITEM(args, 1);
    auto weights = CastPyArg2OptionalValue(weights_obj, "bincount", 1);

    // Parse Attributes
    PyObject *minlength_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value minlength;

    if (PyObject_CheckIRValue(minlength_obj)) {
      minlength = CastPyArg2Value(minlength_obj, "bincount", 2);
    } else {
      int minlength_tmp = CastPyArg2Int(minlength_obj, "bincount", 2);
      minlength = paddle::dialect::full(std::vector<int64_t>{1}, minlength_tmp,
                                        phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("bincount");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bincount(x, weights, minlength);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_binomial(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add binomial op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *count_obj = PyTuple_GET_ITEM(args, 0);
    auto count = CastPyArg2Value(count_obj, "binomial", 0);
    PyObject *prob_obj = PyTuple_GET_ITEM(args, 1);
    auto prob = CastPyArg2Value(prob_obj, "binomial", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("binomial");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::binomial(count, prob);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_and(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_and op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_and", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_and", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_and");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_and(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_and_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_and_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_and_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_and_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_and_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_and_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_left_shift(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_left_shift op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_left_shift", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_left_shift", 1);

    // Parse Attributes
    PyObject *is_arithmetic_obj = PyTuple_GET_ITEM(args, 2);
    bool is_arithmetic =
        CastPyArg2Boolean(is_arithmetic_obj, "bitwise_left_shift", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_left_shift");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::bitwise_left_shift(x, y, is_arithmetic);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_left_shift_(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_left_shift_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_left_shift_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_left_shift_", 1);

    // Parse Attributes
    PyObject *is_arithmetic_obj = PyTuple_GET_ITEM(args, 2);
    bool is_arithmetic =
        CastPyArg2Boolean(is_arithmetic_obj, "bitwise_left_shift_", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_left_shift_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::bitwise_left_shift_(x, y, is_arithmetic);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_not(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_not op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_not", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_not");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_not(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_not_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_not_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_not_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_not_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_not_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_or(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_or op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_or", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_or", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_or");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_or(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_or_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_or_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_or_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_or_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_or_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_or_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_right_shift(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_right_shift op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_right_shift", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_right_shift", 1);

    // Parse Attributes
    PyObject *is_arithmetic_obj = PyTuple_GET_ITEM(args, 2);
    bool is_arithmetic =
        CastPyArg2Boolean(is_arithmetic_obj, "bitwise_right_shift", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_right_shift");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::bitwise_right_shift(x, y, is_arithmetic);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_right_shift_(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_right_shift_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_right_shift_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_right_shift_", 1);

    // Parse Attributes
    PyObject *is_arithmetic_obj = PyTuple_GET_ITEM(args, 2);
    bool is_arithmetic =
        CastPyArg2Boolean(is_arithmetic_obj, "bitwise_right_shift_", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_right_shift_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::bitwise_right_shift_(x, y, is_arithmetic);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_xor(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_xor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_xor", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_xor", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_xor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_xor(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bitwise_xor_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add bitwise_xor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bitwise_xor_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bitwise_xor_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bitwise_xor_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bitwise_xor_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_bmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add bmm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "bmm", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "bmm", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("bmm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::bmm(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_box_coder(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add box_coder op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *prior_box_obj = PyTuple_GET_ITEM(args, 0);
    auto prior_box = CastPyArg2Value(prior_box_obj, "box_coder", 0);
    PyObject *prior_box_var_obj = PyTuple_GET_ITEM(args, 1);
    auto prior_box_var =
        CastPyArg2OptionalValue(prior_box_var_obj, "box_coder", 1);
    PyObject *target_box_obj = PyTuple_GET_ITEM(args, 2);
    auto target_box = CastPyArg2Value(target_box_obj, "box_coder", 2);

    // Parse Attributes
    PyObject *code_type_obj = PyTuple_GET_ITEM(args, 3);
    std::string code_type = CastPyArg2String(code_type_obj, "box_coder", 3);
    PyObject *box_normalized_obj = PyTuple_GET_ITEM(args, 4);
    bool box_normalized = CastPyArg2Boolean(box_normalized_obj, "box_coder", 4);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 5);
    int axis = CastPyArg2Int(axis_obj, "box_coder", 5);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<float> variance =
        CastPyArg2Floats(variance_obj, "box_coder", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("box_coder");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::box_coder(prior_box, prior_box_var, target_box,
                                   code_type, box_normalized, axis, variance);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_broadcast_tensors(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add broadcast_tensors op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2VectorOfValue(input_obj, "broadcast_tensors", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("broadcast_tensors");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::broadcast_tensors(input);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_ceil(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add ceil op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "ceil", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("ceil");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::ceil(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_ceil_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add ceil_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "ceil_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("ceil_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::ceil_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_celu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add celu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "celu", 0);

    // Parse Attributes
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "celu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("celu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::celu(x, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_check_finite_and_unscale_(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add check_finite_and_unscale_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "check_finite_and_unscale_", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "check_finite_and_unscale_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("check_finite_and_unscale_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::check_finite_and_unscale_(x, scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_check_numerics(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add check_numerics op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *tensor_obj = PyTuple_GET_ITEM(args, 0);
    auto tensor = CastPyArg2Value(tensor_obj, "check_numerics", 0);

    // Parse Attributes
    PyObject *op_type_obj = PyTuple_GET_ITEM(args, 1);
    std::string op_type = CastPyArg2String(op_type_obj, "check_numerics", 1);
    PyObject *var_name_obj = PyTuple_GET_ITEM(args, 2);
    std::string var_name = CastPyArg2String(var_name_obj, "check_numerics", 2);
    PyObject *check_nan_inf_level_obj = PyTuple_GET_ITEM(args, 3);
    int check_nan_inf_level =
        CastPyArg2Int(check_nan_inf_level_obj, "check_numerics", 3);
    PyObject *stack_height_limit_obj = PyTuple_GET_ITEM(args, 4);
    int stack_height_limit =
        CastPyArg2Int(stack_height_limit_obj, "check_numerics", 4);
    PyObject *output_dir_obj = PyTuple_GET_ITEM(args, 5);
    std::string output_dir =
        CastPyArg2String(output_dir_obj, "check_numerics", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("check_numerics");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::check_numerics(
        tensor, op_type, var_name, check_nan_inf_level, stack_height_limit,
        output_dir);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cholesky(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add cholesky op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cholesky", 0);

    // Parse Attributes
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 1);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("cholesky");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cholesky(x, upper);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cholesky_solve(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add cholesky_solve op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cholesky_solve", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "cholesky_solve", 1);

    // Parse Attributes
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky_solve", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("cholesky_solve");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cholesky_solve(x, y, upper);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_class_center_sample(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add class_center_sample op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *label_obj = PyTuple_GET_ITEM(args, 0);
    auto label = CastPyArg2Value(label_obj, "class_center_sample", 0);

    // Parse Attributes
    PyObject *num_classes_obj = PyTuple_GET_ITEM(args, 1);
    int num_classes = CastPyArg2Int(num_classes_obj, "class_center_sample", 1);
    PyObject *num_samples_obj = PyTuple_GET_ITEM(args, 2);
    int num_samples = CastPyArg2Int(num_samples_obj, "class_center_sample", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "class_center_sample", 3);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 4);
    int rank = CastPyArg2Int(rank_obj, "class_center_sample", 4);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 5);
    int nranks = CastPyArg2Int(nranks_obj, "class_center_sample", 5);
    PyObject *fix_seed_obj = PyTuple_GET_ITEM(args, 6);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "class_center_sample", 6);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 7);
    int seed = CastPyArg2Int(seed_obj, "class_center_sample", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("class_center_sample");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::class_center_sample(
        label, num_classes, num_samples, ring_id, rank, nranks, fix_seed, seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_clip(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add clip op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "clip", 0);

    // Parse Attributes
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value min;

    pir::Value max;

    if (PyObject_CheckIRValue(min_obj)) {
      min = CastPyArg2Value(min_obj, "clip", 1);
    } else {
      float min_tmp = CastPyArg2Float(min_obj, "clip", 1);
      min = paddle::dialect::full(std::vector<int64_t>{1}, min_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(max_obj)) {
      max = CastPyArg2Value(max_obj, "clip", 2);
    } else {
      float max_tmp = CastPyArg2Float(max_obj, "clip", 2);
      max = paddle::dialect::full(std::vector<int64_t>{1}, max_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("clip");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::clip(x, min, max);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_clip_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add clip_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "clip_", 0);

    // Parse Attributes
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value min;

    pir::Value max;

    if (PyObject_CheckIRValue(min_obj)) {
      min = CastPyArg2Value(min_obj, "clip_", 1);
    } else {
      float min_tmp = CastPyArg2Float(min_obj, "clip_", 1);
      min = paddle::dialect::full(std::vector<int64_t>{1}, min_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(max_obj)) {
      max = CastPyArg2Value(max_obj, "clip_", 2);
    } else {
      float max_tmp = CastPyArg2Float(max_obj, "clip_", 2);
      max = paddle::dialect::full(std::vector<int64_t>{1}, max_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("clip_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::clip_(x, min, max);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_clip_by_norm(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add clip_by_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "clip_by_norm", 0);

    // Parse Attributes
    PyObject *max_norm_obj = PyTuple_GET_ITEM(args, 1);
    float max_norm = CastPyArg2Float(max_norm_obj, "clip_by_norm", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("clip_by_norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::clip_by_norm(x, max_norm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_coalesce_tensor(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add coalesce_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2VectorOfValue(input_obj, "coalesce_tensor", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "coalesce_tensor", 1);
    PyObject *copy_data_obj = PyTuple_GET_ITEM(args, 2);
    bool copy_data = CastPyArg2Boolean(copy_data_obj, "coalesce_tensor", 2);
    PyObject *set_constant_obj = PyTuple_GET_ITEM(args, 3);
    bool set_constant =
        CastPyArg2Boolean(set_constant_obj, "coalesce_tensor", 3);
    PyObject *persist_output_obj = PyTuple_GET_ITEM(args, 4);
    bool persist_output =
        CastPyArg2Boolean(persist_output_obj, "coalesce_tensor", 4);
    PyObject *constant_obj = PyTuple_GET_ITEM(args, 5);
    float constant = CastPyArg2Float(constant_obj, "coalesce_tensor", 5);
    PyObject *use_align_obj = PyTuple_GET_ITEM(args, 6);
    bool use_align = CastPyArg2Boolean(use_align_obj, "coalesce_tensor", 6);
    PyObject *align_size_obj = PyTuple_GET_ITEM(args, 7);
    int align_size = CastPyArg2Int(align_size_obj, "coalesce_tensor", 7);
    PyObject *size_of_dtype_obj = PyTuple_GET_ITEM(args, 8);
    int size_of_dtype = CastPyArg2Int(size_of_dtype_obj, "coalesce_tensor", 8);
    PyObject *concated_shapes_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<int64_t> concated_shapes =
        CastPyArg2Longs(concated_shapes_obj, "coalesce_tensor", 9);
    PyObject *concated_ranks_obj = PyTuple_GET_ITEM(args, 10);
    std::vector<int64_t> concated_ranks =
        CastPyArg2Longs(concated_ranks_obj, "coalesce_tensor", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("coalesce_tensor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::coalesce_tensor(
        input, dtype, copy_data, set_constant, persist_output, constant,
        use_align, align_size, size_of_dtype, concated_shapes, concated_ranks);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_complex(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add complex op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *real_obj = PyTuple_GET_ITEM(args, 0);
    auto real = CastPyArg2Value(real_obj, "complex", 0);
    PyObject *imag_obj = PyTuple_GET_ITEM(args, 1);
    auto imag = CastPyArg2Value(imag_obj, "complex", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("complex");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::complex(real, imag);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add concat op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "concat", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "concat", 1);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "concat", 1);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("concat");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::concat(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conj(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add conj op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "conj", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("conj");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conj(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conv2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add conv2d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "conv2d", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "conv2d", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv2d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv2d", 3);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv2d", 4);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv2d", 5);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 6);
    int groups = CastPyArg2Int(groups_obj, "conv2d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv2d", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("conv2d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conv2d(
        input, filter, strides, paddings, padding_algorithm, dilations, groups,
        data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conv3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add conv3d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "conv3d", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "conv3d", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d", 3);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv3d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv3d", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv3d", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("conv3d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conv3d(
        input, filter, strides, paddings, padding_algorithm, groups, dilations,
        data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conv3d_transpose(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add conv3d_transpose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "conv3d_transpose", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "conv3d_transpose", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "conv3d_transpose", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "conv3d_transpose", 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "conv3d_transpose", 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "conv3d_transpose", 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv3d_transpose", 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "conv3d_transpose", 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "conv3d_transpose", 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "conv3d_transpose", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("conv3d_transpose");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conv3d_transpose(
        x, filter, strides, paddings, output_padding, output_size,
        padding_algorithm, groups, dilations, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_copysign(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add copysign op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "copysign", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "copysign", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("copysign");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::copysign(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_copysign_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add copysign_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "copysign_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "copysign_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("copysign_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::copysign_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cos(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cos op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cos", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("cos");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cos(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cos_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cos_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cos_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("cos_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cos_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cosh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cosh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("cosh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cosh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cosh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cosh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cosh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("cosh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cosh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_crop(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add crop op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "crop", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *offsets_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value shape;

    pir::Value offsets;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "crop", 1);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "crop", 1);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "crop", 1);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(offsets_obj)) {
      offsets = CastPyArg2Value(offsets_obj, "crop", 2);
    } else if (PyObject_CheckIRVectorOfValue(offsets_obj)) {
      std::vector<pir::Value> offsets_tmp =
          CastPyArg2VectorOfValue(offsets_obj, "crop", 2);
      offsets = paddle::dialect::stack(offsets_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> offsets_tmp =
          CastPyArg2Longs(offsets_obj, "crop", 2);
      offsets = paddle::dialect::full_int_array(
          offsets_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("crop");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::crop(x, shape, offsets);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cross(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cross op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cross", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "cross", 1);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "cross", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("cross");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cross(x, y, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cross_entropy_with_softmax(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add cross_entropy_with_softmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "cross_entropy_with_softmax", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "cross_entropy_with_softmax", 1);

    // Parse Attributes
    PyObject *soft_label_obj = PyTuple_GET_ITEM(args, 2);
    bool soft_label =
        CastPyArg2Boolean(soft_label_obj, "cross_entropy_with_softmax", 2);
    PyObject *use_softmax_obj = PyTuple_GET_ITEM(args, 3);
    bool use_softmax =
        CastPyArg2Boolean(use_softmax_obj, "cross_entropy_with_softmax", 3);
    PyObject *numeric_stable_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool numeric_stable_mode = CastPyArg2Boolean(
        numeric_stable_mode_obj, "cross_entropy_with_softmax", 4);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 5);
    int ignore_index =
        CastPyArg2Int(ignore_index_obj, "cross_entropy_with_softmax", 5);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 6);
    int axis = CastPyArg2Int(axis_obj, "cross_entropy_with_softmax", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("cross_entropy_with_softmax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cross_entropy_with_softmax(
        input, label, soft_label, use_softmax, numeric_stable_mode,
        ignore_index, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cross_entropy_with_softmax_(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add cross_entropy_with_softmax_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "cross_entropy_with_softmax_", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "cross_entropy_with_softmax_", 1);

    // Parse Attributes
    PyObject *soft_label_obj = PyTuple_GET_ITEM(args, 2);
    bool soft_label =
        CastPyArg2Boolean(soft_label_obj, "cross_entropy_with_softmax_", 2);
    PyObject *use_softmax_obj = PyTuple_GET_ITEM(args, 3);
    bool use_softmax =
        CastPyArg2Boolean(use_softmax_obj, "cross_entropy_with_softmax_", 3);
    PyObject *numeric_stable_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool numeric_stable_mode = CastPyArg2Boolean(
        numeric_stable_mode_obj, "cross_entropy_with_softmax_", 4);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 5);
    int ignore_index =
        CastPyArg2Int(ignore_index_obj, "cross_entropy_with_softmax_", 5);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 6);
    int axis = CastPyArg2Int(axis_obj, "cross_entropy_with_softmax_", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("cross_entropy_with_softmax_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cross_entropy_with_softmax_(
        input, label, soft_label, use_softmax, numeric_stable_mode,
        ignore_index, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cummax(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cummax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cummax", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "cummax", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "cummax", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("cummax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cummax(x, axis, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cummin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cummin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cummin", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "cummin", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "cummin", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("cummin");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cummin(x, axis, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cumprod(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cumprod op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cumprod", 0);

    // Parse Attributes
    PyObject *dim_obj = PyTuple_GET_ITEM(args, 1);
    int dim = CastPyArg2Int(dim_obj, "cumprod", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("cumprod");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cumprod(x, dim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cumprod_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add cumprod_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cumprod_", 0);

    // Parse Attributes
    PyObject *dim_obj = PyTuple_GET_ITEM(args, 1);
    int dim = CastPyArg2Int(dim_obj, "cumprod_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("cumprod_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cumprod_(x, dim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cumsum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cumsum", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *reverse_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "cumsum", 1);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "cumsum", 1);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }
    bool flatten = CastPyArg2Boolean(flatten_obj, "cumsum", 2);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "cumsum", 3);
    bool reverse = CastPyArg2Boolean(reverse_obj, "cumsum", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("cumsum");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::cumsum(x, axis, flatten, exclusive, reverse);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cumsum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cumsum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cumsum_", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *reverse_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "cumsum_", 1);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "cumsum_", 1);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }
    bool flatten = CastPyArg2Boolean(flatten_obj, "cumsum_", 2);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "cumsum_", 3);
    bool reverse = CastPyArg2Boolean(reverse_obj, "cumsum_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("cumsum_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::cumsum_(x, axis, flatten, exclusive, reverse);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_data(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add data op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 0);
    std::string name = CastPyArg2String(name_obj, "data", 0);
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "data", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "data", 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);
    Place place = CastPyArg2Place(place_obj, "data", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("data");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::data(name, shape, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_depthwise_conv2d(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add depthwise_conv2d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "depthwise_conv2d", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "depthwise_conv2d", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "depthwise_conv2d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "depthwise_conv2d", 3);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "depthwise_conv2d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "depthwise_conv2d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format =
        CastPyArg2String(data_format_obj, "depthwise_conv2d", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("depthwise_conv2d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::depthwise_conv2d(
        input, filter, strides, paddings, padding_algorithm, groups, dilations,
        data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_det(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add det op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "det", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("det");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::det(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_diag(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add diag op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "diag", 0);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diag", 1);
    PyObject *padding_value_obj = PyTuple_GET_ITEM(args, 2);
    float padding_value = CastPyArg2Float(padding_value_obj, "diag", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("diag");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::diag(x, offset, padding_value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_diag_embed(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add diag_embed op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "diag_embed", 0);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diag_embed", 1);
    PyObject *dim1_obj = PyTuple_GET_ITEM(args, 2);
    int dim1 = CastPyArg2Int(dim1_obj, "diag_embed", 2);
    PyObject *dim2_obj = PyTuple_GET_ITEM(args, 3);
    int dim2 = CastPyArg2Int(dim2_obj, "diag_embed", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("diag_embed");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::diag_embed(input, offset, dim1, dim2);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_diagonal(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add diagonal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "diagonal", 0);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diagonal", 1);
    PyObject *axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "diagonal", 2);
    PyObject *axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "diagonal", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("diagonal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::diagonal(x, offset, axis1, axis2);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_digamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add digamma op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "digamma", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("digamma");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::digamma(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_digamma_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add digamma_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "digamma_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("digamma_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::digamma_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dirichlet(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add dirichlet op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 0);
    auto alpha = CastPyArg2Value(alpha_obj, "dirichlet", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("dirichlet");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dirichlet(alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dist(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add dist op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "dist", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "dist", 1);

    // Parse Attributes
    PyObject *p_obj = PyTuple_GET_ITEM(args, 2);
    float p = CastPyArg2Float(p_obj, "dist", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("dist");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dist(x, y, p);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dot(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add dot op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "dot", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "dot", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("dot");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dot(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_edit_distance(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add edit_distance op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *hyps_obj = PyTuple_GET_ITEM(args, 0);
    auto hyps = CastPyArg2Value(hyps_obj, "edit_distance", 0);
    PyObject *refs_obj = PyTuple_GET_ITEM(args, 1);
    auto refs = CastPyArg2Value(refs_obj, "edit_distance", 1);
    PyObject *hypslength_obj = PyTuple_GET_ITEM(args, 2);
    auto hypslength =
        CastPyArg2OptionalValue(hypslength_obj, "edit_distance", 2);
    PyObject *refslength_obj = PyTuple_GET_ITEM(args, 3);
    auto refslength =
        CastPyArg2OptionalValue(refslength_obj, "edit_distance", 3);

    // Parse Attributes
    PyObject *normalized_obj = PyTuple_GET_ITEM(args, 4);
    bool normalized = CastPyArg2Boolean(normalized_obj, "edit_distance", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("edit_distance");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::edit_distance(
        hyps, refs, hypslength, refslength, normalized);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_eig(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add eig op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "eig", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("eig");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::eig(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_eigh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add eigh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "eigh", 0);

    // Parse Attributes
    PyObject *UPLO_obj = PyTuple_GET_ITEM(args, 1);
    std::string UPLO = CastPyArg2String(UPLO_obj, "eigh", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("eigh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::eigh(x, UPLO);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_eigvals(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add eigvals op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "eigvals", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("eigvals");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::eigvals(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_eigvalsh(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add eigvalsh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "eigvalsh", 0);

    // Parse Attributes
    PyObject *uplo_obj = PyTuple_GET_ITEM(args, 1);
    std::string uplo = CastPyArg2String(uplo_obj, "eigvalsh", 1);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 2);
    bool is_test = CastPyArg2Boolean(is_test_obj, "eigvalsh", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("eigvalsh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::eigvalsh(x, uplo, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_elu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add elu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "elu", 0);

    // Parse Attributes
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "elu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("elu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::elu(x, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_elu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add elu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "elu_", 0);

    // Parse Attributes
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "elu_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("elu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::elu_(x, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_equal_all(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add equal_all op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "equal_all", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "equal_all", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("equal_all");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::equal_all(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_erf(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add erf op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "erf", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("erf");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::erf(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_erf_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add erf_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "erf_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("erf_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::erf_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_erfinv(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add erfinv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "erfinv", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("erfinv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::erfinv(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_erfinv_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add erfinv_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "erfinv_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("erfinv_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::erfinv_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_exp(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add exp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "exp", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("exp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::exp(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_exp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add exp_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "exp_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("exp_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::exp_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_expand(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add expand op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "expand", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "expand", 1);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "expand", 1);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "expand", 1);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("expand");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::expand(x, shape);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_expand_as(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add expand_as op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "expand_as", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2OptionalValue(y_obj, "expand_as", 1);

    // Parse Attributes
    PyObject *target_shape_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> target_shape =
        CastPyArg2Ints(target_shape_obj, "expand_as", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("expand_as");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::expand_as(x, y, target_shape);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_expm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add expm1 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "expm1", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("expm1");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::expm1(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_expm1_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add expm1_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "expm1_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("expm1_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::expm1_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fft_c2c op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fft_c2c", 0);

    // Parse Attributes
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_c2c", 1);
    PyObject *normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization =
        CastPyArg2String(normalization_obj, "fft_c2c", 2);
    PyObject *forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_c2c", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("fft_c2c");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fft_c2c(x, axes, normalization, forward);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fft_c2r op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fft_c2r", 0);

    // Parse Attributes
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_c2r", 1);
    PyObject *normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization =
        CastPyArg2String(normalization_obj, "fft_c2r", 2);
    PyObject *forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_c2r", 3);
    PyObject *last_dim_size_obj = PyTuple_GET_ITEM(args, 4);
    int64_t last_dim_size = CastPyArg2Long(last_dim_size_obj, "fft_c2r", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fft_c2r");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fft_c2r(x, axes, normalization,
                                                   forward, last_dim_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fft_r2c op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fft_r2c", 0);

    // Parse Attributes
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_r2c", 1);
    PyObject *normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization =
        CastPyArg2String(normalization_obj, "fft_r2c", 2);
    PyObject *forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_r2c", 3);
    PyObject *onesided_obj = PyTuple_GET_ITEM(args, 4);
    bool onesided = CastPyArg2Boolean(onesided_obj, "fft_r2c", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fft_r2c");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fft_r2c(x, axes, normalization, forward, onesided);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value value;

    if (PyObject_CheckIRValue(value_obj)) {
      value = CastPyArg2Value(value_obj, "fill", 1);
    } else {
      float value_tmp = CastPyArg2Float(value_obj, "fill", 1);
      value = paddle::dialect::full(std::vector<int64_t>{1}, value_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("fill");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fill(x, value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill_", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value value;

    if (PyObject_CheckIRValue(value_obj)) {
      value = CastPyArg2Value(value_obj, "fill_", 1);
    } else {
      float value_tmp = CastPyArg2Float(value_obj, "fill_", 1);
      value = paddle::dialect::full(std::vector<int64_t>{1}, value_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("fill_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fill_(x, value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill_diagonal(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill_diagonal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill_diagonal", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "fill_diagonal", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "fill_diagonal", 2);
    PyObject *wrap_obj = PyTuple_GET_ITEM(args, 3);
    bool wrap = CastPyArg2Boolean(wrap_obj, "fill_diagonal", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("fill_diagonal");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fill_diagonal(x, value, offset, wrap);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill_diagonal_(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill_diagonal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill_diagonal_", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "fill_diagonal_", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "fill_diagonal_", 2);
    PyObject *wrap_obj = PyTuple_GET_ITEM(args, 3);
    bool wrap = CastPyArg2Boolean(wrap_obj, "fill_diagonal_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("fill_diagonal_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fill_diagonal_(x, value, offset, wrap);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill_diagonal_tensor(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill_diagonal_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill_diagonal_tensor", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fill_diagonal_tensor", 1);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int64_t offset = CastPyArg2Long(offset_obj, "fill_diagonal_tensor", 2);
    PyObject *dim1_obj = PyTuple_GET_ITEM(args, 3);
    int dim1 = CastPyArg2Int(dim1_obj, "fill_diagonal_tensor", 3);
    PyObject *dim2_obj = PyTuple_GET_ITEM(args, 4);
    int dim2 = CastPyArg2Int(dim2_obj, "fill_diagonal_tensor", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fill_diagonal_tensor");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fill_diagonal_tensor(x, y, offset, dim1, dim2);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fill_diagonal_tensor_(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add fill_diagonal_tensor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fill_diagonal_tensor_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fill_diagonal_tensor_", 1);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int64_t offset = CastPyArg2Long(offset_obj, "fill_diagonal_tensor_", 2);
    PyObject *dim1_obj = PyTuple_GET_ITEM(args, 3);
    int dim1 = CastPyArg2Int(dim1_obj, "fill_diagonal_tensor_", 3);
    PyObject *dim2_obj = PyTuple_GET_ITEM(args, 4);
    int dim2 = CastPyArg2Int(dim2_obj, "fill_diagonal_tensor_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fill_diagonal_tensor_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fill_diagonal_tensor_(x, y, offset, dim1, dim2);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flash_attn(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add flash_attn op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *q_obj = PyTuple_GET_ITEM(args, 0);
    auto q = CastPyArg2Value(q_obj, "flash_attn", 0);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    auto k = CastPyArg2Value(k_obj, "flash_attn", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v = CastPyArg2Value(v_obj, "flash_attn", 2);
    PyObject *fixed_seed_offset_obj = PyTuple_GET_ITEM(args, 3);
    auto fixed_seed_offset =
        CastPyArg2OptionalValue(fixed_seed_offset_obj, "flash_attn", 3);
    PyObject *attn_mask_obj = PyTuple_GET_ITEM(args, 4);
    auto attn_mask = CastPyArg2OptionalValue(attn_mask_obj, "flash_attn", 4);

    // Parse Attributes
    PyObject *dropout_obj = PyTuple_GET_ITEM(args, 5);
    float dropout = CastPyArg2Float(dropout_obj, "flash_attn", 5);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 6);
    bool causal = CastPyArg2Boolean(causal_obj, "flash_attn", 6);
    PyObject *return_softmax_obj = PyTuple_GET_ITEM(args, 7);
    bool return_softmax =
        CastPyArg2Boolean(return_softmax_obj, "flash_attn", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "flash_attn", 8);
    PyObject *rng_name_obj = PyTuple_GET_ITEM(args, 9);
    std::string rng_name = CastPyArg2String(rng_name_obj, "flash_attn", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("flash_attn");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flash_attn(
        q, k, v, fixed_seed_offset, attn_mask, dropout, causal, return_softmax,
        is_test, rng_name);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flash_attn_unpadded(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add flash_attn_unpadded op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *q_obj = PyTuple_GET_ITEM(args, 0);
    auto q = CastPyArg2Value(q_obj, "flash_attn_unpadded", 0);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    auto k = CastPyArg2Value(k_obj, "flash_attn_unpadded", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v = CastPyArg2Value(v_obj, "flash_attn_unpadded", 2);
    PyObject *cu_seqlens_q_obj = PyTuple_GET_ITEM(args, 3);
    auto cu_seqlens_q =
        CastPyArg2Value(cu_seqlens_q_obj, "flash_attn_unpadded", 3);
    PyObject *cu_seqlens_k_obj = PyTuple_GET_ITEM(args, 4);
    auto cu_seqlens_k =
        CastPyArg2Value(cu_seqlens_k_obj, "flash_attn_unpadded", 4);
    PyObject *fixed_seed_offset_obj = PyTuple_GET_ITEM(args, 5);
    auto fixed_seed_offset = CastPyArg2OptionalValue(fixed_seed_offset_obj,
                                                     "flash_attn_unpadded", 5);
    PyObject *attn_mask_obj = PyTuple_GET_ITEM(args, 6);
    auto attn_mask =
        CastPyArg2OptionalValue(attn_mask_obj, "flash_attn_unpadded", 6);

    // Parse Attributes
    PyObject *max_seqlen_q_obj = PyTuple_GET_ITEM(args, 7);
    int64_t max_seqlen_q =
        CastPyArg2Long(max_seqlen_q_obj, "flash_attn_unpadded", 7);
    PyObject *max_seqlen_k_obj = PyTuple_GET_ITEM(args, 8);
    int64_t max_seqlen_k =
        CastPyArg2Long(max_seqlen_k_obj, "flash_attn_unpadded", 8);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 9);
    float scale = CastPyArg2Float(scale_obj, "flash_attn_unpadded", 9);
    PyObject *dropout_obj = PyTuple_GET_ITEM(args, 10);
    float dropout = CastPyArg2Float(dropout_obj, "flash_attn_unpadded", 10);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 11);
    bool causal = CastPyArg2Boolean(causal_obj, "flash_attn_unpadded", 11);
    PyObject *return_softmax_obj = PyTuple_GET_ITEM(args, 12);
    bool return_softmax =
        CastPyArg2Boolean(return_softmax_obj, "flash_attn_unpadded", 12);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 13);
    bool is_test = CastPyArg2Boolean(is_test_obj, "flash_attn_unpadded", 13);
    PyObject *rng_name_obj = PyTuple_GET_ITEM(args, 14);
    std::string rng_name =
        CastPyArg2String(rng_name_obj, "flash_attn_unpadded", 14);

    // Call ir static api
    CallStackRecorder callstack_recorder("flash_attn_unpadded");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flash_attn_unpadded(
        q, k, v, cu_seqlens_q, cu_seqlens_k, fixed_seed_offset, attn_mask,
        max_seqlen_q, max_seqlen_k, scale, dropout, causal, return_softmax,
        is_test, rng_name);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flash_attn_with_sparse_mask(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add flash_attn_with_sparse_mask op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *q_obj = PyTuple_GET_ITEM(args, 0);
    auto q = CastPyArg2Value(q_obj, "flash_attn_with_sparse_mask", 0);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    auto k = CastPyArg2Value(k_obj, "flash_attn_with_sparse_mask", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v = CastPyArg2Value(v_obj, "flash_attn_with_sparse_mask", 2);
    PyObject *attn_mask_start_row_indices_obj = PyTuple_GET_ITEM(args, 3);
    auto attn_mask_start_row_indices = CastPyArg2Value(
        attn_mask_start_row_indices_obj, "flash_attn_with_sparse_mask", 3);
    PyObject *fixed_seed_offset_obj = PyTuple_GET_ITEM(args, 4);
    auto fixed_seed_offset = CastPyArg2OptionalValue(
        fixed_seed_offset_obj, "flash_attn_with_sparse_mask", 4);

    // Parse Attributes
    PyObject *dropout_obj = PyTuple_GET_ITEM(args, 5);
    float dropout =
        CastPyArg2Float(dropout_obj, "flash_attn_with_sparse_mask", 5);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 6);
    bool causal =
        CastPyArg2Boolean(causal_obj, "flash_attn_with_sparse_mask", 6);
    PyObject *attn_mask_start_row_obj = PyTuple_GET_ITEM(args, 7);
    int attn_mask_start_row = CastPyArg2Int(attn_mask_start_row_obj,
                                            "flash_attn_with_sparse_mask", 7);
    PyObject *return_softmax_obj = PyTuple_GET_ITEM(args, 8);
    bool return_softmax =
        CastPyArg2Boolean(return_softmax_obj, "flash_attn_with_sparse_mask", 8);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 9);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "flash_attn_with_sparse_mask", 9);
    PyObject *rng_name_obj = PyTuple_GET_ITEM(args, 10);
    std::string rng_name =
        CastPyArg2String(rng_name_obj, "flash_attn_with_sparse_mask", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("flash_attn_with_sparse_mask");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flash_attn_with_sparse_mask(
        q, k, v, attn_mask_start_row_indices, fixed_seed_offset, dropout,
        causal, attn_mask_start_row, return_softmax, is_test, rng_name);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flatten(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add flatten op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "flatten", 0);

    // Parse Attributes
    PyObject *start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten", 1);
    PyObject *stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("flatten");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flatten(x, start_axis, stop_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flatten_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add flatten_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "flatten_", 0);

    // Parse Attributes
    PyObject *start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten_", 1);
    PyObject *stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten_", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("flatten_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flatten_(x, start_axis, stop_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_flip(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add flip op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "flip", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "flip", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("flip");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::flip(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_floor(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add floor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "floor", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("floor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::floor(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_floor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add floor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "floor_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("floor_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::floor_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fmax", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fmax", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("fmax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fmax(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fmin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fmin", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fmin", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("fmin");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fmin(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fold(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fold op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fold", 0);

    // Parse Attributes
    PyObject *output_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> output_sizes = CastPyArg2Ints(output_sizes_obj, "fold", 1);
    PyObject *kernel_sizes_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> kernel_sizes = CastPyArg2Ints(kernel_sizes_obj, "fold", 2);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "fold", 3);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "fold", 4);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "fold", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fold");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fold(x, output_sizes, kernel_sizes,
                                                strides, paddings, dilations);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fractional_max_pool2d(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add fractional_max_pool2d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fractional_max_pool2d", 0);

    // Parse Attributes
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "fractional_max_pool2d", 1);
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "fractional_max_pool2d", 2);
    PyObject *random_u_obj = PyTuple_GET_ITEM(args, 3);
    float random_u = CastPyArg2Float(random_u_obj, "fractional_max_pool2d", 3);
    PyObject *return_mask_obj = PyTuple_GET_ITEM(args, 4);
    bool return_mask =
        CastPyArg2Boolean(return_mask_obj, "fractional_max_pool2d", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fractional_max_pool2d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fractional_max_pool2d(
        x, output_size, kernel_size, random_u, return_mask);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fractional_max_pool3d(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add fractional_max_pool3d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fractional_max_pool3d", 0);

    // Parse Attributes
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "fractional_max_pool3d", 1);
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "fractional_max_pool3d", 2);
    PyObject *random_u_obj = PyTuple_GET_ITEM(args, 3);
    float random_u = CastPyArg2Float(random_u_obj, "fractional_max_pool3d", 3);
    PyObject *return_mask_obj = PyTuple_GET_ITEM(args, 4);
    bool return_mask =
        CastPyArg2Boolean(return_mask_obj, "fractional_max_pool3d", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fractional_max_pool3d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fractional_max_pool3d(
        x, output_size, kernel_size, random_u, return_mask);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add frame op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "frame", 0);

    // Parse Attributes
    PyObject *frame_length_obj = PyTuple_GET_ITEM(args, 1);
    int frame_length = CastPyArg2Int(frame_length_obj, "frame", 1);
    PyObject *hop_length_obj = PyTuple_GET_ITEM(args, 2);
    int hop_length = CastPyArg2Int(hop_length_obj, "frame", 2);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "frame", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("frame");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::frame(x, frame_length, hop_length, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full_int_array(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add full_int_array op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int64_t> value =
        CastPyArg2Longs(value_obj, "full_int_array", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "full_int_array", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    Place place = CastPyArg2Place(place_obj, "full_int_array", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("full_int_array");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::full_int_array(value, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gammaincc(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add gammaincc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gammaincc", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "gammaincc", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gammaincc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gammaincc(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gammaincc_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add gammaincc_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gammaincc_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "gammaincc_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gammaincc_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gammaincc_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gammaln(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add gammaln op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gammaln", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gammaln");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gammaln(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gammaln_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add gammaln_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gammaln_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gammaln_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gammaln_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gather(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add gather op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gather", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "gather", 1);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "gather", 2);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "gather", 2);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("gather");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gather(x, index, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gather_nd(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add gather_nd op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gather_nd", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "gather_nd", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gather_nd");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gather_nd(x, index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gather_tree(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add gather_tree op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2Value(ids_obj, "gather_tree", 0);
    PyObject *parents_obj = PyTuple_GET_ITEM(args, 1);
    auto parents = CastPyArg2Value(parents_obj, "gather_tree", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("gather_tree");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gather_tree(ids, parents);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gaussian_inplace(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add gaussian_inplace op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gaussian_inplace", 0);

    // Parse Attributes
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "gaussian_inplace", 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "gaussian_inplace", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "gaussian_inplace", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("gaussian_inplace");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gaussian_inplace(x, mean, std, seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gaussian_inplace_(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add gaussian_inplace_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gaussian_inplace_", 0);

    // Parse Attributes
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "gaussian_inplace_", 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "gaussian_inplace_", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "gaussian_inplace_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("gaussian_inplace_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::gaussian_inplace_(x, mean, std, seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add gelu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gelu", 0);

    // Parse Attributes
    PyObject *approximate_obj = PyTuple_GET_ITEM(args, 1);
    bool approximate = CastPyArg2Boolean(approximate_obj, "gelu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("gelu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::gelu(x, approximate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_generate_proposals(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add generate_proposals op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *scores_obj = PyTuple_GET_ITEM(args, 0);
    auto scores = CastPyArg2Value(scores_obj, "generate_proposals", 0);
    PyObject *bbox_deltas_obj = PyTuple_GET_ITEM(args, 1);
    auto bbox_deltas =
        CastPyArg2Value(bbox_deltas_obj, "generate_proposals", 1);
    PyObject *im_shape_obj = PyTuple_GET_ITEM(args, 2);
    auto im_shape = CastPyArg2Value(im_shape_obj, "generate_proposals", 2);
    PyObject *anchors_obj = PyTuple_GET_ITEM(args, 3);
    auto anchors = CastPyArg2Value(anchors_obj, "generate_proposals", 3);
    PyObject *variances_obj = PyTuple_GET_ITEM(args, 4);
    auto variances = CastPyArg2Value(variances_obj, "generate_proposals", 4);

    // Parse Attributes
    PyObject *pre_nms_top_n_obj = PyTuple_GET_ITEM(args, 5);
    int pre_nms_top_n =
        CastPyArg2Int(pre_nms_top_n_obj, "generate_proposals", 5);
    PyObject *post_nms_top_n_obj = PyTuple_GET_ITEM(args, 6);
    int post_nms_top_n =
        CastPyArg2Int(post_nms_top_n_obj, "generate_proposals", 6);
    PyObject *nms_thresh_obj = PyTuple_GET_ITEM(args, 7);
    float nms_thresh = CastPyArg2Float(nms_thresh_obj, "generate_proposals", 7);
    PyObject *min_size_obj = PyTuple_GET_ITEM(args, 8);
    float min_size = CastPyArg2Float(min_size_obj, "generate_proposals", 8);
    PyObject *eta_obj = PyTuple_GET_ITEM(args, 9);
    float eta = CastPyArg2Float(eta_obj, "generate_proposals", 9);
    PyObject *pixel_offset_obj = PyTuple_GET_ITEM(args, 10);
    bool pixel_offset =
        CastPyArg2Boolean(pixel_offset_obj, "generate_proposals", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("generate_proposals");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::generate_proposals(
        scores, bbox_deltas, im_shape, anchors, variances, pre_nms_top_n,
        post_nms_top_n, nms_thresh, min_size, eta, pixel_offset);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_graph_khop_sampler(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add graph_khop_sampler op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *row_obj = PyTuple_GET_ITEM(args, 0);
    auto row = CastPyArg2Value(row_obj, "graph_khop_sampler", 0);
    PyObject *colptr_obj = PyTuple_GET_ITEM(args, 1);
    auto colptr = CastPyArg2Value(colptr_obj, "graph_khop_sampler", 1);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 2);
    auto x = CastPyArg2Value(x_obj, "graph_khop_sampler", 2);
    PyObject *eids_obj = PyTuple_GET_ITEM(args, 3);
    auto eids = CastPyArg2OptionalValue(eids_obj, "graph_khop_sampler", 3);

    // Parse Attributes
    PyObject *sample_sizes_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> sample_sizes =
        CastPyArg2Ints(sample_sizes_obj, "graph_khop_sampler", 4);
    PyObject *return_eids_obj = PyTuple_GET_ITEM(args, 5);
    bool return_eids =
        CastPyArg2Boolean(return_eids_obj, "graph_khop_sampler", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("graph_khop_sampler");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::graph_khop_sampler(
        row, colptr, x, eids, sample_sizes, return_eids);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_graph_sample_neighbors(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add graph_sample_neighbors op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *row_obj = PyTuple_GET_ITEM(args, 0);
    auto row = CastPyArg2Value(row_obj, "graph_sample_neighbors", 0);
    PyObject *colptr_obj = PyTuple_GET_ITEM(args, 1);
    auto colptr = CastPyArg2Value(colptr_obj, "graph_sample_neighbors", 1);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 2);
    auto x = CastPyArg2Value(x_obj, "graph_sample_neighbors", 2);
    PyObject *eids_obj = PyTuple_GET_ITEM(args, 3);
    auto eids = CastPyArg2OptionalValue(eids_obj, "graph_sample_neighbors", 3);
    PyObject *perm_buffer_obj = PyTuple_GET_ITEM(args, 4);
    auto perm_buffer =
        CastPyArg2OptionalValue(perm_buffer_obj, "graph_sample_neighbors", 4);

    // Parse Attributes
    PyObject *sample_size_obj = PyTuple_GET_ITEM(args, 5);
    int sample_size =
        CastPyArg2Int(sample_size_obj, "graph_sample_neighbors", 5);
    PyObject *return_eids_obj = PyTuple_GET_ITEM(args, 6);
    bool return_eids =
        CastPyArg2Boolean(return_eids_obj, "graph_sample_neighbors", 6);
    PyObject *flag_perm_buffer_obj = PyTuple_GET_ITEM(args, 7);
    bool flag_perm_buffer =
        CastPyArg2Boolean(flag_perm_buffer_obj, "graph_sample_neighbors", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("graph_sample_neighbors");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::graph_sample_neighbors(
        row, colptr, x, eids, perm_buffer, sample_size, return_eids,
        flag_perm_buffer);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_grid_sample(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add grid_sample op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "grid_sample", 0);
    PyObject *grid_obj = PyTuple_GET_ITEM(args, 1);
    auto grid = CastPyArg2Value(grid_obj, "grid_sample", 1);

    // Parse Attributes
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 2);
    std::string mode = CastPyArg2String(mode_obj, "grid_sample", 2);
    PyObject *padding_mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string padding_mode =
        CastPyArg2String(padding_mode_obj, "grid_sample", 3);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 4);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "grid_sample", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("grid_sample");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::grid_sample(
        x, grid, mode, padding_mode, align_corners);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_group_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add group_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "group_norm", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2OptionalValue(scale_obj, "group_norm", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "group_norm", 2);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "group_norm", 3);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 4);
    int groups = CastPyArg2Int(groups_obj, "group_norm", 4);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 5);
    std::string data_format =
        CastPyArg2String(data_format_obj, "group_norm", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("group_norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::group_norm(x, scale, bias, epsilon,
                                                      groups, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gumbel_softmax(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add gumbel_softmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "gumbel_softmax", 0);

    // Parse Attributes
    PyObject *temperature_obj = PyTuple_GET_ITEM(args, 1);
    float temperature = CastPyArg2Float(temperature_obj, "gumbel_softmax", 1);
    PyObject *hard_obj = PyTuple_GET_ITEM(args, 2);
    bool hard = CastPyArg2Boolean(hard_obj, "gumbel_softmax", 2);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "gumbel_softmax", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("gumbel_softmax");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::gumbel_softmax(x, temperature, hard, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hardshrink(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add hardshrink op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hardshrink", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "hardshrink", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("hardshrink");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hardshrink(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hardsigmoid(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add hardsigmoid op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hardsigmoid", 0);

    // Parse Attributes
    PyObject *slope_obj = PyTuple_GET_ITEM(args, 1);
    float slope = CastPyArg2Float(slope_obj, "hardsigmoid", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    float offset = CastPyArg2Float(offset_obj, "hardsigmoid", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("hardsigmoid");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hardsigmoid(x, slope, offset);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hardtanh(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add hardtanh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hardtanh", 0);

    // Parse Attributes
    PyObject *t_min_obj = PyTuple_GET_ITEM(args, 1);
    float t_min = CastPyArg2Float(t_min_obj, "hardtanh", 1);
    PyObject *t_max_obj = PyTuple_GET_ITEM(args, 2);
    float t_max = CastPyArg2Float(t_max_obj, "hardtanh", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("hardtanh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hardtanh(x, t_min, t_max);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hardtanh_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add hardtanh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hardtanh_", 0);

    // Parse Attributes
    PyObject *t_min_obj = PyTuple_GET_ITEM(args, 1);
    float t_min = CastPyArg2Float(t_min_obj, "hardtanh_", 1);
    PyObject *t_max_obj = PyTuple_GET_ITEM(args, 2);
    float t_max = CastPyArg2Float(t_max_obj, "hardtanh_", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("hardtanh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hardtanh_(x, t_min, t_max);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_heaviside(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add heaviside op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "heaviside", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "heaviside", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("heaviside");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::heaviside(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_histogram(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add histogram op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "histogram", 0);

    // Parse Attributes
    PyObject *bins_obj = PyTuple_GET_ITEM(args, 1);
    int64_t bins = CastPyArg2Long(bins_obj, "histogram", 1);
    PyObject *min_obj = PyTuple_GET_ITEM(args, 2);
    int min = CastPyArg2Int(min_obj, "histogram", 2);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 3);
    int max = CastPyArg2Int(max_obj, "histogram", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("histogram");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::histogram(input, bins, min, max);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_huber_loss(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add huber_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "huber_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "huber_loss", 1);

    // Parse Attributes
    PyObject *delta_obj = PyTuple_GET_ITEM(args, 2);
    float delta = CastPyArg2Float(delta_obj, "huber_loss", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("huber_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::huber_loss(input, label, delta);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_i0(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add i0 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "i0", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("i0");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::i0(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_i0_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add i0_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "i0_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("i0_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::i0_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_i0e(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add i0e op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "i0e", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("i0e");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::i0e(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_i1(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add i1 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "i1", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("i1");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::i1(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_i1e(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add i1e op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "i1e", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("i1e");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::i1e(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_identity_loss(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add identity_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "identity_loss", 0);

    // Parse Attributes
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 1);
    int reduction = CastPyArg2Int(reduction_obj, "identity_loss", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("identity_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::identity_loss(x, reduction);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_identity_loss_(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add identity_loss_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "identity_loss_", 0);

    // Parse Attributes
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 1);
    int reduction = CastPyArg2Int(reduction_obj, "identity_loss_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("identity_loss_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::identity_loss_(x, reduction);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_imag(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add imag op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "imag", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("imag");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::imag(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_add(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_add", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "index_add", 1);
    PyObject *add_value_obj = PyTuple_GET_ITEM(args, 2);
    auto add_value = CastPyArg2Value(add_value_obj, "index_add", 2);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "index_add", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::index_add(x, index, add_value, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_add_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_add_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_add_", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "index_add_", 1);
    PyObject *add_value_obj = PyTuple_GET_ITEM(args, 2);
    auto add_value = CastPyArg2Value(add_value_obj, "index_add_", 2);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "index_add_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_add_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::index_add_(x, index, add_value, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_put(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_put op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_put", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2VectorOfValue(indices_obj, "index_put", 1);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 2);
    auto value = CastPyArg2Value(value_obj, "index_put", 2);

    // Parse Attributes
    PyObject *accumulate_obj = PyTuple_GET_ITEM(args, 3);
    bool accumulate = CastPyArg2Boolean(accumulate_obj, "index_put", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_put");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::index_put(x, indices, value, accumulate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_put_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_put_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_put_", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2VectorOfValue(indices_obj, "index_put_", 1);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 2);
    auto value = CastPyArg2Value(value_obj, "index_put_", 2);

    // Parse Attributes
    PyObject *accumulate_obj = PyTuple_GET_ITEM(args, 3);
    bool accumulate = CastPyArg2Boolean(accumulate_obj, "index_put_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_put_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::index_put_(x, indices, value, accumulate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_sample(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_sample op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_sample", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "index_sample", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("index_sample");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::index_sample(x, index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_select(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_select op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_select", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "index_select", 1);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "index_select", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_select");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::index_select(x, index, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_index_select_strided(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add index_select_strided op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "index_select_strided", 0);

    // Parse Attributes
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    int64_t index = CastPyArg2Long(index_obj, "index_select_strided", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "index_select_strided", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("index_select_strided");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::index_select_strided(x, index, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_instance_norm(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add instance_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "instance_norm", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2OptionalValue(scale_obj, "instance_norm", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "instance_norm", 2);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "instance_norm", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("instance_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::instance_norm(x, scale, bias, epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_inverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add inverse op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "inverse", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("inverse");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::inverse(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_is_empty(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add is_empty op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "is_empty", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("is_empty");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::is_empty(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_isclose(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add isclose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "isclose", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "isclose", 1);

    // Parse Attributes
    PyObject *rtol_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *atol_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *equal_nan_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value rtol;

    pir::Value atol;

    if (PyObject_CheckIRValue(rtol_obj)) {
      rtol = CastPyArg2Value(rtol_obj, "isclose", 2);
    } else {
      double rtol_tmp = CastPyArg2Double(rtol_obj, "isclose", 2);
      rtol = paddle::dialect::full(std::vector<int64_t>{1}, rtol_tmp,
                                   phi::DataType::FLOAT64, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(atol_obj)) {
      atol = CastPyArg2Value(atol_obj, "isclose", 3);
    } else {
      double atol_tmp = CastPyArg2Double(atol_obj, "isclose", 3);
      atol = paddle::dialect::full(std::vector<int64_t>{1}, atol_tmp,
                                   phi::DataType::FLOAT64, phi::CPUPlace());
    }
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "isclose", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("isclose");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::isclose(x, y, rtol, atol, equal_nan);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_isfinite(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add isfinite op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "isfinite", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("isfinite");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::isfinite(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_isinf(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add isinf op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "isinf", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("isinf");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::isinf(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_isnan(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add isnan op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "isnan", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("isnan");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::isnan(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_kldiv_loss(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add kldiv_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "kldiv_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "kldiv_loss", 1);

    // Parse Attributes
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 2);
    std::string reduction = CastPyArg2String(reduction_obj, "kldiv_loss", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("kldiv_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::kldiv_loss(x, label, reduction);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_kron(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add kron op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "kron", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "kron", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("kron");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::kron(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_kthvalue(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add kthvalue op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "kthvalue", 0);

    // Parse Attributes
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    int k = CastPyArg2Int(k_obj, "kthvalue", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "kthvalue", 2);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 3);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "kthvalue", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("kthvalue");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::kthvalue(x, k, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_label_smooth(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add label_smooth op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *label_obj = PyTuple_GET_ITEM(args, 0);
    auto label = CastPyArg2Value(label_obj, "label_smooth", 0);
    PyObject *prior_dist_obj = PyTuple_GET_ITEM(args, 1);
    auto prior_dist =
        CastPyArg2OptionalValue(prior_dist_obj, "label_smooth", 1);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "label_smooth", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("label_smooth");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::label_smooth(label, prior_dist, epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lamb_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lamb_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "lamb_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "lamb_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "lamb_", 2);
    PyObject *moment1_obj = PyTuple_GET_ITEM(args, 3);
    auto moment1 = CastPyArg2Value(moment1_obj, "lamb_", 3);
    PyObject *moment2_obj = PyTuple_GET_ITEM(args, 4);
    auto moment2 = CastPyArg2Value(moment2_obj, "lamb_", 4);
    PyObject *beta1_pow_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pow = CastPyArg2Value(beta1_pow_obj, "lamb_", 5);
    PyObject *beta2_pow_obj = PyTuple_GET_ITEM(args, 6);
    auto beta2_pow = CastPyArg2Value(beta2_pow_obj, "lamb_", 6);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 7);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "lamb_", 7);
    PyObject *skip_update_obj = PyTuple_GET_ITEM(args, 8);
    auto skip_update = CastPyArg2OptionalValue(skip_update_obj, "lamb_", 8);

    // Parse Attributes
    PyObject *weight_decay_obj = PyTuple_GET_ITEM(args, 9);
    float weight_decay = CastPyArg2Float(weight_decay_obj, "lamb_", 9);
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 10);
    float beta1 = CastPyArg2Float(beta1_obj, "lamb_", 10);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 11);
    float beta2 = CastPyArg2Float(beta2_obj, "lamb_", 11);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 12);
    float epsilon = CastPyArg2Float(epsilon_obj, "lamb_", 12);
    PyObject *always_adapt_obj = PyTuple_GET_ITEM(args, 13);
    bool always_adapt = CastPyArg2Boolean(always_adapt_obj, "lamb_", 13);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 14);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "lamb_", 14);

    // Call ir static api
    CallStackRecorder callstack_recorder("lamb_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lamb_(
        param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow,
        master_param, skip_update, weight_decay, beta1, beta2, epsilon,
        always_adapt, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_layer_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add layer_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "layer_norm", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2OptionalValue(scale_obj, "layer_norm", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "layer_norm", 2);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "layer_norm", 3);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 4);
    int begin_norm_axis = CastPyArg2Int(begin_norm_axis_obj, "layer_norm", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("layer_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::layer_norm(x, scale, bias, epsilon, begin_norm_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_leaky_relu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add leaky_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "leaky_relu", 0);

    // Parse Attributes
    PyObject *negative_slope_obj = PyTuple_GET_ITEM(args, 1);
    float negative_slope = CastPyArg2Float(negative_slope_obj, "leaky_relu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("leaky_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::leaky_relu(x, negative_slope);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_leaky_relu_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add leaky_relu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "leaky_relu_", 0);

    // Parse Attributes
    PyObject *negative_slope_obj = PyTuple_GET_ITEM(args, 1);
    float negative_slope =
        CastPyArg2Float(negative_slope_obj, "leaky_relu_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("leaky_relu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::leaky_relu_(x, negative_slope);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lerp(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lerp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lerp", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "lerp", 1);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 2);
    auto weight = CastPyArg2Value(weight_obj, "lerp", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("lerp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lerp(x, y, weight);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lerp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lerp_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lerp_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "lerp_", 1);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 2);
    auto weight = CastPyArg2Value(weight_obj, "lerp_", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("lerp_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lerp_(x, y, weight);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lgamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lgamma op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lgamma", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("lgamma");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lgamma(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lgamma_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lgamma_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lgamma_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("lgamma_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lgamma_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_linear_interp(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add linear_interp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "linear_interp", 0);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 1);
    auto out_size = CastPyArg2OptionalValue(out_size_obj, "linear_interp", 1);
    PyObject *size_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto size_tensor =
        CastPyArg2OptionalVectorOfValue(size_tensor_obj, "linear_interp", 2);
    PyObject *scale_tensor_obj = PyTuple_GET_ITEM(args, 3);
    auto scale_tensor =
        CastPyArg2OptionalValue(scale_tensor_obj, "linear_interp", 3);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "linear_interp", 4);
    PyObject *out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "linear_interp", 5);
    PyObject *out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "linear_interp", 6);
    PyObject *out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "linear_interp", 7);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "linear_interp", 8);
    PyObject *interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method =
        CastPyArg2String(interp_method_obj, "linear_interp", 9);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners =
        CastPyArg2Boolean(align_corners_obj, "linear_interp", 10);
    PyObject *align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "linear_interp", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("linear_interp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::linear_interp(
        x, out_size, size_tensor, scale_tensor, data_format, out_d, out_h,
        out_w, scale, interp_method, align_corners, align_mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_llm_int8_linear(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add llm_int8_linear op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "llm_int8_linear", 0);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 1);
    auto weight = CastPyArg2Value(weight_obj, "llm_int8_linear", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "llm_int8_linear", 2);
    PyObject *weight_scale_obj = PyTuple_GET_ITEM(args, 3);
    auto weight_scale = CastPyArg2Value(weight_scale_obj, "llm_int8_linear", 3);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 4);
    float threshold = CastPyArg2Float(threshold_obj, "llm_int8_linear", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("llm_int8_linear");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::llm_int8_linear(
        x, weight, bias, weight_scale, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log10(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log10 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log10", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log10");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log10(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log10_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log10_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log10_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log10_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log10_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log1p(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log1p op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log1p", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log1p");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log1p(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log1p_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log1p_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log1p_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log1p_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log1p_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log2(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log2", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log2");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log2(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log2_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add log2_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log2_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("log2_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log2_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log_loss(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add log_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "log_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "log_loss", 1);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "log_loss", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("log_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log_loss(input, label, epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_log_softmax(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add log_softmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "log_softmax", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "log_softmax", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("log_softmax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::log_softmax(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logcumsumexp(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add logcumsumexp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logcumsumexp", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "logcumsumexp", 1);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "logcumsumexp", 2);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 3);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "logcumsumexp", 3);
    PyObject *reverse_obj = PyTuple_GET_ITEM(args, 4);
    bool reverse = CastPyArg2Boolean(reverse_obj, "logcumsumexp", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("logcumsumexp");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::logcumsumexp(x, axis, flatten, exclusive, reverse);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_and(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_and op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_and", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_and", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_and");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_and(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_and_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_and_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_and_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_and_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_and_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_and_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_not(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_not op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_not", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_not");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_not(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_not_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_not_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_not_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_not_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_not_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_or(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_or op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_or", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_or", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_or");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_or(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_or_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_or_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_or_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_or_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_or_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_or_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_xor(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_xor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_xor", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_xor", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_xor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_xor(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logical_xor_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add logical_xor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logical_xor_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "logical_xor_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logical_xor_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logical_xor_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logit(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add logit op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logit", 0);

    // Parse Attributes
    PyObject *eps_obj = PyTuple_GET_ITEM(args, 1);
    float eps = CastPyArg2Float(eps_obj, "logit", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("logit");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logit(x, eps);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logit_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add logit_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logit_", 0);

    // Parse Attributes
    PyObject *eps_obj = PyTuple_GET_ITEM(args, 1);
    float eps = CastPyArg2Float(eps_obj, "logit_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("logit_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logit_(x, eps);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logsigmoid(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add logsigmoid op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logsigmoid", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("logsigmoid");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::logsigmoid(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lstsq(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lstsq op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lstsq", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "lstsq", 1);

    // Parse Attributes
    PyObject *rcond_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *driver_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value rcond;

    if (PyObject_CheckIRValue(rcond_obj)) {
      rcond = CastPyArg2Value(rcond_obj, "lstsq", 2);
    } else {
      float rcond_tmp = CastPyArg2Float(rcond_obj, "lstsq", 2);
      rcond = paddle::dialect::full(std::vector<int64_t>{1}, rcond_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    std::string driver = CastPyArg2String(driver_obj, "lstsq", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("lstsq");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lstsq(x, y, rcond, driver);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lu", 0);

    // Parse Attributes
    PyObject *pivot_obj = PyTuple_GET_ITEM(args, 1);
    bool pivot = CastPyArg2Boolean(pivot_obj, "lu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("lu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lu(x, pivot);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lu_", 0);

    // Parse Attributes
    PyObject *pivot_obj = PyTuple_GET_ITEM(args, 1);
    bool pivot = CastPyArg2Boolean(pivot_obj, "lu_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("lu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lu_(x, pivot);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lu_unpack(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add lu_unpack op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lu_unpack", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "lu_unpack", 1);

    // Parse Attributes
    PyObject *unpack_ludata_obj = PyTuple_GET_ITEM(args, 2);
    bool unpack_ludata = CastPyArg2Boolean(unpack_ludata_obj, "lu_unpack", 2);
    PyObject *unpack_pivots_obj = PyTuple_GET_ITEM(args, 3);
    bool unpack_pivots = CastPyArg2Boolean(unpack_pivots_obj, "lu_unpack", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("lu_unpack");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::lu_unpack(x, y, unpack_ludata, unpack_pivots);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_margin_cross_entropy(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add margin_cross_entropy op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *logits_obj = PyTuple_GET_ITEM(args, 0);
    auto logits = CastPyArg2Value(logits_obj, "margin_cross_entropy", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "margin_cross_entropy", 1);

    // Parse Attributes
    PyObject *return_softmax_obj = PyTuple_GET_ITEM(args, 2);
    bool return_softmax =
        CastPyArg2Boolean(return_softmax_obj, "margin_cross_entropy", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "margin_cross_entropy", 3);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 4);
    int rank = CastPyArg2Int(rank_obj, "margin_cross_entropy", 4);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 5);
    int nranks = CastPyArg2Int(nranks_obj, "margin_cross_entropy", 5);
    PyObject *margin1_obj = PyTuple_GET_ITEM(args, 6);
    float margin1 = CastPyArg2Float(margin1_obj, "margin_cross_entropy", 6);
    PyObject *margin2_obj = PyTuple_GET_ITEM(args, 7);
    float margin2 = CastPyArg2Float(margin2_obj, "margin_cross_entropy", 7);
    PyObject *margin3_obj = PyTuple_GET_ITEM(args, 8);
    float margin3 = CastPyArg2Float(margin3_obj, "margin_cross_entropy", 8);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 9);
    float scale = CastPyArg2Float(scale_obj, "margin_cross_entropy", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("margin_cross_entropy");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::margin_cross_entropy(
        logits, label, return_softmax, ring_id, rank, nranks, margin1, margin2,
        margin3, scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_masked_multihead_attention_(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add masked_multihead_attention_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "masked_multihead_attention_", 0);
    PyObject *cache_kv_obj = PyTuple_GET_ITEM(args, 1);
    auto cache_kv =
        CastPyArg2Value(cache_kv_obj, "masked_multihead_attention_", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias =
        CastPyArg2OptionalValue(bias_obj, "masked_multihead_attention_", 2);
    PyObject *src_mask_obj = PyTuple_GET_ITEM(args, 3);
    auto src_mask =
        CastPyArg2OptionalValue(src_mask_obj, "masked_multihead_attention_", 3);
    PyObject *cum_offsets_obj = PyTuple_GET_ITEM(args, 4);
    auto cum_offsets = CastPyArg2OptionalValue(
        cum_offsets_obj, "masked_multihead_attention_", 4);
    PyObject *sequence_lengths_obj = PyTuple_GET_ITEM(args, 5);
    auto sequence_lengths = CastPyArg2OptionalValue(
        sequence_lengths_obj, "masked_multihead_attention_", 5);
    PyObject *rotary_tensor_obj = PyTuple_GET_ITEM(args, 6);
    auto rotary_tensor = CastPyArg2OptionalValue(
        rotary_tensor_obj, "masked_multihead_attention_", 6);
    PyObject *beam_cache_offset_obj = PyTuple_GET_ITEM(args, 7);
    auto beam_cache_offset = CastPyArg2OptionalValue(
        beam_cache_offset_obj, "masked_multihead_attention_", 7);
    PyObject *qkv_out_scale_obj = PyTuple_GET_ITEM(args, 8);
    auto qkv_out_scale = CastPyArg2OptionalValue(
        qkv_out_scale_obj, "masked_multihead_attention_", 8);
    PyObject *out_shift_obj = PyTuple_GET_ITEM(args, 9);
    auto out_shift = CastPyArg2OptionalValue(out_shift_obj,
                                             "masked_multihead_attention_", 9);
    PyObject *out_smooth_obj = PyTuple_GET_ITEM(args, 10);
    auto out_smooth = CastPyArg2OptionalValue(
        out_smooth_obj, "masked_multihead_attention_", 10);

    // Parse Attributes
    PyObject *seq_len_obj = PyTuple_GET_ITEM(args, 11);
    int seq_len = CastPyArg2Int(seq_len_obj, "masked_multihead_attention_", 11);
    PyObject *rotary_emb_dims_obj = PyTuple_GET_ITEM(args, 12);
    int rotary_emb_dims =
        CastPyArg2Int(rotary_emb_dims_obj, "masked_multihead_attention_", 12);
    PyObject *use_neox_rotary_style_obj = PyTuple_GET_ITEM(args, 13);
    bool use_neox_rotary_style = CastPyArg2Boolean(
        use_neox_rotary_style_obj, "masked_multihead_attention_", 13);
    PyObject *compute_dtype_obj = PyTuple_GET_ITEM(args, 14);
    std::string compute_dtype =
        CastPyArg2String(compute_dtype_obj, "masked_multihead_attention_", 14);
    PyObject *out_scale_obj = PyTuple_GET_ITEM(args, 15);
    float out_scale =
        CastPyArg2Float(out_scale_obj, "masked_multihead_attention_", 15);
    PyObject *quant_round_type_obj = PyTuple_GET_ITEM(args, 16);
    int quant_round_type =
        CastPyArg2Int(quant_round_type_obj, "masked_multihead_attention_", 16);
    PyObject *quant_max_bound_obj = PyTuple_GET_ITEM(args, 17);
    float quant_max_bound =
        CastPyArg2Float(quant_max_bound_obj, "masked_multihead_attention_", 17);
    PyObject *quant_min_bound_obj = PyTuple_GET_ITEM(args, 18);
    float quant_min_bound =
        CastPyArg2Float(quant_min_bound_obj, "masked_multihead_attention_", 18);

    // Call ir static api
    CallStackRecorder callstack_recorder("masked_multihead_attention_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::masked_multihead_attention_(
        x, cache_kv, bias, src_mask, cum_offsets, sequence_lengths,
        rotary_tensor, beam_cache_offset, qkv_out_scale, out_shift, out_smooth,
        seq_len, rotary_emb_dims, use_neox_rotary_style, compute_dtype,
        out_scale, quant_round_type, quant_max_bound, quant_min_bound);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_masked_select(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add masked_select op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "masked_select", 0);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 1);
    auto mask = CastPyArg2Value(mask_obj, "masked_select", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("masked_select");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::masked_select(x, mask);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matrix_nms(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add matrix_nms op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *bboxes_obj = PyTuple_GET_ITEM(args, 0);
    auto bboxes = CastPyArg2Value(bboxes_obj, "matrix_nms", 0);
    PyObject *scores_obj = PyTuple_GET_ITEM(args, 1);
    auto scores = CastPyArg2Value(scores_obj, "matrix_nms", 1);

    // Parse Attributes
    PyObject *score_threshold_obj = PyTuple_GET_ITEM(args, 2);
    float score_threshold =
        CastPyArg2Float(score_threshold_obj, "matrix_nms", 2);
    PyObject *nms_top_k_obj = PyTuple_GET_ITEM(args, 3);
    int nms_top_k = CastPyArg2Int(nms_top_k_obj, "matrix_nms", 3);
    PyObject *keep_top_k_obj = PyTuple_GET_ITEM(args, 4);
    int keep_top_k = CastPyArg2Int(keep_top_k_obj, "matrix_nms", 4);
    PyObject *post_threshold_obj = PyTuple_GET_ITEM(args, 5);
    float post_threshold = CastPyArg2Float(post_threshold_obj, "matrix_nms", 5);
    PyObject *use_gaussian_obj = PyTuple_GET_ITEM(args, 6);
    bool use_gaussian = CastPyArg2Boolean(use_gaussian_obj, "matrix_nms", 6);
    PyObject *gaussian_sigma_obj = PyTuple_GET_ITEM(args, 7);
    float gaussian_sigma = CastPyArg2Float(gaussian_sigma_obj, "matrix_nms", 7);
    PyObject *background_label_obj = PyTuple_GET_ITEM(args, 8);
    int background_label = CastPyArg2Int(background_label_obj, "matrix_nms", 8);
    PyObject *normalized_obj = PyTuple_GET_ITEM(args, 9);
    bool normalized = CastPyArg2Boolean(normalized_obj, "matrix_nms", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("matrix_nms");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::matrix_nms(
        bboxes, scores, score_threshold, nms_top_k, keep_top_k, post_threshold,
        use_gaussian, gaussian_sigma, background_label, normalized);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matrix_power(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add matrix_power op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "matrix_power", 0);

    // Parse Attributes
    PyObject *n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "matrix_power", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("matrix_power");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::matrix_power(x, n);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_max_pool2d_with_index(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add max_pool2d_with_index op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "max_pool2d_with_index", 0);

    // Parse Attributes
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "max_pool2d_with_index", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "max_pool2d_with_index", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "max_pool2d_with_index", 3);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "max_pool2d_with_index", 4);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool2d_with_index", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("max_pool2d_with_index");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::max_pool2d_with_index(
        x, kernel_size, strides, paddings, global_pooling, adaptive);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_max_pool3d_with_index(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add max_pool3d_with_index op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "max_pool3d_with_index", 0);

    // Parse Attributes
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "max_pool3d_with_index", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "max_pool3d_with_index", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "max_pool3d_with_index", 3);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "max_pool3d_with_index", 4);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool3d_with_index", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("max_pool3d_with_index");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::max_pool3d_with_index(
        x, kernel_size, strides, paddings, global_pooling, adaptive);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_maxout(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add maxout op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "maxout", 0);

    // Parse Attributes
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 1);
    int groups = CastPyArg2Int(groups_obj, "maxout", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "maxout", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("maxout");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::maxout(x, groups, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_mean_all(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add mean_all op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "mean_all", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("mean_all");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::mean_all(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_memory_efficient_attention(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add memory_efficient_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *query_obj = PyTuple_GET_ITEM(args, 0);
    auto query = CastPyArg2Value(query_obj, "memory_efficient_attention", 0);
    PyObject *key_obj = PyTuple_GET_ITEM(args, 1);
    auto key = CastPyArg2Value(key_obj, "memory_efficient_attention", 1);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 2);
    auto value = CastPyArg2Value(value_obj, "memory_efficient_attention", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias =
        CastPyArg2OptionalValue(bias_obj, "memory_efficient_attention", 3);
    PyObject *cu_seqlens_q_obj = PyTuple_GET_ITEM(args, 4);
    auto cu_seqlens_q = CastPyArg2OptionalValue(
        cu_seqlens_q_obj, "memory_efficient_attention", 4);
    PyObject *cu_seqlens_k_obj = PyTuple_GET_ITEM(args, 5);
    auto cu_seqlens_k = CastPyArg2OptionalValue(
        cu_seqlens_k_obj, "memory_efficient_attention", 5);
    PyObject *causal_diagonal_obj = PyTuple_GET_ITEM(args, 6);
    auto causal_diagonal = CastPyArg2OptionalValue(
        causal_diagonal_obj, "memory_efficient_attention", 6);
    PyObject *seqlen_k_obj = PyTuple_GET_ITEM(args, 7);
    auto seqlen_k =
        CastPyArg2OptionalValue(seqlen_k_obj, "memory_efficient_attention", 7);

    // Parse Attributes
    PyObject *max_seqlen_q_obj = PyTuple_GET_ITEM(args, 8);
    float max_seqlen_q =
        CastPyArg2Float(max_seqlen_q_obj, "memory_efficient_attention", 8);
    PyObject *max_seqlen_k_obj = PyTuple_GET_ITEM(args, 9);
    float max_seqlen_k =
        CastPyArg2Float(max_seqlen_k_obj, "memory_efficient_attention", 9);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 10);
    bool causal =
        CastPyArg2Boolean(causal_obj, "memory_efficient_attention", 10);
    PyObject *dropout_p_obj = PyTuple_GET_ITEM(args, 11);
    double dropout_p =
        CastPyArg2Double(dropout_p_obj, "memory_efficient_attention", 11);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 12);
    float scale = CastPyArg2Float(scale_obj, "memory_efficient_attention", 12);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 13);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "memory_efficient_attention", 13);

    // Call ir static api
    CallStackRecorder callstack_recorder("memory_efficient_attention");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::memory_efficient_attention(
        query, key, value, bias, cu_seqlens_q, cu_seqlens_k, causal_diagonal,
        seqlen_k, max_seqlen_q, max_seqlen_k, causal, dropout_p, scale,
        is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_merge_selected_rows(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add merge_selected_rows op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "merge_selected_rows", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("merge_selected_rows");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::merge_selected_rows(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_merged_adam_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add merged_adam_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2VectorOfValue(param_obj, "merged_adam_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2VectorOfValue(grad_obj, "merged_adam_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate =
        CastPyArg2VectorOfValue(learning_rate_obj, "merged_adam_", 2);
    PyObject *moment1_obj = PyTuple_GET_ITEM(args, 3);
    auto moment1 = CastPyArg2VectorOfValue(moment1_obj, "merged_adam_", 3);
    PyObject *moment2_obj = PyTuple_GET_ITEM(args, 4);
    auto moment2 = CastPyArg2VectorOfValue(moment2_obj, "merged_adam_", 4);
    PyObject *beta1_pow_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pow = CastPyArg2VectorOfValue(beta1_pow_obj, "merged_adam_", 5);
    PyObject *beta2_pow_obj = PyTuple_GET_ITEM(args, 6);
    auto beta2_pow = CastPyArg2VectorOfValue(beta2_pow_obj, "merged_adam_", 6);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 7);
    auto master_param =
        CastPyArg2OptionalVectorOfValue(master_param_obj, "merged_adam_", 7);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 10);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 11);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 12);

    // Check for mutable attrs
    pir::Value beta1;

    pir::Value beta2;

    pir::Value epsilon;

    if (PyObject_CheckIRValue(beta1_obj)) {
      beta1 = CastPyArg2Value(beta1_obj, "merged_adam_", 8);
    } else {
      float beta1_tmp = CastPyArg2Float(beta1_obj, "merged_adam_", 8);
      beta1 = paddle::dialect::full(std::vector<int64_t>{1}, beta1_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(beta2_obj)) {
      beta2 = CastPyArg2Value(beta2_obj, "merged_adam_", 9);
    } else {
      float beta2_tmp = CastPyArg2Float(beta2_obj, "merged_adam_", 9);
      beta2 = paddle::dialect::full(std::vector<int64_t>{1}, beta2_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(epsilon_obj)) {
      epsilon = CastPyArg2Value(epsilon_obj, "merged_adam_", 10);
    } else {
      float epsilon_tmp = CastPyArg2Float(epsilon_obj, "merged_adam_", 10);
      epsilon = paddle::dialect::full(std::vector<int64_t>{1}, epsilon_tmp,
                                      phi::DataType::FLOAT32, phi::CPUPlace());
    }
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "merged_adam_", 11);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "merged_adam_", 12);

    // Call ir static api
    CallStackRecorder callstack_recorder("merged_adam_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::merged_adam_(
        param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow,
        master_param, beta1, beta2, epsilon, multi_precision,
        use_global_beta_pow);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_merged_momentum_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add merged_momentum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2VectorOfValue(param_obj, "merged_momentum_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2VectorOfValue(grad_obj, "merged_momentum_", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity =
        CastPyArg2VectorOfValue(velocity_obj, "merged_momentum_", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate =
        CastPyArg2VectorOfValue(learning_rate_obj, "merged_momentum_", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param = CastPyArg2OptionalVectorOfValue(master_param_obj,
                                                        "merged_momentum_", 4);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "merged_momentum_", 5);
    PyObject *use_nesterov_obj = PyTuple_GET_ITEM(args, 6);
    bool use_nesterov =
        CastPyArg2Boolean(use_nesterov_obj, "merged_momentum_", 6);
    PyObject *regularization_method_obj = PyTuple_GET_ITEM(args, 7);
    std::vector<std::string> regularization_method =
        CastPyArg2Strings(regularization_method_obj, "merged_momentum_", 7);
    PyObject *regularization_coeff_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> regularization_coeff =
        CastPyArg2Floats(regularization_coeff_obj, "merged_momentum_", 8);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "merged_momentum_", 9);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad =
        CastPyArg2Float(rescale_grad_obj, "merged_momentum_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("merged_momentum_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::merged_momentum_(
        param, grad, velocity, learning_rate, master_param, mu, use_nesterov,
        regularization_method, regularization_coeff, multi_precision,
        rescale_grad);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_meshgrid(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add meshgrid op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *inputs_obj = PyTuple_GET_ITEM(args, 0);
    auto inputs = CastPyArg2VectorOfValue(inputs_obj, "meshgrid", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("meshgrid");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::meshgrid(inputs);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_mode(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add mode op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "mode", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "mode", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "mode", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("mode");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::mode(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_momentum_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add momentum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "momentum_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "momentum_", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity = CastPyArg2Value(velocity_obj, "momentum_", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "momentum_", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "momentum_", 4);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "momentum_", 5);
    PyObject *use_nesterov_obj = PyTuple_GET_ITEM(args, 6);
    bool use_nesterov = CastPyArg2Boolean(use_nesterov_obj, "momentum_", 6);
    PyObject *regularization_method_obj = PyTuple_GET_ITEM(args, 7);
    std::string regularization_method =
        CastPyArg2String(regularization_method_obj, "momentum_", 7);
    PyObject *regularization_coeff_obj = PyTuple_GET_ITEM(args, 8);
    float regularization_coeff =
        CastPyArg2Float(regularization_coeff_obj, "momentum_", 8);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "momentum_", 9);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad = CastPyArg2Float(rescale_grad_obj, "momentum_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("momentum_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::momentum_(
        param, grad, velocity, learning_rate, master_param, mu, use_nesterov,
        regularization_method, regularization_coeff, multi_precision,
        rescale_grad);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multi_dot(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add multi_dot op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "multi_dot", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("multi_dot");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multi_dot(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multiclass_nms3(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add multiclass_nms3 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *bboxes_obj = PyTuple_GET_ITEM(args, 0);
    auto bboxes = CastPyArg2Value(bboxes_obj, "multiclass_nms3", 0);
    PyObject *scores_obj = PyTuple_GET_ITEM(args, 1);
    auto scores = CastPyArg2Value(scores_obj, "multiclass_nms3", 1);
    PyObject *rois_num_obj = PyTuple_GET_ITEM(args, 2);
    auto rois_num = CastPyArg2OptionalValue(rois_num_obj, "multiclass_nms3", 2);

    // Parse Attributes
    PyObject *score_threshold_obj = PyTuple_GET_ITEM(args, 3);
    float score_threshold =
        CastPyArg2Float(score_threshold_obj, "multiclass_nms3", 3);
    PyObject *nms_top_k_obj = PyTuple_GET_ITEM(args, 4);
    int nms_top_k = CastPyArg2Int(nms_top_k_obj, "multiclass_nms3", 4);
    PyObject *keep_top_k_obj = PyTuple_GET_ITEM(args, 5);
    int keep_top_k = CastPyArg2Int(keep_top_k_obj, "multiclass_nms3", 5);
    PyObject *nms_threshold_obj = PyTuple_GET_ITEM(args, 6);
    float nms_threshold =
        CastPyArg2Float(nms_threshold_obj, "multiclass_nms3", 6);
    PyObject *normalized_obj = PyTuple_GET_ITEM(args, 7);
    bool normalized = CastPyArg2Boolean(normalized_obj, "multiclass_nms3", 7);
    PyObject *nms_eta_obj = PyTuple_GET_ITEM(args, 8);
    float nms_eta = CastPyArg2Float(nms_eta_obj, "multiclass_nms3", 8);
    PyObject *background_label_obj = PyTuple_GET_ITEM(args, 9);
    int background_label =
        CastPyArg2Int(background_label_obj, "multiclass_nms3", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("multiclass_nms3");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multiclass_nms3(
        bboxes, scores, rois_num, score_threshold, nms_top_k, keep_top_k,
        nms_threshold, normalized, nms_eta, background_label);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multinomial(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add multinomial op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "multinomial", 0);

    // Parse Attributes
    PyObject *num_samples_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *replacement_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value num_samples;

    if (PyObject_CheckIRValue(num_samples_obj)) {
      num_samples = CastPyArg2Value(num_samples_obj, "multinomial", 1);
    } else {
      int num_samples_tmp = CastPyArg2Int(num_samples_obj, "multinomial", 1);
      num_samples =
          paddle::dialect::full(std::vector<int64_t>{1}, num_samples_tmp,
                                phi::DataType::INT32, phi::CPUPlace());
    }
    bool replacement = CastPyArg2Boolean(replacement_obj, "multinomial", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("multinomial");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::multinomial(x, num_samples, replacement);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multiplex(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add multiplex op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *inputs_obj = PyTuple_GET_ITEM(args, 0);
    auto inputs = CastPyArg2VectorOfValue(inputs_obj, "multiplex", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "multiplex", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("multiplex");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multiplex(inputs, index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_mv(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add mv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "mv", 0);
    PyObject *vec_obj = PyTuple_GET_ITEM(args, 1);
    auto vec = CastPyArg2Value(vec_obj, "mv", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("mv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::mv(x, vec);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nanmedian(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add nanmedian op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nanmedian", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "nanmedian", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "nanmedian", 2);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string mode = CastPyArg2String(mode_obj, "nanmedian", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("nanmedian");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nanmedian(x, axis, keepdim, mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nearest_interp(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add nearest_interp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nearest_interp", 0);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 1);
    auto out_size = CastPyArg2OptionalValue(out_size_obj, "nearest_interp", 1);
    PyObject *size_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto size_tensor =
        CastPyArg2OptionalVectorOfValue(size_tensor_obj, "nearest_interp", 2);
    PyObject *scale_tensor_obj = PyTuple_GET_ITEM(args, 3);
    auto scale_tensor =
        CastPyArg2OptionalValue(scale_tensor_obj, "nearest_interp", 3);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "nearest_interp", 4);
    PyObject *out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "nearest_interp", 5);
    PyObject *out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "nearest_interp", 6);
    PyObject *out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "nearest_interp", 7);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "nearest_interp", 8);
    PyObject *interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method =
        CastPyArg2String(interp_method_obj, "nearest_interp", 9);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners =
        CastPyArg2Boolean(align_corners_obj, "nearest_interp", 10);
    PyObject *align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "nearest_interp", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("nearest_interp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nearest_interp(
        x, out_size, size_tensor, scale_tensor, data_format, out_d, out_h,
        out_w, scale, interp_method, align_corners, align_mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nextafter(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add nextafter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nextafter", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "nextafter", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("nextafter");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nextafter(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nll_loss(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add nll_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "nll_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "nll_loss", 1);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 2);
    auto weight = CastPyArg2OptionalValue(weight_obj, "nll_loss", 2);

    // Parse Attributes
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 3);
    int64_t ignore_index = CastPyArg2Long(ignore_index_obj, "nll_loss", 3);
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduction = CastPyArg2String(reduction_obj, "nll_loss", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("nll_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nll_loss(input, label, weight,
                                                    ignore_index, reduction);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nms(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add nms op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nms", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "nms", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("nms");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nms(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nonzero(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add nonzero op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *condition_obj = PyTuple_GET_ITEM(args, 0);
    auto condition = CastPyArg2Value(condition_obj, "nonzero", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("nonzero");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nonzero(condition);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_npu_identity(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add npu_identity op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "npu_identity", 0);

    // Parse Attributes
    PyObject *format_obj = PyTuple_GET_ITEM(args, 1);
    int format = CastPyArg2Int(format_obj, "npu_identity", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("npu_identity");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::npu_identity(x, format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_numel(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add numel op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "numel", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("numel");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::numel(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_overlap_add(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add overlap_add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "overlap_add", 0);

    // Parse Attributes
    PyObject *hop_length_obj = PyTuple_GET_ITEM(args, 1);
    int hop_length = CastPyArg2Int(hop_length_obj, "overlap_add", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "overlap_add", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("overlap_add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::overlap_add(x, hop_length, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_p_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add p_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "p_norm", 0);

    // Parse Attributes
    PyObject *porder_obj = PyTuple_GET_ITEM(args, 1);
    float porder = CastPyArg2Float(porder_obj, "p_norm", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "p_norm", 2);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "p_norm", 3);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 4);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "p_norm", 4);
    PyObject *asvector_obj = PyTuple_GET_ITEM(args, 5);
    bool asvector = CastPyArg2Boolean(asvector_obj, "p_norm", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("p_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::p_norm(x, porder, axis, epsilon, keepdim, asvector);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pad3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pad3d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pad3d", 0);

    // Parse Attributes
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *pad_value_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value paddings;

    if (PyObject_CheckIRValue(paddings_obj)) {
      paddings = CastPyArg2Value(paddings_obj, "pad3d", 1);
    } else if (PyObject_CheckIRVectorOfValue(paddings_obj)) {
      std::vector<pir::Value> paddings_tmp =
          CastPyArg2VectorOfValue(paddings_obj, "pad3d", 1);
      paddings = paddle::dialect::stack(paddings_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> paddings_tmp =
          CastPyArg2Longs(paddings_obj, "pad3d", 1);
      paddings = paddle::dialect::full_int_array(
          paddings_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::string mode = CastPyArg2String(mode_obj, "pad3d", 2);
    float pad_value = CastPyArg2Float(pad_value_obj, "pad3d", 3);
    std::string data_format = CastPyArg2String(data_format_obj, "pad3d", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("pad3d");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::pad3d(x, paddings, mode, pad_value, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pixel_shuffle(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add pixel_shuffle op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pixel_shuffle", 0);

    // Parse Attributes
    PyObject *upscale_factor_obj = PyTuple_GET_ITEM(args, 1);
    int upscale_factor = CastPyArg2Int(upscale_factor_obj, "pixel_shuffle", 1);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format =
        CastPyArg2String(data_format_obj, "pixel_shuffle", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("pixel_shuffle");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::pixel_shuffle(x, upscale_factor, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pixel_unshuffle(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add pixel_unshuffle op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pixel_unshuffle", 0);

    // Parse Attributes
    PyObject *downscale_factor_obj = PyTuple_GET_ITEM(args, 1);
    int downscale_factor =
        CastPyArg2Int(downscale_factor_obj, "pixel_unshuffle", 1);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format =
        CastPyArg2String(data_format_obj, "pixel_unshuffle", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("pixel_unshuffle");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::pixel_unshuffle(x, downscale_factor, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_poisson(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add poisson op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "poisson", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("poisson");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::poisson(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_polygamma(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add polygamma op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "polygamma", 0);

    // Parse Attributes
    PyObject *n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "polygamma", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("polygamma");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::polygamma(x, n);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_polygamma_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add polygamma_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "polygamma_", 0);

    // Parse Attributes
    PyObject *n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "polygamma_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("polygamma_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::polygamma_(x, n);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pow op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pow", 0);

    // Parse Attributes
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    float y = CastPyArg2Float(y_obj, "pow", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("pow");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::pow(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pow_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pow_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pow_", 0);

    // Parse Attributes
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    float y = CastPyArg2Float(y_obj, "pow_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("pow_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::pow_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_prelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add prelu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "prelu", 0);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    auto alpha = CastPyArg2Value(alpha_obj, "prelu", 1);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format = CastPyArg2String(data_format_obj, "prelu", 2);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string mode = CastPyArg2String(mode_obj, "prelu", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("prelu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::prelu(x, alpha, data_format, mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_prior_box(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add prior_box op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "prior_box", 0);
    PyObject *image_obj = PyTuple_GET_ITEM(args, 1);
    auto image = CastPyArg2Value(image_obj, "prior_box", 1);

    // Parse Attributes
    PyObject *min_sizes_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<float> min_sizes =
        CastPyArg2Floats(min_sizes_obj, "prior_box", 2);
    PyObject *max_sizes_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<float> max_sizes =
        CastPyArg2Floats(max_sizes_obj, "prior_box", 3);
    PyObject *aspect_ratios_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<float> aspect_ratios =
        CastPyArg2Floats(aspect_ratios_obj, "prior_box", 4);
    PyObject *variances_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<float> variances =
        CastPyArg2Floats(variances_obj, "prior_box", 5);
    PyObject *flip_obj = PyTuple_GET_ITEM(args, 6);
    bool flip = CastPyArg2Boolean(flip_obj, "prior_box", 6);
    PyObject *clip_obj = PyTuple_GET_ITEM(args, 7);
    bool clip = CastPyArg2Boolean(clip_obj, "prior_box", 7);
    PyObject *step_w_obj = PyTuple_GET_ITEM(args, 8);
    float step_w = CastPyArg2Float(step_w_obj, "prior_box", 8);
    PyObject *step_h_obj = PyTuple_GET_ITEM(args, 9);
    float step_h = CastPyArg2Float(step_h_obj, "prior_box", 9);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 10);
    float offset = CastPyArg2Float(offset_obj, "prior_box", 10);
    PyObject *min_max_aspect_ratios_order_obj = PyTuple_GET_ITEM(args, 11);
    bool min_max_aspect_ratios_order =
        CastPyArg2Boolean(min_max_aspect_ratios_order_obj, "prior_box", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("prior_box");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::prior_box(
        input, image, min_sizes, max_sizes, aspect_ratios, variances, flip,
        clip, step_w, step_h, offset, min_max_aspect_ratios_order);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_psroi_pool(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add psroi_pool op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "psroi_pool", 0);
    PyObject *boxes_obj = PyTuple_GET_ITEM(args, 1);
    auto boxes = CastPyArg2Value(boxes_obj, "psroi_pool", 1);
    PyObject *boxes_num_obj = PyTuple_GET_ITEM(args, 2);
    auto boxes_num = CastPyArg2OptionalValue(boxes_num_obj, "psroi_pool", 2);

    // Parse Attributes
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "psroi_pool", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "psroi_pool", 4);
    PyObject *output_channels_obj = PyTuple_GET_ITEM(args, 5);
    int output_channels = CastPyArg2Int(output_channels_obj, "psroi_pool", 5);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 6);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "psroi_pool", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("psroi_pool");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::psroi_pool(
        x, boxes, boxes_num, pooled_height, pooled_width, output_channels,
        spatial_scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_put_along_axis(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add put_along_axis op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *arr_obj = PyTuple_GET_ITEM(args, 0);
    auto arr = CastPyArg2Value(arr_obj, "put_along_axis", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "put_along_axis", 1);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 2);
    auto values = CastPyArg2Value(values_obj, "put_along_axis", 2);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "put_along_axis", 3);
    PyObject *reduce_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduce = CastPyArg2String(reduce_obj, "put_along_axis", 4);
    PyObject *include_self_obj = PyTuple_GET_ITEM(args, 5);
    bool include_self =
        CastPyArg2Boolean(include_self_obj, "put_along_axis", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("put_along_axis");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::put_along_axis(
        arr, indices, values, axis, reduce, include_self);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_put_along_axis_(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add put_along_axis_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *arr_obj = PyTuple_GET_ITEM(args, 0);
    auto arr = CastPyArg2Value(arr_obj, "put_along_axis_", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "put_along_axis_", 1);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 2);
    auto values = CastPyArg2Value(values_obj, "put_along_axis_", 2);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "put_along_axis_", 3);
    PyObject *reduce_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduce = CastPyArg2String(reduce_obj, "put_along_axis_", 4);
    PyObject *include_self_obj = PyTuple_GET_ITEM(args, 5);
    bool include_self =
        CastPyArg2Boolean(include_self_obj, "put_along_axis_", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("put_along_axis_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::put_along_axis_(
        arr, indices, values, axis, reduce, include_self);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_qr(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add qr op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "qr", 0);

    // Parse Attributes
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 1);
    std::string mode = CastPyArg2String(mode_obj, "qr", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("qr");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::qr(x, mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_real(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add real op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "real", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("real");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::real(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reciprocal(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add reciprocal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reciprocal", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("reciprocal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reciprocal(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reciprocal_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add reciprocal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reciprocal_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("reciprocal_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reciprocal_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reindex_graph(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add reindex_graph op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reindex_graph", 0);
    PyObject *neighbors_obj = PyTuple_GET_ITEM(args, 1);
    auto neighbors = CastPyArg2Value(neighbors_obj, "reindex_graph", 1);
    PyObject *count_obj = PyTuple_GET_ITEM(args, 2);
    auto count = CastPyArg2Value(count_obj, "reindex_graph", 2);
    PyObject *hashtable_value_obj = PyTuple_GET_ITEM(args, 3);
    auto hashtable_value =
        CastPyArg2OptionalValue(hashtable_value_obj, "reindex_graph", 3);
    PyObject *hashtable_index_obj = PyTuple_GET_ITEM(args, 4);
    auto hashtable_index =
        CastPyArg2OptionalValue(hashtable_index_obj, "reindex_graph", 4);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("reindex_graph");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reindex_graph(
        x, neighbors, count, hashtable_value, hashtable_index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "relu", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::relu(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_relu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add relu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "relu_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("relu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::relu_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_relu6(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add relu6 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "relu6", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("relu6");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::relu6(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_renorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add renorm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "renorm", 0);

    // Parse Attributes
    PyObject *p_obj = PyTuple_GET_ITEM(args, 1);
    float p = CastPyArg2Float(p_obj, "renorm", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "renorm", 2);
    PyObject *max_norm_obj = PyTuple_GET_ITEM(args, 3);
    float max_norm = CastPyArg2Float(max_norm_obj, "renorm", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("renorm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::renorm(x, p, axis, max_norm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_renorm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add renorm_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "renorm_", 0);

    // Parse Attributes
    PyObject *p_obj = PyTuple_GET_ITEM(args, 1);
    float p = CastPyArg2Float(p_obj, "renorm_", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "renorm_", 2);
    PyObject *max_norm_obj = PyTuple_GET_ITEM(args, 3);
    float max_norm = CastPyArg2Float(max_norm_obj, "renorm_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("renorm_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::renorm_(x, p, axis, max_norm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add reverse op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reverse", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "reverse", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "reverse", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "reverse", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("reverse");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reverse(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rms_norm(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add rms_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rms_norm", 0);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 1);
    auto bias = CastPyArg2OptionalValue(bias_obj, "rms_norm", 1);
    PyObject *residual_obj = PyTuple_GET_ITEM(args, 2);
    auto residual = CastPyArg2OptionalValue(residual_obj, "rms_norm", 2);
    PyObject *norm_weight_obj = PyTuple_GET_ITEM(args, 3);
    auto norm_weight = CastPyArg2Value(norm_weight_obj, "rms_norm", 3);
    PyObject *norm_bias_obj = PyTuple_GET_ITEM(args, 4);
    auto norm_bias = CastPyArg2OptionalValue(norm_bias_obj, "rms_norm", 4);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon = CastPyArg2Float(epsilon_obj, "rms_norm", 5);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 6);
    int begin_norm_axis = CastPyArg2Int(begin_norm_axis_obj, "rms_norm", 6);
    PyObject *quant_scale_obj = PyTuple_GET_ITEM(args, 7);
    float quant_scale = CastPyArg2Float(quant_scale_obj, "rms_norm", 7);
    PyObject *quant_round_type_obj = PyTuple_GET_ITEM(args, 8);
    int quant_round_type = CastPyArg2Int(quant_round_type_obj, "rms_norm", 8);
    PyObject *quant_max_bound_obj = PyTuple_GET_ITEM(args, 9);
    float quant_max_bound = CastPyArg2Float(quant_max_bound_obj, "rms_norm", 9);
    PyObject *quant_min_bound_obj = PyTuple_GET_ITEM(args, 10);
    float quant_min_bound =
        CastPyArg2Float(quant_min_bound_obj, "rms_norm", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("rms_norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rms_norm(
        x, bias, residual, norm_weight, norm_bias, epsilon, begin_norm_axis,
        quant_scale, quant_round_type, quant_max_bound, quant_min_bound);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rmsprop_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add rmsprop_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "rmsprop_", 0);
    PyObject *mean_square_obj = PyTuple_GET_ITEM(args, 1);
    auto mean_square = CastPyArg2Value(mean_square_obj, "rmsprop_", 1);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 2);
    auto grad = CastPyArg2Value(grad_obj, "rmsprop_", 2);
    PyObject *moment_obj = PyTuple_GET_ITEM(args, 3);
    auto moment = CastPyArg2Value(moment_obj, "rmsprop_", 3);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 4);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "rmsprop_", 4);
    PyObject *mean_grad_obj = PyTuple_GET_ITEM(args, 5);
    auto mean_grad = CastPyArg2OptionalValue(mean_grad_obj, "rmsprop_", 5);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 6);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "rmsprop_", 6);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "rmsprop_", 7);
    PyObject *decay_obj = PyTuple_GET_ITEM(args, 8);
    float decay = CastPyArg2Float(decay_obj, "rmsprop_", 8);
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 9);
    float momentum = CastPyArg2Float(momentum_obj, "rmsprop_", 9);
    PyObject *centered_obj = PyTuple_GET_ITEM(args, 10);
    bool centered = CastPyArg2Boolean(centered_obj, "rmsprop_", 10);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 11);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "rmsprop_", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("rmsprop_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rmsprop_(
        param, mean_square, grad, moment, learning_rate, mean_grad,
        master_param, epsilon, decay, momentum, centered, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_roi_align(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add roi_align op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "roi_align", 0);
    PyObject *boxes_obj = PyTuple_GET_ITEM(args, 1);
    auto boxes = CastPyArg2Value(boxes_obj, "roi_align", 1);
    PyObject *boxes_num_obj = PyTuple_GET_ITEM(args, 2);
    auto boxes_num = CastPyArg2OptionalValue(boxes_num_obj, "roi_align", 2);

    // Parse Attributes
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_align", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_align", 4);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_align", 5);
    PyObject *sampling_ratio_obj = PyTuple_GET_ITEM(args, 6);
    int sampling_ratio = CastPyArg2Int(sampling_ratio_obj, "roi_align", 6);
    PyObject *aligned_obj = PyTuple_GET_ITEM(args, 7);
    bool aligned = CastPyArg2Boolean(aligned_obj, "roi_align", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("roi_align");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::roi_align(
        x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale,
        sampling_ratio, aligned);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_roi_pool(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add roi_pool op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "roi_pool", 0);
    PyObject *boxes_obj = PyTuple_GET_ITEM(args, 1);
    auto boxes = CastPyArg2Value(boxes_obj, "roi_pool", 1);
    PyObject *boxes_num_obj = PyTuple_GET_ITEM(args, 2);
    auto boxes_num = CastPyArg2OptionalValue(boxes_num_obj, "roi_pool", 2);

    // Parse Attributes
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_pool", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_pool", 4);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_pool", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("roi_pool");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::roi_pool(
        x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_roll(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add roll op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "roll", 0);

    // Parse Attributes
    PyObject *shifts_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value shifts;

    if (PyObject_CheckIRValue(shifts_obj)) {
      shifts = CastPyArg2Value(shifts_obj, "roll", 1);
    } else if (PyObject_CheckIRVectorOfValue(shifts_obj)) {
      std::vector<pir::Value> shifts_tmp =
          CastPyArg2VectorOfValue(shifts_obj, "roll", 1);
      shifts = paddle::dialect::stack(shifts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shifts_tmp = CastPyArg2Longs(shifts_obj, "roll", 1);
      shifts = paddle::dialect::full_int_array(shifts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "roll", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("roll");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::roll(x, shifts, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_round(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add round op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "round", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("round");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::round(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_round_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add round_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "round_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("round_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::round_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rprop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rprop_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "rprop_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "rprop_", 1);
    PyObject *prev_obj = PyTuple_GET_ITEM(args, 2);
    auto prev = CastPyArg2Value(prev_obj, "rprop_", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "rprop_", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "rprop_", 4);
    PyObject *learning_rate_range_obj = PyTuple_GET_ITEM(args, 5);
    auto learning_rate_range =
        CastPyArg2Value(learning_rate_range_obj, "rprop_", 5);
    PyObject *etas_obj = PyTuple_GET_ITEM(args, 6);
    auto etas = CastPyArg2Value(etas_obj, "rprop_", 6);

    // Parse Attributes
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 7);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "rprop_", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("rprop_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::rprop_(param, grad, prev, learning_rate, master_param,
                                learning_rate_range, etas, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rsqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rsqrt op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rsqrt", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("rsqrt");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rsqrt(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rsqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rsqrt_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rsqrt_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("rsqrt_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rsqrt_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add scale op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "scale", 0);

    // Parse Attributes
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value scale;

    if (PyObject_CheckIRValue(scale_obj)) {
      scale = CastPyArg2Value(scale_obj, "scale", 1);
    } else {
      float scale_tmp = CastPyArg2Float(scale_obj, "scale", 1);
      scale = paddle::dialect::full(std::vector<int64_t>{1}, scale_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("scale");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::scale(x, scale, bias, bias_after_scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_scale_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add scale_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "scale_", 0);

    // Parse Attributes
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value scale;

    if (PyObject_CheckIRValue(scale_obj)) {
      scale = CastPyArg2Value(scale_obj, "scale_", 1);
    } else {
      float scale_tmp = CastPyArg2Float(scale_obj, "scale_", 1);
      scale = paddle::dialect::full(std::vector<int64_t>{1}, scale_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    float bias = CastPyArg2Float(bias_obj, "scale_", 2);
    bool bias_after_scale =
        CastPyArg2Boolean(bias_after_scale_obj, "scale_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("scale_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::scale_(x, scale, bias, bias_after_scale);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_scatter(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add scatter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "scatter", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "scatter", 1);
    PyObject *updates_obj = PyTuple_GET_ITEM(args, 2);
    auto updates = CastPyArg2Value(updates_obj, "scatter", 2);

    // Parse Attributes
    PyObject *overwrite_obj = PyTuple_GET_ITEM(args, 3);
    bool overwrite = CastPyArg2Boolean(overwrite_obj, "scatter", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("scatter");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::scatter(x, index, updates, overwrite);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_scatter_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add scatter_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "scatter_", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "scatter_", 1);
    PyObject *updates_obj = PyTuple_GET_ITEM(args, 2);
    auto updates = CastPyArg2Value(updates_obj, "scatter_", 2);

    // Parse Attributes
    PyObject *overwrite_obj = PyTuple_GET_ITEM(args, 3);
    bool overwrite = CastPyArg2Boolean(overwrite_obj, "scatter_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("scatter_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::scatter_(x, index, updates, overwrite);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_scatter_nd_add(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add scatter_nd_add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "scatter_nd_add", 0);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 1);
    auto index = CastPyArg2Value(index_obj, "scatter_nd_add", 1);
    PyObject *updates_obj = PyTuple_GET_ITEM(args, 2);
    auto updates = CastPyArg2Value(updates_obj, "scatter_nd_add", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("scatter_nd_add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::scatter_nd_add(x, index, updates);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_searchsorted(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add searchsorted op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *sorted_sequence_obj = PyTuple_GET_ITEM(args, 0);
    auto sorted_sequence =
        CastPyArg2Value(sorted_sequence_obj, "searchsorted", 0);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 1);
    auto values = CastPyArg2Value(values_obj, "searchsorted", 1);

    // Parse Attributes
    PyObject *out_int32_obj = PyTuple_GET_ITEM(args, 2);
    bool out_int32 = CastPyArg2Boolean(out_int32_obj, "searchsorted", 2);
    PyObject *right_obj = PyTuple_GET_ITEM(args, 3);
    bool right = CastPyArg2Boolean(right_obj, "searchsorted", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("searchsorted");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::searchsorted(sorted_sequence, values,
                                                        out_int32, right);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_segment_pool(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add segment_pool op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "segment_pool", 0);
    PyObject *segment_ids_obj = PyTuple_GET_ITEM(args, 1);
    auto segment_ids = CastPyArg2Value(segment_ids_obj, "segment_pool", 1);

    // Parse Attributes
    PyObject *pooltype_obj = PyTuple_GET_ITEM(args, 2);
    std::string pooltype = CastPyArg2String(pooltype_obj, "segment_pool", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("segment_pool");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::segment_pool(x, segment_ids, pooltype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_selu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add selu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "selu", 0);

    // Parse Attributes
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    float scale = CastPyArg2Float(scale_obj, "selu", 1);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 2);
    float alpha = CastPyArg2Float(alpha_obj, "selu", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("selu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::selu(x, scale, alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_send_u_recv(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add send_u_recv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "send_u_recv", 0);
    PyObject *src_index_obj = PyTuple_GET_ITEM(args, 1);
    auto src_index = CastPyArg2Value(src_index_obj, "send_u_recv", 1);
    PyObject *dst_index_obj = PyTuple_GET_ITEM(args, 2);
    auto dst_index = CastPyArg2Value(dst_index_obj, "send_u_recv", 2);

    // Parse Attributes
    PyObject *reduce_op_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value out_size;

    std::string reduce_op = CastPyArg2String(reduce_op_obj, "send_u_recv", 3);
    if (PyObject_CheckIRValue(out_size_obj)) {
      out_size = CastPyArg2Value(out_size_obj, "send_u_recv", 4);
    } else if (PyObject_CheckIRVectorOfValue(out_size_obj)) {
      std::vector<pir::Value> out_size_tmp =
          CastPyArg2VectorOfValue(out_size_obj, "send_u_recv", 4);
      out_size = paddle::dialect::stack(out_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> out_size_tmp =
          CastPyArg2Longs(out_size_obj, "send_u_recv", 4);
      out_size = paddle::dialect::full_int_array(
          out_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("send_u_recv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::send_u_recv(x, src_index, dst_index,
                                                       out_size, reduce_op);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_send_ue_recv(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add send_ue_recv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "send_ue_recv", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "send_ue_recv", 1);
    PyObject *src_index_obj = PyTuple_GET_ITEM(args, 2);
    auto src_index = CastPyArg2Value(src_index_obj, "send_ue_recv", 2);
    PyObject *dst_index_obj = PyTuple_GET_ITEM(args, 3);
    auto dst_index = CastPyArg2Value(dst_index_obj, "send_ue_recv", 3);

    // Parse Attributes
    PyObject *message_op_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *reduce_op_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 6);

    // Check for mutable attrs
    pir::Value out_size;

    std::string message_op =
        CastPyArg2String(message_op_obj, "send_ue_recv", 4);
    std::string reduce_op = CastPyArg2String(reduce_op_obj, "send_ue_recv", 5);
    if (PyObject_CheckIRValue(out_size_obj)) {
      out_size = CastPyArg2Value(out_size_obj, "send_ue_recv", 6);
    } else if (PyObject_CheckIRVectorOfValue(out_size_obj)) {
      std::vector<pir::Value> out_size_tmp =
          CastPyArg2VectorOfValue(out_size_obj, "send_ue_recv", 6);
      out_size = paddle::dialect::stack(out_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> out_size_tmp =
          CastPyArg2Longs(out_size_obj, "send_ue_recv", 6);
      out_size = paddle::dialect::full_int_array(
          out_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("send_ue_recv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::send_ue_recv(
        x, y, src_index, dst_index, out_size, message_op, reduce_op);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_send_uv(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add send_uv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "send_uv", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "send_uv", 1);
    PyObject *src_index_obj = PyTuple_GET_ITEM(args, 2);
    auto src_index = CastPyArg2Value(src_index_obj, "send_uv", 2);
    PyObject *dst_index_obj = PyTuple_GET_ITEM(args, 3);
    auto dst_index = CastPyArg2Value(dst_index_obj, "send_uv", 3);

    // Parse Attributes
    PyObject *message_op_obj = PyTuple_GET_ITEM(args, 4);
    std::string message_op = CastPyArg2String(message_op_obj, "send_uv", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("send_uv");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::send_uv(x, y, src_index, dst_index, message_op);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sgd_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sgd_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "sgd_", 0);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 1);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "sgd_", 1);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 2);
    auto grad = CastPyArg2Value(grad_obj, "sgd_", 2);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 3);
    auto master_param = CastPyArg2OptionalValue(master_param_obj, "sgd_", 3);

    // Parse Attributes
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 4);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "sgd_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("sgd_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sgd_(param, learning_rate, grad,
                                                master_param, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_shape(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add shape op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "shape", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("shape");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::shape(input);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_shard_index(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add shard_index op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "shard_index", 0);

    // Parse Attributes
    PyObject *index_num_obj = PyTuple_GET_ITEM(args, 1);
    int index_num = CastPyArg2Int(index_num_obj, "shard_index", 1);
    PyObject *nshards_obj = PyTuple_GET_ITEM(args, 2);
    int nshards = CastPyArg2Int(nshards_obj, "shard_index", 2);
    PyObject *shard_id_obj = PyTuple_GET_ITEM(args, 3);
    int shard_id = CastPyArg2Int(shard_id_obj, "shard_index", 3);
    PyObject *ignore_value_obj = PyTuple_GET_ITEM(args, 4);
    int ignore_value = CastPyArg2Int(ignore_value_obj, "shard_index", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("shard_index");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::shard_index(
        input, index_num, nshards, shard_id, ignore_value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sigmoid op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sigmoid", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sigmoid");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sigmoid(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sigmoid_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add sigmoid_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sigmoid_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sigmoid_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sigmoid_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sigmoid_cross_entropy_with_logits(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add sigmoid_cross_entropy_with_logits op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sigmoid_cross_entropy_with_logits", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label =
        CastPyArg2Value(label_obj, "sigmoid_cross_entropy_with_logits", 1);
    PyObject *pos_weight_obj = PyTuple_GET_ITEM(args, 2);
    auto pos_weight = CastPyArg2OptionalValue(
        pos_weight_obj, "sigmoid_cross_entropy_with_logits", 2);

    // Parse Attributes
    PyObject *normalize_obj = PyTuple_GET_ITEM(args, 3);
    bool normalize = CastPyArg2Boolean(normalize_obj,
                                       "sigmoid_cross_entropy_with_logits", 3);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 4);
    int ignore_index =
        CastPyArg2Int(ignore_index_obj, "sigmoid_cross_entropy_with_logits", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("sigmoid_cross_entropy_with_logits");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sigmoid_cross_entropy_with_logits(
        x, label, pos_weight, normalize, ignore_index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sigmoid_cross_entropy_with_logits_(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add sigmoid_cross_entropy_with_logits_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sigmoid_cross_entropy_with_logits_", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label =
        CastPyArg2Value(label_obj, "sigmoid_cross_entropy_with_logits_", 1);
    PyObject *pos_weight_obj = PyTuple_GET_ITEM(args, 2);
    auto pos_weight = CastPyArg2OptionalValue(
        pos_weight_obj, "sigmoid_cross_entropy_with_logits_", 2);

    // Parse Attributes
    PyObject *normalize_obj = PyTuple_GET_ITEM(args, 3);
    bool normalize = CastPyArg2Boolean(normalize_obj,
                                       "sigmoid_cross_entropy_with_logits_", 3);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 4);
    int ignore_index = CastPyArg2Int(ignore_index_obj,
                                     "sigmoid_cross_entropy_with_logits_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("sigmoid_cross_entropy_with_logits_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sigmoid_cross_entropy_with_logits_(
        x, label, pos_weight, normalize, ignore_index);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sign(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sign op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sign", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sign");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sign(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_silu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add silu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "silu", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("silu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::silu(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sin", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sin");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sin(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sin_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sin_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sin_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sin_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sin_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sinh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sinh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sinh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sinh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sinh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sinh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sinh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sinh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sinh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_slogdet(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add slogdet op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "slogdet", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("slogdet");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::slogdet(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_softplus(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add softplus op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "softplus", 0);

    // Parse Attributes
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 1);
    float beta = CastPyArg2Float(beta_obj, "softplus", 1);
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 2);
    float threshold = CastPyArg2Float(threshold_obj, "softplus", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("softplus");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::softplus(x, beta, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_softshrink(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add softshrink op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "softshrink", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "softshrink", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("softshrink");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::softshrink(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_softsign(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add softsign op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "softsign", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("softsign");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::softsign(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add solve op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "solve", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "solve", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("solve");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::solve(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_spectral_norm(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add spectral_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 0);
    auto weight = CastPyArg2Value(weight_obj, "spectral_norm", 0);
    PyObject *u_obj = PyTuple_GET_ITEM(args, 1);
    auto u = CastPyArg2Value(u_obj, "spectral_norm", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v = CastPyArg2Value(v_obj, "spectral_norm", 2);

    // Parse Attributes
    PyObject *dim_obj = PyTuple_GET_ITEM(args, 3);
    int dim = CastPyArg2Int(dim_obj, "spectral_norm", 3);
    PyObject *power_iters_obj = PyTuple_GET_ITEM(args, 4);
    int power_iters = CastPyArg2Int(power_iters_obj, "spectral_norm", 4);
    PyObject *eps_obj = PyTuple_GET_ITEM(args, 5);
    float eps = CastPyArg2Float(eps_obj, "spectral_norm", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("spectral_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::spectral_norm(weight, u, v, dim, power_iters, eps);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sqrt op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sqrt", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sqrt");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sqrt(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sqrt_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sqrt_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("sqrt_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sqrt_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_square(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add square op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "square", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("square");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::square(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_squared_l2_norm(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add squared_l2_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "squared_l2_norm", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("squared_l2_norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::squared_l2_norm(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_squeeze(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add squeeze op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "squeeze", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "squeeze", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "squeeze", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "squeeze", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("squeeze");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::squeeze(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_squeeze_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add squeeze_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "squeeze_", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "squeeze_", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "squeeze_", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "squeeze_", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("squeeze_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::squeeze_(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_stack(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add stack op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "stack", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "stack", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("stack");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::stack(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_standard_gamma(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add standard_gamma op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "standard_gamma", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("standard_gamma");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::standard_gamma(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_stanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add stanh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "stanh", 0);

    // Parse Attributes
    PyObject *scale_a_obj = PyTuple_GET_ITEM(args, 1);
    float scale_a = CastPyArg2Float(scale_a_obj, "stanh", 1);
    PyObject *scale_b_obj = PyTuple_GET_ITEM(args, 2);
    float scale_b = CastPyArg2Float(scale_b_obj, "stanh", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("stanh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::stanh(x, scale_a, scale_b);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_svd(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add svd op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "svd", 0);

    // Parse Attributes
    PyObject *full_matrices_obj = PyTuple_GET_ITEM(args, 1);
    bool full_matrices = CastPyArg2Boolean(full_matrices_obj, "svd", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("svd");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::svd(x, full_matrices);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_swiglu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add swiglu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "swiglu", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2OptionalValue(y_obj, "swiglu", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("swiglu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::swiglu(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_take_along_axis(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add take_along_axis op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *arr_obj = PyTuple_GET_ITEM(args, 0);
    auto arr = CastPyArg2Value(arr_obj, "take_along_axis", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "take_along_axis", 1);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "take_along_axis", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("take_along_axis");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::take_along_axis(arr, indices, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tan(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tan op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tan", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("tan");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tan(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tan_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tan_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tan_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("tan_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tan_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tanh op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tanh", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("tanh");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tanh(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tanh_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tanh_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("tanh_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tanh_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tanh_shrink(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add tanh_shrink op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tanh_shrink", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("tanh_shrink");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tanh_shrink(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_temporal_shift(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add temporal_shift op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "temporal_shift", 0);

    // Parse Attributes
    PyObject *seg_num_obj = PyTuple_GET_ITEM(args, 1);
    int seg_num = CastPyArg2Int(seg_num_obj, "temporal_shift", 1);
    PyObject *shift_ratio_obj = PyTuple_GET_ITEM(args, 2);
    float shift_ratio = CastPyArg2Float(shift_ratio_obj, "temporal_shift", 2);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 3);
    std::string data_format =
        CastPyArg2String(data_format_obj, "temporal_shift", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("temporal_shift");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::temporal_shift(x, seg_num, shift_ratio, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tensor_unfold(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add tensor_unfold op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "tensor_unfold", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int64_t axis = CastPyArg2Long(axis_obj, "tensor_unfold", 1);
    PyObject *size_obj = PyTuple_GET_ITEM(args, 2);
    int64_t size = CastPyArg2Long(size_obj, "tensor_unfold", 2);
    PyObject *step_obj = PyTuple_GET_ITEM(args, 3);
    int64_t step = CastPyArg2Long(step_obj, "tensor_unfold", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("tensor_unfold");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::tensor_unfold(input, axis, size, step);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_thresholded_relu(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add thresholded_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "thresholded_relu", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "thresholded_relu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("thresholded_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::thresholded_relu(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_thresholded_relu_(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add thresholded_relu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "thresholded_relu_", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "thresholded_relu_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("thresholded_relu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::thresholded_relu_(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_top_p_sampling(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add top_p_sampling op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "top_p_sampling", 0);
    PyObject *ps_obj = PyTuple_GET_ITEM(args, 1);
    auto ps = CastPyArg2Value(ps_obj, "top_p_sampling", 1);
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 2);
    auto threshold =
        CastPyArg2OptionalValue(threshold_obj, "top_p_sampling", 2);

    // Parse Attributes
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "top_p_sampling", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("top_p_sampling");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::top_p_sampling(x, ps, threshold, seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_topk(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add topk op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "topk", 0);

    // Parse Attributes
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *largest_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *sorted_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value k;

    if (PyObject_CheckIRValue(k_obj)) {
      k = CastPyArg2Value(k_obj, "topk", 1);
    } else {
      int k_tmp = CastPyArg2Int(k_obj, "topk", 1);
      k = paddle::dialect::full(std::vector<int64_t>{1}, k_tmp,
                                phi::DataType::INT32, phi::CPUPlace());
    }
    int axis = CastPyArg2Int(axis_obj, "topk", 2);
    bool largest = CastPyArg2Boolean(largest_obj, "topk", 3);
    bool sorted = CastPyArg2Boolean(sorted_obj, "topk", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("topk");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::topk(x, k, axis, largest, sorted);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_trace(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add trace op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "trace", 0);

    // Parse Attributes
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "trace", 1);
    PyObject *axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "trace", 2);
    PyObject *axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "trace", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("trace");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::trace(x, offset, axis1, axis2);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_triangular_solve(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add triangular_solve op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "triangular_solve", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "triangular_solve", 1);

    // Parse Attributes
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "triangular_solve", 2);
    PyObject *transpose_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose = CastPyArg2Boolean(transpose_obj, "triangular_solve", 3);
    PyObject *unitriangular_obj = PyTuple_GET_ITEM(args, 4);
    bool unitriangular =
        CastPyArg2Boolean(unitriangular_obj, "triangular_solve", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("triangular_solve");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::triangular_solve(
        x, y, upper, transpose, unitriangular);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_trilinear_interp(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add trilinear_interp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "trilinear_interp", 0);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 1);
    auto out_size =
        CastPyArg2OptionalValue(out_size_obj, "trilinear_interp", 1);
    PyObject *size_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto size_tensor =
        CastPyArg2OptionalVectorOfValue(size_tensor_obj, "trilinear_interp", 2);
    PyObject *scale_tensor_obj = PyTuple_GET_ITEM(args, 3);
    auto scale_tensor =
        CastPyArg2OptionalValue(scale_tensor_obj, "trilinear_interp", 3);

    // Parse Attributes
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "trilinear_interp", 4);
    PyObject *out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "trilinear_interp", 5);
    PyObject *out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "trilinear_interp", 6);
    PyObject *out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "trilinear_interp", 7);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale =
        CastPyArg2Floats(scale_obj, "trilinear_interp", 8);
    PyObject *interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method =
        CastPyArg2String(interp_method_obj, "trilinear_interp", 9);
    PyObject *align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners =
        CastPyArg2Boolean(align_corners_obj, "trilinear_interp", 10);
    PyObject *align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "trilinear_interp", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("trilinear_interp");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::trilinear_interp(
        x, out_size, size_tensor, scale_tensor, data_format, out_d, out_h,
        out_w, scale, interp_method, align_corners, align_mode);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_trunc(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add trunc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "trunc", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("trunc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::trunc(input);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_trunc_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add trunc_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "trunc_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("trunc_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::trunc_(input);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unbind(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add unbind op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "unbind", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "unbind", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("unbind");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unbind(input, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unfold(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add unfold op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unfold", 0);

    // Parse Attributes
    PyObject *kernel_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_sizes =
        CastPyArg2Ints(kernel_sizes_obj, "unfold", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unfold", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "unfold", 3);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "unfold", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("unfold");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::unfold(x, kernel_sizes, strides, paddings, dilations);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_uniform_inplace(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add uniform_inplace op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "uniform_inplace", 0);

    // Parse Attributes
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    float min = CastPyArg2Float(min_obj, "uniform_inplace", 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);
    float max = CastPyArg2Float(max_obj, "uniform_inplace", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "uniform_inplace", 3);
    PyObject *diag_num_obj = PyTuple_GET_ITEM(args, 4);
    int diag_num = CastPyArg2Int(diag_num_obj, "uniform_inplace", 4);
    PyObject *diag_step_obj = PyTuple_GET_ITEM(args, 5);
    int diag_step = CastPyArg2Int(diag_step_obj, "uniform_inplace", 5);
    PyObject *diag_val_obj = PyTuple_GET_ITEM(args, 6);
    float diag_val = CastPyArg2Float(diag_val_obj, "uniform_inplace", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("uniform_inplace");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::uniform_inplace(
        x, min, max, seed, diag_num, diag_step, diag_val);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_uniform_inplace_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add uniform_inplace_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "uniform_inplace_", 0);

    // Parse Attributes
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    float min = CastPyArg2Float(min_obj, "uniform_inplace_", 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);
    float max = CastPyArg2Float(max_obj, "uniform_inplace_", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "uniform_inplace_", 3);
    PyObject *diag_num_obj = PyTuple_GET_ITEM(args, 4);
    int diag_num = CastPyArg2Int(diag_num_obj, "uniform_inplace_", 4);
    PyObject *diag_step_obj = PyTuple_GET_ITEM(args, 5);
    int diag_step = CastPyArg2Int(diag_step_obj, "uniform_inplace_", 5);
    PyObject *diag_val_obj = PyTuple_GET_ITEM(args, 6);
    float diag_val = CastPyArg2Float(diag_val_obj, "uniform_inplace_", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("uniform_inplace_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::uniform_inplace_(
        x, min, max, seed, diag_num, diag_step, diag_val);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unique_consecutive(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add unique_consecutive op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unique_consecutive", 0);

    // Parse Attributes
    PyObject *return_inverse_obj = PyTuple_GET_ITEM(args, 1);
    bool return_inverse =
        CastPyArg2Boolean(return_inverse_obj, "unique_consecutive", 1);
    PyObject *return_counts_obj = PyTuple_GET_ITEM(args, 2);
    bool return_counts =
        CastPyArg2Boolean(return_counts_obj, "unique_consecutive", 2);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "unique_consecutive", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "unique_consecutive", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("unique_consecutive");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unique_consecutive(
        x, return_inverse, return_counts, axis, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unpool3d(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add unpool3d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unpool3d", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "unpool3d", 1);

    // Parse Attributes
    PyObject *ksize_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> ksize = CastPyArg2Ints(ksize_obj, "unpool3d", 2);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unpool3d", 3);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "unpool3d", 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "unpool3d", 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "unpool3d", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("unpool3d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unpool3d(
        x, indices, ksize, strides, paddings, output_size, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unsqueeze(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add unsqueeze op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unsqueeze", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "unsqueeze", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "unsqueeze", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "unsqueeze", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("unsqueeze");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unsqueeze(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unsqueeze_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add unsqueeze_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unsqueeze_", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "unsqueeze_", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "unsqueeze_", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp =
          CastPyArg2Longs(axis_obj, "unsqueeze_", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("unsqueeze_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unsqueeze_(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unstack(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add unstack op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unstack", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "unstack", 1);
    PyObject *num_obj = PyTuple_GET_ITEM(args, 2);
    int num = CastPyArg2Int(num_obj, "unstack", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("unstack");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unstack(x, axis, num);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_update_loss_scaling_(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add update_loss_scaling_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "update_loss_scaling_", 0);
    PyObject *found_infinite_obj = PyTuple_GET_ITEM(args, 1);
    auto found_infinite =
        CastPyArg2Value(found_infinite_obj, "update_loss_scaling_", 1);
    PyObject *prev_loss_scaling_obj = PyTuple_GET_ITEM(args, 2);
    auto prev_loss_scaling =
        CastPyArg2Value(prev_loss_scaling_obj, "update_loss_scaling_", 2);
    PyObject *in_good_steps_obj = PyTuple_GET_ITEM(args, 3);
    auto in_good_steps =
        CastPyArg2Value(in_good_steps_obj, "update_loss_scaling_", 3);
    PyObject *in_bad_steps_obj = PyTuple_GET_ITEM(args, 4);
    auto in_bad_steps =
        CastPyArg2Value(in_bad_steps_obj, "update_loss_scaling_", 4);

    // Parse Attributes
    PyObject *incr_every_n_steps_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *decr_every_n_nan_or_inf_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *incr_ratio_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *decr_ratio_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *stop_update_obj = PyTuple_GET_ITEM(args, 9);

    // Check for mutable attrs
    pir::Value stop_update;

    int incr_every_n_steps =
        CastPyArg2Int(incr_every_n_steps_obj, "update_loss_scaling_", 5);
    int decr_every_n_nan_or_inf =
        CastPyArg2Int(decr_every_n_nan_or_inf_obj, "update_loss_scaling_", 6);
    float incr_ratio =
        CastPyArg2Float(incr_ratio_obj, "update_loss_scaling_", 7);
    float decr_ratio =
        CastPyArg2Float(decr_ratio_obj, "update_loss_scaling_", 8);
    if (PyObject_CheckIRValue(stop_update_obj)) {
      stop_update = CastPyArg2Value(stop_update_obj, "update_loss_scaling_", 9);
    } else {
      bool stop_update_tmp =
          CastPyArg2Boolean(stop_update_obj, "update_loss_scaling_", 9);
      stop_update =
          paddle::dialect::full(std::vector<int64_t>{1}, stop_update_tmp,
                                phi::DataType::BOOL, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("update_loss_scaling_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::update_loss_scaling_(
        x, found_infinite, prev_loss_scaling, in_good_steps, in_bad_steps,
        stop_update, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio,
        decr_ratio);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_view_dtype(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add view_dtype op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "view_dtype", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "view_dtype", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("view_dtype");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::view_dtype(input, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_view_shape(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add view_shape op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "view_shape", 0);

    // Parse Attributes
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "view_shape", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("view_shape");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::view_shape(input, dims);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_viterbi_decode(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add viterbi_decode op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *potentials_obj = PyTuple_GET_ITEM(args, 0);
    auto potentials = CastPyArg2Value(potentials_obj, "viterbi_decode", 0);
    PyObject *transition_params_obj = PyTuple_GET_ITEM(args, 1);
    auto transition_params =
        CastPyArg2Value(transition_params_obj, "viterbi_decode", 1);
    PyObject *lengths_obj = PyTuple_GET_ITEM(args, 2);
    auto lengths = CastPyArg2Value(lengths_obj, "viterbi_decode", 2);

    // Parse Attributes
    PyObject *include_bos_eos_tag_obj = PyTuple_GET_ITEM(args, 3);
    bool include_bos_eos_tag =
        CastPyArg2Boolean(include_bos_eos_tag_obj, "viterbi_decode", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("viterbi_decode");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::viterbi_decode(
        potentials, transition_params, lengths, include_bos_eos_tag);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_warpctc(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add warpctc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *logits_obj = PyTuple_GET_ITEM(args, 0);
    auto logits = CastPyArg2Value(logits_obj, "warpctc", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "warpctc", 1);
    PyObject *logits_length_obj = PyTuple_GET_ITEM(args, 2);
    auto logits_length =
        CastPyArg2OptionalValue(logits_length_obj, "warpctc", 2);
    PyObject *labels_length_obj = PyTuple_GET_ITEM(args, 3);
    auto labels_length =
        CastPyArg2OptionalValue(labels_length_obj, "warpctc", 3);

    // Parse Attributes
    PyObject *blank_obj = PyTuple_GET_ITEM(args, 4);
    int blank = CastPyArg2Int(blank_obj, "warpctc", 4);
    PyObject *norm_by_times_obj = PyTuple_GET_ITEM(args, 5);
    bool norm_by_times = CastPyArg2Boolean(norm_by_times_obj, "warpctc", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("warpctc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::warpctc(
        logits, label, logits_length, labels_length, blank, norm_by_times);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_warprnnt(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add warprnnt op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "warprnnt", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "warprnnt", 1);
    PyObject *input_lengths_obj = PyTuple_GET_ITEM(args, 2);
    auto input_lengths = CastPyArg2Value(input_lengths_obj, "warprnnt", 2);
    PyObject *label_lengths_obj = PyTuple_GET_ITEM(args, 3);
    auto label_lengths = CastPyArg2Value(label_lengths_obj, "warprnnt", 3);

    // Parse Attributes
    PyObject *blank_obj = PyTuple_GET_ITEM(args, 4);
    int blank = CastPyArg2Int(blank_obj, "warprnnt", 4);
    PyObject *fastemit_lambda_obj = PyTuple_GET_ITEM(args, 5);
    float fastemit_lambda = CastPyArg2Float(fastemit_lambda_obj, "warprnnt", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("warprnnt");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::warprnnt(
        input, label, input_lengths, label_lengths, blank, fastemit_lambda);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_weight_dequantize(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add weight_dequantize op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "weight_dequantize", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "weight_dequantize", 1);

    // Parse Attributes
    PyObject *algo_obj = PyTuple_GET_ITEM(args, 2);
    std::string algo = CastPyArg2String(algo_obj, "weight_dequantize", 2);
    PyObject *out_dtype_obj = PyTuple_GET_ITEM(args, 3);
    phi::DataType out_dtype =
        CastPyArg2DataTypeDirectly(out_dtype_obj, "weight_dequantize", 3);
    PyObject *group_size_obj = PyTuple_GET_ITEM(args, 4);
    int group_size = CastPyArg2Int(group_size_obj, "weight_dequantize", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("weight_dequantize");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::weight_dequantize(
        x, scale, algo, out_dtype, group_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_weight_only_linear(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add weight_only_linear op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "weight_only_linear", 0);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 1);
    auto weight = CastPyArg2Value(weight_obj, "weight_only_linear", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "weight_only_linear", 2);
    PyObject *weight_scale_obj = PyTuple_GET_ITEM(args, 3);
    auto weight_scale =
        CastPyArg2Value(weight_scale_obj, "weight_only_linear", 3);

    // Parse Attributes
    PyObject *weight_dtype_obj = PyTuple_GET_ITEM(args, 4);
    std::string weight_dtype =
        CastPyArg2String(weight_dtype_obj, "weight_only_linear", 4);
    PyObject *arch_obj = PyTuple_GET_ITEM(args, 5);
    int arch = CastPyArg2Int(arch_obj, "weight_only_linear", 5);
    PyObject *group_size_obj = PyTuple_GET_ITEM(args, 6);
    int group_size = CastPyArg2Int(group_size_obj, "weight_only_linear", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("weight_only_linear");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::weight_only_linear(
        x, weight, bias, weight_scale, weight_dtype, arch, group_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_weight_quantize(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add weight_quantize op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "weight_quantize", 0);

    // Parse Attributes
    PyObject *algo_obj = PyTuple_GET_ITEM(args, 1);
    std::string algo = CastPyArg2String(algo_obj, "weight_quantize", 1);
    PyObject *arch_obj = PyTuple_GET_ITEM(args, 2);
    int arch = CastPyArg2Int(arch_obj, "weight_quantize", 2);
    PyObject *group_size_obj = PyTuple_GET_ITEM(args, 3);
    int group_size = CastPyArg2Int(group_size_obj, "weight_quantize", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("weight_quantize");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::weight_quantize(x, algo, arch, group_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_weighted_sample_neighbors(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add weighted_sample_neighbors op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *row_obj = PyTuple_GET_ITEM(args, 0);
    auto row = CastPyArg2Value(row_obj, "weighted_sample_neighbors", 0);
    PyObject *colptr_obj = PyTuple_GET_ITEM(args, 1);
    auto colptr = CastPyArg2Value(colptr_obj, "weighted_sample_neighbors", 1);
    PyObject *edge_weight_obj = PyTuple_GET_ITEM(args, 2);
    auto edge_weight =
        CastPyArg2Value(edge_weight_obj, "weighted_sample_neighbors", 2);
    PyObject *input_nodes_obj = PyTuple_GET_ITEM(args, 3);
    auto input_nodes =
        CastPyArg2Value(input_nodes_obj, "weighted_sample_neighbors", 3);
    PyObject *eids_obj = PyTuple_GET_ITEM(args, 4);
    auto eids =
        CastPyArg2OptionalValue(eids_obj, "weighted_sample_neighbors", 4);

    // Parse Attributes
    PyObject *sample_size_obj = PyTuple_GET_ITEM(args, 5);
    int sample_size =
        CastPyArg2Int(sample_size_obj, "weighted_sample_neighbors", 5);
    PyObject *return_eids_obj = PyTuple_GET_ITEM(args, 6);
    bool return_eids =
        CastPyArg2Boolean(return_eids_obj, "weighted_sample_neighbors", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("weighted_sample_neighbors");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::weighted_sample_neighbors(
        row, colptr, edge_weight, input_nodes, eids, sample_size, return_eids);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_where(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add where op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *condition_obj = PyTuple_GET_ITEM(args, 0);
    auto condition = CastPyArg2Value(condition_obj, "where", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "where", 1);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 2);
    auto y = CastPyArg2Value(y_obj, "where", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("where");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::where(condition, x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_where_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add where_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *condition_obj = PyTuple_GET_ITEM(args, 0);
    auto condition = CastPyArg2Value(condition_obj, "where_", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "where_", 1);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 2);
    auto y = CastPyArg2Value(y_obj, "where_", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("where_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::where_(condition, x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_yolo_box(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add yolo_box op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "yolo_box", 0);
    PyObject *img_size_obj = PyTuple_GET_ITEM(args, 1);
    auto img_size = CastPyArg2Value(img_size_obj, "yolo_box", 1);

    // Parse Attributes
    PyObject *anchors_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> anchors = CastPyArg2Ints(anchors_obj, "yolo_box", 2);
    PyObject *class_num_obj = PyTuple_GET_ITEM(args, 3);
    int class_num = CastPyArg2Int(class_num_obj, "yolo_box", 3);
    PyObject *conf_thresh_obj = PyTuple_GET_ITEM(args, 4);
    float conf_thresh = CastPyArg2Float(conf_thresh_obj, "yolo_box", 4);
    PyObject *downsample_ratio_obj = PyTuple_GET_ITEM(args, 5);
    int downsample_ratio = CastPyArg2Int(downsample_ratio_obj, "yolo_box", 5);
    PyObject *clip_bbox_obj = PyTuple_GET_ITEM(args, 6);
    bool clip_bbox = CastPyArg2Boolean(clip_bbox_obj, "yolo_box", 6);
    PyObject *scale_x_y_obj = PyTuple_GET_ITEM(args, 7);
    float scale_x_y = CastPyArg2Float(scale_x_y_obj, "yolo_box", 7);
    PyObject *iou_aware_obj = PyTuple_GET_ITEM(args, 8);
    bool iou_aware = CastPyArg2Boolean(iou_aware_obj, "yolo_box", 8);
    PyObject *iou_aware_factor_obj = PyTuple_GET_ITEM(args, 9);
    float iou_aware_factor =
        CastPyArg2Float(iou_aware_factor_obj, "yolo_box", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("yolo_box");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::yolo_box(
        x, img_size, anchors, class_num, conf_thresh, downsample_ratio,
        clip_bbox, scale_x_y, iou_aware, iou_aware_factor);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_yolo_loss(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add yolo_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "yolo_loss", 0);
    PyObject *gt_box_obj = PyTuple_GET_ITEM(args, 1);
    auto gt_box = CastPyArg2Value(gt_box_obj, "yolo_loss", 1);
    PyObject *gt_label_obj = PyTuple_GET_ITEM(args, 2);
    auto gt_label = CastPyArg2Value(gt_label_obj, "yolo_loss", 2);
    PyObject *gt_score_obj = PyTuple_GET_ITEM(args, 3);
    auto gt_score = CastPyArg2OptionalValue(gt_score_obj, "yolo_loss", 3);

    // Parse Attributes
    PyObject *anchors_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> anchors = CastPyArg2Ints(anchors_obj, "yolo_loss", 4);
    PyObject *anchor_mask_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> anchor_mask =
        CastPyArg2Ints(anchor_mask_obj, "yolo_loss", 5);
    PyObject *class_num_obj = PyTuple_GET_ITEM(args, 6);
    int class_num = CastPyArg2Int(class_num_obj, "yolo_loss", 6);
    PyObject *ignore_thresh_obj = PyTuple_GET_ITEM(args, 7);
    float ignore_thresh = CastPyArg2Float(ignore_thresh_obj, "yolo_loss", 7);
    PyObject *downsample_ratio_obj = PyTuple_GET_ITEM(args, 8);
    int downsample_ratio = CastPyArg2Int(downsample_ratio_obj, "yolo_loss", 8);
    PyObject *use_label_smooth_obj = PyTuple_GET_ITEM(args, 9);
    bool use_label_smooth =
        CastPyArg2Boolean(use_label_smooth_obj, "yolo_loss", 9);
    PyObject *scale_x_y_obj = PyTuple_GET_ITEM(args, 10);
    float scale_x_y = CastPyArg2Float(scale_x_y_obj, "yolo_loss", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("yolo_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::yolo_loss(
        x, gt_box, gt_label, gt_score, anchors, anchor_mask, class_num,
        ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_block_multihead_attention_(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add block_multihead_attention_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *qkv_obj = PyTuple_GET_ITEM(args, 0);
    auto qkv = CastPyArg2Value(qkv_obj, "block_multihead_attention_", 0);
    PyObject *key_cache_obj = PyTuple_GET_ITEM(args, 1);
    auto key_cache =
        CastPyArg2Value(key_cache_obj, "block_multihead_attention_", 1);
    PyObject *value_cache_obj = PyTuple_GET_ITEM(args, 2);
    auto value_cache =
        CastPyArg2Value(value_cache_obj, "block_multihead_attention_", 2);
    PyObject *seq_lens_encoder_obj = PyTuple_GET_ITEM(args, 3);
    auto seq_lens_encoder =
        CastPyArg2Value(seq_lens_encoder_obj, "block_multihead_attention_", 3);
    PyObject *seq_lens_decoder_obj = PyTuple_GET_ITEM(args, 4);
    auto seq_lens_decoder =
        CastPyArg2Value(seq_lens_decoder_obj, "block_multihead_attention_", 4);
    PyObject *seq_lens_this_time_obj = PyTuple_GET_ITEM(args, 5);
    auto seq_lens_this_time = CastPyArg2Value(seq_lens_this_time_obj,
                                              "block_multihead_attention_", 5);
    PyObject *padding_offsets_obj = PyTuple_GET_ITEM(args, 6);
    auto padding_offsets =
        CastPyArg2Value(padding_offsets_obj, "block_multihead_attention_", 6);
    PyObject *cum_offsets_obj = PyTuple_GET_ITEM(args, 7);
    auto cum_offsets =
        CastPyArg2Value(cum_offsets_obj, "block_multihead_attention_", 7);
    PyObject *cu_seqlens_q_obj = PyTuple_GET_ITEM(args, 8);
    auto cu_seqlens_q =
        CastPyArg2Value(cu_seqlens_q_obj, "block_multihead_attention_", 8);
    PyObject *cu_seqlens_k_obj = PyTuple_GET_ITEM(args, 9);
    auto cu_seqlens_k =
        CastPyArg2Value(cu_seqlens_k_obj, "block_multihead_attention_", 9);
    PyObject *block_tables_obj = PyTuple_GET_ITEM(args, 10);
    auto block_tables =
        CastPyArg2Value(block_tables_obj, "block_multihead_attention_", 10);
    PyObject *pre_key_cache_obj = PyTuple_GET_ITEM(args, 11);
    auto pre_key_cache = CastPyArg2OptionalValue(
        pre_key_cache_obj, "block_multihead_attention_", 11);
    PyObject *pre_value_cache_obj = PyTuple_GET_ITEM(args, 12);
    auto pre_value_cache = CastPyArg2OptionalValue(
        pre_value_cache_obj, "block_multihead_attention_", 12);
    PyObject *rope_emb_obj = PyTuple_GET_ITEM(args, 13);
    auto rope_emb =
        CastPyArg2OptionalValue(rope_emb_obj, "block_multihead_attention_", 13);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 14);
    auto mask =
        CastPyArg2OptionalValue(mask_obj, "block_multihead_attention_", 14);
    PyObject *tgt_mask_obj = PyTuple_GET_ITEM(args, 15);
    auto tgt_mask =
        CastPyArg2OptionalValue(tgt_mask_obj, "block_multihead_attention_", 15);
    PyObject *cache_k_quant_scales_obj = PyTuple_GET_ITEM(args, 16);
    auto cache_k_quant_scales = CastPyArg2OptionalValue(
        cache_k_quant_scales_obj, "block_multihead_attention_", 16);
    PyObject *cache_v_quant_scales_obj = PyTuple_GET_ITEM(args, 17);
    auto cache_v_quant_scales = CastPyArg2OptionalValue(
        cache_v_quant_scales_obj, "block_multihead_attention_", 17);
    PyObject *cache_k_dequant_scales_obj = PyTuple_GET_ITEM(args, 18);
    auto cache_k_dequant_scales = CastPyArg2OptionalValue(
        cache_k_dequant_scales_obj, "block_multihead_attention_", 18);
    PyObject *cache_v_dequant_scales_obj = PyTuple_GET_ITEM(args, 19);
    auto cache_v_dequant_scales = CastPyArg2OptionalValue(
        cache_v_dequant_scales_obj, "block_multihead_attention_", 19);
    PyObject *qkv_out_scale_obj = PyTuple_GET_ITEM(args, 20);
    auto qkv_out_scale = CastPyArg2OptionalValue(
        qkv_out_scale_obj, "block_multihead_attention_", 20);
    PyObject *qkv_bias_obj = PyTuple_GET_ITEM(args, 21);
    auto qkv_bias =
        CastPyArg2OptionalValue(qkv_bias_obj, "block_multihead_attention_", 21);
    PyObject *out_shift_obj = PyTuple_GET_ITEM(args, 22);
    auto out_shift = CastPyArg2OptionalValue(out_shift_obj,
                                             "block_multihead_attention_", 22);
    PyObject *out_smooth_obj = PyTuple_GET_ITEM(args, 23);
    auto out_smooth = CastPyArg2OptionalValue(out_smooth_obj,
                                              "block_multihead_attention_", 23);

    // Parse Attributes
    PyObject *max_seq_len_obj = PyTuple_GET_ITEM(args, 24);
    int max_seq_len =
        CastPyArg2Int(max_seq_len_obj, "block_multihead_attention_", 24);
    PyObject *block_size_obj = PyTuple_GET_ITEM(args, 25);
    int block_size =
        CastPyArg2Int(block_size_obj, "block_multihead_attention_", 25);
    PyObject *use_neox_style_obj = PyTuple_GET_ITEM(args, 26);
    bool use_neox_style =
        CastPyArg2Boolean(use_neox_style_obj, "block_multihead_attention_", 26);
    PyObject *dynamic_cachekv_quant_obj = PyTuple_GET_ITEM(args, 27);
    bool dynamic_cachekv_quant = CastPyArg2Boolean(
        dynamic_cachekv_quant_obj, "block_multihead_attention_", 27);
    PyObject *quant_round_type_obj = PyTuple_GET_ITEM(args, 28);
    int quant_round_type =
        CastPyArg2Int(quant_round_type_obj, "block_multihead_attention_", 28);
    PyObject *quant_max_bound_obj = PyTuple_GET_ITEM(args, 29);
    float quant_max_bound =
        CastPyArg2Float(quant_max_bound_obj, "block_multihead_attention_", 29);
    PyObject *quant_min_bound_obj = PyTuple_GET_ITEM(args, 30);
    float quant_min_bound =
        CastPyArg2Float(quant_min_bound_obj, "block_multihead_attention_", 30);
    PyObject *out_scale_obj = PyTuple_GET_ITEM(args, 31);
    float out_scale =
        CastPyArg2Float(out_scale_obj, "block_multihead_attention_", 31);
    PyObject *compute_dtype_obj = PyTuple_GET_ITEM(args, 32);
    std::string compute_dtype =
        CastPyArg2String(compute_dtype_obj, "block_multihead_attention_", 32);

    // Call ir static api
    CallStackRecorder callstack_recorder("block_multihead_attention_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::block_multihead_attention_(
        qkv, key_cache, value_cache, seq_lens_encoder, seq_lens_decoder,
        seq_lens_this_time, padding_offsets, cum_offsets, cu_seqlens_q,
        cu_seqlens_k, block_tables, pre_key_cache, pre_value_cache, rope_emb,
        mask, tgt_mask, cache_k_quant_scales, cache_v_quant_scales,
        cache_k_dequant_scales, cache_v_dequant_scales, qkv_out_scale, qkv_bias,
        out_shift, out_smooth, max_seq_len, block_size, use_neox_style,
        dynamic_cachekv_quant, quant_round_type, quant_max_bound,
        quant_min_bound, out_scale, compute_dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fc(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "fc", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "fc", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "fc", 2);

    // Parse Attributes
    PyObject *in_num_col_dims_obj = PyTuple_GET_ITEM(args, 3);
    int in_num_col_dims = CastPyArg2Int(in_num_col_dims_obj, "fc", 3);
    PyObject *activation_type_obj = PyTuple_GET_ITEM(args, 4);
    std::string activation_type =
        CastPyArg2String(activation_type_obj, "fc", 4);
    PyObject *padding_weights_obj = PyTuple_GET_ITEM(args, 5);
    bool padding_weights = CastPyArg2Boolean(padding_weights_obj, "fc", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fc(input, w, bias, in_num_col_dims,
                                              activation_type, padding_weights);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_bias_act(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_bias_act op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_bias_act", 0);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 1);
    auto bias = CastPyArg2OptionalValue(bias_obj, "fused_bias_act", 1);
    PyObject *dequant_scales_obj = PyTuple_GET_ITEM(args, 2);
    auto dequant_scales =
        CastPyArg2OptionalValue(dequant_scales_obj, "fused_bias_act", 2);
    PyObject *shift_obj = PyTuple_GET_ITEM(args, 3);
    auto shift = CastPyArg2OptionalValue(shift_obj, "fused_bias_act", 3);
    PyObject *smooth_obj = PyTuple_GET_ITEM(args, 4);
    auto smooth = CastPyArg2OptionalValue(smooth_obj, "fused_bias_act", 4);

    // Parse Attributes
    PyObject *act_method_obj = PyTuple_GET_ITEM(args, 5);
    std::string act_method =
        CastPyArg2String(act_method_obj, "fused_bias_act", 5);
    PyObject *compute_dtype_obj = PyTuple_GET_ITEM(args, 6);
    std::string compute_dtype =
        CastPyArg2String(compute_dtype_obj, "fused_bias_act", 6);
    PyObject *quant_scale_obj = PyTuple_GET_ITEM(args, 7);
    float quant_scale = CastPyArg2Float(quant_scale_obj, "fused_bias_act", 7);
    PyObject *quant_round_type_obj = PyTuple_GET_ITEM(args, 8);
    int quant_round_type =
        CastPyArg2Int(quant_round_type_obj, "fused_bias_act", 8);
    PyObject *quant_max_bound_obj = PyTuple_GET_ITEM(args, 9);
    float quant_max_bound =
        CastPyArg2Float(quant_max_bound_obj, "fused_bias_act", 9);
    PyObject *quant_min_bound_obj = PyTuple_GET_ITEM(args, 10);
    float quant_min_bound =
        CastPyArg2Float(quant_min_bound_obj, "fused_bias_act", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_bias_act");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_bias_act(
        x, bias, dequant_scales, shift, smooth, act_method, compute_dtype,
        quant_scale, quant_round_type, quant_max_bound, quant_min_bound);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_bias_dropout_residual_layer_norm(PyObject *self,
                                                            PyObject *args,
                                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_bias_dropout_residual_layer_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x =
        CastPyArg2Value(x_obj, "fused_bias_dropout_residual_layer_norm", 0);
    PyObject *residual_obj = PyTuple_GET_ITEM(args, 1);
    auto residual = CastPyArg2Value(
        residual_obj, "fused_bias_dropout_residual_layer_norm", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(
        bias_obj, "fused_bias_dropout_residual_layer_norm", 2);
    PyObject *ln_scale_obj = PyTuple_GET_ITEM(args, 3);
    auto ln_scale = CastPyArg2OptionalValue(
        ln_scale_obj, "fused_bias_dropout_residual_layer_norm", 3);
    PyObject *ln_bias_obj = PyTuple_GET_ITEM(args, 4);
    auto ln_bias = CastPyArg2OptionalValue(
        ln_bias_obj, "fused_bias_dropout_residual_layer_norm", 4);

    // Parse Attributes
    PyObject *dropout_rate_obj = PyTuple_GET_ITEM(args, 5);
    float dropout_rate = CastPyArg2Float(
        dropout_rate_obj, "fused_bias_dropout_residual_layer_norm", 5);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 6);
    bool is_test = CastPyArg2Boolean(
        is_test_obj, "fused_bias_dropout_residual_layer_norm", 6);
    PyObject *dropout_fix_seed_obj = PyTuple_GET_ITEM(args, 7);
    bool dropout_fix_seed = CastPyArg2Boolean(
        dropout_fix_seed_obj, "fused_bias_dropout_residual_layer_norm", 7);
    PyObject *dropout_seed_obj = PyTuple_GET_ITEM(args, 8);
    int dropout_seed = CastPyArg2Int(
        dropout_seed_obj, "fused_bias_dropout_residual_layer_norm", 8);
    PyObject *dropout_implementation_obj = PyTuple_GET_ITEM(args, 9);
    std::string dropout_implementation =
        CastPyArg2String(dropout_implementation_obj,
                         "fused_bias_dropout_residual_layer_norm", 9);
    PyObject *ln_epsilon_obj = PyTuple_GET_ITEM(args, 10);
    float ln_epsilon = CastPyArg2Float(
        ln_epsilon_obj, "fused_bias_dropout_residual_layer_norm", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder(
        "fused_bias_dropout_residual_layer_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::fused_bias_dropout_residual_layer_norm(
            x, residual, bias, ln_scale, ln_bias, dropout_rate, is_test,
            dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_bias_residual_layernorm(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_bias_residual_layernorm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_bias_residual_layernorm", 0);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 1);
    auto bias =
        CastPyArg2OptionalValue(bias_obj, "fused_bias_residual_layernorm", 1);
    PyObject *residual_obj = PyTuple_GET_ITEM(args, 2);
    auto residual = CastPyArg2OptionalValue(residual_obj,
                                            "fused_bias_residual_layernorm", 2);
    PyObject *norm_weight_obj = PyTuple_GET_ITEM(args, 3);
    auto norm_weight = CastPyArg2OptionalValue(
        norm_weight_obj, "fused_bias_residual_layernorm", 3);
    PyObject *norm_bias_obj = PyTuple_GET_ITEM(args, 4);
    auto norm_bias = CastPyArg2OptionalValue(
        norm_bias_obj, "fused_bias_residual_layernorm", 4);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon =
        CastPyArg2Float(epsilon_obj, "fused_bias_residual_layernorm", 5);
    PyObject *residual_alpha_obj = PyTuple_GET_ITEM(args, 6);
    float residual_alpha =
        CastPyArg2Float(residual_alpha_obj, "fused_bias_residual_layernorm", 6);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 7);
    int begin_norm_axis =
        CastPyArg2Int(begin_norm_axis_obj, "fused_bias_residual_layernorm", 7);
    PyObject *quant_scale_obj = PyTuple_GET_ITEM(args, 8);
    float quant_scale =
        CastPyArg2Float(quant_scale_obj, "fused_bias_residual_layernorm", 8);
    PyObject *quant_round_type_obj = PyTuple_GET_ITEM(args, 9);
    int quant_round_type =
        CastPyArg2Int(quant_round_type_obj, "fused_bias_residual_layernorm", 9);
    PyObject *quant_max_bound_obj = PyTuple_GET_ITEM(args, 10);
    float quant_max_bound = CastPyArg2Float(
        quant_max_bound_obj, "fused_bias_residual_layernorm", 10);
    PyObject *quant_min_bound_obj = PyTuple_GET_ITEM(args, 11);
    float quant_min_bound = CastPyArg2Float(
        quant_min_bound_obj, "fused_bias_residual_layernorm", 11);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_bias_residual_layernorm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_bias_residual_layernorm(
        x, bias, residual, norm_weight, norm_bias, epsilon, residual_alpha,
        begin_norm_axis, quant_scale, quant_round_type, quant_max_bound,
        quant_min_bound);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_conv2d_add_act(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_conv2d_add_act op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "fused_conv2d_add_act", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "fused_conv2d_add_act", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2OptionalValue(bias_obj, "fused_conv2d_add_act", 2);
    PyObject *residual_data_obj = PyTuple_GET_ITEM(args, 3);
    auto residual_data =
        CastPyArg2OptionalValue(residual_data_obj, "fused_conv2d_add_act", 3);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "fused_conv2d_add_act", 4);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "fused_conv2d_add_act", 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "fused_conv2d_add_act", 6);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 7);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "fused_conv2d_add_act", 7);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 8);
    int groups = CastPyArg2Int(groups_obj, "fused_conv2d_add_act", 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "fused_conv2d_add_act", 9);
    PyObject *activation_obj = PyTuple_GET_ITEM(args, 10);
    std::string activation =
        CastPyArg2String(activation_obj, "fused_conv2d_add_act", 10);
    PyObject *split_channels_obj = PyTuple_GET_ITEM(args, 11);
    std::vector<int> split_channels =
        CastPyArg2Ints(split_channels_obj, "fused_conv2d_add_act", 11);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 12);
    bool exhaustive_search =
        CastPyArg2Boolean(exhaustive_search_obj, "fused_conv2d_add_act", 12);
    PyObject *workspace_size_MB_obj = PyTuple_GET_ITEM(args, 13);
    int workspace_size_MB =
        CastPyArg2Int(workspace_size_MB_obj, "fused_conv2d_add_act", 13);
    PyObject *fuse_alpha_obj = PyTuple_GET_ITEM(args, 14);
    float fuse_alpha =
        CastPyArg2Float(fuse_alpha_obj, "fused_conv2d_add_act", 14);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_conv2d_add_act");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_conv2d_add_act(
        input, filter, bias, residual_data, strides, paddings,
        padding_algorithm, dilations, groups, data_format, activation,
        split_channels, exhaustive_search, workspace_size_MB, fuse_alpha);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_dconv_drelu_dbn(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_dconv_drelu_dbn op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *grad_output_obj = PyTuple_GET_ITEM(args, 0);
    auto grad_output =
        CastPyArg2Value(grad_output_obj, "fused_dconv_drelu_dbn", 0);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 1);
    auto weight = CastPyArg2Value(weight_obj, "fused_dconv_drelu_dbn", 1);
    PyObject *grad_output_add_obj = PyTuple_GET_ITEM(args, 2);
    auto grad_output_add = CastPyArg2OptionalValue(grad_output_add_obj,
                                                   "fused_dconv_drelu_dbn", 2);
    PyObject *residual_input_obj = PyTuple_GET_ITEM(args, 3);
    auto residual_input =
        CastPyArg2OptionalValue(residual_input_obj, "fused_dconv_drelu_dbn", 3);
    PyObject *bn1_eqscale_obj = PyTuple_GET_ITEM(args, 4);
    auto bn1_eqscale =
        CastPyArg2OptionalValue(bn1_eqscale_obj, "fused_dconv_drelu_dbn", 4);
    PyObject *bn1_eqbias_obj = PyTuple_GET_ITEM(args, 5);
    auto bn1_eqbias =
        CastPyArg2OptionalValue(bn1_eqbias_obj, "fused_dconv_drelu_dbn", 5);
    PyObject *conv_input_obj = PyTuple_GET_ITEM(args, 6);
    auto conv_input =
        CastPyArg2OptionalValue(conv_input_obj, "fused_dconv_drelu_dbn", 6);
    PyObject *bn1_mean_obj = PyTuple_GET_ITEM(args, 7);
    auto bn1_mean = CastPyArg2Value(bn1_mean_obj, "fused_dconv_drelu_dbn", 7);
    PyObject *bn1_inv_std_obj = PyTuple_GET_ITEM(args, 8);
    auto bn1_inv_std =
        CastPyArg2Value(bn1_inv_std_obj, "fused_dconv_drelu_dbn", 8);
    PyObject *bn1_gamma_obj = PyTuple_GET_ITEM(args, 9);
    auto bn1_gamma = CastPyArg2Value(bn1_gamma_obj, "fused_dconv_drelu_dbn", 9);
    PyObject *bn1_beta_obj = PyTuple_GET_ITEM(args, 10);
    auto bn1_beta = CastPyArg2Value(bn1_beta_obj, "fused_dconv_drelu_dbn", 10);
    PyObject *bn1_input_obj = PyTuple_GET_ITEM(args, 11);
    auto bn1_input =
        CastPyArg2Value(bn1_input_obj, "fused_dconv_drelu_dbn", 11);
    PyObject *bn2_mean_obj = PyTuple_GET_ITEM(args, 12);
    auto bn2_mean =
        CastPyArg2OptionalValue(bn2_mean_obj, "fused_dconv_drelu_dbn", 12);
    PyObject *bn2_inv_std_obj = PyTuple_GET_ITEM(args, 13);
    auto bn2_inv_std =
        CastPyArg2OptionalValue(bn2_inv_std_obj, "fused_dconv_drelu_dbn", 13);
    PyObject *bn2_gamma_obj = PyTuple_GET_ITEM(args, 14);
    auto bn2_gamma =
        CastPyArg2OptionalValue(bn2_gamma_obj, "fused_dconv_drelu_dbn", 14);
    PyObject *bn2_beta_obj = PyTuple_GET_ITEM(args, 15);
    auto bn2_beta =
        CastPyArg2OptionalValue(bn2_beta_obj, "fused_dconv_drelu_dbn", 15);
    PyObject *bn2_input_obj = PyTuple_GET_ITEM(args, 16);
    auto bn2_input =
        CastPyArg2OptionalValue(bn2_input_obj, "fused_dconv_drelu_dbn", 16);

    // Parse Attributes
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 17);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "fused_dconv_drelu_dbn", 17);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 18);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "fused_dconv_drelu_dbn", 18);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 19);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "fused_dconv_drelu_dbn", 19);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 20);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "fused_dconv_drelu_dbn", 20);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 21);
    int groups = CastPyArg2Int(groups_obj, "fused_dconv_drelu_dbn", 21);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 22);
    std::string data_format =
        CastPyArg2String(data_format_obj, "fused_dconv_drelu_dbn", 22);
    PyObject *fuse_shortcut_obj = PyTuple_GET_ITEM(args, 23);
    bool fuse_shortcut =
        CastPyArg2Boolean(fuse_shortcut_obj, "fused_dconv_drelu_dbn", 23);
    PyObject *fuse_dual_obj = PyTuple_GET_ITEM(args, 24);
    bool fuse_dual =
        CastPyArg2Boolean(fuse_dual_obj, "fused_dconv_drelu_dbn", 24);
    PyObject *fuse_add_obj = PyTuple_GET_ITEM(args, 25);
    bool fuse_add =
        CastPyArg2Boolean(fuse_add_obj, "fused_dconv_drelu_dbn", 25);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 26);
    bool exhaustive_search =
        CastPyArg2Boolean(exhaustive_search_obj, "fused_dconv_drelu_dbn", 26);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_dconv_drelu_dbn");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_dconv_drelu_dbn(
        grad_output, weight, grad_output_add, residual_input, bn1_eqscale,
        bn1_eqbias, conv_input, bn1_mean, bn1_inv_std, bn1_gamma, bn1_beta,
        bn1_input, bn2_mean, bn2_inv_std, bn2_gamma, bn2_beta, bn2_input,
        paddings, dilations, strides, padding_algorithm, groups, data_format,
        fuse_shortcut, fuse_dual, fuse_add, exhaustive_search);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_dot_product_attention(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_dot_product_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *q_obj = PyTuple_GET_ITEM(args, 0);
    auto q = CastPyArg2Value(q_obj, "fused_dot_product_attention", 0);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    auto k = CastPyArg2Value(k_obj, "fused_dot_product_attention", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v = CastPyArg2Value(v_obj, "fused_dot_product_attention", 2);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 3);
    auto mask = CastPyArg2Value(mask_obj, "fused_dot_product_attention", 3);

    // Parse Attributes
    PyObject *scaling_factor_obj = PyTuple_GET_ITEM(args, 4);
    float scaling_factor =
        CastPyArg2Float(scaling_factor_obj, "fused_dot_product_attention", 4);
    PyObject *dropout_probability_obj = PyTuple_GET_ITEM(args, 5);
    float dropout_probability = CastPyArg2Float(
        dropout_probability_obj, "fused_dot_product_attention", 5);
    PyObject *is_training_obj = PyTuple_GET_ITEM(args, 6);
    bool is_training =
        CastPyArg2Boolean(is_training_obj, "fused_dot_product_attention", 6);
    PyObject *is_causal_masking_obj = PyTuple_GET_ITEM(args, 7);
    bool is_causal_masking = CastPyArg2Boolean(
        is_causal_masking_obj, "fused_dot_product_attention", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_dot_product_attention");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_dot_product_attention(
        q, k, v, mask, scaling_factor, dropout_probability, is_training,
        is_causal_masking);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_dropout_add(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_dropout_add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_dropout_add", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fused_dropout_add", 1);
    PyObject *seed_tensor_obj = PyTuple_GET_ITEM(args, 2);
    auto seed_tensor =
        CastPyArg2OptionalValue(seed_tensor_obj, "fused_dropout_add", 2);

    // Parse Attributes
    PyObject *p_obj = PyTuple_GET_ITEM(args, 3);
    float p = CastPyArg2Float(p_obj, "fused_dropout_add", 3);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 4);
    bool is_test = CastPyArg2Boolean(is_test_obj, "fused_dropout_add", 4);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 5);
    std::string mode = CastPyArg2String(mode_obj, "fused_dropout_add", 5);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 6);
    int seed = CastPyArg2Int(seed_obj, "fused_dropout_add", 6);
    PyObject *fix_seed_obj = PyTuple_GET_ITEM(args, 7);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "fused_dropout_add", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_dropout_add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_dropout_add(
        x, y, seed_tensor, p, is_test, mode, seed, fix_seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_embedding_eltwise_layernorm(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_embedding_eltwise_layernorm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj,
                                       "fused_embedding_eltwise_layernorm", 0);
    PyObject *embs_obj = PyTuple_GET_ITEM(args, 1);
    auto embs = CastPyArg2VectorOfValue(embs_obj,
                                        "fused_embedding_eltwise_layernorm", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias =
        CastPyArg2Value(bias_obj, "fused_embedding_eltwise_layernorm", 2);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 3);
    auto scale =
        CastPyArg2Value(scale_obj, "fused_embedding_eltwise_layernorm", 3);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 4);
    float epsilon =
        CastPyArg2Float(epsilon_obj, "fused_embedding_eltwise_layernorm", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_embedding_eltwise_layernorm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_embedding_eltwise_layernorm(
        ids, embs, bias, scale, epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_fc_elementwise_layernorm(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_fc_elementwise_layernorm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_fc_elementwise_layernorm", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "fused_fc_elementwise_layernorm", 1);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 2);
    auto y = CastPyArg2Value(y_obj, "fused_fc_elementwise_layernorm", 2);
    PyObject *bias0_obj = PyTuple_GET_ITEM(args, 3);
    auto bias0 =
        CastPyArg2OptionalValue(bias0_obj, "fused_fc_elementwise_layernorm", 3);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 4);
    auto scale =
        CastPyArg2OptionalValue(scale_obj, "fused_fc_elementwise_layernorm", 4);
    PyObject *bias1_obj = PyTuple_GET_ITEM(args, 5);
    auto bias1 =
        CastPyArg2OptionalValue(bias1_obj, "fused_fc_elementwise_layernorm", 5);

    // Parse Attributes
    PyObject *x_num_col_dims_obj = PyTuple_GET_ITEM(args, 6);
    int x_num_col_dims =
        CastPyArg2Int(x_num_col_dims_obj, "fused_fc_elementwise_layernorm", 6);
    PyObject *activation_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string activation_type = CastPyArg2String(
        activation_type_obj, "fused_fc_elementwise_layernorm", 7);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 8);
    float epsilon =
        CastPyArg2Float(epsilon_obj, "fused_fc_elementwise_layernorm", 8);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 9);
    int begin_norm_axis =
        CastPyArg2Int(begin_norm_axis_obj, "fused_fc_elementwise_layernorm", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_fc_elementwise_layernorm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_fc_elementwise_layernorm(
        x, w, y, bias0, scale, bias1, x_num_col_dims, activation_type, epsilon,
        begin_norm_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_linear_param_grad_add(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_linear_param_grad_add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_linear_param_grad_add", 0);
    PyObject *dout_obj = PyTuple_GET_ITEM(args, 1);
    auto dout = CastPyArg2Value(dout_obj, "fused_linear_param_grad_add", 1);
    PyObject *dweight_obj = PyTuple_GET_ITEM(args, 2);
    auto dweight =
        CastPyArg2OptionalValue(dweight_obj, "fused_linear_param_grad_add", 2);
    PyObject *dbias_obj = PyTuple_GET_ITEM(args, 3);
    auto dbias =
        CastPyArg2OptionalValue(dbias_obj, "fused_linear_param_grad_add", 3);

    // Parse Attributes
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 4);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj,
                                             "fused_linear_param_grad_add", 4);
    PyObject *has_bias_obj = PyTuple_GET_ITEM(args, 5);
    bool has_bias =
        CastPyArg2Boolean(has_bias_obj, "fused_linear_param_grad_add", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_linear_param_grad_add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_linear_param_grad_add(
        x, dout, dweight, dbias, multi_precision, has_bias);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_rotary_position_embedding(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_rotary_position_embedding op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *q_obj = PyTuple_GET_ITEM(args, 0);
    auto q = CastPyArg2Value(q_obj, "fused_rotary_position_embedding", 0);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    auto k =
        CastPyArg2OptionalValue(k_obj, "fused_rotary_position_embedding", 1);
    PyObject *v_obj = PyTuple_GET_ITEM(args, 2);
    auto v =
        CastPyArg2OptionalValue(v_obj, "fused_rotary_position_embedding", 2);
    PyObject *sin_obj = PyTuple_GET_ITEM(args, 3);
    auto sin =
        CastPyArg2OptionalValue(sin_obj, "fused_rotary_position_embedding", 3);
    PyObject *cos_obj = PyTuple_GET_ITEM(args, 4);
    auto cos =
        CastPyArg2OptionalValue(cos_obj, "fused_rotary_position_embedding", 4);
    PyObject *position_ids_obj = PyTuple_GET_ITEM(args, 5);
    auto position_ids = CastPyArg2OptionalValue(
        position_ids_obj, "fused_rotary_position_embedding", 5);

    // Parse Attributes
    PyObject *use_neox_rotary_style_obj = PyTuple_GET_ITEM(args, 6);
    bool use_neox_rotary_style = CastPyArg2Boolean(
        use_neox_rotary_style_obj, "fused_rotary_position_embedding", 6);
    PyObject *time_major_obj = PyTuple_GET_ITEM(args, 7);
    bool time_major =
        CastPyArg2Boolean(time_major_obj, "fused_rotary_position_embedding", 7);
    PyObject *rotary_emb_base_obj = PyTuple_GET_ITEM(args, 8);
    float rotary_emb_base = CastPyArg2Float(
        rotary_emb_base_obj, "fused_rotary_position_embedding", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_rotary_position_embedding");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_rotary_position_embedding(
        q, k, v, sin, cos, position_ids, use_neox_rotary_style, time_major,
        rotary_emb_base);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_scale_bias_add_relu(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_scale_bias_add_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x1_obj = PyTuple_GET_ITEM(args, 0);
    auto x1 = CastPyArg2Value(x1_obj, "fused_scale_bias_add_relu", 0);
    PyObject *scale1_obj = PyTuple_GET_ITEM(args, 1);
    auto scale1 = CastPyArg2Value(scale1_obj, "fused_scale_bias_add_relu", 1);
    PyObject *bias1_obj = PyTuple_GET_ITEM(args, 2);
    auto bias1 = CastPyArg2Value(bias1_obj, "fused_scale_bias_add_relu", 2);
    PyObject *x2_obj = PyTuple_GET_ITEM(args, 3);
    auto x2 = CastPyArg2Value(x2_obj, "fused_scale_bias_add_relu", 3);
    PyObject *scale2_obj = PyTuple_GET_ITEM(args, 4);
    auto scale2 =
        CastPyArg2OptionalValue(scale2_obj, "fused_scale_bias_add_relu", 4);
    PyObject *bias2_obj = PyTuple_GET_ITEM(args, 5);
    auto bias2 =
        CastPyArg2OptionalValue(bias2_obj, "fused_scale_bias_add_relu", 5);

    // Parse Attributes
    PyObject *fuse_dual_obj = PyTuple_GET_ITEM(args, 6);
    bool fuse_dual =
        CastPyArg2Boolean(fuse_dual_obj, "fused_scale_bias_add_relu", 6);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 7);
    bool exhaustive_search = CastPyArg2Boolean(exhaustive_search_obj,
                                               "fused_scale_bias_add_relu", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_scale_bias_add_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_scale_bias_add_relu(
        x1, scale1, bias1, x2, scale2, bias2, fuse_dual, exhaustive_search);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_scale_bias_relu_conv_bn(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_scale_bias_relu_conv_bn op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_scale_bias_relu_conv_bn", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "fused_scale_bias_relu_conv_bn", 1);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 2);
    auto scale =
        CastPyArg2OptionalValue(scale_obj, "fused_scale_bias_relu_conv_bn", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias =
        CastPyArg2OptionalValue(bias_obj, "fused_scale_bias_relu_conv_bn", 3);
    PyObject *bn_scale_obj = PyTuple_GET_ITEM(args, 4);
    auto bn_scale =
        CastPyArg2Value(bn_scale_obj, "fused_scale_bias_relu_conv_bn", 4);
    PyObject *bn_bias_obj = PyTuple_GET_ITEM(args, 5);
    auto bn_bias =
        CastPyArg2Value(bn_bias_obj, "fused_scale_bias_relu_conv_bn", 5);
    PyObject *input_running_mean_obj = PyTuple_GET_ITEM(args, 6);
    auto input_running_mean = CastPyArg2Value(
        input_running_mean_obj, "fused_scale_bias_relu_conv_bn", 6);
    PyObject *input_running_var_obj = PyTuple_GET_ITEM(args, 7);
    auto input_running_var = CastPyArg2Value(
        input_running_var_obj, "fused_scale_bias_relu_conv_bn", 7);

    // Parse Attributes
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "fused_scale_bias_relu_conv_bn", 8);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "fused_scale_bias_relu_conv_bn", 9);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 10);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "fused_scale_bias_relu_conv_bn", 10);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 11);
    std::string padding_algorithm = CastPyArg2String(
        padding_algorithm_obj, "fused_scale_bias_relu_conv_bn", 11);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 12);
    int groups = CastPyArg2Int(groups_obj, "fused_scale_bias_relu_conv_bn", 12);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 13);
    std::string data_format =
        CastPyArg2String(data_format_obj, "fused_scale_bias_relu_conv_bn", 13);
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 14);
    float momentum =
        CastPyArg2Float(momentum_obj, "fused_scale_bias_relu_conv_bn", 14);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 15);
    float epsilon =
        CastPyArg2Float(epsilon_obj, "fused_scale_bias_relu_conv_bn", 15);
    PyObject *fuse_prologue_obj = PyTuple_GET_ITEM(args, 16);
    bool fuse_prologue = CastPyArg2Boolean(fuse_prologue_obj,
                                           "fused_scale_bias_relu_conv_bn", 16);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 17);
    bool exhaustive_search = CastPyArg2Boolean(
        exhaustive_search_obj, "fused_scale_bias_relu_conv_bn", 17);
    PyObject *accumulation_count_obj = PyTuple_GET_ITEM(args, 18);
    int64_t accumulation_count = CastPyArg2Long(
        accumulation_count_obj, "fused_scale_bias_relu_conv_bn", 18);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_scale_bias_relu_conv_bn");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_scale_bias_relu_conv_bn(
        x, w, scale, bias, bn_scale, bn_bias, input_running_mean,
        input_running_var, paddings, dilations, strides, padding_algorithm,
        groups, data_format, momentum, epsilon, fuse_prologue,
        exhaustive_search, accumulation_count);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_gru(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_gru op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fusion_gru", 0);
    PyObject *h0_obj = PyTuple_GET_ITEM(args, 1);
    auto h0 = CastPyArg2OptionalValue(h0_obj, "fusion_gru", 1);
    PyObject *weight_x_obj = PyTuple_GET_ITEM(args, 2);
    auto weight_x = CastPyArg2Value(weight_x_obj, "fusion_gru", 2);
    PyObject *weight_h_obj = PyTuple_GET_ITEM(args, 3);
    auto weight_h = CastPyArg2Value(weight_h_obj, "fusion_gru", 3);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 4);
    auto bias = CastPyArg2OptionalValue(bias_obj, "fusion_gru", 4);

    // Parse Attributes
    PyObject *activation_obj = PyTuple_GET_ITEM(args, 5);
    std::string activation = CastPyArg2String(activation_obj, "fusion_gru", 5);
    PyObject *gate_activation_obj = PyTuple_GET_ITEM(args, 6);
    std::string gate_activation =
        CastPyArg2String(gate_activation_obj, "fusion_gru", 6);
    PyObject *is_reverse_obj = PyTuple_GET_ITEM(args, 7);
    bool is_reverse = CastPyArg2Boolean(is_reverse_obj, "fusion_gru", 7);
    PyObject *use_seq_obj = PyTuple_GET_ITEM(args, 8);
    bool use_seq = CastPyArg2Boolean(use_seq_obj, "fusion_gru", 8);
    PyObject *origin_mode_obj = PyTuple_GET_ITEM(args, 9);
    bool origin_mode = CastPyArg2Boolean(origin_mode_obj, "fusion_gru", 9);
    PyObject *force_fp32_output_obj = PyTuple_GET_ITEM(args, 10);
    bool force_fp32_output =
        CastPyArg2Boolean(force_fp32_output_obj, "fusion_gru", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_gru");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_gru(
        x, h0, weight_x, weight_h, bias, activation, gate_activation,
        is_reverse, use_seq, origin_mode, force_fp32_output);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_repeated_fc_relu(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_repeated_fc_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fusion_repeated_fc_relu", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2VectorOfValue(w_obj, "fusion_repeated_fc_relu", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2VectorOfValue(bias_obj, "fusion_repeated_fc_relu", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_repeated_fc_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_repeated_fc_relu(x, w, bias);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_seqconv_eltadd_relu(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_seqconv_eltadd_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fusion_seqconv_eltadd_relu", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "fusion_seqconv_eltadd_relu", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "fusion_seqconv_eltadd_relu", 2);

    // Parse Attributes
    PyObject *context_length_obj = PyTuple_GET_ITEM(args, 3);
    int context_length =
        CastPyArg2Int(context_length_obj, "fusion_seqconv_eltadd_relu", 3);
    PyObject *context_start_obj = PyTuple_GET_ITEM(args, 4);
    int context_start =
        CastPyArg2Int(context_start_obj, "fusion_seqconv_eltadd_relu", 4);
    PyObject *context_stride_obj = PyTuple_GET_ITEM(args, 5);
    int context_stride =
        CastPyArg2Int(context_stride_obj, "fusion_seqconv_eltadd_relu", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_seqconv_eltadd_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_seqconv_eltadd_relu(
        x, filter, bias, context_length, context_start, context_stride);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_seqexpand_concat_fc(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_seqexpand_concat_fc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "fusion_seqexpand_concat_fc", 0);
    PyObject *fc_weight_obj = PyTuple_GET_ITEM(args, 1);
    auto fc_weight =
        CastPyArg2Value(fc_weight_obj, "fusion_seqexpand_concat_fc", 1);
    PyObject *fc_bias_obj = PyTuple_GET_ITEM(args, 2);
    auto fc_bias =
        CastPyArg2OptionalValue(fc_bias_obj, "fusion_seqexpand_concat_fc", 2);

    // Parse Attributes
    PyObject *fc_activation_obj = PyTuple_GET_ITEM(args, 3);
    std::string fc_activation =
        CastPyArg2String(fc_activation_obj, "fusion_seqexpand_concat_fc", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_seqexpand_concat_fc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_seqexpand_concat_fc(
        x, fc_weight, fc_bias, fc_activation);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_squared_mat_sub(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_squared_mat_sub op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fusion_squared_mat_sub", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fusion_squared_mat_sub", 1);

    // Parse Attributes
    PyObject *scalar_obj = PyTuple_GET_ITEM(args, 2);
    float scalar = CastPyArg2Float(scalar_obj, "fusion_squared_mat_sub", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_squared_mat_sub");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_squared_mat_sub(x, y, scalar);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fusion_transpose_flatten_concat(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add fusion_transpose_flatten_concat op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x =
        CastPyArg2VectorOfValue(x_obj, "fusion_transpose_flatten_concat", 0);

    // Parse Attributes
    PyObject *trans_axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> trans_axis =
        CastPyArg2Ints(trans_axis_obj, "fusion_transpose_flatten_concat", 1);
    PyObject *flatten_axis_obj = PyTuple_GET_ITEM(args, 2);
    int flatten_axis =
        CastPyArg2Int(flatten_axis_obj, "fusion_transpose_flatten_concat", 2);
    PyObject *concat_axis_obj = PyTuple_GET_ITEM(args, 3);
    int concat_axis =
        CastPyArg2Int(concat_axis_obj, "fusion_transpose_flatten_concat", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("fusion_transpose_flatten_concat");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fusion_transpose_flatten_concat(
        x, trans_axis, flatten_axis, concat_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_max_pool2d_v2(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add max_pool2d_v2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "max_pool2d_v2", 0);

    // Parse Attributes
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "max_pool2d_v2", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "max_pool2d_v2", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "max_pool2d_v2", 3);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format =
        CastPyArg2String(data_format_obj, "max_pool2d_v2", 4);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 5);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "max_pool2d_v2", 5);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 6);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool2d_v2", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("max_pool2d_v2");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::max_pool2d_v2(x, kernel_size, strides, paddings,
                                       data_format, global_pooling, adaptive);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multihead_matmul(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add multihead_matmul op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "multihead_matmul", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "multihead_matmul", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "multihead_matmul", 2);
    PyObject *bias_qk_obj = PyTuple_GET_ITEM(args, 3);
    auto bias_qk = CastPyArg2OptionalValue(bias_qk_obj, "multihead_matmul", 3);

    // Parse Attributes
    PyObject *transpose_q_obj = PyTuple_GET_ITEM(args, 4);
    bool transpose_q =
        CastPyArg2Boolean(transpose_q_obj, "multihead_matmul", 4);
    PyObject *transpose_k_obj = PyTuple_GET_ITEM(args, 5);
    bool transpose_k =
        CastPyArg2Boolean(transpose_k_obj, "multihead_matmul", 5);
    PyObject *transpose_v_obj = PyTuple_GET_ITEM(args, 6);
    bool transpose_v =
        CastPyArg2Boolean(transpose_v_obj, "multihead_matmul", 6);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 7);
    float alpha = CastPyArg2Float(alpha_obj, "multihead_matmul", 7);
    PyObject *head_number_obj = PyTuple_GET_ITEM(args, 8);
    int head_number = CastPyArg2Int(head_number_obj, "multihead_matmul", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("multihead_matmul");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multihead_matmul(
        input, w, bias, bias_qk, transpose_q, transpose_k, transpose_v, alpha,
        head_number);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_self_dp_attention(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add self_dp_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "self_dp_attention", 0);

    // Parse Attributes
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "self_dp_attention", 1);
    PyObject *head_number_obj = PyTuple_GET_ITEM(args, 2);
    int head_number = CastPyArg2Int(head_number_obj, "self_dp_attention", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("self_dp_attention");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::self_dp_attention(x, alpha, head_number);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_skip_layernorm(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add skip_layernorm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "skip_layernorm", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "skip_layernorm", 1);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 2);
    auto scale = CastPyArg2Value(scale_obj, "skip_layernorm", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2Value(bias_obj, "skip_layernorm", 3);

    // Parse Attributes
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 4);
    float epsilon = CastPyArg2Float(epsilon_obj, "skip_layernorm", 4);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 5);
    int begin_norm_axis =
        CastPyArg2Int(begin_norm_axis_obj, "skip_layernorm", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("skip_layernorm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::skip_layernorm(
        x, y, scale, bias, epsilon, begin_norm_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_squeeze_excitation_block(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add squeeze_excitation_block op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "squeeze_excitation_block", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "squeeze_excitation_block", 1);
    PyObject *filter_max_obj = PyTuple_GET_ITEM(args, 2);
    auto filter_max =
        CastPyArg2Value(filter_max_obj, "squeeze_excitation_block", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias =
        CastPyArg2OptionalValue(bias_obj, "squeeze_excitation_block", 3);
    PyObject *branch_obj = PyTuple_GET_ITEM(args, 4);
    auto branch =
        CastPyArg2OptionalValue(branch_obj, "squeeze_excitation_block", 4);

    // Parse Attributes
    PyObject *act_type_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> act_type =
        CastPyArg2Ints(act_type_obj, "squeeze_excitation_block", 5);
    PyObject *act_param_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<float> act_param =
        CastPyArg2Floats(act_param_obj, "squeeze_excitation_block", 6);
    PyObject *filter_dims_obj = PyTuple_GET_ITEM(args, 7);
    std::vector<int> filter_dims =
        CastPyArg2Ints(filter_dims_obj, "squeeze_excitation_block", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("squeeze_excitation_block");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::squeeze_excitation_block(
        x, filter, filter_max, bias, branch, act_type, act_param, filter_dims);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_variable_length_memory_efficient_attention(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add variable_length_memory_efficient_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *query_obj = PyTuple_GET_ITEM(args, 0);
    auto query = CastPyArg2Value(
        query_obj, "variable_length_memory_efficient_attention", 0);
    PyObject *key_obj = PyTuple_GET_ITEM(args, 1);
    auto key = CastPyArg2Value(key_obj,
                               "variable_length_memory_efficient_attention", 1);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 2);
    auto value = CastPyArg2Value(
        value_obj, "variable_length_memory_efficient_attention", 2);
    PyObject *seq_lens_obj = PyTuple_GET_ITEM(args, 3);
    auto seq_lens = CastPyArg2Value(
        seq_lens_obj, "variable_length_memory_efficient_attention", 3);
    PyObject *kv_seq_lens_obj = PyTuple_GET_ITEM(args, 4);
    auto kv_seq_lens = CastPyArg2Value(
        kv_seq_lens_obj, "variable_length_memory_efficient_attention", 4);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 5);
    auto mask = CastPyArg2OptionalValue(
        mask_obj, "variable_length_memory_efficient_attention", 5);

    // Parse Attributes
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 6);
    float scale = CastPyArg2Float(
        scale_obj, "variable_length_memory_efficient_attention", 6);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 7);
    bool causal = CastPyArg2Boolean(
        causal_obj, "variable_length_memory_efficient_attention", 7);
    PyObject *pre_cache_length_obj = PyTuple_GET_ITEM(args, 8);
    int pre_cache_length = CastPyArg2Int(
        pre_cache_length_obj, "variable_length_memory_efficient_attention", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder(
        "variable_length_memory_efficient_attention");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::variable_length_memory_efficient_attention(
            query, key, value, seq_lens, kv_seq_lens, mask, scale, causal,
            pre_cache_length);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_adadelta_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add adadelta_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "adadelta_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "adadelta_", 1);
    PyObject *avg_squared_grad_obj = PyTuple_GET_ITEM(args, 2);
    auto avg_squared_grad =
        CastPyArg2Value(avg_squared_grad_obj, "adadelta_", 2);
    PyObject *avg_squared_update_obj = PyTuple_GET_ITEM(args, 3);
    auto avg_squared_update =
        CastPyArg2Value(avg_squared_update_obj, "adadelta_", 3);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 4);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "adadelta_", 4);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 5);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "adadelta_", 5);

    // Parse Attributes
    PyObject *rho_obj = PyTuple_GET_ITEM(args, 6);
    float rho = CastPyArg2Float(rho_obj, "adadelta_", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "adadelta_", 7);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 8);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "adadelta_", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("adadelta_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::adadelta_(
        param, grad, avg_squared_grad, avg_squared_update, learning_rate,
        master_param, rho, epsilon, multi_precision);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add add op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "add", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "add", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("add");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::add(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_add_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add add_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "add_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "add_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("add_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::add_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_add_n(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add add_n op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *inputs_obj = PyTuple_GET_ITEM(args, 0);
    auto inputs = CastPyArg2VectorOfValue(inputs_obj, "add_n", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("add_n");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::add_n(inputs);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add all op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "all", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "all", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "all", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("all");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::all(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_all_reduce(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add all_reduce op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "all_reduce", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "all_reduce", 1);
    PyObject *reduce_type_obj = PyTuple_GET_ITEM(args, 2);
    int reduce_type = CastPyArg2Int(reduce_type_obj, "all_reduce", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("all_reduce");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::all_reduce(x, ring_id, reduce_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_all_reduce_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add all_reduce_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "all_reduce_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "all_reduce_", 1);
    PyObject *reduce_type_obj = PyTuple_GET_ITEM(args, 2);
    int reduce_type = CastPyArg2Int(reduce_type_obj, "all_reduce_", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("all_reduce_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::all_reduce_(x, ring_id, reduce_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_amax(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add amax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "amax", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "amax", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "amax", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("amax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::amax(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_amin(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add amin op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "amin", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "amin", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "amin", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("amin");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::amin(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_any(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add any op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "any", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "any", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "any", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("any");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::any(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "assign", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("assign");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::assign(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "assign_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("assign_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::assign_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign_out_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign_out_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "assign_out_", 0);
    PyObject *output_obj = PyTuple_GET_ITEM(args, 1);
    auto output = CastPyArg2Value(output_obj, "assign_out_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("assign_out_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::assign_out_(x, output);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign_pos(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign_pos op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "assign_pos", 0);
    PyObject *cum_count_obj = PyTuple_GET_ITEM(args, 1);
    auto cum_count = CastPyArg2Value(cum_count_obj, "assign_pos", 1);
    PyObject *eff_num_len_obj = PyTuple_GET_ITEM(args, 2);
    auto eff_num_len = CastPyArg2Value(eff_num_len_obj, "assign_pos", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("assign_pos");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::assign_pos(x, cum_count, eff_num_len);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign_value(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign_value op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int> shape = CastPyArg2Ints(shape_obj, "assign_value", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "assign_value", 1);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<phi::Scalar> values =
        CastPyArg2ScalarArray(values_obj, "assign_value", 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);
    Place place = CastPyArg2Place(place_obj, "assign_value", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("assign_value");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::assign_value(shape, dtype, values, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_assign_value_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add assign_value_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *output_obj = PyTuple_GET_ITEM(args, 0);
    auto output = CastPyArg2Value(output_obj, "assign_value_", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape = CastPyArg2Ints(shape_obj, "assign_value_", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "assign_value_", 2);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<phi::Scalar> values =
        CastPyArg2ScalarArray(values_obj, "assign_value_", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    Place place = CastPyArg2Place(place_obj, "assign_value_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("assign_value_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::assign_value_(output, shape, dtype, values, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_batch_fc(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add batch_fc op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "batch_fc", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "batch_fc", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "batch_fc", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("batch_fc");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::batch_fc(input, w, bias);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_batch_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add batch_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "batch_norm", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    auto mean = CastPyArg2Value(mean_obj, "batch_norm", 1);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 2);
    auto variance = CastPyArg2Value(variance_obj, "batch_norm", 2);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 3);
    auto scale = CastPyArg2OptionalValue(scale_obj, "batch_norm", 3);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 4);
    auto bias = CastPyArg2OptionalValue(bias_obj, "batch_norm", 4);

    // Parse Attributes
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 5);
    bool is_test = CastPyArg2Boolean(is_test_obj, "batch_norm", 5);
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 6);
    float momentum = CastPyArg2Float(momentum_obj, "batch_norm", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "batch_norm", 7);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 8);
    std::string data_format =
        CastPyArg2String(data_format_obj, "batch_norm", 8);
    PyObject *use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats =
        CastPyArg2Boolean(use_global_stats_obj, "batch_norm", 9);
    PyObject *trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics =
        CastPyArg2Boolean(trainable_statistics_obj, "batch_norm", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("batch_norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::batch_norm(
        x, mean, variance, scale, bias, is_test, momentum, epsilon, data_format,
        use_global_stats, trainable_statistics);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_batch_norm_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add batch_norm_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "batch_norm_", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    auto mean = CastPyArg2Value(mean_obj, "batch_norm_", 1);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 2);
    auto variance = CastPyArg2Value(variance_obj, "batch_norm_", 2);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 3);
    auto scale = CastPyArg2OptionalValue(scale_obj, "batch_norm_", 3);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 4);
    auto bias = CastPyArg2OptionalValue(bias_obj, "batch_norm_", 4);

    // Parse Attributes
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 5);
    bool is_test = CastPyArg2Boolean(is_test_obj, "batch_norm_", 5);
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 6);
    float momentum = CastPyArg2Float(momentum_obj, "batch_norm_", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "batch_norm_", 7);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 8);
    std::string data_format =
        CastPyArg2String(data_format_obj, "batch_norm_", 8);
    PyObject *use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats =
        CastPyArg2Boolean(use_global_stats_obj, "batch_norm_", 9);
    PyObject *trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics =
        CastPyArg2Boolean(trainable_statistics_obj, "batch_norm_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("batch_norm_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::batch_norm_(
        x, mean, variance, scale, bias, is_test, momentum, epsilon, data_format,
        use_global_stats, trainable_statistics);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allgather(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allgather op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allgather", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allgather", 1);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 2);
    int nranks = CastPyArg2Int(nranks_obj, "c_allgather", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allgather", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allgather");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_allgather(x, ring_id, nranks, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_avg(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_avg op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_avg", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_avg", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_avg", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_avg", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_avg");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_avg(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_avg_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_avg_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_avg_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_avg_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_avg_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_avg_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_avg_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_avg_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_max(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_max op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_max", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_max", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_max", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_max", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_max");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_max(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_max_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_max_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_max_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_max_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_max_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_max_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_max_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_max_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_min(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_min op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_min", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_min", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_min", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_min", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_min");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_min(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_min_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_min_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_min_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_min_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_min_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_min_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_min_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_min_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_prod(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_prod op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_prod", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_prod", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_prod", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_prod", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_prod");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_prod(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_prod_(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_prod_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_prod_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_prod_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_prod_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_prod_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_prod_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_prod_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_sum(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_sum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_sum", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_sum", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_sum", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_sum", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_sum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_sum(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_allreduce_sum_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_allreduce_sum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_allreduce_sum_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_allreduce_sum_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_allreduce_sum_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_allreduce_sum_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_allreduce_sum_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_allreduce_sum_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_broadcast(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_broadcast op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_broadcast", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_broadcast", 1);
    PyObject *root_obj = PyTuple_GET_ITEM(args, 2);
    int root = CastPyArg2Int(root_obj, "c_broadcast", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_broadcast", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_broadcast");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_broadcast(x, ring_id, root, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_broadcast_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_broadcast_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_broadcast_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_broadcast_", 1);
    PyObject *root_obj = PyTuple_GET_ITEM(args, 2);
    int root = CastPyArg2Int(root_obj, "c_broadcast_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_broadcast_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_broadcast_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_broadcast_(x, ring_id, root, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_concat(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_concat op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_concat", 0);

    // Parse Attributes
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 1);
    int rank = CastPyArg2Int(rank_obj, "c_concat", 1);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 2);
    int nranks = CastPyArg2Int(nranks_obj, "c_concat", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_concat", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_concat", 4);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 5);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_concat", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_concat");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_concat(
        x, rank, nranks, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_embedding(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_embedding op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 0);
    auto weight = CastPyArg2Value(weight_obj, "c_embedding", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "c_embedding", 1);

    // Parse Attributes
    PyObject *start_index_obj = PyTuple_GET_ITEM(args, 2);
    int64_t start_index = CastPyArg2Long(start_index_obj, "c_embedding", 2);
    PyObject *vocab_size_obj = PyTuple_GET_ITEM(args, 3);
    int64_t vocab_size = CastPyArg2Long(vocab_size_obj, "c_embedding", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_embedding");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_embedding(weight, x, start_index, vocab_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_identity(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_identity op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_identity", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_identity", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_identity", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_identity", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_identity");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_identity(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_identity_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_identity_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_identity_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_identity_", 1);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 2);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_identity_", 2);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 3);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_identity_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_identity_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_identity_(
        x, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_avg(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_avg op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_avg", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_avg", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_avg", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_avg", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_avg");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_avg(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_avg_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_avg_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_avg_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_avg_", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_avg_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_avg_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_avg_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_avg_(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_max(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_max op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_max", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_max", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_max", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_max", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_max");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_max(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_max_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_max_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_max_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_max_", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_max_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_max_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_max_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_max_(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_min(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_min op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_min", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_min", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_min", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_min", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_min");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_min(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_min_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_min_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_min_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_min_", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_min_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_min_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_min_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_min_(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_prod(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_prod op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_prod", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_prod", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_prod", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_prod", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_prod");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_prod(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_prod_(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_prod_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_prod_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_prod_", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_prod_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_prod_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_prod_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_prod_(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_sum(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_sum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_sum", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_sum", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_sum", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_sum", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_sum");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_sum(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reduce_sum_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reduce_sum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reduce_sum_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reduce_sum_", 1);
    PyObject *root_id_obj = PyTuple_GET_ITEM(args, 2);
    int root_id = CastPyArg2Int(root_id_obj, "c_reduce_sum_", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reduce_sum_", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reduce_sum_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reduce_sum_(x, ring_id, root_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_reducescatter(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_reducescatter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_reducescatter", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_reducescatter", 1);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 2);
    int nranks = CastPyArg2Int(nranks_obj, "c_reducescatter", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_reducescatter", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_reducescatter");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_reducescatter(x, ring_id, nranks, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_scatter(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_scatter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_scatter", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_scatter", 1);
    PyObject *root_obj = PyTuple_GET_ITEM(args, 2);
    int root = CastPyArg2Int(root_obj, "c_scatter", 2);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 3);
    int nranks = CastPyArg2Int(nranks_obj, "c_scatter", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "c_scatter", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_scatter");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::c_scatter(x, ring_id, root, nranks, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_split(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_split op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_split", 0);

    // Parse Attributes
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 1);
    int rank = CastPyArg2Int(rank_obj, "c_split", 1);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 2);
    int nranks = CastPyArg2Int(nranks_obj, "c_split", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_split", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream = CastPyArg2Boolean(use_calc_stream_obj, "c_split", 4);
    PyObject *use_model_parallel_obj = PyTuple_GET_ITEM(args, 5);
    bool use_model_parallel =
        CastPyArg2Boolean(use_model_parallel_obj, "c_split", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_split");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_split(
        x, rank, nranks, ring_id, use_calc_stream, use_model_parallel);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_sync_calc_stream(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_sync_calc_stream op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_sync_calc_stream", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("c_sync_calc_stream");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_sync_calc_stream(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_sync_calc_stream_(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_sync_calc_stream_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_sync_calc_stream_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("c_sync_calc_stream_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_sync_calc_stream_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_sync_comm_stream(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_sync_comm_stream op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_sync_comm_stream", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_sync_comm_stream", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_sync_comm_stream");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_sync_comm_stream(x, ring_id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_sync_comm_stream_(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_sync_comm_stream_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "c_sync_comm_stream_", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_sync_comm_stream_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_sync_comm_stream_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_sync_comm_stream_(x, ring_id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cast(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cast op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cast", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "cast", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("cast");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cast(x, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_cast_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add cast_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "cast_", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "cast_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("cast_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::cast_(x, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_channel_shuffle(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add channel_shuffle op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "channel_shuffle", 0);

    // Parse Attributes
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 1);
    int groups = CastPyArg2Int(groups_obj, "channel_shuffle", 1);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format =
        CastPyArg2String(data_format_obj, "channel_shuffle", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("channel_shuffle");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::channel_shuffle(x, groups, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_coalesce_tensor_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add coalesce_tensor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2VectorOfValue(input_obj, "coalesce_tensor_", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "coalesce_tensor_", 1);
    PyObject *copy_data_obj = PyTuple_GET_ITEM(args, 2);
    bool copy_data = CastPyArg2Boolean(copy_data_obj, "coalesce_tensor_", 2);
    PyObject *set_constant_obj = PyTuple_GET_ITEM(args, 3);
    bool set_constant =
        CastPyArg2Boolean(set_constant_obj, "coalesce_tensor_", 3);
    PyObject *persist_output_obj = PyTuple_GET_ITEM(args, 4);
    bool persist_output =
        CastPyArg2Boolean(persist_output_obj, "coalesce_tensor_", 4);
    PyObject *constant_obj = PyTuple_GET_ITEM(args, 5);
    float constant = CastPyArg2Float(constant_obj, "coalesce_tensor_", 5);
    PyObject *use_align_obj = PyTuple_GET_ITEM(args, 6);
    bool use_align = CastPyArg2Boolean(use_align_obj, "coalesce_tensor_", 6);
    PyObject *align_size_obj = PyTuple_GET_ITEM(args, 7);
    int align_size = CastPyArg2Int(align_size_obj, "coalesce_tensor_", 7);
    PyObject *size_of_dtype_obj = PyTuple_GET_ITEM(args, 8);
    int size_of_dtype = CastPyArg2Int(size_of_dtype_obj, "coalesce_tensor_", 8);
    PyObject *concated_shapes_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<int64_t> concated_shapes =
        CastPyArg2Longs(concated_shapes_obj, "coalesce_tensor_", 9);
    PyObject *concated_ranks_obj = PyTuple_GET_ITEM(args, 10);
    std::vector<int64_t> concated_ranks =
        CastPyArg2Longs(concated_ranks_obj, "coalesce_tensor_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("coalesce_tensor_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::coalesce_tensor_(
        input, dtype, copy_data, set_constant, persist_output, constant,
        use_align, align_size, size_of_dtype, concated_shapes, concated_ranks);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conv2d_transpose(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add conv2d_transpose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "conv2d_transpose", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "conv2d_transpose", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);

    // Check for mutable attrs
    pir::Value output_size;

    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "conv2d_transpose", 2);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "conv2d_transpose", 3);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "conv2d_transpose", 4);
    if (PyObject_CheckIRValue(output_size_obj)) {
      output_size = CastPyArg2Value(output_size_obj, "conv2d_transpose", 5);
    } else if (PyObject_CheckIRVectorOfValue(output_size_obj)) {
      std::vector<pir::Value> output_size_tmp =
          CastPyArg2VectorOfValue(output_size_obj, "conv2d_transpose", 5);
      output_size = paddle::dialect::stack(output_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> output_size_tmp =
          CastPyArg2Longs(output_size_obj, "conv2d_transpose", 5);
      output_size = paddle::dialect::full_int_array(
          output_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv2d_transpose", 6);
    int groups = CastPyArg2Int(groups_obj, "conv2d_transpose", 7);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "conv2d_transpose", 8);
    std::string data_format =
        CastPyArg2String(data_format_obj, "conv2d_transpose", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("conv2d_transpose");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conv2d_transpose(
        x, filter, output_size, strides, paddings, output_padding,
        padding_algorithm, groups, dilations, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_conv2d_transpose_bias(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add conv2d_transpose_bias op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "conv2d_transpose_bias", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "conv2d_transpose_bias", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "conv2d_transpose_bias", 2);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 10);

    // Check for mutable attrs
    pir::Value output_size;

    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "conv2d_transpose_bias", 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "conv2d_transpose_bias", 4);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "conv2d_transpose_bias", 5);
    if (PyObject_CheckIRValue(output_size_obj)) {
      output_size =
          CastPyArg2Value(output_size_obj, "conv2d_transpose_bias", 6);
    } else if (PyObject_CheckIRVectorOfValue(output_size_obj)) {
      std::vector<pir::Value> output_size_tmp =
          CastPyArg2VectorOfValue(output_size_obj, "conv2d_transpose_bias", 6);
      output_size = paddle::dialect::stack(output_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> output_size_tmp =
          CastPyArg2Longs(output_size_obj, "conv2d_transpose_bias", 6);
      output_size = paddle::dialect::full_int_array(
          output_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv2d_transpose_bias", 7);
    int groups = CastPyArg2Int(groups_obj, "conv2d_transpose_bias", 8);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "conv2d_transpose_bias", 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "conv2d_transpose_bias", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("conv2d_transpose_bias");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::conv2d_transpose_bias(
        x, filter, bias, output_size, strides, paddings, output_padding,
        padding_algorithm, groups, dilations, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_decayed_adagrad(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add decayed_adagrad op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "decayed_adagrad", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "decayed_adagrad", 1);
    PyObject *moment_obj = PyTuple_GET_ITEM(args, 2);
    auto moment = CastPyArg2Value(moment_obj, "decayed_adagrad", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate =
        CastPyArg2Value(learning_rate_obj, "decayed_adagrad", 3);

    // Parse Attributes
    PyObject *decay_obj = PyTuple_GET_ITEM(args, 4);
    float decay = CastPyArg2Float(decay_obj, "decayed_adagrad", 4);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon = CastPyArg2Float(epsilon_obj, "decayed_adagrad", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("decayed_adagrad");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::decayed_adagrad(
        param, grad, moment, learning_rate, decay, epsilon);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_decode_jpeg(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add decode_jpeg op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "decode_jpeg", 0);

    // Parse Attributes
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 1);
    std::string mode = CastPyArg2String(mode_obj, "decode_jpeg", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    Place place = CastPyArg2Place(place_obj, "decode_jpeg", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("decode_jpeg");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::decode_jpeg(x, mode, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_deformable_conv(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add deformable_conv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "deformable_conv", 0);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    auto offset = CastPyArg2Value(offset_obj, "deformable_conv", 1);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 2);
    auto filter = CastPyArg2Value(filter_obj, "deformable_conv", 2);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 3);
    auto mask = CastPyArg2OptionalValue(mask_obj, "deformable_conv", 3);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "deformable_conv", 4);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "deformable_conv", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "deformable_conv", 6);
    PyObject *deformable_groups_obj = PyTuple_GET_ITEM(args, 7);
    int deformable_groups =
        CastPyArg2Int(deformable_groups_obj, "deformable_conv", 7);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 8);
    int groups = CastPyArg2Int(groups_obj, "deformable_conv", 8);
    PyObject *im2col_step_obj = PyTuple_GET_ITEM(args, 9);
    int im2col_step = CastPyArg2Int(im2col_step_obj, "deformable_conv", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("deformable_conv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::deformable_conv(
        x, offset, filter, mask, strides, paddings, dilations,
        deformable_groups, groups, im2col_step);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_depthwise_conv2d_transpose(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add depthwise_conv2d_transpose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "depthwise_conv2d_transpose", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "depthwise_conv2d_transpose", 1);

    // Parse Attributes
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);

    // Check for mutable attrs
    pir::Value output_size;

    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "depthwise_conv2d_transpose", 2);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "depthwise_conv2d_transpose", 3);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "depthwise_conv2d_transpose", 4);
    if (PyObject_CheckIRValue(output_size_obj)) {
      output_size =
          CastPyArg2Value(output_size_obj, "depthwise_conv2d_transpose", 5);
    } else if (PyObject_CheckIRVectorOfValue(output_size_obj)) {
      std::vector<pir::Value> output_size_tmp = CastPyArg2VectorOfValue(
          output_size_obj, "depthwise_conv2d_transpose", 5);
      output_size = paddle::dialect::stack(output_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> output_size_tmp =
          CastPyArg2Longs(output_size_obj, "depthwise_conv2d_transpose", 5);
      output_size = paddle::dialect::full_int_array(
          output_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::string padding_algorithm = CastPyArg2String(
        padding_algorithm_obj, "depthwise_conv2d_transpose", 6);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d_transpose", 7);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "depthwise_conv2d_transpose", 8);
    std::string data_format =
        CastPyArg2String(data_format_obj, "depthwise_conv2d_transpose", 9);

    // Call ir static api
    CallStackRecorder callstack_recorder("depthwise_conv2d_transpose");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::depthwise_conv2d_transpose(
        x, filter, output_size, strides, paddings, output_padding,
        padding_algorithm, groups, dilations, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dequantize_linear(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add dequantize_linear op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "dequantize_linear", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "dequantize_linear", 1);
    PyObject *zero_point_obj = PyTuple_GET_ITEM(args, 2);
    auto zero_point = CastPyArg2Value(zero_point_obj, "dequantize_linear", 2);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 3);
    auto in_accum =
        CastPyArg2OptionalValue(in_accum_obj, "dequantize_linear", 3);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 4);
    auto in_state =
        CastPyArg2OptionalValue(in_state_obj, "dequantize_linear", 4);

    // Parse Attributes
    PyObject *quant_axis_obj = PyTuple_GET_ITEM(args, 5);
    int quant_axis = CastPyArg2Int(quant_axis_obj, "dequantize_linear", 5);
    PyObject *bit_length_obj = PyTuple_GET_ITEM(args, 6);
    int bit_length = CastPyArg2Int(bit_length_obj, "dequantize_linear", 6);
    PyObject *round_type_obj = PyTuple_GET_ITEM(args, 7);
    int round_type = CastPyArg2Int(round_type_obj, "dequantize_linear", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "dequantize_linear", 8);
    PyObject *only_observer_obj = PyTuple_GET_ITEM(args, 9);
    bool only_observer =
        CastPyArg2Boolean(only_observer_obj, "dequantize_linear", 9);
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 10);
    float moving_rate =
        CastPyArg2Float(moving_rate_obj, "dequantize_linear", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("dequantize_linear");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dequantize_linear(
        x, scale, zero_point, in_accum, in_state, quant_axis, bit_length,
        round_type, is_test, only_observer, moving_rate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dequantize_linear_(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add dequantize_linear_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "dequantize_linear_", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "dequantize_linear_", 1);
    PyObject *zero_point_obj = PyTuple_GET_ITEM(args, 2);
    auto zero_point = CastPyArg2Value(zero_point_obj, "dequantize_linear_", 2);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 3);
    auto in_accum =
        CastPyArg2OptionalValue(in_accum_obj, "dequantize_linear_", 3);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 4);
    auto in_state =
        CastPyArg2OptionalValue(in_state_obj, "dequantize_linear_", 4);

    // Parse Attributes
    PyObject *quant_axis_obj = PyTuple_GET_ITEM(args, 5);
    int quant_axis = CastPyArg2Int(quant_axis_obj, "dequantize_linear_", 5);
    PyObject *bit_length_obj = PyTuple_GET_ITEM(args, 6);
    int bit_length = CastPyArg2Int(bit_length_obj, "dequantize_linear_", 6);
    PyObject *round_type_obj = PyTuple_GET_ITEM(args, 7);
    int round_type = CastPyArg2Int(round_type_obj, "dequantize_linear_", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "dequantize_linear_", 8);
    PyObject *only_observer_obj = PyTuple_GET_ITEM(args, 9);
    bool only_observer =
        CastPyArg2Boolean(only_observer_obj, "dequantize_linear_", 9);
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 10);
    float moving_rate =
        CastPyArg2Float(moving_rate_obj, "dequantize_linear_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("dequantize_linear_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dequantize_linear_(
        x, scale, zero_point, in_accum, in_state, quant_axis, bit_length,
        round_type, is_test, only_observer, moving_rate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dgc_momentum(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add dgc_momentum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "dgc_momentum", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "dgc_momentum", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity = CastPyArg2Value(velocity_obj, "dgc_momentum", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "dgc_momentum", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "dgc_momentum", 4);
    PyObject *current_step_tensor_obj = PyTuple_GET_ITEM(args, 5);
    auto current_step_tensor =
        CastPyArg2Value(current_step_tensor_obj, "dgc_momentum", 5);
    PyObject *nranks_tensor_obj = PyTuple_GET_ITEM(args, 6);
    auto nranks_tensor = CastPyArg2Value(nranks_tensor_obj, "dgc_momentum", 6);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 7);
    float mu = CastPyArg2Float(mu_obj, "dgc_momentum", 7);
    PyObject *use_nesterov_obj = PyTuple_GET_ITEM(args, 8);
    bool use_nesterov = CastPyArg2Boolean(use_nesterov_obj, "dgc_momentum", 8);
    PyObject *regularization_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string regularization_method =
        CastPyArg2String(regularization_method_obj, "dgc_momentum", 9);
    PyObject *regularization_coeff_obj = PyTuple_GET_ITEM(args, 10);
    float regularization_coeff =
        CastPyArg2Float(regularization_coeff_obj, "dgc_momentum", 10);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 11);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "dgc_momentum", 11);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 12);
    float rescale_grad = CastPyArg2Float(rescale_grad_obj, "dgc_momentum", 12);
    PyObject *rampup_begin_step_obj = PyTuple_GET_ITEM(args, 13);
    float rampup_begin_step =
        CastPyArg2Float(rampup_begin_step_obj, "dgc_momentum", 13);

    // Call ir static api
    CallStackRecorder callstack_recorder("dgc_momentum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dgc_momentum(
        param, grad, velocity, learning_rate, master_param, current_step_tensor,
        nranks_tensor, mu, use_nesterov, regularization_method,
        regularization_coeff, multi_precision, rescale_grad, rampup_begin_step);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_disable_check_model_nan_inf(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add disable_check_model_nan_inf op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "disable_check_model_nan_inf", 0);

    // Parse Attributes
    PyObject *flag_obj = PyTuple_GET_ITEM(args, 1);
    int flag = CastPyArg2Int(flag_obj, "disable_check_model_nan_inf", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("disable_check_model_nan_inf");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::disable_check_model_nan_inf(x, flag);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_distribute_fpn_proposals(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add distribute_fpn_proposals op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *fpn_rois_obj = PyTuple_GET_ITEM(args, 0);
    auto fpn_rois =
        CastPyArg2Value(fpn_rois_obj, "distribute_fpn_proposals", 0);
    PyObject *rois_num_obj = PyTuple_GET_ITEM(args, 1);
    auto rois_num =
        CastPyArg2OptionalValue(rois_num_obj, "distribute_fpn_proposals", 1);

    // Parse Attributes
    PyObject *min_level_obj = PyTuple_GET_ITEM(args, 2);
    int min_level = CastPyArg2Int(min_level_obj, "distribute_fpn_proposals", 2);
    PyObject *max_level_obj = PyTuple_GET_ITEM(args, 3);
    int max_level = CastPyArg2Int(max_level_obj, "distribute_fpn_proposals", 3);
    PyObject *refer_level_obj = PyTuple_GET_ITEM(args, 4);
    int refer_level =
        CastPyArg2Int(refer_level_obj, "distribute_fpn_proposals", 4);
    PyObject *refer_scale_obj = PyTuple_GET_ITEM(args, 5);
    int refer_scale =
        CastPyArg2Int(refer_scale_obj, "distribute_fpn_proposals", 5);
    PyObject *pixel_offset_obj = PyTuple_GET_ITEM(args, 6);
    bool pixel_offset =
        CastPyArg2Boolean(pixel_offset_obj, "distribute_fpn_proposals", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("distribute_fpn_proposals");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::distribute_fpn_proposals(
        fpn_rois, rois_num, min_level, max_level, refer_level, refer_scale,
        pixel_offset);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_distributed_fused_lamb_init(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add distributed_fused_lamb_init op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param =
        CastPyArg2VectorOfValue(param_obj, "distributed_fused_lamb_init", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad =
        CastPyArg2VectorOfValue(grad_obj, "distributed_fused_lamb_init", 1);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 2);
    float beta1 = CastPyArg2Float(beta1_obj, "distributed_fused_lamb_init", 2);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 3);
    float beta2 = CastPyArg2Float(beta2_obj, "distributed_fused_lamb_init", 3);
    PyObject *apply_weight_decay_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> apply_weight_decay = CastPyArg2Ints(
        apply_weight_decay_obj, "distributed_fused_lamb_init", 4);
    PyObject *alignment_obj = PyTuple_GET_ITEM(args, 5);
    int alignment =
        CastPyArg2Int(alignment_obj, "distributed_fused_lamb_init", 5);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 6);
    int rank = CastPyArg2Int(rank_obj, "distributed_fused_lamb_init", 6);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 7);
    int nranks = CastPyArg2Int(nranks_obj, "distributed_fused_lamb_init", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("distributed_fused_lamb_init");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::distributed_fused_lamb_init(
        param, grad, beta1, beta2, apply_weight_decay, alignment, rank, nranks);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_distributed_fused_lamb_init_(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add distributed_fused_lamb_init_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param =
        CastPyArg2VectorOfValue(param_obj, "distributed_fused_lamb_init_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad =
        CastPyArg2VectorOfValue(grad_obj, "distributed_fused_lamb_init_", 1);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 2);
    float beta1 = CastPyArg2Float(beta1_obj, "distributed_fused_lamb_init_", 2);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 3);
    float beta2 = CastPyArg2Float(beta2_obj, "distributed_fused_lamb_init_", 3);
    PyObject *apply_weight_decay_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> apply_weight_decay = CastPyArg2Ints(
        apply_weight_decay_obj, "distributed_fused_lamb_init_", 4);
    PyObject *alignment_obj = PyTuple_GET_ITEM(args, 5);
    int alignment =
        CastPyArg2Int(alignment_obj, "distributed_fused_lamb_init_", 5);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 6);
    int rank = CastPyArg2Int(rank_obj, "distributed_fused_lamb_init_", 6);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 7);
    int nranks = CastPyArg2Int(nranks_obj, "distributed_fused_lamb_init_", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("distributed_fused_lamb_init_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::distributed_fused_lamb_init_(
        param, grad, beta1, beta2, apply_weight_decay, alignment, rank, nranks);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_distributed_lookup_table(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add distributed_lookup_table op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj, "distributed_lookup_table", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2Value(w_obj, "distributed_lookup_table", 1);

    // Parse Attributes
    PyObject *table_id_obj = PyTuple_GET_ITEM(args, 2);
    int table_id = CastPyArg2Int(table_id_obj, "distributed_lookup_table", 2);
    PyObject *is_distributed_obj = PyTuple_GET_ITEM(args, 3);
    bool is_distributed =
        CastPyArg2Boolean(is_distributed_obj, "distributed_lookup_table", 3);
    PyObject *lookup_table_version_obj = PyTuple_GET_ITEM(args, 4);
    std::string lookup_table_version = CastPyArg2String(
        lookup_table_version_obj, "distributed_lookup_table", 4);
    PyObject *padding_idx_obj = PyTuple_GET_ITEM(args, 5);
    int64_t padding_idx =
        CastPyArg2Long(padding_idx_obj, "distributed_lookup_table", 5);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 6);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "distributed_lookup_table", 6);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 7);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "distributed_lookup_table", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("distributed_lookup_table");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::distributed_lookup_table(
        ids, w, table_id, is_distributed, lookup_table_version, padding_idx,
        dtype, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_distributed_push_sparse(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add distributed_push_sparse op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj, "distributed_push_sparse", 0);
    PyObject *shows_obj = PyTuple_GET_ITEM(args, 1);
    auto shows =
        CastPyArg2VectorOfValue(shows_obj, "distributed_push_sparse", 1);
    PyObject *clicks_obj = PyTuple_GET_ITEM(args, 2);
    auto clicks =
        CastPyArg2VectorOfValue(clicks_obj, "distributed_push_sparse", 2);

    // Parse Attributes
    PyObject *table_id_obj = PyTuple_GET_ITEM(args, 3);
    int table_id = CastPyArg2Int(table_id_obj, "distributed_push_sparse", 3);
    PyObject *size_obj = PyTuple_GET_ITEM(args, 4);
    int size = CastPyArg2Int(size_obj, "distributed_push_sparse", 4);
    PyObject *is_distributed_obj = PyTuple_GET_ITEM(args, 5);
    bool is_distributed =
        CastPyArg2Boolean(is_distributed_obj, "distributed_push_sparse", 5);
    PyObject *push_sparse_version_obj = PyTuple_GET_ITEM(args, 6);
    std::string push_sparse_version =
        CastPyArg2String(push_sparse_version_obj, "distributed_push_sparse", 6);
    PyObject *padding_idx_obj = PyTuple_GET_ITEM(args, 7);
    int64_t padding_idx =
        CastPyArg2Long(padding_idx_obj, "distributed_push_sparse", 7);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 8);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "distributed_push_sparse", 8);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 9);
    bool is_test = CastPyArg2Boolean(is_test_obj, "distributed_push_sparse", 9);
    PyObject *use_cvm_op_obj = PyTuple_GET_ITEM(args, 10);
    bool use_cvm_op =
        CastPyArg2Boolean(use_cvm_op_obj, "distributed_push_sparse", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("distributed_push_sparse");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::distributed_push_sparse(
        ids, shows, clicks, table_id, size, is_distributed, push_sparse_version,
        padding_idx, dtype, is_test, use_cvm_op);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add divide op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "divide", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "divide", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("divide");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::divide(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_divide_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add divide_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "divide_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "divide_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("divide_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::divide_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dropout(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add dropout op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "dropout", 0);
    PyObject *seed_tensor_obj = PyTuple_GET_ITEM(args, 1);
    auto seed_tensor = CastPyArg2OptionalValue(seed_tensor_obj, "dropout", 1);

    // Parse Attributes
    PyObject *p_obj = PyTuple_GET_ITEM(args, 2);
    float p = CastPyArg2Float(p_obj, "dropout", 2);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "dropout", 3);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 4);
    std::string mode = CastPyArg2String(mode_obj, "dropout", 4);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 5);
    int seed = CastPyArg2Int(seed_obj, "dropout", 5);
    PyObject *fix_seed_obj = PyTuple_GET_ITEM(args, 6);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "dropout", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("dropout");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dropout(x, seed_tensor, p, is_test,
                                                   mode, seed, fix_seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_einsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add einsum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "einsum", 0);

    // Parse Attributes
    PyObject *equation_obj = PyTuple_GET_ITEM(args, 1);
    std::string equation = CastPyArg2String(equation_obj, "einsum", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("einsum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::einsum(x, equation);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_elementwise_pow(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add elementwise_pow op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "elementwise_pow", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "elementwise_pow", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("elementwise_pow");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::elementwise_pow(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_embedding(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add embedding op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "embedding", 0);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 1);
    auto weight = CastPyArg2Value(weight_obj, "embedding", 1);

    // Parse Attributes
    PyObject *padding_idx_obj = PyTuple_GET_ITEM(args, 2);
    int64_t padding_idx = CastPyArg2Long(padding_idx_obj, "embedding", 2);
    PyObject *sparse_obj = PyTuple_GET_ITEM(args, 3);
    bool sparse = CastPyArg2Boolean(sparse_obj, "embedding", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("embedding");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::embedding(x, weight, padding_idx, sparse);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add empty op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "empty", 0);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "empty", 0);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "empty", 0);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "empty", 1);
    Place place = CastPyArg2Place(place_obj, "empty", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("empty");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::empty(shape, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_empty_like(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add empty_like op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "empty_like", 0);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "empty_like", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    Place place = CastPyArg2Place(place_obj, "empty_like", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("empty_like");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::empty_like(x, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_enable_check_model_nan_inf(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add enable_check_model_nan_inf op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "enable_check_model_nan_inf", 0);

    // Parse Attributes
    PyObject *flag_obj = PyTuple_GET_ITEM(args, 1);
    int flag = CastPyArg2Int(flag_obj, "enable_check_model_nan_inf", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("enable_check_model_nan_inf");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::enable_check_model_nan_inf(x, flag);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add equal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "equal", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "equal", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("equal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::equal(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_equal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add equal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "equal_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "equal_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("equal_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::equal_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_exponential_(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add exponential_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "exponential_", 0);

    // Parse Attributes
    PyObject *lam_obj = PyTuple_GET_ITEM(args, 1);
    float lam = CastPyArg2Float(lam_obj, "exponential_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("exponential_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::exponential_(x, lam);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_eye(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add eye op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *num_rows_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *num_columns_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value num_rows;

    pir::Value num_columns;

    if (PyObject_CheckIRValue(num_rows_obj)) {
      num_rows = CastPyArg2Value(num_rows_obj, "eye", 0);
    } else {
      float num_rows_tmp = CastPyArg2Float(num_rows_obj, "eye", 0);
      num_rows = paddle::dialect::full(std::vector<int64_t>{1}, num_rows_tmp,
                                       phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(num_columns_obj)) {
      num_columns = CastPyArg2Value(num_columns_obj, "eye", 1);
    } else {
      float num_columns_tmp = CastPyArg2Float(num_columns_obj, "eye", 1);
      num_columns =
          paddle::dialect::full(std::vector<int64_t>{1}, num_columns_tmp,
                                phi::DataType::FLOAT32, phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "eye", 2);
    Place place = CastPyArg2Place(place_obj, "eye", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("eye");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::eye(num_rows, num_columns, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fetch(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add fetch op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fetch", 0);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 1);
    std::string name = CastPyArg2String(name_obj, "fetch", 1);
    PyObject *col_obj = PyTuple_GET_ITEM(args, 2);
    int col = CastPyArg2Int(col_obj, "fetch", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("fetch");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fetch(x, name, col);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_floor_divide(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add floor_divide op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "floor_divide", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "floor_divide", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("floor_divide");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::floor_divide(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_floor_divide_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add floor_divide_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "floor_divide_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "floor_divide_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("floor_divide_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::floor_divide_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_frobenius_norm(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add frobenius_norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "frobenius_norm", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "frobenius_norm", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "frobenius_norm", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp =
          CastPyArg2Longs(axis_obj, "frobenius_norm", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "frobenius_norm", 2);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "frobenius_norm", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("frobenius_norm");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::frobenius_norm(x, axis, keep_dim, reduce_all);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add full_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *output_obj = PyTuple_GET_ITEM(args, 0);
    auto output = CastPyArg2Value(output_obj, "full_", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "full_", 1);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 2);
    float value = CastPyArg2Float(value_obj, "full_", 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "full_", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    Place place = CastPyArg2Place(place_obj, "full_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("full_");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::full_(output, shape, value, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full_batch_size_like(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add full_batch_size_like op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "full_batch_size_like", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape =
        CastPyArg2Ints(shape_obj, "full_batch_size_like", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "full_batch_size_like", 2);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 3);
    float value = CastPyArg2Float(value_obj, "full_batch_size_like", 3);
    PyObject *input_dim_idx_obj = PyTuple_GET_ITEM(args, 4);
    int input_dim_idx =
        CastPyArg2Int(input_dim_idx_obj, "full_batch_size_like", 4);
    PyObject *output_dim_idx_obj = PyTuple_GET_ITEM(args, 5);
    int output_dim_idx =
        CastPyArg2Int(output_dim_idx_obj, "full_batch_size_like", 5);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 6);
    Place place = CastPyArg2Place(place_obj, "full_batch_size_like", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("full_batch_size_like");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::full_batch_size_like(
        input, shape, dtype, value, input_dim_idx, output_dim_idx, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full_like(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add full_like op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "full_like", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value value;

    if (PyObject_CheckIRValue(value_obj)) {
      value = CastPyArg2Value(value_obj, "full_like", 1);
    } else {
      float value_tmp = CastPyArg2Float(value_obj, "full_like", 1);
      value = paddle::dialect::full(std::vector<int64_t>{1}, value_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "full_like", 2);
    Place place = CastPyArg2Place(place_obj, "full_like", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("full_like");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::full_like(x, value, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full_with_tensor(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add full_with_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *value_obj = PyTuple_GET_ITEM(args, 0);
    auto value = CastPyArg2Value(value_obj, "full_with_tensor", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "full_with_tensor", 1);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "full_with_tensor", 1);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp =
          CastPyArg2Longs(shape_obj, "full_with_tensor", 1);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "full_with_tensor", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("full_with_tensor");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::full_with_tensor(value, shape, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_adam_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_adam_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *params_obj = PyTuple_GET_ITEM(args, 0);
    auto params = CastPyArg2VectorOfValue(params_obj, "fused_adam_", 0);
    PyObject *grads_obj = PyTuple_GET_ITEM(args, 1);
    auto grads = CastPyArg2VectorOfValue(grads_obj, "fused_adam_", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "fused_adam_", 2);
    PyObject *moments1_obj = PyTuple_GET_ITEM(args, 3);
    auto moments1 = CastPyArg2VectorOfValue(moments1_obj, "fused_adam_", 3);
    PyObject *moments2_obj = PyTuple_GET_ITEM(args, 4);
    auto moments2 = CastPyArg2VectorOfValue(moments2_obj, "fused_adam_", 4);
    PyObject *beta1_pows_obj = PyTuple_GET_ITEM(args, 5);
    auto beta1_pows = CastPyArg2VectorOfValue(beta1_pows_obj, "fused_adam_", 5);
    PyObject *beta2_pows_obj = PyTuple_GET_ITEM(args, 6);
    auto beta2_pows = CastPyArg2VectorOfValue(beta2_pows_obj, "fused_adam_", 6);
    PyObject *master_params_obj = PyTuple_GET_ITEM(args, 7);
    auto master_params =
        CastPyArg2OptionalVectorOfValue(master_params_obj, "fused_adam_", 7);
    PyObject *skip_update_obj = PyTuple_GET_ITEM(args, 8);
    auto skip_update =
        CastPyArg2OptionalValue(skip_update_obj, "fused_adam_", 8);

    // Parse Attributes
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 9);
    float beta1 = CastPyArg2Float(beta1_obj, "fused_adam_", 9);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 10);
    float beta2 = CastPyArg2Float(beta2_obj, "fused_adam_", 10);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 11);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_adam_", 11);
    PyObject *chunk_size_obj = PyTuple_GET_ITEM(args, 12);
    int chunk_size = CastPyArg2Int(chunk_size_obj, "fused_adam_", 12);
    PyObject *weight_decay_obj = PyTuple_GET_ITEM(args, 13);
    float weight_decay = CastPyArg2Float(weight_decay_obj, "fused_adam_", 13);
    PyObject *use_adamw_obj = PyTuple_GET_ITEM(args, 14);
    bool use_adamw = CastPyArg2Boolean(use_adamw_obj, "fused_adam_", 14);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 15);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "fused_adam_", 15);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 16);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "fused_adam_", 16);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_adam_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_adam_(
        params, grads, learning_rate, moments1, moments2, beta1_pows,
        beta2_pows, master_params, skip_update, beta1, beta2, epsilon,
        chunk_size, weight_decay, use_adamw, multi_precision,
        use_global_beta_pow);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_batch_norm_act(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_batch_norm_act op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_batch_norm_act", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "fused_batch_norm_act", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "fused_batch_norm_act", 2);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 3);
    auto mean = CastPyArg2Value(mean_obj, "fused_batch_norm_act", 3);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 4);
    auto variance = CastPyArg2Value(variance_obj, "fused_batch_norm_act", 4);

    // Parse Attributes
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 5);
    float momentum = CastPyArg2Float(momentum_obj, "fused_batch_norm_act", 5);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_batch_norm_act", 6);
    PyObject *act_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string act_type =
        CastPyArg2String(act_type_obj, "fused_batch_norm_act", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_batch_norm_act");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_batch_norm_act(
        x, scale, bias, mean, variance, momentum, epsilon, act_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_batch_norm_act_(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_batch_norm_act_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_batch_norm_act_", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "fused_batch_norm_act_", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    auto bias = CastPyArg2Value(bias_obj, "fused_batch_norm_act_", 2);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 3);
    auto mean = CastPyArg2Value(mean_obj, "fused_batch_norm_act_", 3);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 4);
    auto variance = CastPyArg2Value(variance_obj, "fused_batch_norm_act_", 4);

    // Parse Attributes
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 5);
    float momentum = CastPyArg2Float(momentum_obj, "fused_batch_norm_act_", 5);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_batch_norm_act_", 6);
    PyObject *act_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string act_type =
        CastPyArg2String(act_type_obj, "fused_batch_norm_act_", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_batch_norm_act_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_batch_norm_act_(
        x, scale, bias, mean, variance, momentum, epsilon, act_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_bn_add_activation(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_bn_add_activation op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_bn_add_activation", 0);
    PyObject *z_obj = PyTuple_GET_ITEM(args, 1);
    auto z = CastPyArg2Value(z_obj, "fused_bn_add_activation", 1);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 2);
    auto scale = CastPyArg2Value(scale_obj, "fused_bn_add_activation", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2Value(bias_obj, "fused_bn_add_activation", 3);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 4);
    auto mean = CastPyArg2Value(mean_obj, "fused_bn_add_activation", 4);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 5);
    auto variance = CastPyArg2Value(variance_obj, "fused_bn_add_activation", 5);

    // Parse Attributes
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 6);
    float momentum =
        CastPyArg2Float(momentum_obj, "fused_bn_add_activation", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_bn_add_activation", 7);
    PyObject *act_type_obj = PyTuple_GET_ITEM(args, 8);
    std::string act_type =
        CastPyArg2String(act_type_obj, "fused_bn_add_activation", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_bn_add_activation");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_bn_add_activation(
        x, z, scale, bias, mean, variance, momentum, epsilon, act_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_bn_add_activation_(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_bn_add_activation_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_bn_add_activation_", 0);
    PyObject *z_obj = PyTuple_GET_ITEM(args, 1);
    auto z = CastPyArg2Value(z_obj, "fused_bn_add_activation_", 1);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 2);
    auto scale = CastPyArg2Value(scale_obj, "fused_bn_add_activation_", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2Value(bias_obj, "fused_bn_add_activation_", 3);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 4);
    auto mean = CastPyArg2Value(mean_obj, "fused_bn_add_activation_", 4);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 5);
    auto variance =
        CastPyArg2Value(variance_obj, "fused_bn_add_activation_", 5);

    // Parse Attributes
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 6);
    float momentum =
        CastPyArg2Float(momentum_obj, "fused_bn_add_activation_", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_bn_add_activation_", 7);
    PyObject *act_type_obj = PyTuple_GET_ITEM(args, 8);
    std::string act_type =
        CastPyArg2String(act_type_obj, "fused_bn_add_activation_", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_bn_add_activation_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_bn_add_activation_(
        x, z, scale, bias, mean, variance, momentum, epsilon, act_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_multi_transformer(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_multi_transformer op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_multi_transformer", 0);
    PyObject *ln_scales_obj = PyTuple_GET_ITEM(args, 1);
    auto ln_scales =
        CastPyArg2VectorOfValue(ln_scales_obj, "fused_multi_transformer", 1);
    PyObject *ln_biases_obj = PyTuple_GET_ITEM(args, 2);
    auto ln_biases =
        CastPyArg2VectorOfValue(ln_biases_obj, "fused_multi_transformer", 2);
    PyObject *qkv_weights_obj = PyTuple_GET_ITEM(args, 3);
    auto qkv_weights =
        CastPyArg2VectorOfValue(qkv_weights_obj, "fused_multi_transformer", 3);
    PyObject *qkv_biases_obj = PyTuple_GET_ITEM(args, 4);
    auto qkv_biases = CastPyArg2OptionalVectorOfValue(
        qkv_biases_obj, "fused_multi_transformer", 4);
    PyObject *cache_kvs_obj = PyTuple_GET_ITEM(args, 5);
    auto cache_kvs = CastPyArg2OptionalVectorOfValue(
        cache_kvs_obj, "fused_multi_transformer", 5);
    PyObject *pre_caches_obj = PyTuple_GET_ITEM(args, 6);
    auto pre_caches = CastPyArg2OptionalVectorOfValue(
        pre_caches_obj, "fused_multi_transformer", 6);
    PyObject *rotary_tensor_obj = PyTuple_GET_ITEM(args, 7);
    auto rotary_tensor = CastPyArg2OptionalValue(rotary_tensor_obj,
                                                 "fused_multi_transformer", 7);
    PyObject *time_step_obj = PyTuple_GET_ITEM(args, 8);
    auto time_step =
        CastPyArg2OptionalValue(time_step_obj, "fused_multi_transformer", 8);
    PyObject *seq_lengths_obj = PyTuple_GET_ITEM(args, 9);
    auto seq_lengths =
        CastPyArg2OptionalValue(seq_lengths_obj, "fused_multi_transformer", 9);
    PyObject *src_mask_obj = PyTuple_GET_ITEM(args, 10);
    auto src_mask =
        CastPyArg2OptionalValue(src_mask_obj, "fused_multi_transformer", 10);
    PyObject *out_linear_weights_obj = PyTuple_GET_ITEM(args, 11);
    auto out_linear_weights = CastPyArg2VectorOfValue(
        out_linear_weights_obj, "fused_multi_transformer", 11);
    PyObject *out_linear_biases_obj = PyTuple_GET_ITEM(args, 12);
    auto out_linear_biases = CastPyArg2OptionalVectorOfValue(
        out_linear_biases_obj, "fused_multi_transformer", 12);
    PyObject *ffn_ln_scales_obj = PyTuple_GET_ITEM(args, 13);
    auto ffn_ln_scales = CastPyArg2VectorOfValue(ffn_ln_scales_obj,
                                                 "fused_multi_transformer", 13);
    PyObject *ffn_ln_biases_obj = PyTuple_GET_ITEM(args, 14);
    auto ffn_ln_biases = CastPyArg2VectorOfValue(ffn_ln_biases_obj,
                                                 "fused_multi_transformer", 14);
    PyObject *ffn1_weights_obj = PyTuple_GET_ITEM(args, 15);
    auto ffn1_weights = CastPyArg2VectorOfValue(ffn1_weights_obj,
                                                "fused_multi_transformer", 15);
    PyObject *ffn1_biases_obj = PyTuple_GET_ITEM(args, 16);
    auto ffn1_biases = CastPyArg2OptionalVectorOfValue(
        ffn1_biases_obj, "fused_multi_transformer", 16);
    PyObject *ffn2_weights_obj = PyTuple_GET_ITEM(args, 17);
    auto ffn2_weights = CastPyArg2VectorOfValue(ffn2_weights_obj,
                                                "fused_multi_transformer", 17);
    PyObject *ffn2_biases_obj = PyTuple_GET_ITEM(args, 18);
    auto ffn2_biases = CastPyArg2OptionalVectorOfValue(
        ffn2_biases_obj, "fused_multi_transformer", 18);

    // Parse Attributes
    PyObject *pre_layer_norm_obj = PyTuple_GET_ITEM(args, 19);
    bool pre_layer_norm =
        CastPyArg2Boolean(pre_layer_norm_obj, "fused_multi_transformer", 19);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 20);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_multi_transformer", 20);
    PyObject *dropout_rate_obj = PyTuple_GET_ITEM(args, 21);
    float dropout_rate =
        CastPyArg2Float(dropout_rate_obj, "fused_multi_transformer", 21);
    PyObject *rotary_emb_dims_obj = PyTuple_GET_ITEM(args, 22);
    int rotary_emb_dims =
        CastPyArg2Int(rotary_emb_dims_obj, "fused_multi_transformer", 22);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 23);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "fused_multi_transformer", 23);
    PyObject *dropout_implementation_obj = PyTuple_GET_ITEM(args, 24);
    std::string dropout_implementation = CastPyArg2String(
        dropout_implementation_obj, "fused_multi_transformer", 24);
    PyObject *act_method_obj = PyTuple_GET_ITEM(args, 25);
    std::string act_method =
        CastPyArg2String(act_method_obj, "fused_multi_transformer", 25);
    PyObject *trans_qkvw_obj = PyTuple_GET_ITEM(args, 26);
    bool trans_qkvw =
        CastPyArg2Boolean(trans_qkvw_obj, "fused_multi_transformer", 26);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 27);
    int ring_id = CastPyArg2Int(ring_id_obj, "fused_multi_transformer", 27);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_multi_transformer");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_multi_transformer(
        x, ln_scales, ln_biases, qkv_weights, qkv_biases, cache_kvs, pre_caches,
        rotary_tensor, time_step, seq_lengths, src_mask, out_linear_weights,
        out_linear_biases, ffn_ln_scales, ffn_ln_biases, ffn1_weights,
        ffn1_biases, ffn2_weights, ffn2_biases, pre_layer_norm, epsilon,
        dropout_rate, rotary_emb_dims, is_test, dropout_implementation,
        act_method, trans_qkvw, ring_id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_softmax_mask(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_softmax_mask op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_softmax_mask", 0);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 1);
    auto mask = CastPyArg2Value(mask_obj, "fused_softmax_mask", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_softmax_mask");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_softmax_mask(x, mask);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_softmax_mask_upper_triangle(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_softmax_mask_upper_triangle op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *X_obj = PyTuple_GET_ITEM(args, 0);
    auto X = CastPyArg2Value(X_obj, "fused_softmax_mask_upper_triangle", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_softmax_mask_upper_triangle");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_softmax_mask_upper_triangle(X);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_token_prune(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_token_prune op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *attn_obj = PyTuple_GET_ITEM(args, 0);
    auto attn = CastPyArg2Value(attn_obj, "fused_token_prune", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "fused_token_prune", 1);
    PyObject *mask_obj = PyTuple_GET_ITEM(args, 2);
    auto mask = CastPyArg2Value(mask_obj, "fused_token_prune", 2);
    PyObject *new_mask_obj = PyTuple_GET_ITEM(args, 3);
    auto new_mask = CastPyArg2Value(new_mask_obj, "fused_token_prune", 3);

    // Parse Attributes
    PyObject *keep_first_token_obj = PyTuple_GET_ITEM(args, 4);
    bool keep_first_token =
        CastPyArg2Boolean(keep_first_token_obj, "fused_token_prune", 4);
    PyObject *keep_order_obj = PyTuple_GET_ITEM(args, 5);
    bool keep_order = CastPyArg2Boolean(keep_order_obj, "fused_token_prune", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_token_prune");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_token_prune(
        attn, x, mask, new_mask, keep_first_token, keep_order);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_gaussian(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add gaussian op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "gaussian", 0);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "gaussian", 0);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp =
          CastPyArg2Longs(shape_obj, "gaussian", 0);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    float mean = CastPyArg2Float(mean_obj, "gaussian", 1);
    float std = CastPyArg2Float(std_obj, "gaussian", 2);
    int seed = CastPyArg2Int(seed_obj, "gaussian", 3);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "gaussian", 4);
    Place place = CastPyArg2Place(place_obj, "gaussian", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("gaussian");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::gaussian(shape, mean, std, seed, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_get_tensor_from_selected_rows(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add get_tensor_from_selected_rows op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "get_tensor_from_selected_rows", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("get_tensor_from_selected_rows");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::get_tensor_from_selected_rows(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_global_scatter(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add global_scatter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "global_scatter", 0);
    PyObject *local_count_obj = PyTuple_GET_ITEM(args, 1);
    auto local_count = CastPyArg2Value(local_count_obj, "global_scatter", 1);
    PyObject *global_count_obj = PyTuple_GET_ITEM(args, 2);
    auto global_count = CastPyArg2Value(global_count_obj, "global_scatter", 2);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "global_scatter", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "global_scatter", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("global_scatter");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::global_scatter(
        x, local_count, global_count, ring_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_greater_equal(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add greater_equal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "greater_equal", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "greater_equal", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("greater_equal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::greater_equal(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_greater_equal_(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add greater_equal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "greater_equal_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "greater_equal_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("greater_equal_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::greater_equal_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_greater_than(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add greater_than op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "greater_than", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "greater_than", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("greater_than");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::greater_than(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_greater_than_(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add greater_than_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "greater_than_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "greater_than_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("greater_than_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::greater_than_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hardswish(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add hardswish op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hardswish", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("hardswish");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hardswish(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_hsigmoid_loss(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add hsigmoid_loss op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "hsigmoid_loss", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "hsigmoid_loss", 1);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 2);
    auto w = CastPyArg2Value(w_obj, "hsigmoid_loss", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2OptionalValue(bias_obj, "hsigmoid_loss", 3);
    PyObject *path_obj = PyTuple_GET_ITEM(args, 4);
    auto path = CastPyArg2OptionalValue(path_obj, "hsigmoid_loss", 4);
    PyObject *code_obj = PyTuple_GET_ITEM(args, 5);
    auto code = CastPyArg2OptionalValue(code_obj, "hsigmoid_loss", 5);

    // Parse Attributes
    PyObject *num_classes_obj = PyTuple_GET_ITEM(args, 6);
    int num_classes = CastPyArg2Int(num_classes_obj, "hsigmoid_loss", 6);
    PyObject *is_sparse_obj = PyTuple_GET_ITEM(args, 7);
    bool is_sparse = CastPyArg2Boolean(is_sparse_obj, "hsigmoid_loss", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("hsigmoid_loss");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::hsigmoid_loss(
        x, label, w, bias, path, code, num_classes, is_sparse);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_increment(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add increment op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "increment", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "increment", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("increment");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::increment(x, value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_increment_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add increment_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "increment_", 0);

    // Parse Attributes
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "increment_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("increment_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::increment_(x, value);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_less_equal(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add less_equal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "less_equal", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "less_equal", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("less_equal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::less_equal(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_less_equal_(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add less_equal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "less_equal_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "less_equal_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("less_equal_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::less_equal_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_less_than(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add less_than op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "less_than", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "less_than", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("less_than");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::less_than(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_less_than_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add less_than_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "less_than_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "less_than_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("less_than_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::less_than_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_limit_by_capacity(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add limit_by_capacity op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *expert_count_obj = PyTuple_GET_ITEM(args, 0);
    auto expert_count =
        CastPyArg2Value(expert_count_obj, "limit_by_capacity", 0);
    PyObject *capacity_obj = PyTuple_GET_ITEM(args, 1);
    auto capacity = CastPyArg2Value(capacity_obj, "limit_by_capacity", 1);

    // Parse Attributes
    PyObject *n_worker_obj = PyTuple_GET_ITEM(args, 2);
    int n_worker = CastPyArg2Int(n_worker_obj, "limit_by_capacity", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("limit_by_capacity");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::limit_by_capacity(expert_count, capacity, n_worker);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_linspace(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add linspace op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *start_obj = PyTuple_GET_ITEM(args, 0);
    auto start = CastPyArg2Value(start_obj, "linspace", 0);
    PyObject *stop_obj = PyTuple_GET_ITEM(args, 1);
    auto stop = CastPyArg2Value(stop_obj, "linspace", 1);
    PyObject *number_obj = PyTuple_GET_ITEM(args, 2);
    auto number = CastPyArg2Value(number_obj, "linspace", 2);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "linspace", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    Place place = CastPyArg2Place(place_obj, "linspace", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("linspace");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::linspace(start, stop, number, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logspace(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add logspace op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *start_obj = PyTuple_GET_ITEM(args, 0);
    auto start = CastPyArg2Value(start_obj, "logspace", 0);
    PyObject *stop_obj = PyTuple_GET_ITEM(args, 1);
    auto stop = CastPyArg2Value(stop_obj, "logspace", 1);
    PyObject *num_obj = PyTuple_GET_ITEM(args, 2);
    auto num = CastPyArg2Value(num_obj, "logspace", 2);
    PyObject *base_obj = PyTuple_GET_ITEM(args, 3);
    auto base = CastPyArg2Value(base_obj, "logspace", 3);

    // Parse Attributes
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "logspace", 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);
    Place place = CastPyArg2Place(place_obj, "logspace", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("logspace");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::logspace(start, stop, num, base, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_logsumexp(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add logsumexp op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "logsumexp", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "logsumexp", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "logsumexp", 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "logsumexp", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("logsumexp");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::logsumexp(x, axis, keepdim, reduce_all);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lrn(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add lrn op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "lrn", 0);

    // Parse Attributes
    PyObject *n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "lrn", 1);
    PyObject *k_obj = PyTuple_GET_ITEM(args, 2);
    float k = CastPyArg2Float(k_obj, "lrn", 2);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 3);
    float alpha = CastPyArg2Float(alpha_obj, "lrn", 3);
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 4);
    float beta = CastPyArg2Float(beta_obj, "lrn", 4);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 5);
    std::string data_format = CastPyArg2String(data_format_obj, "lrn", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("lrn");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::lrn(x, n, k, alpha, beta, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add matmul op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "matmul", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "matmul", 1);

    // Parse Attributes
    PyObject *transpose_x_obj = PyTuple_GET_ITEM(args, 2);
    bool transpose_x = CastPyArg2Boolean(transpose_x_obj, "matmul", 2);
    PyObject *transpose_y_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose_y = CastPyArg2Boolean(transpose_y_obj, "matmul", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("matmul");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::matmul(x, y, transpose_x, transpose_y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matmul_with_flatten(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add matmul_with_flatten op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "matmul_with_flatten", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "matmul_with_flatten", 1);

    // Parse Attributes
    PyObject *x_num_col_dims_obj = PyTuple_GET_ITEM(args, 2);
    int x_num_col_dims =
        CastPyArg2Int(x_num_col_dims_obj, "matmul_with_flatten", 2);
    PyObject *y_num_col_dims_obj = PyTuple_GET_ITEM(args, 3);
    int y_num_col_dims =
        CastPyArg2Int(y_num_col_dims_obj, "matmul_with_flatten", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("matmul_with_flatten");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::matmul_with_flatten(
        x, y, x_num_col_dims, y_num_col_dims);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matrix_rank(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add matrix_rank op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "matrix_rank", 0);

    // Parse Attributes
    PyObject *tol_obj = PyTuple_GET_ITEM(args, 1);
    float tol = CastPyArg2Float(tol_obj, "matrix_rank", 1);
    PyObject *use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol =
        CastPyArg2Boolean(use_default_tol_obj, "matrix_rank", 2);
    PyObject *hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("matrix_rank");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::matrix_rank(x, tol, use_default_tol, hermitian);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_matrix_rank_tol(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add matrix_rank_tol op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "matrix_rank_tol", 0);
    PyObject *atol_tensor_obj = PyTuple_GET_ITEM(args, 1);
    auto atol_tensor = CastPyArg2Value(atol_tensor_obj, "matrix_rank_tol", 1);

    // Parse Attributes
    PyObject *use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol =
        CastPyArg2Boolean(use_default_tol_obj, "matrix_rank_tol", 2);
    PyObject *hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank_tol", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("matrix_rank_tol");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::matrix_rank_tol(
        x, atol_tensor, use_default_tol, hermitian);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add max op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "max", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "max", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "max", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "max", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "max", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("max");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::max(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_maximum(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add maximum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "maximum", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "maximum", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("maximum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::maximum(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add mean op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "mean", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "mean", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "mean", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("mean");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::mean(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_memcpy(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add memcpy op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "memcpy", 0);

    // Parse Attributes
    PyObject *dst_place_type_obj = PyTuple_GET_ITEM(args, 1);
    int dst_place_type = CastPyArg2Int(dst_place_type_obj, "memcpy", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("memcpy");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::memcpy(x, dst_place_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_memcpy_d2h(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add memcpy_d2h op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "memcpy_d2h", 0);

    // Parse Attributes
    PyObject *dst_place_type_obj = PyTuple_GET_ITEM(args, 1);
    int dst_place_type = CastPyArg2Int(dst_place_type_obj, "memcpy_d2h", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("memcpy_d2h");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::memcpy_d2h(x, dst_place_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_memcpy_h2d(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add memcpy_h2d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "memcpy_h2d", 0);

    // Parse Attributes
    PyObject *dst_place_type_obj = PyTuple_GET_ITEM(args, 1);
    int dst_place_type = CastPyArg2Int(dst_place_type_obj, "memcpy_h2d", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("memcpy_h2d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::memcpy_h2d(x, dst_place_type);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_min(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add min op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "min", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "min", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "min", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "min", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "min", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("min");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::min(x, axis, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_minimum(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add minimum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "minimum", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "minimum", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("minimum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::minimum(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_mish(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add mish op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "mish", 0);

    // Parse Attributes
    PyObject *lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "mish", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("mish");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::mish(x, lambda);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multiply(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add multiply op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "multiply", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "multiply", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("multiply");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multiply(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_multiply_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add multiply_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "multiply_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "multiply_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("multiply_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::multiply_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nop(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add nop op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nop", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("nop");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nop(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add nop_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "nop_", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("nop_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nop_(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add norm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "norm", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "norm", 1);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "norm", 2);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "norm", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("norm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::norm(x, axis, epsilon, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_not_equal(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add not_equal op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "not_equal", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "not_equal", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("not_equal");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::not_equal(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_not_equal_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add not_equal_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "not_equal_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "not_equal_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("not_equal_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::not_equal_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_one_hot(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add one_hot op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "one_hot", 0);

    // Parse Attributes
    PyObject *num_classes_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value num_classes;

    if (PyObject_CheckIRValue(num_classes_obj)) {
      num_classes = CastPyArg2Value(num_classes_obj, "one_hot", 1);
    } else {
      int num_classes_tmp = CastPyArg2Int(num_classes_obj, "one_hot", 1);
      num_classes =
          paddle::dialect::full(std::vector<int64_t>{1}, num_classes_tmp,
                                phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("one_hot");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::one_hot(x, num_classes);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pad(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pad op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pad", 0);

    // Parse Attributes
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *pad_value_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value pad_value;

    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pad", 1);
    if (PyObject_CheckIRValue(pad_value_obj)) {
      pad_value = CastPyArg2Value(pad_value_obj, "pad", 2);
    } else {
      float pad_value_tmp = CastPyArg2Float(pad_value_obj, "pad", 2);
      pad_value =
          paddle::dialect::full(std::vector<int64_t>{1}, pad_value_tmp,
                                phi::DataType::FLOAT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("pad");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::pad(x, pad_value, paddings);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_allgather(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_allgather op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "partial_allgather", 0);

    // Parse Attributes
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 1);
    int nranks = CastPyArg2Int(nranks_obj, "partial_allgather", 1);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 2);
    int rank = CastPyArg2Int(rank_obj, "partial_allgather", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "partial_allgather", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "partial_allgather", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_allgather");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::partial_allgather(
        x, nranks, rank, ring_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_allgather_(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_allgather_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "partial_allgather_", 0);

    // Parse Attributes
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 1);
    int nranks = CastPyArg2Int(nranks_obj, "partial_allgather_", 1);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 2);
    int rank = CastPyArg2Int(rank_obj, "partial_allgather_", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "partial_allgather_", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "partial_allgather_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_allgather_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::partial_allgather_(
        x, nranks, rank, ring_id, use_calc_stream);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_concat(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_concat op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "partial_concat", 0);

    // Parse Attributes
    PyObject *start_index_obj = PyTuple_GET_ITEM(args, 1);
    int start_index = CastPyArg2Int(start_index_obj, "partial_concat", 1);
    PyObject *length_obj = PyTuple_GET_ITEM(args, 2);
    int length = CastPyArg2Int(length_obj, "partial_concat", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_concat");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::partial_concat(x, start_index, length);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_recv(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_recv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 0);
    int ring_id = CastPyArg2Int(ring_id_obj, "partial_recv", 0);
    PyObject *peer_obj = PyTuple_GET_ITEM(args, 1);
    int peer = CastPyArg2Int(peer_obj, "partial_recv", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "partial_recv", 2);
    PyObject *out_shape_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> out_shape =
        CastPyArg2Ints(out_shape_obj, "partial_recv", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "partial_recv", 4);
    PyObject *num_obj = PyTuple_GET_ITEM(args, 5);
    int num = CastPyArg2Int(num_obj, "partial_recv", 5);
    PyObject *id_obj = PyTuple_GET_ITEM(args, 6);
    int id = CastPyArg2Int(id_obj, "partial_recv", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_recv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::partial_recv(
        ring_id, peer, dtype, out_shape, use_calc_stream, num, id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_sum(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_sum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "partial_sum", 0);

    // Parse Attributes
    PyObject *start_index_obj = PyTuple_GET_ITEM(args, 1);
    int start_index = CastPyArg2Int(start_index_obj, "partial_sum", 1);
    PyObject *length_obj = PyTuple_GET_ITEM(args, 2);
    int length = CastPyArg2Int(length_obj, "partial_sum", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_sum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::partial_sum(x, start_index, length);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pool2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pool2d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pool2d", 0);

    // Parse Attributes
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);

    // Check for mutable attrs
    pir::Value kernel_size;

    if (PyObject_CheckIRValue(kernel_size_obj)) {
      kernel_size = CastPyArg2Value(kernel_size_obj, "pool2d", 1);
    } else if (PyObject_CheckIRVectorOfValue(kernel_size_obj)) {
      std::vector<pir::Value> kernel_size_tmp =
          CastPyArg2VectorOfValue(kernel_size_obj, "pool2d", 1);
      kernel_size = paddle::dialect::stack(kernel_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> kernel_size_tmp =
          CastPyArg2Longs(kernel_size_obj, "pool2d", 1);
      kernel_size = paddle::dialect::full_int_array(
          kernel_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool2d", 2);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool2d", 3);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool2d", 4);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool2d", 5);
    std::string data_format = CastPyArg2String(data_format_obj, "pool2d", 6);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool2d", 7);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool2d", 8);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool2d", 9);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "pool2d", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("pool2d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::pool2d(
        x, kernel_size, strides, paddings, ceil_mode, exclusive, data_format,
        pooling_type, global_pooling, adaptive, padding_algorithm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_pool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add pool3d op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "pool3d", 0);

    // Parse Attributes
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool3d", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool3d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool3d", 3);
    PyObject *ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool3d", 4);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool3d", 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool3d", 6);
    PyObject *pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool3d", 7);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool3d", 8);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool3d", 9);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "pool3d", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("pool3d");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::pool3d(
        x, kernel_size, strides, paddings, ceil_mode, exclusive, data_format,
        pooling_type, global_pooling, adaptive, padding_algorithm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_print(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add print op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *in_obj = PyTuple_GET_ITEM(args, 0);
    auto in = CastPyArg2Value(in_obj, "print", 0);

    // Parse Attributes
    PyObject *first_n_obj = PyTuple_GET_ITEM(args, 1);
    int first_n = CastPyArg2Int(first_n_obj, "print", 1);
    PyObject *message_obj = PyTuple_GET_ITEM(args, 2);
    std::string message = CastPyArg2String(message_obj, "print", 2);
    PyObject *summarize_obj = PyTuple_GET_ITEM(args, 3);
    int summarize = CastPyArg2Int(summarize_obj, "print", 3);
    PyObject *print_tensor_name_obj = PyTuple_GET_ITEM(args, 4);
    bool print_tensor_name =
        CastPyArg2Boolean(print_tensor_name_obj, "print", 4);
    PyObject *print_tensor_type_obj = PyTuple_GET_ITEM(args, 5);
    bool print_tensor_type =
        CastPyArg2Boolean(print_tensor_type_obj, "print", 5);
    PyObject *print_tensor_shape_obj = PyTuple_GET_ITEM(args, 6);
    bool print_tensor_shape =
        CastPyArg2Boolean(print_tensor_shape_obj, "print", 6);
    PyObject *print_tensor_layout_obj = PyTuple_GET_ITEM(args, 7);
    bool print_tensor_layout =
        CastPyArg2Boolean(print_tensor_layout_obj, "print", 7);
    PyObject *print_tensor_lod_obj = PyTuple_GET_ITEM(args, 8);
    bool print_tensor_lod = CastPyArg2Boolean(print_tensor_lod_obj, "print", 8);
    PyObject *print_phase_obj = PyTuple_GET_ITEM(args, 9);
    std::string print_phase = CastPyArg2String(print_phase_obj, "print", 9);
    PyObject *is_forward_obj = PyTuple_GET_ITEM(args, 10);
    bool is_forward = CastPyArg2Boolean(is_forward_obj, "print", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("print");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::print(
        in, first_n, message, summarize, print_tensor_name, print_tensor_type,
        print_tensor_shape, print_tensor_layout, print_tensor_lod, print_phase,
        is_forward);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_prod(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add prod op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "prod", 0);

    // Parse Attributes
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value dims;

    if (PyObject_CheckIRValue(dims_obj)) {
      dims = CastPyArg2Value(dims_obj, "prod", 1);
    } else if (PyObject_CheckIRVectorOfValue(dims_obj)) {
      std::vector<pir::Value> dims_tmp =
          CastPyArg2VectorOfValue(dims_obj, "prod", 1);
      dims = paddle::dialect::stack(dims_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> dims_tmp = CastPyArg2Longs(dims_obj, "prod", 1);
      dims = paddle::dialect::full_int_array(dims_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "prod", 2);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "prod", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("prod");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::prod(x, dims, keep_dim, reduce_all);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_prune_gate_by_capacity(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add prune_gate_by_capacity op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *gate_idx_obj = PyTuple_GET_ITEM(args, 0);
    auto gate_idx = CastPyArg2Value(gate_idx_obj, "prune_gate_by_capacity", 0);
    PyObject *expert_count_obj = PyTuple_GET_ITEM(args, 1);
    auto expert_count =
        CastPyArg2Value(expert_count_obj, "prune_gate_by_capacity", 1);

    // Parse Attributes
    PyObject *n_expert_obj = PyTuple_GET_ITEM(args, 2);
    int64_t n_expert =
        CastPyArg2Long(n_expert_obj, "prune_gate_by_capacity", 2);
    PyObject *n_worker_obj = PyTuple_GET_ITEM(args, 3);
    int64_t n_worker =
        CastPyArg2Long(n_worker_obj, "prune_gate_by_capacity", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("prune_gate_by_capacity");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::prune_gate_by_capacity(
        gate_idx, expert_count, n_expert, n_worker);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_push_dense(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add push_dense op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj, "push_dense", 0);

    // Parse Attributes
    PyObject *table_id_obj = PyTuple_GET_ITEM(args, 1);
    int table_id = CastPyArg2Int(table_id_obj, "push_dense", 1);
    PyObject *scale_data_norm_obj = PyTuple_GET_ITEM(args, 2);
    float scale_data_norm =
        CastPyArg2Float(scale_data_norm_obj, "push_dense", 2);
    PyObject *input_names_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<std::string> input_names =
        CastPyArg2Strings(input_names_obj, "push_dense", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("push_dense");
    callstack_recorder.Record();
    paddle::dialect::push_dense(ids, table_id, scale_data_norm, input_names);
    callstack_recorder.AttachToOps();
    return nullptr;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_push_sparse_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add push_sparse_v2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj, "push_sparse_v2", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2VectorOfValue(w_obj, "push_sparse_v2", 1);
    PyObject *out_grad_in_obj = PyTuple_GET_ITEM(args, 2);
    auto out_grad_in =
        CastPyArg2VectorOfValue(out_grad_in_obj, "push_sparse_v2", 2);

    // Parse Attributes
    PyObject *embeddingdim_obj = PyTuple_GET_ITEM(args, 3);
    int embeddingdim = CastPyArg2Int(embeddingdim_obj, "push_sparse_v2", 3);
    PyObject *tableid_obj = PyTuple_GET_ITEM(args, 4);
    int tableid = CastPyArg2Int(tableid_obj, "push_sparse_v2", 4);
    PyObject *accessorclass_obj = PyTuple_GET_ITEM(args, 5);
    std::string accessorclass =
        CastPyArg2String(accessorclass_obj, "push_sparse_v2", 5);
    PyObject *ctrlabelname_obj = PyTuple_GET_ITEM(args, 6);
    std::string ctrlabelname =
        CastPyArg2String(ctrlabelname_obj, "push_sparse_v2", 6);
    PyObject *paddingid_obj = PyTuple_GET_ITEM(args, 7);
    int paddingid = CastPyArg2Int(paddingid_obj, "push_sparse_v2", 7);
    PyObject *scalesparsegrad_obj = PyTuple_GET_ITEM(args, 8);
    bool scalesparsegrad =
        CastPyArg2Boolean(scalesparsegrad_obj, "push_sparse_v2", 8);
    PyObject *inputnames_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<std::string> inputnames =
        CastPyArg2Strings(inputnames_obj, "push_sparse_v2", 9);
    PyObject *is_distributed_obj = PyTuple_GET_ITEM(args, 10);
    bool is_distributed =
        CastPyArg2Boolean(is_distributed_obj, "push_sparse_v2", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("push_sparse_v2");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::push_sparse_v2(
        ids, w, out_grad_in, embeddingdim, tableid, accessorclass, ctrlabelname,
        paddingid, scalesparsegrad, inputnames, is_distributed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_push_sparse_v2_(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add push_sparse_v2_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *ids_obj = PyTuple_GET_ITEM(args, 0);
    auto ids = CastPyArg2VectorOfValue(ids_obj, "push_sparse_v2_", 0);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 1);
    auto w = CastPyArg2VectorOfValue(w_obj, "push_sparse_v2_", 1);
    PyObject *out_grad_in_obj = PyTuple_GET_ITEM(args, 2);
    auto out_grad_in =
        CastPyArg2VectorOfValue(out_grad_in_obj, "push_sparse_v2_", 2);

    // Parse Attributes
    PyObject *embeddingdim_obj = PyTuple_GET_ITEM(args, 3);
    int embeddingdim = CastPyArg2Int(embeddingdim_obj, "push_sparse_v2_", 3);
    PyObject *tableid_obj = PyTuple_GET_ITEM(args, 4);
    int tableid = CastPyArg2Int(tableid_obj, "push_sparse_v2_", 4);
    PyObject *accessorclass_obj = PyTuple_GET_ITEM(args, 5);
    std::string accessorclass =
        CastPyArg2String(accessorclass_obj, "push_sparse_v2_", 5);
    PyObject *ctrlabelname_obj = PyTuple_GET_ITEM(args, 6);
    std::string ctrlabelname =
        CastPyArg2String(ctrlabelname_obj, "push_sparse_v2_", 6);
    PyObject *paddingid_obj = PyTuple_GET_ITEM(args, 7);
    int paddingid = CastPyArg2Int(paddingid_obj, "push_sparse_v2_", 7);
    PyObject *scalesparsegrad_obj = PyTuple_GET_ITEM(args, 8);
    bool scalesparsegrad =
        CastPyArg2Boolean(scalesparsegrad_obj, "push_sparse_v2_", 8);
    PyObject *inputnames_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<std::string> inputnames =
        CastPyArg2Strings(inputnames_obj, "push_sparse_v2_", 9);
    PyObject *is_distributed_obj = PyTuple_GET_ITEM(args, 10);
    bool is_distributed =
        CastPyArg2Boolean(is_distributed_obj, "push_sparse_v2_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("push_sparse_v2_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::push_sparse_v2_(
        ids, w, out_grad_in, embeddingdim, tableid, accessorclass, ctrlabelname,
        paddingid, scalesparsegrad, inputnames, is_distributed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_quantize_linear(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add quantize_linear op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "quantize_linear", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "quantize_linear", 1);
    PyObject *zero_point_obj = PyTuple_GET_ITEM(args, 2);
    auto zero_point = CastPyArg2Value(zero_point_obj, "quantize_linear", 2);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 3);
    auto in_accum = CastPyArg2OptionalValue(in_accum_obj, "quantize_linear", 3);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 4);
    auto in_state = CastPyArg2OptionalValue(in_state_obj, "quantize_linear", 4);

    // Parse Attributes
    PyObject *quant_axis_obj = PyTuple_GET_ITEM(args, 5);
    int quant_axis = CastPyArg2Int(quant_axis_obj, "quantize_linear", 5);
    PyObject *bit_length_obj = PyTuple_GET_ITEM(args, 6);
    int bit_length = CastPyArg2Int(bit_length_obj, "quantize_linear", 6);
    PyObject *round_type_obj = PyTuple_GET_ITEM(args, 7);
    int round_type = CastPyArg2Int(round_type_obj, "quantize_linear", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "quantize_linear", 8);
    PyObject *only_observer_obj = PyTuple_GET_ITEM(args, 9);
    bool only_observer =
        CastPyArg2Boolean(only_observer_obj, "quantize_linear", 9);
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 10);
    float moving_rate = CastPyArg2Float(moving_rate_obj, "quantize_linear", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("quantize_linear");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::quantize_linear(
        x, scale, zero_point, in_accum, in_state, quant_axis, bit_length,
        round_type, is_test, only_observer, moving_rate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_quantize_linear_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add quantize_linear_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "quantize_linear_", 0);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    auto scale = CastPyArg2Value(scale_obj, "quantize_linear_", 1);
    PyObject *zero_point_obj = PyTuple_GET_ITEM(args, 2);
    auto zero_point = CastPyArg2Value(zero_point_obj, "quantize_linear_", 2);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 3);
    auto in_accum =
        CastPyArg2OptionalValue(in_accum_obj, "quantize_linear_", 3);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 4);
    auto in_state =
        CastPyArg2OptionalValue(in_state_obj, "quantize_linear_", 4);

    // Parse Attributes
    PyObject *quant_axis_obj = PyTuple_GET_ITEM(args, 5);
    int quant_axis = CastPyArg2Int(quant_axis_obj, "quantize_linear_", 5);
    PyObject *bit_length_obj = PyTuple_GET_ITEM(args, 6);
    int bit_length = CastPyArg2Int(bit_length_obj, "quantize_linear_", 6);
    PyObject *round_type_obj = PyTuple_GET_ITEM(args, 7);
    int round_type = CastPyArg2Int(round_type_obj, "quantize_linear_", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "quantize_linear_", 8);
    PyObject *only_observer_obj = PyTuple_GET_ITEM(args, 9);
    bool only_observer =
        CastPyArg2Boolean(only_observer_obj, "quantize_linear_", 9);
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 10);
    float moving_rate =
        CastPyArg2Float(moving_rate_obj, "quantize_linear_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("quantize_linear_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::quantize_linear_(
        x, scale, zero_point, in_accum, in_state, quant_axis, bit_length,
        round_type, is_test, only_observer, moving_rate);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_randint(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add randint op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *low_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *high_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value shape;

    int low = CastPyArg2Int(low_obj, "randint", 0);
    int high = CastPyArg2Int(high_obj, "randint", 1);
    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "randint", 2);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "randint", 2);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "randint", 2);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "randint", 3);
    Place place = CastPyArg2Place(place_obj, "randint", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("randint");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::randint(shape, low, high, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_random_routing(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add random_routing op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *prob_obj = PyTuple_GET_ITEM(args, 0);
    auto prob = CastPyArg2Value(prob_obj, "random_routing", 0);
    PyObject *topk_value_obj = PyTuple_GET_ITEM(args, 1);
    auto topk_value = CastPyArg2Value(topk_value_obj, "random_routing", 1);
    PyObject *topk_idx_obj = PyTuple_GET_ITEM(args, 2);
    auto topk_idx = CastPyArg2Value(topk_idx_obj, "random_routing", 2);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("random_routing");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::random_routing(prob, topk_value, topk_idx);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_randperm(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add randperm op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *n_obj = PyTuple_GET_ITEM(args, 0);
    int n = CastPyArg2Int(n_obj, "randperm", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "randperm", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    Place place = CastPyArg2Place(place_obj, "randperm", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("randperm");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::randperm(n, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rank_attention(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add rank_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rank_attention", 0);
    PyObject *rank_offset_obj = PyTuple_GET_ITEM(args, 1);
    auto rank_offset = CastPyArg2Value(rank_offset_obj, "rank_attention", 1);
    PyObject *rank_param_obj = PyTuple_GET_ITEM(args, 2);
    auto rank_param = CastPyArg2Value(rank_param_obj, "rank_attention", 2);

    // Parse Attributes
    PyObject *max_rank_obj = PyTuple_GET_ITEM(args, 3);
    int max_rank = CastPyArg2Int(max_rank_obj, "rank_attention", 3);
    PyObject *max_size_obj = PyTuple_GET_ITEM(args, 4);
    int max_size = CastPyArg2Int(max_size_obj, "rank_attention", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("rank_attention");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rank_attention(
        x, rank_offset, rank_param, max_rank, max_size);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_read_file(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add read_file op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *filename_obj = PyTuple_GET_ITEM(args, 0);
    std::string filename = CastPyArg2String(filename_obj, "read_file", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "read_file", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    Place place = CastPyArg2Place(place_obj, "read_file", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("read_file");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::read_file(filename, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_recv_v2(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add recv_v2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *out_shape_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int> out_shape = CastPyArg2Ints(out_shape_obj, "recv_v2", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "recv_v2", 1);
    PyObject *peer_obj = PyTuple_GET_ITEM(args, 2);
    int peer = CastPyArg2Int(peer_obj, "recv_v2", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "recv_v2", 3);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 4);
    bool use_calc_stream = CastPyArg2Boolean(use_calc_stream_obj, "recv_v2", 4);
    PyObject *dynamic_shape_obj = PyTuple_GET_ITEM(args, 5);
    bool dynamic_shape = CastPyArg2Boolean(dynamic_shape_obj, "recv_v2", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("recv_v2");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::recv_v2(
        out_shape, dtype, peer, ring_id, use_calc_stream, dynamic_shape);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_remainder(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add remainder op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "remainder", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "remainder", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("remainder");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::remainder(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_remainder_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add remainder_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "remainder_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "remainder_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("remainder_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::remainder_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_repeat_interleave(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add repeat_interleave op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "repeat_interleave", 0);

    // Parse Attributes
    PyObject *repeats_obj = PyTuple_GET_ITEM(args, 1);
    int repeats = CastPyArg2Int(repeats_obj, "repeat_interleave", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "repeat_interleave", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("repeat_interleave");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::repeat_interleave(x, repeats, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_repeat_interleave_with_tensor_index(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add repeat_interleave_with_tensor_index op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "repeat_interleave_with_tensor_index", 0);
    PyObject *repeats_obj = PyTuple_GET_ITEM(args, 1);
    auto repeats =
        CastPyArg2Value(repeats_obj, "repeat_interleave_with_tensor_index", 1);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis =
        CastPyArg2Int(axis_obj, "repeat_interleave_with_tensor_index", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("repeat_interleave_with_tensor_index");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::repeat_interleave_with_tensor_index(x, repeats, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reshape(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add reshape op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reshape", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "reshape", 1);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "reshape", 1);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "reshape", 1);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("reshape");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reshape(x, shape);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_reshape_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add reshape_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "reshape_", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value shape;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "reshape_", 1);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "reshape_", 1);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp =
          CastPyArg2Longs(shape_obj, "reshape_", 1);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("reshape_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::reshape_(x, shape);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rnn(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rnn op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rnn", 0);
    PyObject *pre_state_obj = PyTuple_GET_ITEM(args, 1);
    auto pre_state = CastPyArg2VectorOfValue(pre_state_obj, "rnn", 1);
    PyObject *weight_list_obj = PyTuple_GET_ITEM(args, 2);
    auto weight_list = CastPyArg2VectorOfValue(weight_list_obj, "rnn", 2);
    PyObject *sequence_length_obj = PyTuple_GET_ITEM(args, 3);
    auto sequence_length =
        CastPyArg2OptionalValue(sequence_length_obj, "rnn", 3);
    PyObject *dropout_state_in_obj = PyTuple_GET_ITEM(args, 4);
    auto dropout_state_in = CastPyArg2Value(dropout_state_in_obj, "rnn", 4);

    // Parse Attributes
    PyObject *dropout_prob_obj = PyTuple_GET_ITEM(args, 5);
    float dropout_prob = CastPyArg2Float(dropout_prob_obj, "rnn", 5);
    PyObject *is_bidirec_obj = PyTuple_GET_ITEM(args, 6);
    bool is_bidirec = CastPyArg2Boolean(is_bidirec_obj, "rnn", 6);
    PyObject *input_size_obj = PyTuple_GET_ITEM(args, 7);
    int input_size = CastPyArg2Int(input_size_obj, "rnn", 7);
    PyObject *hidden_size_obj = PyTuple_GET_ITEM(args, 8);
    int hidden_size = CastPyArg2Int(hidden_size_obj, "rnn", 8);
    PyObject *num_layers_obj = PyTuple_GET_ITEM(args, 9);
    int num_layers = CastPyArg2Int(num_layers_obj, "rnn", 9);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 10);
    std::string mode = CastPyArg2String(mode_obj, "rnn", 10);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 11);
    int seed = CastPyArg2Int(seed_obj, "rnn", 11);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 12);
    bool is_test = CastPyArg2Boolean(is_test_obj, "rnn", 12);

    // Call ir static api
    CallStackRecorder callstack_recorder("rnn");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rnn(
        x, pre_state, weight_list, sequence_length, dropout_state_in,
        dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode,
        seed, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rnn_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rnn_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rnn_", 0);
    PyObject *pre_state_obj = PyTuple_GET_ITEM(args, 1);
    auto pre_state = CastPyArg2VectorOfValue(pre_state_obj, "rnn_", 1);
    PyObject *weight_list_obj = PyTuple_GET_ITEM(args, 2);
    auto weight_list = CastPyArg2VectorOfValue(weight_list_obj, "rnn_", 2);
    PyObject *sequence_length_obj = PyTuple_GET_ITEM(args, 3);
    auto sequence_length =
        CastPyArg2OptionalValue(sequence_length_obj, "rnn_", 3);
    PyObject *dropout_state_in_obj = PyTuple_GET_ITEM(args, 4);
    auto dropout_state_in = CastPyArg2Value(dropout_state_in_obj, "rnn_", 4);

    // Parse Attributes
    PyObject *dropout_prob_obj = PyTuple_GET_ITEM(args, 5);
    float dropout_prob = CastPyArg2Float(dropout_prob_obj, "rnn_", 5);
    PyObject *is_bidirec_obj = PyTuple_GET_ITEM(args, 6);
    bool is_bidirec = CastPyArg2Boolean(is_bidirec_obj, "rnn_", 6);
    PyObject *input_size_obj = PyTuple_GET_ITEM(args, 7);
    int input_size = CastPyArg2Int(input_size_obj, "rnn_", 7);
    PyObject *hidden_size_obj = PyTuple_GET_ITEM(args, 8);
    int hidden_size = CastPyArg2Int(hidden_size_obj, "rnn_", 8);
    PyObject *num_layers_obj = PyTuple_GET_ITEM(args, 9);
    int num_layers = CastPyArg2Int(num_layers_obj, "rnn_", 9);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 10);
    std::string mode = CastPyArg2String(mode_obj, "rnn_", 10);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 11);
    int seed = CastPyArg2Int(seed_obj, "rnn_", 11);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 12);
    bool is_test = CastPyArg2Boolean(is_test_obj, "rnn_", 12);

    // Call ir static api
    CallStackRecorder callstack_recorder("rnn_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rnn_(
        x, pre_state, weight_list, sequence_length, dropout_state_in,
        dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode,
        seed, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_row_conv(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add row_conv op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "row_conv", 0);
    PyObject *filter_obj = PyTuple_GET_ITEM(args, 1);
    auto filter = CastPyArg2Value(filter_obj, "row_conv", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("row_conv");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::row_conv(x, filter);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_rrelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add rrelu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "rrelu", 0);

    // Parse Attributes
    PyObject *lower_obj = PyTuple_GET_ITEM(args, 1);
    float lower = CastPyArg2Float(lower_obj, "rrelu", 1);
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 2);
    float upper = CastPyArg2Float(upper_obj, "rrelu", 2);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "rrelu", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("rrelu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::rrelu(x, lower, upper, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_seed(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add seed op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 0);
    int seed = CastPyArg2Int(seed_obj, "seed", 0);
    PyObject *deterministic_obj = PyTuple_GET_ITEM(args, 1);
    bool deterministic = CastPyArg2Boolean(deterministic_obj, "seed", 1);
    PyObject *rng_name_obj = PyTuple_GET_ITEM(args, 2);
    std::string rng_name = CastPyArg2String(rng_name_obj, "seed", 2);
    PyObject *force_cpu_obj = PyTuple_GET_ITEM(args, 3);
    bool force_cpu = CastPyArg2Boolean(force_cpu_obj, "seed", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("seed");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::seed(seed, deterministic, rng_name, force_cpu);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_send_v2(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add send_v2 op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "send_v2", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "send_v2", 1);
    PyObject *peer_obj = PyTuple_GET_ITEM(args, 2);
    int peer = CastPyArg2Int(peer_obj, "send_v2", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream = CastPyArg2Boolean(use_calc_stream_obj, "send_v2", 3);
    PyObject *dynamic_shape_obj = PyTuple_GET_ITEM(args, 4);
    bool dynamic_shape = CastPyArg2Boolean(dynamic_shape_obj, "send_v2", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("send_v2");
    callstack_recorder.Record();
    paddle::dialect::send_v2(x, ring_id, peer, use_calc_stream, dynamic_shape);
    callstack_recorder.AttachToOps();
    return nullptr;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_set_value(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_value op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "set_value", 0);

    // Parse Attributes
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *steps_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *decrease_axes_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *none_axes_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 8);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    pir::Value steps;

    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "set_value", 1);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "set_value", 1);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "set_value", 1);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "set_value", 2);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "set_value", 2);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp = CastPyArg2Longs(ends_obj, "set_value", 2);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(steps_obj)) {
      steps = CastPyArg2Value(steps_obj, "set_value", 3);
    } else if (PyObject_CheckIRVectorOfValue(steps_obj)) {
      std::vector<pir::Value> steps_tmp =
          CastPyArg2VectorOfValue(steps_obj, "set_value", 3);
      steps = paddle::dialect::stack(steps_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> steps_tmp =
          CastPyArg2Longs(steps_obj, "set_value", 3);
      steps = paddle::dialect::full_int_array(steps_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "set_value", 4);
    std::vector<int64_t> decrease_axes =
        CastPyArg2Longs(decrease_axes_obj, "set_value", 5);
    std::vector<int64_t> none_axes =
        CastPyArg2Longs(none_axes_obj, "set_value", 6);
    std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "set_value", 7);
    std::vector<phi::Scalar> values =
        CastPyArg2ScalarArray(values_obj, "set_value", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("set_value");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::set_value(
        x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_set_value_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_value_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "set_value_", 0);

    // Parse Attributes
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *steps_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *decrease_axes_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *none_axes_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 8);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    pir::Value steps;

    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "set_value_", 1);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "set_value_", 1);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "set_value_", 1);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "set_value_", 2);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "set_value_", 2);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp =
          CastPyArg2Longs(ends_obj, "set_value_", 2);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(steps_obj)) {
      steps = CastPyArg2Value(steps_obj, "set_value_", 3);
    } else if (PyObject_CheckIRVectorOfValue(steps_obj)) {
      std::vector<pir::Value> steps_tmp =
          CastPyArg2VectorOfValue(steps_obj, "set_value_", 3);
      steps = paddle::dialect::stack(steps_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> steps_tmp =
          CastPyArg2Longs(steps_obj, "set_value_", 3);
      steps = paddle::dialect::full_int_array(steps_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "set_value_", 4);
    std::vector<int64_t> decrease_axes =
        CastPyArg2Longs(decrease_axes_obj, "set_value_", 5);
    std::vector<int64_t> none_axes =
        CastPyArg2Longs(none_axes_obj, "set_value_", 6);
    std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "set_value_", 7);
    std::vector<phi::Scalar> values =
        CastPyArg2ScalarArray(values_obj, "set_value_", 8);

    // Call ir static api
    CallStackRecorder callstack_recorder("set_value_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::set_value_(
        x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_set_value_with_tensor(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_value_with_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "set_value_with_tensor", 0);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 1);
    auto values = CastPyArg2Value(values_obj, "set_value_with_tensor", 1);

    // Parse Attributes
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *steps_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *decrease_axes_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *none_axes_obj = PyTuple_GET_ITEM(args, 7);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    pir::Value steps;

    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "set_value_with_tensor", 2);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "set_value_with_tensor", 2);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "set_value_with_tensor", 2);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "set_value_with_tensor", 3);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "set_value_with_tensor", 3);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp =
          CastPyArg2Longs(ends_obj, "set_value_with_tensor", 3);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(steps_obj)) {
      steps = CastPyArg2Value(steps_obj, "set_value_with_tensor", 4);
    } else if (PyObject_CheckIRVectorOfValue(steps_obj)) {
      std::vector<pir::Value> steps_tmp =
          CastPyArg2VectorOfValue(steps_obj, "set_value_with_tensor", 4);
      steps = paddle::dialect::stack(steps_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> steps_tmp =
          CastPyArg2Longs(steps_obj, "set_value_with_tensor", 4);
      steps = paddle::dialect::full_int_array(steps_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    std::vector<int64_t> axes =
        CastPyArg2Longs(axes_obj, "set_value_with_tensor", 5);
    std::vector<int64_t> decrease_axes =
        CastPyArg2Longs(decrease_axes_obj, "set_value_with_tensor", 6);
    std::vector<int64_t> none_axes =
        CastPyArg2Longs(none_axes_obj, "set_value_with_tensor", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("set_value_with_tensor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::set_value_with_tensor(
        x, values, starts, ends, steps, axes, decrease_axes, none_axes);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_set_value_with_tensor_(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_value_with_tensor_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "set_value_with_tensor_", 0);
    PyObject *values_obj = PyTuple_GET_ITEM(args, 1);
    auto values = CastPyArg2Value(values_obj, "set_value_with_tensor_", 1);

    // Parse Attributes
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *steps_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *decrease_axes_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *none_axes_obj = PyTuple_GET_ITEM(args, 7);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    pir::Value steps;

    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "set_value_with_tensor_", 2);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "set_value_with_tensor_", 2);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "set_value_with_tensor_", 2);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "set_value_with_tensor_", 3);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "set_value_with_tensor_", 3);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp =
          CastPyArg2Longs(ends_obj, "set_value_with_tensor_", 3);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(steps_obj)) {
      steps = CastPyArg2Value(steps_obj, "set_value_with_tensor_", 4);
    } else if (PyObject_CheckIRVectorOfValue(steps_obj)) {
      std::vector<pir::Value> steps_tmp =
          CastPyArg2VectorOfValue(steps_obj, "set_value_with_tensor_", 4);
      steps = paddle::dialect::stack(steps_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> steps_tmp =
          CastPyArg2Longs(steps_obj, "set_value_with_tensor_", 4);
      steps = paddle::dialect::full_int_array(steps_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    std::vector<int64_t> axes =
        CastPyArg2Longs(axes_obj, "set_value_with_tensor_", 5);
    std::vector<int64_t> decrease_axes =
        CastPyArg2Longs(decrease_axes_obj, "set_value_with_tensor_", 6);
    std::vector<int64_t> none_axes =
        CastPyArg2Longs(none_axes_obj, "set_value_with_tensor_", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("set_value_with_tensor_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::set_value_with_tensor_(
        x, values, starts, ends, steps, axes, decrease_axes, none_axes);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_shadow_feed(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add shadow_feed op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "shadow_feed", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("shadow_feed");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::shadow_feed(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_shadow_feed_tensors(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add shadow_feed_tensors op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2VectorOfValue(x_obj, "shadow_feed_tensors", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("shadow_feed_tensors");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::shadow_feed_tensors(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_share_data(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add share_data op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "share_data", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("share_data");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::share_data(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_shuffle_batch(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add shuffle_batch op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "shuffle_batch", 0);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 1);
    auto seed = CastPyArg2Value(seed_obj, "shuffle_batch", 1);

    // Parse Attributes
    PyObject *startup_seed_obj = PyTuple_GET_ITEM(args, 2);
    int startup_seed = CastPyArg2Int(startup_seed_obj, "shuffle_batch", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("shuffle_batch");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::shuffle_batch(x, seed, startup_seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add slice op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "slice", 0);

    // Parse Attributes
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *infer_flags_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *decrease_axis_obj = PyTuple_GET_ITEM(args, 5);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "slice", 1);
    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "slice", 2);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "slice", 2);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp = CastPyArg2Longs(starts_obj, "slice", 2);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "slice", 3);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "slice", 3);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp = CastPyArg2Longs(ends_obj, "slice", 3);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    std::vector<int64_t> infer_flags =
        CastPyArg2Longs(infer_flags_obj, "slice", 4);
    std::vector<int64_t> decrease_axis =
        CastPyArg2Longs(decrease_axis_obj, "slice", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("slice");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::slice(input, starts, ends, axes,
                                                 infer_flags, decrease_axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_soft_relu(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add soft_relu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "soft_relu", 0);

    // Parse Attributes
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "soft_relu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("soft_relu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::soft_relu(x, threshold);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add softmax op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "softmax", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("softmax");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::softmax(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_softmax_(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add softmax_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "softmax_", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("softmax_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::softmax_(x, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_split(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add split op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "split", 0);

    // Parse Attributes
    PyObject *sections_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value sections;

    pir::Value axis;

    if (PyObject_CheckIRValue(sections_obj)) {
      sections = CastPyArg2Value(sections_obj, "split", 1);
    } else if (PyObject_CheckIRVectorOfValue(sections_obj)) {
      std::vector<pir::Value> sections_tmp =
          CastPyArg2VectorOfValue(sections_obj, "split", 1);
      sections = paddle::dialect::stack(sections_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> sections_tmp =
          CastPyArg2Longs(sections_obj, "split", 1);
      sections = paddle::dialect::full_int_array(
          sections_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "split", 2);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "split", 2);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("split");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::split(x, sections, axis);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_split_with_num(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add split_with_num op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "split_with_num", 0);

    // Parse Attributes
    PyObject *num_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value axis;

    int num = CastPyArg2Int(num_obj, "split_with_num", 1);
    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "split_with_num", 2);
    } else {
      int axis_tmp = CastPyArg2Int(axis_obj, "split_with_num", 2);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::INT32, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("split_with_num");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::split_with_num(x, axis, num);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_strided_slice(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add strided_slice op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "strided_slice", 0);

    // Parse Attributes
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value starts;

    pir::Value ends;

    pir::Value strides;

    std::vector<int> axes = CastPyArg2Ints(axes_obj, "strided_slice", 1);
    if (PyObject_CheckIRValue(starts_obj)) {
      starts = CastPyArg2Value(starts_obj, "strided_slice", 2);
    } else if (PyObject_CheckIRVectorOfValue(starts_obj)) {
      std::vector<pir::Value> starts_tmp =
          CastPyArg2VectorOfValue(starts_obj, "strided_slice", 2);
      starts = paddle::dialect::stack(starts_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> starts_tmp =
          CastPyArg2Longs(starts_obj, "strided_slice", 2);
      starts = paddle::dialect::full_int_array(starts_tmp, phi::DataType::INT64,
                                               phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(ends_obj)) {
      ends = CastPyArg2Value(ends_obj, "strided_slice", 3);
    } else if (PyObject_CheckIRVectorOfValue(ends_obj)) {
      std::vector<pir::Value> ends_tmp =
          CastPyArg2VectorOfValue(ends_obj, "strided_slice", 3);
      ends = paddle::dialect::stack(ends_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> ends_tmp =
          CastPyArg2Longs(ends_obj, "strided_slice", 3);
      ends = paddle::dialect::full_int_array(ends_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(strides_obj)) {
      strides = CastPyArg2Value(strides_obj, "strided_slice", 4);
    } else if (PyObject_CheckIRVectorOfValue(strides_obj)) {
      std::vector<pir::Value> strides_tmp =
          CastPyArg2VectorOfValue(strides_obj, "strided_slice", 4);
      strides = paddle::dialect::stack(strides_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> strides_tmp =
          CastPyArg2Longs(strides_obj, "strided_slice", 4);
      strides = paddle::dialect::full_int_array(
          strides_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("strided_slice");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::strided_slice(x, starts, ends, strides, axes);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_subtract(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  try {
    VLOG(6) << "Add subtract op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "subtract", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "subtract", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("subtract");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::subtract(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_subtract_(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add subtract_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "subtract_", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "subtract_", 1);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("subtract_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::subtract_(x, y);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add sum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sum", 0);

    // Parse Attributes
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 3);

    // Check for mutable attrs
    pir::Value axis;

    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "sum", 1);
    } else if (PyObject_CheckIRVectorOfValue(axis_obj)) {
      std::vector<pir::Value> axis_tmp =
          CastPyArg2VectorOfValue(axis_obj, "sum", 1);
      axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "sum", 1);
      axis = paddle::dialect::full_int_array(axis_tmp, phi::DataType::INT64,
                                             phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "sum", 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "sum", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("sum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sum(x, axis, dtype, keepdim);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_swish(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add swish op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "swish", 0);

    // Parse Attributes

    // Call ir static api
    CallStackRecorder callstack_recorder("swish");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::swish(x);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sync_batch_norm_(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add sync_batch_norm_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sync_batch_norm_", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    auto mean = CastPyArg2Value(mean_obj, "sync_batch_norm_", 1);
    PyObject *variance_obj = PyTuple_GET_ITEM(args, 2);
    auto variance = CastPyArg2Value(variance_obj, "sync_batch_norm_", 2);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 3);
    auto scale = CastPyArg2Value(scale_obj, "sync_batch_norm_", 3);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 4);
    auto bias = CastPyArg2Value(bias_obj, "sync_batch_norm_", 4);

    // Parse Attributes
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 5);
    bool is_test = CastPyArg2Boolean(is_test_obj, "sync_batch_norm_", 5);
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 6);
    float momentum = CastPyArg2Float(momentum_obj, "sync_batch_norm_", 6);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 7);
    float epsilon = CastPyArg2Float(epsilon_obj, "sync_batch_norm_", 7);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 8);
    std::string data_format =
        CastPyArg2String(data_format_obj, "sync_batch_norm_", 8);
    PyObject *use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats =
        CastPyArg2Boolean(use_global_stats_obj, "sync_batch_norm_", 9);
    PyObject *trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics =
        CastPyArg2Boolean(trainable_statistics_obj, "sync_batch_norm_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("sync_batch_norm_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sync_batch_norm_(
        x, mean, variance, scale, bias, is_test, momentum, epsilon, data_format,
        use_global_stats, trainable_statistics);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tdm_sampler(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  try {
    VLOG(6) << "Add tdm_sampler op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tdm_sampler", 0);
    PyObject *travel_obj = PyTuple_GET_ITEM(args, 1);
    auto travel = CastPyArg2Value(travel_obj, "tdm_sampler", 1);
    PyObject *layer_obj = PyTuple_GET_ITEM(args, 2);
    auto layer = CastPyArg2Value(layer_obj, "tdm_sampler", 2);

    // Parse Attributes
    PyObject *output_positive_obj = PyTuple_GET_ITEM(args, 3);
    bool output_positive =
        CastPyArg2Boolean(output_positive_obj, "tdm_sampler", 3);
    PyObject *neg_samples_num_list_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> neg_samples_num_list =
        CastPyArg2Ints(neg_samples_num_list_obj, "tdm_sampler", 4);
    PyObject *layer_offset_lod_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> layer_offset_lod =
        CastPyArg2Ints(layer_offset_lod_obj, "tdm_sampler", 5);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 6);
    int seed = CastPyArg2Int(seed_obj, "tdm_sampler", 6);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 7);
    int dtype = CastPyArg2Int(dtype_obj, "tdm_sampler", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("tdm_sampler");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tdm_sampler(
        x, travel, layer, output_positive, neg_samples_num_list,
        layer_offset_lod, seed, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tile(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tile op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tile", 0);

    // Parse Attributes
    PyObject *repeat_times_obj = PyTuple_GET_ITEM(args, 1);

    // Check for mutable attrs
    pir::Value repeat_times;

    if (PyObject_CheckIRValue(repeat_times_obj)) {
      repeat_times = CastPyArg2Value(repeat_times_obj, "tile", 1);
    } else if (PyObject_CheckIRVectorOfValue(repeat_times_obj)) {
      std::vector<pir::Value> repeat_times_tmp =
          CastPyArg2VectorOfValue(repeat_times_obj, "tile", 1);
      repeat_times = paddle::dialect::stack(repeat_times_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> repeat_times_tmp =
          CastPyArg2Longs(repeat_times_obj, "tile", 1);
      repeat_times = paddle::dialect::full_int_array(
          repeat_times_tmp, phi::DataType::INT64, phi::CPUPlace());
    }

    // Call ir static api
    CallStackRecorder callstack_recorder("tile");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tile(x, repeat_times);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_trans_layout(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add trans_layout op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "trans_layout", 0);

    // Parse Attributes
    PyObject *perm_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> perm = CastPyArg2Ints(perm_obj, "trans_layout", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("trans_layout");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::trans_layout(x, perm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_transpose(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add transpose op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "transpose", 0);

    // Parse Attributes
    PyObject *perm_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> perm = CastPyArg2Ints(perm_obj, "transpose", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("transpose");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::transpose(x, perm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_transpose_(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  try {
    VLOG(6) << "Add transpose_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "transpose_", 0);

    // Parse Attributes
    PyObject *perm_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> perm = CastPyArg2Ints(perm_obj, "transpose_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("transpose_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::transpose_(x, perm);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tril(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tril op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tril", 0);

    // Parse Attributes
    PyObject *diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "tril", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("tril");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tril(x, diagonal);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tril_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add tril_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "tril_", 0);

    // Parse Attributes
    PyObject *diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "tril_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("tril_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::tril_(x, diagonal);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_tril_indices(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add tril_indices op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *rows_obj = PyTuple_GET_ITEM(args, 0);
    int rows = CastPyArg2Int(rows_obj, "tril_indices", 0);
    PyObject *cols_obj = PyTuple_GET_ITEM(args, 1);
    int cols = CastPyArg2Int(cols_obj, "tril_indices", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "tril_indices", 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "tril_indices", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    Place place = CastPyArg2Place(place_obj, "tril_indices", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("tril_indices");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::tril_indices(rows, cols, offset, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_triu(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add triu op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "triu", 0);

    // Parse Attributes
    PyObject *diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "triu", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("triu");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::triu(x, diagonal);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_triu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add triu_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "triu_", 0);

    // Parse Attributes
    PyObject *diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "triu_", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("triu_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::triu_(x, diagonal);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_triu_indices(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add triu_indices op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *row_obj = PyTuple_GET_ITEM(args, 0);
    int row = CastPyArg2Int(row_obj, "triu_indices", 0);
    PyObject *col_obj = PyTuple_GET_ITEM(args, 1);
    int col = CastPyArg2Int(col_obj, "triu_indices", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "triu_indices", 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "triu_indices", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    Place place = CastPyArg2Place(place_obj, "triu_indices", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("triu_indices");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::triu_indices(row, col, offset, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_truncated_gaussian_random(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  try {
    VLOG(6) << "Add truncated_gaussian_random op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int> shape =
        CastPyArg2Ints(shape_obj, "truncated_gaussian_random", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "truncated_gaussian_random", 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "truncated_gaussian_random", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "truncated_gaussian_random", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "truncated_gaussian_random", 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);
    Place place = CastPyArg2Place(place_obj, "truncated_gaussian_random", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("truncated_gaussian_random");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::truncated_gaussian_random(
        shape, mean, std, seed, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_uniform(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add uniform op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *min_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);

    // Check for mutable attrs
    pir::Value shape;

    pir::Value min;

    pir::Value max;

    if (PyObject_CheckIRValue(shape_obj)) {
      shape = CastPyArg2Value(shape_obj, "uniform", 0);
    } else if (PyObject_CheckIRVectorOfValue(shape_obj)) {
      std::vector<pir::Value> shape_tmp =
          CastPyArg2VectorOfValue(shape_obj, "uniform", 0);
      shape = paddle::dialect::stack(shape_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "uniform", 0);
      shape = paddle::dialect::full_int_array(shape_tmp, phi::DataType::INT64,
                                              phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "uniform", 1);
    if (PyObject_CheckIRValue(min_obj)) {
      min = CastPyArg2Value(min_obj, "uniform", 2);
    } else {
      float min_tmp = CastPyArg2Float(min_obj, "uniform", 2);
      min = paddle::dialect::full(std::vector<int64_t>{1}, min_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(max_obj)) {
      max = CastPyArg2Value(max_obj, "uniform", 3);
    } else {
      float max_tmp = CastPyArg2Float(max_obj, "uniform", 3);
      max = paddle::dialect::full(std::vector<int64_t>{1}, max_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }
    int seed = CastPyArg2Int(seed_obj, "uniform", 4);
    Place place = CastPyArg2Place(place_obj, "uniform", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("uniform");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::uniform(shape, min, max, dtype, seed, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_uniform_random_batch_size_like(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add uniform_random_batch_size_like op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input =
        CastPyArg2Value(input_obj, "uniform_random_batch_size_like", 0);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape =
        CastPyArg2Ints(shape_obj, "uniform_random_batch_size_like", 1);
    PyObject *input_dim_idx_obj = PyTuple_GET_ITEM(args, 2);
    int input_dim_idx =
        CastPyArg2Int(input_dim_idx_obj, "uniform_random_batch_size_like", 2);
    PyObject *output_dim_idx_obj = PyTuple_GET_ITEM(args, 3);
    int output_dim_idx =
        CastPyArg2Int(output_dim_idx_obj, "uniform_random_batch_size_like", 3);
    PyObject *min_obj = PyTuple_GET_ITEM(args, 4);
    float min = CastPyArg2Float(min_obj, "uniform_random_batch_size_like", 4);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 5);
    float max = CastPyArg2Float(max_obj, "uniform_random_batch_size_like", 5);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 6);
    int seed = CastPyArg2Int(seed_obj, "uniform_random_batch_size_like", 6);
    PyObject *diag_num_obj = PyTuple_GET_ITEM(args, 7);
    int diag_num =
        CastPyArg2Int(diag_num_obj, "uniform_random_batch_size_like", 7);
    PyObject *diag_step_obj = PyTuple_GET_ITEM(args, 8);
    int diag_step =
        CastPyArg2Int(diag_step_obj, "uniform_random_batch_size_like", 8);
    PyObject *diag_val_obj = PyTuple_GET_ITEM(args, 9);
    float diag_val =
        CastPyArg2Float(diag_val_obj, "uniform_random_batch_size_like", 9);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 10);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(
        dtype_obj, "uniform_random_batch_size_like", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("uniform_random_batch_size_like");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::uniform_random_batch_size_like(
        input, shape, input_dim_idx, output_dim_idx, min, max, seed, diag_num,
        diag_step, diag_val, dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unique(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add unique op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unique", 0);

    // Parse Attributes
    PyObject *return_index_obj = PyTuple_GET_ITEM(args, 1);
    bool return_index = CastPyArg2Boolean(return_index_obj, "unique", 1);
    PyObject *return_inverse_obj = PyTuple_GET_ITEM(args, 2);
    bool return_inverse = CastPyArg2Boolean(return_inverse_obj, "unique", 2);
    PyObject *return_counts_obj = PyTuple_GET_ITEM(args, 3);
    bool return_counts = CastPyArg2Boolean(return_counts_obj, "unique", 3);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "unique", 4);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 5);
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "unique", 5);
    PyObject *is_sorted_obj = PyTuple_GET_ITEM(args, 6);
    bool is_sorted = CastPyArg2Boolean(is_sorted_obj, "unique", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("unique");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unique(
        x, return_index, return_inverse, return_counts, axis, dtype, is_sorted);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_unpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add unpool op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "unpool", 0);
    PyObject *indices_obj = PyTuple_GET_ITEM(args, 1);
    auto indices = CastPyArg2Value(indices_obj, "unpool", 1);

    // Parse Attributes
    PyObject *ksize_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *padding_obj = PyTuple_GET_ITEM(args, 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);

    // Check for mutable attrs
    pir::Value output_size;

    std::vector<int> ksize = CastPyArg2Ints(ksize_obj, "unpool", 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unpool", 3);
    std::vector<int> padding = CastPyArg2Ints(padding_obj, "unpool", 4);
    if (PyObject_CheckIRValue(output_size_obj)) {
      output_size = CastPyArg2Value(output_size_obj, "unpool", 5);
    } else if (PyObject_CheckIRVectorOfValue(output_size_obj)) {
      std::vector<pir::Value> output_size_tmp =
          CastPyArg2VectorOfValue(output_size_obj, "unpool", 5);
      output_size = paddle::dialect::stack(output_size_tmp, /*axis*/ 0);

    } else {
      std::vector<int64_t> output_size_tmp =
          CastPyArg2Longs(output_size_obj, "unpool", 5);
      output_size = paddle::dialect::full_int_array(
          output_size_tmp, phi::DataType::INT64, phi::CPUPlace());
    }
    std::string data_format = CastPyArg2String(data_format_obj, "unpool", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("unpool");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::unpool(
        x, indices, output_size, ksize, strides, padding, data_format);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_c_softmax_with_cross_entropy(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add c_softmax_with_cross_entropy op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *logits_obj = PyTuple_GET_ITEM(args, 0);
    auto logits =
        CastPyArg2Value(logits_obj, "c_softmax_with_cross_entropy", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "c_softmax_with_cross_entropy", 1);

    // Parse Attributes
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 2);
    int64_t ignore_index =
        CastPyArg2Long(ignore_index_obj, "c_softmax_with_cross_entropy", 2);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "c_softmax_with_cross_entropy", 3);
    PyObject *rank_obj = PyTuple_GET_ITEM(args, 4);
    int rank = CastPyArg2Int(rank_obj, "c_softmax_with_cross_entropy", 4);
    PyObject *nranks_obj = PyTuple_GET_ITEM(args, 5);
    int nranks = CastPyArg2Int(nranks_obj, "c_softmax_with_cross_entropy", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("c_softmax_with_cross_entropy");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::c_softmax_with_cross_entropy(
        logits, label, ignore_index, ring_id, rank, nranks);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_dpsgd(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add dpsgd op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "dpsgd", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "dpsgd", 1);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 2);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "dpsgd", 2);

    // Parse Attributes
    PyObject *clip_obj = PyTuple_GET_ITEM(args, 3);
    float clip = CastPyArg2Float(clip_obj, "dpsgd", 3);
    PyObject *batch_size_obj = PyTuple_GET_ITEM(args, 4);
    float batch_size = CastPyArg2Float(batch_size_obj, "dpsgd", 4);
    PyObject *sigma_obj = PyTuple_GET_ITEM(args, 5);
    float sigma = CastPyArg2Float(sigma_obj, "dpsgd", 5);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 6);
    int seed = CastPyArg2Int(seed_obj, "dpsgd", 6);

    // Call ir static api
    CallStackRecorder callstack_recorder("dpsgd");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::dpsgd(param, grad, learning_rate,
                                                 clip, batch_size, sigma, seed);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_ftrl(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add ftrl op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "ftrl", 0);
    PyObject *squared_accumulator_obj = PyTuple_GET_ITEM(args, 1);
    auto squared_accumulator =
        CastPyArg2Value(squared_accumulator_obj, "ftrl", 1);
    PyObject *linear_accumulator_obj = PyTuple_GET_ITEM(args, 2);
    auto linear_accumulator =
        CastPyArg2Value(linear_accumulator_obj, "ftrl", 2);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 3);
    auto grad = CastPyArg2Value(grad_obj, "ftrl", 3);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 4);
    auto learning_rate = CastPyArg2Value(learning_rate_obj, "ftrl", 4);

    // Parse Attributes
    PyObject *l1_obj = PyTuple_GET_ITEM(args, 5);
    float l1 = CastPyArg2Float(l1_obj, "ftrl", 5);
    PyObject *l2_obj = PyTuple_GET_ITEM(args, 6);
    float l2 = CastPyArg2Float(l2_obj, "ftrl", 6);
    PyObject *lr_power_obj = PyTuple_GET_ITEM(args, 7);
    float lr_power = CastPyArg2Float(lr_power_obj, "ftrl", 7);

    // Call ir static api
    CallStackRecorder callstack_recorder("ftrl");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::ftrl(param, squared_accumulator, linear_accumulator,
                              grad, learning_rate, l1, l2, lr_power);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_attention(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_attention op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_attention", 0);
    PyObject *ln_scale_obj = PyTuple_GET_ITEM(args, 1);
    auto ln_scale = CastPyArg2OptionalValue(ln_scale_obj, "fused_attention", 1);
    PyObject *ln_bias_obj = PyTuple_GET_ITEM(args, 2);
    auto ln_bias = CastPyArg2OptionalValue(ln_bias_obj, "fused_attention", 2);
    PyObject *qkv_weight_obj = PyTuple_GET_ITEM(args, 3);
    auto qkv_weight = CastPyArg2Value(qkv_weight_obj, "fused_attention", 3);
    PyObject *qkv_bias_obj = PyTuple_GET_ITEM(args, 4);
    auto qkv_bias = CastPyArg2OptionalValue(qkv_bias_obj, "fused_attention", 4);
    PyObject *cache_kv_obj = PyTuple_GET_ITEM(args, 5);
    auto cache_kv = CastPyArg2OptionalValue(cache_kv_obj, "fused_attention", 5);
    PyObject *src_mask_obj = PyTuple_GET_ITEM(args, 6);
    auto src_mask = CastPyArg2OptionalValue(src_mask_obj, "fused_attention", 6);
    PyObject *out_linear_weight_obj = PyTuple_GET_ITEM(args, 7);
    auto out_linear_weight =
        CastPyArg2Value(out_linear_weight_obj, "fused_attention", 7);
    PyObject *out_linear_bias_obj = PyTuple_GET_ITEM(args, 8);
    auto out_linear_bias =
        CastPyArg2OptionalValue(out_linear_bias_obj, "fused_attention", 8);
    PyObject *ln_scale_2_obj = PyTuple_GET_ITEM(args, 9);
    auto ln_scale_2 =
        CastPyArg2OptionalValue(ln_scale_2_obj, "fused_attention", 9);
    PyObject *ln_bias_2_obj = PyTuple_GET_ITEM(args, 10);
    auto ln_bias_2 =
        CastPyArg2OptionalValue(ln_bias_2_obj, "fused_attention", 10);

    // Parse Attributes
    PyObject *num_heads_obj = PyTuple_GET_ITEM(args, 11);
    int num_heads = CastPyArg2Int(num_heads_obj, "fused_attention", 11);
    PyObject *transpose_qkv_wb_obj = PyTuple_GET_ITEM(args, 12);
    bool transpose_qkv_wb =
        CastPyArg2Boolean(transpose_qkv_wb_obj, "fused_attention", 12);
    PyObject *pre_layer_norm_obj = PyTuple_GET_ITEM(args, 13);
    bool pre_layer_norm =
        CastPyArg2Boolean(pre_layer_norm_obj, "fused_attention", 13);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 14);
    float epsilon = CastPyArg2Float(epsilon_obj, "fused_attention", 14);
    PyObject *attn_dropout_rate_obj = PyTuple_GET_ITEM(args, 15);
    float attn_dropout_rate =
        CastPyArg2Float(attn_dropout_rate_obj, "fused_attention", 15);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 16);
    bool is_test = CastPyArg2Boolean(is_test_obj, "fused_attention", 16);
    PyObject *attn_dropout_fix_seed_obj = PyTuple_GET_ITEM(args, 17);
    bool attn_dropout_fix_seed =
        CastPyArg2Boolean(attn_dropout_fix_seed_obj, "fused_attention", 17);
    PyObject *attn_dropout_seed_obj = PyTuple_GET_ITEM(args, 18);
    int attn_dropout_seed =
        CastPyArg2Int(attn_dropout_seed_obj, "fused_attention", 18);
    PyObject *attn_dropout_implementation_obj = PyTuple_GET_ITEM(args, 19);
    std::string attn_dropout_implementation = CastPyArg2String(
        attn_dropout_implementation_obj, "fused_attention", 19);
    PyObject *dropout_rate_obj = PyTuple_GET_ITEM(args, 20);
    float dropout_rate =
        CastPyArg2Float(dropout_rate_obj, "fused_attention", 20);
    PyObject *dropout_fix_seed_obj = PyTuple_GET_ITEM(args, 21);
    bool dropout_fix_seed =
        CastPyArg2Boolean(dropout_fix_seed_obj, "fused_attention", 21);
    PyObject *dropout_seed_obj = PyTuple_GET_ITEM(args, 22);
    int dropout_seed = CastPyArg2Int(dropout_seed_obj, "fused_attention", 22);
    PyObject *dropout_implementation_obj = PyTuple_GET_ITEM(args, 23);
    std::string dropout_implementation =
        CastPyArg2String(dropout_implementation_obj, "fused_attention", 23);
    PyObject *ln_epsilon_obj = PyTuple_GET_ITEM(args, 24);
    float ln_epsilon = CastPyArg2Float(ln_epsilon_obj, "fused_attention", 24);
    PyObject *add_residual_obj = PyTuple_GET_ITEM(args, 25);
    bool add_residual =
        CastPyArg2Boolean(add_residual_obj, "fused_attention", 25);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 26);
    int ring_id = CastPyArg2Int(ring_id_obj, "fused_attention", 26);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_attention");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_attention(
        x, ln_scale, ln_bias, qkv_weight, qkv_bias, cache_kv, src_mask,
        out_linear_weight, out_linear_bias, ln_scale_2, ln_bias_2, num_heads,
        transpose_qkv_wb, pre_layer_norm, epsilon, attn_dropout_rate, is_test,
        attn_dropout_fix_seed, attn_dropout_seed, attn_dropout_implementation,
        dropout_rate, dropout_fix_seed, dropout_seed, dropout_implementation,
        ln_epsilon, add_residual, ring_id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_elemwise_add_activation(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_elemwise_add_activation op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_elemwise_add_activation", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "fused_elemwise_add_activation", 1);

    // Parse Attributes
    PyObject *functor_list_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<std::string> functor_list =
        CastPyArg2Strings(functor_list_obj, "fused_elemwise_add_activation", 2);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 3);
    float scale =
        CastPyArg2Float(scale_obj, "fused_elemwise_add_activation", 3);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 4);
    int axis = CastPyArg2Int(axis_obj, "fused_elemwise_add_activation", 4);
    PyObject *save_intermediate_out_obj = PyTuple_GET_ITEM(args, 5);
    bool save_intermediate_out = CastPyArg2Boolean(
        save_intermediate_out_obj, "fused_elemwise_add_activation", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_elemwise_add_activation");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_elemwise_add_activation(
        x, y, functor_list, scale, axis, save_intermediate_out);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_fused_feedforward(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add fused_feedforward op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "fused_feedforward", 0);
    PyObject *dropout1_seed_obj = PyTuple_GET_ITEM(args, 1);
    auto dropout1_seed =
        CastPyArg2OptionalValue(dropout1_seed_obj, "fused_feedforward", 1);
    PyObject *dropout2_seed_obj = PyTuple_GET_ITEM(args, 2);
    auto dropout2_seed =
        CastPyArg2OptionalValue(dropout2_seed_obj, "fused_feedforward", 2);
    PyObject *linear1_weight_obj = PyTuple_GET_ITEM(args, 3);
    auto linear1_weight =
        CastPyArg2Value(linear1_weight_obj, "fused_feedforward", 3);
    PyObject *linear1_bias_obj = PyTuple_GET_ITEM(args, 4);
    auto linear1_bias =
        CastPyArg2OptionalValue(linear1_bias_obj, "fused_feedforward", 4);
    PyObject *linear2_weight_obj = PyTuple_GET_ITEM(args, 5);
    auto linear2_weight =
        CastPyArg2Value(linear2_weight_obj, "fused_feedforward", 5);
    PyObject *linear2_bias_obj = PyTuple_GET_ITEM(args, 6);
    auto linear2_bias =
        CastPyArg2OptionalValue(linear2_bias_obj, "fused_feedforward", 6);
    PyObject *ln1_scale_obj = PyTuple_GET_ITEM(args, 7);
    auto ln1_scale =
        CastPyArg2OptionalValue(ln1_scale_obj, "fused_feedforward", 7);
    PyObject *ln1_bias_obj = PyTuple_GET_ITEM(args, 8);
    auto ln1_bias =
        CastPyArg2OptionalValue(ln1_bias_obj, "fused_feedforward", 8);
    PyObject *ln2_scale_obj = PyTuple_GET_ITEM(args, 9);
    auto ln2_scale =
        CastPyArg2OptionalValue(ln2_scale_obj, "fused_feedforward", 9);
    PyObject *ln2_bias_obj = PyTuple_GET_ITEM(args, 10);
    auto ln2_bias =
        CastPyArg2OptionalValue(ln2_bias_obj, "fused_feedforward", 10);

    // Parse Attributes
    PyObject *pre_layer_norm_obj = PyTuple_GET_ITEM(args, 11);
    bool pre_layer_norm =
        CastPyArg2Boolean(pre_layer_norm_obj, "fused_feedforward", 11);
    PyObject *ln1_epsilon_obj = PyTuple_GET_ITEM(args, 12);
    float ln1_epsilon =
        CastPyArg2Float(ln1_epsilon_obj, "fused_feedforward", 12);
    PyObject *ln2_epsilon_obj = PyTuple_GET_ITEM(args, 13);
    float ln2_epsilon =
        CastPyArg2Float(ln2_epsilon_obj, "fused_feedforward", 13);
    PyObject *act_method_obj = PyTuple_GET_ITEM(args, 14);
    std::string act_method =
        CastPyArg2String(act_method_obj, "fused_feedforward", 14);
    PyObject *dropout1_prob_obj = PyTuple_GET_ITEM(args, 15);
    float dropout1_prob =
        CastPyArg2Float(dropout1_prob_obj, "fused_feedforward", 15);
    PyObject *dropout2_prob_obj = PyTuple_GET_ITEM(args, 16);
    float dropout2_prob =
        CastPyArg2Float(dropout2_prob_obj, "fused_feedforward", 16);
    PyObject *dropout1_implementation_obj = PyTuple_GET_ITEM(args, 17);
    std::string dropout1_implementation =
        CastPyArg2String(dropout1_implementation_obj, "fused_feedforward", 17);
    PyObject *dropout2_implementation_obj = PyTuple_GET_ITEM(args, 18);
    std::string dropout2_implementation =
        CastPyArg2String(dropout2_implementation_obj, "fused_feedforward", 18);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 19);
    bool is_test = CastPyArg2Boolean(is_test_obj, "fused_feedforward", 19);
    PyObject *dropout1_fix_seed_obj = PyTuple_GET_ITEM(args, 20);
    bool dropout1_fix_seed =
        CastPyArg2Boolean(dropout1_fix_seed_obj, "fused_feedforward", 20);
    PyObject *dropout2_fix_seed_obj = PyTuple_GET_ITEM(args, 21);
    bool dropout2_fix_seed =
        CastPyArg2Boolean(dropout2_fix_seed_obj, "fused_feedforward", 21);
    PyObject *dropout1_seed_val_obj = PyTuple_GET_ITEM(args, 22);
    int dropout1_seed_val =
        CastPyArg2Int(dropout1_seed_val_obj, "fused_feedforward", 22);
    PyObject *dropout2_seed_val_obj = PyTuple_GET_ITEM(args, 23);
    int dropout2_seed_val =
        CastPyArg2Int(dropout2_seed_val_obj, "fused_feedforward", 23);
    PyObject *add_residual_obj = PyTuple_GET_ITEM(args, 24);
    bool add_residual =
        CastPyArg2Boolean(add_residual_obj, "fused_feedforward", 24);
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 25);
    int ring_id = CastPyArg2Int(ring_id_obj, "fused_feedforward", 25);

    // Call ir static api
    CallStackRecorder callstack_recorder("fused_feedforward");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::fused_feedforward(
        x, dropout1_seed, dropout2_seed, linear1_weight, linear1_bias,
        linear2_weight, linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
        pre_layer_norm, ln1_epsilon, ln2_epsilon, act_method, dropout1_prob,
        dropout2_prob, dropout1_implementation, dropout2_implementation,
        is_test, dropout1_fix_seed, dropout2_fix_seed, dropout1_seed_val,
        dropout2_seed_val, add_residual, ring_id);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lars_momentum(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add lars_momentum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2VectorOfValue(param_obj, "lars_momentum", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2VectorOfValue(grad_obj, "lars_momentum", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity = CastPyArg2VectorOfValue(velocity_obj, "lars_momentum", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate =
        CastPyArg2VectorOfValue(learning_rate_obj, "lars_momentum", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param =
        CastPyArg2OptionalVectorOfValue(master_param_obj, "lars_momentum", 4);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "lars_momentum", 5);
    PyObject *lars_coeff_obj = PyTuple_GET_ITEM(args, 6);
    float lars_coeff = CastPyArg2Float(lars_coeff_obj, "lars_momentum", 6);
    PyObject *lars_weight_decay_obj = PyTuple_GET_ITEM(args, 7);
    std::vector<float> lars_weight_decay =
        CastPyArg2Floats(lars_weight_decay_obj, "lars_momentum", 7);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 8);
    float epsilon = CastPyArg2Float(epsilon_obj, "lars_momentum", 8);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "lars_momentum", 9);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad = CastPyArg2Float(rescale_grad_obj, "lars_momentum", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("lars_momentum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lars_momentum(
        param, grad, velocity, learning_rate, master_param, mu, lars_coeff,
        lars_weight_decay, epsilon, multi_precision, rescale_grad);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_lars_momentum_(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add lars_momentum_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2VectorOfValue(param_obj, "lars_momentum_", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2VectorOfValue(grad_obj, "lars_momentum_", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity = CastPyArg2VectorOfValue(velocity_obj, "lars_momentum_", 2);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 3);
    auto learning_rate =
        CastPyArg2VectorOfValue(learning_rate_obj, "lars_momentum_", 3);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 4);
    auto master_param =
        CastPyArg2OptionalVectorOfValue(master_param_obj, "lars_momentum_", 4);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "lars_momentum_", 5);
    PyObject *lars_coeff_obj = PyTuple_GET_ITEM(args, 6);
    float lars_coeff = CastPyArg2Float(lars_coeff_obj, "lars_momentum_", 6);
    PyObject *lars_weight_decay_obj = PyTuple_GET_ITEM(args, 7);
    std::vector<float> lars_weight_decay =
        CastPyArg2Floats(lars_weight_decay_obj, "lars_momentum_", 7);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 8);
    float epsilon = CastPyArg2Float(epsilon_obj, "lars_momentum_", 8);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "lars_momentum_", 9);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad =
        CastPyArg2Float(rescale_grad_obj, "lars_momentum_", 10);

    // Call ir static api
    CallStackRecorder callstack_recorder("lars_momentum_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::lars_momentum_(
        param, grad, velocity, learning_rate, master_param, mu, lars_coeff,
        lars_weight_decay, epsilon, multi_precision, rescale_grad);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_match_matrix_tensor(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add match_matrix_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "match_matrix_tensor", 0);
    PyObject *y_obj = PyTuple_GET_ITEM(args, 1);
    auto y = CastPyArg2Value(y_obj, "match_matrix_tensor", 1);
    PyObject *w_obj = PyTuple_GET_ITEM(args, 2);
    auto w = CastPyArg2Value(w_obj, "match_matrix_tensor", 2);

    // Parse Attributes
    PyObject *dim_t_obj = PyTuple_GET_ITEM(args, 3);
    int dim_t = CastPyArg2Int(dim_t_obj, "match_matrix_tensor", 3);

    // Call ir static api
    CallStackRecorder callstack_recorder("match_matrix_tensor");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::match_matrix_tensor(x, y, w, dim_t);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_moving_average_abs_max_scale(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add moving_average_abs_max_scale op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "moving_average_abs_max_scale", 0);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 1);
    auto in_accum = CastPyArg2OptionalValue(in_accum_obj,
                                            "moving_average_abs_max_scale", 1);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 2);
    auto in_state = CastPyArg2OptionalValue(in_state_obj,
                                            "moving_average_abs_max_scale", 2);

    // Parse Attributes
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 3);
    float moving_rate =
        CastPyArg2Float(moving_rate_obj, "moving_average_abs_max_scale", 3);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 4);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "moving_average_abs_max_scale", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("moving_average_abs_max_scale");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::moving_average_abs_max_scale(
        x, in_accum, in_state, moving_rate, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_moving_average_abs_max_scale_(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add moving_average_abs_max_scale_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "moving_average_abs_max_scale_", 0);
    PyObject *in_accum_obj = PyTuple_GET_ITEM(args, 1);
    auto in_accum = CastPyArg2OptionalValue(in_accum_obj,
                                            "moving_average_abs_max_scale_", 1);
    PyObject *in_state_obj = PyTuple_GET_ITEM(args, 2);
    auto in_state = CastPyArg2OptionalValue(in_state_obj,
                                            "moving_average_abs_max_scale_", 2);

    // Parse Attributes
    PyObject *moving_rate_obj = PyTuple_GET_ITEM(args, 3);
    float moving_rate =
        CastPyArg2Float(moving_rate_obj, "moving_average_abs_max_scale_", 3);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 4);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "moving_average_abs_max_scale_", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("moving_average_abs_max_scale_");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::moving_average_abs_max_scale_(
        x, in_accum, in_state, moving_rate, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_nce(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add nce op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "nce", 0);
    PyObject *label_obj = PyTuple_GET_ITEM(args, 1);
    auto label = CastPyArg2Value(label_obj, "nce", 1);
    PyObject *weight_obj = PyTuple_GET_ITEM(args, 2);
    auto weight = CastPyArg2Value(weight_obj, "nce", 2);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 3);
    auto bias = CastPyArg2OptionalValue(bias_obj, "nce", 3);
    PyObject *sample_weight_obj = PyTuple_GET_ITEM(args, 4);
    auto sample_weight = CastPyArg2OptionalValue(sample_weight_obj, "nce", 4);
    PyObject *custom_dist_probs_obj = PyTuple_GET_ITEM(args, 5);
    auto custom_dist_probs =
        CastPyArg2OptionalValue(custom_dist_probs_obj, "nce", 5);
    PyObject *custom_dist_alias_obj = PyTuple_GET_ITEM(args, 6);
    auto custom_dist_alias =
        CastPyArg2OptionalValue(custom_dist_alias_obj, "nce", 6);
    PyObject *custom_dist_alias_probs_obj = PyTuple_GET_ITEM(args, 7);
    auto custom_dist_alias_probs =
        CastPyArg2OptionalValue(custom_dist_alias_probs_obj, "nce", 7);

    // Parse Attributes
    PyObject *num_total_classes_obj = PyTuple_GET_ITEM(args, 8);
    int num_total_classes = CastPyArg2Int(num_total_classes_obj, "nce", 8);
    PyObject *custom_neg_classes_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<int> custom_neg_classes =
        CastPyArg2Ints(custom_neg_classes_obj, "nce", 9);
    PyObject *num_neg_samples_obj = PyTuple_GET_ITEM(args, 10);
    int num_neg_samples = CastPyArg2Int(num_neg_samples_obj, "nce", 10);
    PyObject *sampler_obj = PyTuple_GET_ITEM(args, 11);
    int sampler = CastPyArg2Int(sampler_obj, "nce", 11);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 12);
    int seed = CastPyArg2Int(seed_obj, "nce", 12);
    PyObject *is_sparse_obj = PyTuple_GET_ITEM(args, 13);
    bool is_sparse = CastPyArg2Boolean(is_sparse_obj, "nce", 13);
    PyObject *remote_prefetch_obj = PyTuple_GET_ITEM(args, 14);
    bool remote_prefetch = CastPyArg2Boolean(remote_prefetch_obj, "nce", 14);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 15);
    bool is_test = CastPyArg2Boolean(is_test_obj, "nce", 15);

    // Call ir static api
    CallStackRecorder callstack_recorder("nce");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::nce(
        input, label, weight, bias, sample_weight, custom_dist_probs,
        custom_dist_alias, custom_dist_alias_probs, num_total_classes,
        custom_neg_classes, num_neg_samples, sampler, seed, is_sparse,
        remote_prefetch, is_test);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_number_count(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add number_count op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *numbers_obj = PyTuple_GET_ITEM(args, 0);
    auto numbers = CastPyArg2Value(numbers_obj, "number_count", 0);

    // Parse Attributes
    PyObject *upper_range_obj = PyTuple_GET_ITEM(args, 1);
    int upper_range = CastPyArg2Int(upper_range_obj, "number_count", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("number_count");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::number_count(numbers, upper_range);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_onednn_to_paddle_layout(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  try {
    VLOG(6) << "Add onednn_to_paddle_layout op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "onednn_to_paddle_layout", 0);

    // Parse Attributes
    PyObject *dst_layout_obj = PyTuple_GET_ITEM(args, 1);
    int dst_layout =
        CastPyArg2Int(dst_layout_obj, "onednn_to_paddle_layout", 1);

    // Call ir static api
    CallStackRecorder callstack_recorder("onednn_to_paddle_layout");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::onednn_to_paddle_layout(x, dst_layout);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_partial_send(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  try {
    VLOG(6) << "Add partial_send op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "partial_send", 0);

    // Parse Attributes
    PyObject *ring_id_obj = PyTuple_GET_ITEM(args, 1);
    int ring_id = CastPyArg2Int(ring_id_obj, "partial_send", 1);
    PyObject *peer_obj = PyTuple_GET_ITEM(args, 2);
    int peer = CastPyArg2Int(peer_obj, "partial_send", 2);
    PyObject *use_calc_stream_obj = PyTuple_GET_ITEM(args, 3);
    bool use_calc_stream =
        CastPyArg2Boolean(use_calc_stream_obj, "partial_send", 3);
    PyObject *num_obj = PyTuple_GET_ITEM(args, 4);
    int num = CastPyArg2Int(num_obj, "partial_send", 4);
    PyObject *id_obj = PyTuple_GET_ITEM(args, 5);
    int id = CastPyArg2Int(id_obj, "partial_send", 5);

    // Call ir static api
    CallStackRecorder callstack_recorder("partial_send");
    callstack_recorder.Record();
    paddle::dialect::partial_send(x, ring_id, peer, use_calc_stream, num, id);
    callstack_recorder.AttachToOps();
    return nullptr;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sparse_momentum(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  try {
    VLOG(6) << "Add sparse_momentum op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *param_obj = PyTuple_GET_ITEM(args, 0);
    auto param = CastPyArg2Value(param_obj, "sparse_momentum", 0);
    PyObject *grad_obj = PyTuple_GET_ITEM(args, 1);
    auto grad = CastPyArg2Value(grad_obj, "sparse_momentum", 1);
    PyObject *velocity_obj = PyTuple_GET_ITEM(args, 2);
    auto velocity = CastPyArg2Value(velocity_obj, "sparse_momentum", 2);
    PyObject *index_obj = PyTuple_GET_ITEM(args, 3);
    auto index = CastPyArg2Value(index_obj, "sparse_momentum", 3);
    PyObject *learning_rate_obj = PyTuple_GET_ITEM(args, 4);
    auto learning_rate =
        CastPyArg2Value(learning_rate_obj, "sparse_momentum", 4);
    PyObject *master_param_obj = PyTuple_GET_ITEM(args, 5);
    auto master_param =
        CastPyArg2OptionalValue(master_param_obj, "sparse_momentum", 5);

    // Parse Attributes
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 6);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 7);
    PyObject *use_nesterov_obj = PyTuple_GET_ITEM(args, 8);
    PyObject *regularization_method_obj = PyTuple_GET_ITEM(args, 9);
    PyObject *regularization_coeff_obj = PyTuple_GET_ITEM(args, 10);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 11);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 12);

    // Check for mutable attrs
    pir::Value axis;

    float mu = CastPyArg2Float(mu_obj, "sparse_momentum", 6);
    if (PyObject_CheckIRValue(axis_obj)) {
      axis = CastPyArg2Value(axis_obj, "sparse_momentum", 7);
    } else {
      float axis_tmp = CastPyArg2Float(axis_obj, "sparse_momentum", 7);
      axis = paddle::dialect::full(std::vector<int64_t>{1}, axis_tmp,
                                   phi::DataType::FLOAT32, phi::CPUPlace());
    }
    bool use_nesterov =
        CastPyArg2Boolean(use_nesterov_obj, "sparse_momentum", 8);
    std::string regularization_method =
        CastPyArg2String(regularization_method_obj, "sparse_momentum", 9);
    float regularization_coeff =
        CastPyArg2Float(regularization_coeff_obj, "sparse_momentum", 10);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "sparse_momentum", 11);
    float rescale_grad =
        CastPyArg2Float(rescale_grad_obj, "sparse_momentum", 12);

    // Call ir static api
    CallStackRecorder callstack_recorder("sparse_momentum");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sparse_momentum(
        param, grad, velocity, index, learning_rate, master_param, axis, mu,
        use_nesterov, regularization_method, regularization_coeff,
        multi_precision, rescale_grad);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_arange(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add arange op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args

    // Parse Attributes
    PyObject *start_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *end_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *step_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);

    // Check for mutable attrs
    pir::Value start;

    pir::Value end;

    pir::Value step;

    if (PyObject_CheckIRValue(start_obj)) {
      start = CastPyArg2Value(start_obj, "arange", 0);
    } else {
      float start_tmp = CastPyArg2Float(start_obj, "arange", 0);
      start = paddle::dialect::full(std::vector<int64_t>{1}, start_tmp,
                                    phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(end_obj)) {
      end = CastPyArg2Value(end_obj, "arange", 1);
    } else {
      float end_tmp = CastPyArg2Float(end_obj, "arange", 1);
      end = paddle::dialect::full(std::vector<int64_t>{1}, end_tmp,
                                  phi::DataType::FLOAT32, phi::CPUPlace());
    }
    if (PyObject_CheckIRValue(step_obj)) {
      step = CastPyArg2Value(step_obj, "arange", 2);
    } else {
      float step_tmp = CastPyArg2Float(step_obj, "arange", 2);
      step = paddle::dialect::full(std::vector<int64_t>{1}, step_tmp,
                                   phi::DataType::FLOAT32, phi::CPUPlace());
    }
    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "arange", 3);
    Place place = CastPyArg2Place(place_obj, "arange", 4);

    // Call ir static api
    CallStackRecorder callstack_recorder("arange");
    callstack_recorder.Record();
    auto static_api_out =
        paddle::dialect::arange(start, end, step, dtype, place);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_sequence_mask(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  try {
    VLOG(6) << "Add sequence_mask op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "sequence_mask", 0);

    // Parse Attributes
    PyObject *max_len_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *out_dtype_obj = PyTuple_GET_ITEM(args, 2);

    // Check for mutable attrs
    pir::Value max_len;

    if (PyObject_CheckIRValue(max_len_obj)) {
      max_len = CastPyArg2Value(max_len_obj, "sequence_mask", 1);
    } else {
      int max_len_tmp = CastPyArg2Int(max_len_obj, "sequence_mask", 1);
      max_len = paddle::dialect::full(std::vector<int64_t>{1}, max_len_tmp,
                                      phi::DataType::INT32, phi::CPUPlace());
    }
    phi::DataType out_dtype =
        CastPyArg2DataTypeDirectly(out_dtype_obj, "sequence_mask", 2);

    // Call ir static api
    CallStackRecorder callstack_recorder("sequence_mask");
    callstack_recorder.Record();
    auto static_api_out = paddle::dialect::sequence_mask(x, max_len, out_dtype);
    callstack_recorder.AttachToOps();
    return ToPyObject(static_api_out);

  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

}  // namespace pybind

}  // namespace paddle
