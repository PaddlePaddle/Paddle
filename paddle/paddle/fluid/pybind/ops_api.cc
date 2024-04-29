
#include <pybind11/pybind11.h>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/pybind/eager_op_function.h"
#include "paddle/fluid/pybind/manual_static_op_function.h"
#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/phi/core/enforce.h"


namespace paddle {

namespace pybind {


static PyObject *abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs";
    return static_api_abs(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs";
    return eager_api_abs(self, args, kwargs);
  }
}
static PyObject *abs_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs_";
    return static_api_abs_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs_";
    return eager_api_abs_(self, args, kwargs);
  }
}
static PyObject *accuracy(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_accuracy";
    return static_api_accuracy(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_accuracy";
    return eager_api_accuracy(self, args, kwargs);
  }
}
static PyObject *acos(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos";
    return static_api_acos(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos";
    return eager_api_acos(self, args, kwargs);
  }
}
static PyObject *acos_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos_";
    return static_api_acos_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos_";
    return eager_api_acos_(self, args, kwargs);
  }
}
static PyObject *acosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh";
    return static_api_acosh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh";
    return eager_api_acosh(self, args, kwargs);
  }
}
static PyObject *acosh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh_";
    return static_api_acosh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh_";
    return eager_api_acosh_(self, args, kwargs);
  }
}
static PyObject *adagrad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adagrad_";
    return static_api_adagrad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adagrad_";
    return eager_api_adagrad_(self, args, kwargs);
  }
}
static PyObject *adam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adam_";
    return static_api_adam_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adam_";
    return eager_api_adam_(self, args, kwargs);
  }
}
static PyObject *adamax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adamax_";
    return static_api_adamax_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adamax_";
    return eager_api_adamax_(self, args, kwargs);
  }
}
static PyObject *adamw_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adamw_";
    return static_api_adamw_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adamw_";
    return eager_api_adamw_(self, args, kwargs);
  }
}
static PyObject *addmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_addmm";
    return static_api_addmm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_addmm";
    return eager_api_addmm(self, args, kwargs);
  }
}
static PyObject *addmm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_addmm_";
    return static_api_addmm_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_addmm_";
    return eager_api_addmm_(self, args, kwargs);
  }
}
static PyObject *affine_grid(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_grid";
    return static_api_affine_grid(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_grid";
    return eager_api_affine_grid(self, args, kwargs);
  }
}
static PyObject *allclose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_allclose";
    return static_api_allclose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_allclose";
    return eager_api_allclose(self, args, kwargs);
  }
}
static PyObject *angle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_angle";
    return static_api_angle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_angle";
    return eager_api_angle(self, args, kwargs);
  }
}
static PyObject *apply_per_channel_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_apply_per_channel_scale";
    return static_api_apply_per_channel_scale(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_apply_per_channel_scale";
    return eager_api_apply_per_channel_scale(self, args, kwargs);
  }
}
static PyObject *argmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_argmax";
    return static_api_argmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_argmax";
    return eager_api_argmax(self, args, kwargs);
  }
}
static PyObject *argmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_argmin";
    return static_api_argmin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_argmin";
    return eager_api_argmin(self, args, kwargs);
  }
}
static PyObject *argsort(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_argsort";
    return static_api_argsort(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_argsort";
    return eager_api_argsort(self, args, kwargs);
  }
}
static PyObject *as_complex(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_as_complex";
    return static_api_as_complex(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_as_complex";
    return eager_api_as_complex(self, args, kwargs);
  }
}
static PyObject *as_real(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_as_real";
    return static_api_as_real(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_as_real";
    return eager_api_as_real(self, args, kwargs);
  }
}
static PyObject *as_strided(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_as_strided";
    return static_api_as_strided(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_as_strided";
    return eager_api_as_strided(self, args, kwargs);
  }
}
static PyObject *asgd_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asgd_";
    return static_api_asgd_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asgd_";
    return eager_api_asgd_(self, args, kwargs);
  }
}
static PyObject *asin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin";
    return static_api_asin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin";
    return eager_api_asin(self, args, kwargs);
  }
}
static PyObject *asin_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin_";
    return static_api_asin_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin_";
    return eager_api_asin_(self, args, kwargs);
  }
}
static PyObject *asinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh";
    return static_api_asinh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh";
    return eager_api_asinh(self, args, kwargs);
  }
}
static PyObject *asinh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh_";
    return static_api_asinh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh_";
    return eager_api_asinh_(self, args, kwargs);
  }
}
static PyObject *atan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan";
    return static_api_atan(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan";
    return eager_api_atan(self, args, kwargs);
  }
}
static PyObject *atan_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan_";
    return static_api_atan_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan_";
    return eager_api_atan_(self, args, kwargs);
  }
}
static PyObject *atan2(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan2";
    return static_api_atan2(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan2";
    return eager_api_atan2(self, args, kwargs);
  }
}
static PyObject *atanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh";
    return static_api_atanh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh";
    return eager_api_atanh(self, args, kwargs);
  }
}
static PyObject *atanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh_";
    return static_api_atanh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh_";
    return eager_api_atanh_(self, args, kwargs);
  }
}
static PyObject *auc(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_auc";
    return static_api_auc(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_auc";
    return eager_api_auc(self, args, kwargs);
  }
}
static PyObject *average_accumulates_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_average_accumulates_";
    return static_api_average_accumulates_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_average_accumulates_";
    return eager_api_average_accumulates_(self, args, kwargs);
  }
}
static PyObject *bce_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bce_loss";
    return static_api_bce_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bce_loss";
    return eager_api_bce_loss(self, args, kwargs);
  }
}
static PyObject *bce_loss_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bce_loss_";
    return static_api_bce_loss_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bce_loss_";
    return eager_api_bce_loss_(self, args, kwargs);
  }
}
static PyObject *bernoulli(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bernoulli";
    return static_api_bernoulli(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bernoulli";
    return eager_api_bernoulli(self, args, kwargs);
  }
}
static PyObject *bicubic_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bicubic_interp";
    return static_api_bicubic_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bicubic_interp";
    return eager_api_bicubic_interp(self, args, kwargs);
  }
}
static PyObject *bilinear(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bilinear";
    return static_api_bilinear(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bilinear";
    return eager_api_bilinear(self, args, kwargs);
  }
}
static PyObject *bilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bilinear_interp";
    return static_api_bilinear_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bilinear_interp";
    return eager_api_bilinear_interp(self, args, kwargs);
  }
}
static PyObject *bincount(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bincount";
    return static_api_bincount(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bincount";
    return eager_api_bincount(self, args, kwargs);
  }
}
static PyObject *binomial(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_binomial";
    return static_api_binomial(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_binomial";
    return eager_api_binomial(self, args, kwargs);
  }
}
static PyObject *bitwise_and(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_and";
    return static_api_bitwise_and(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_and";
    return eager_api_bitwise_and(self, args, kwargs);
  }
}
static PyObject *bitwise_and_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_and_";
    return static_api_bitwise_and_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_and_";
    return eager_api_bitwise_and_(self, args, kwargs);
  }
}
static PyObject *bitwise_left_shift(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_left_shift";
    return static_api_bitwise_left_shift(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_left_shift";
    return eager_api_bitwise_left_shift(self, args, kwargs);
  }
}
static PyObject *bitwise_left_shift_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_left_shift_";
    return static_api_bitwise_left_shift_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_left_shift_";
    return eager_api_bitwise_left_shift_(self, args, kwargs);
  }
}
static PyObject *bitwise_not(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_not";
    return static_api_bitwise_not(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_not";
    return eager_api_bitwise_not(self, args, kwargs);
  }
}
static PyObject *bitwise_not_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_not_";
    return static_api_bitwise_not_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_not_";
    return eager_api_bitwise_not_(self, args, kwargs);
  }
}
static PyObject *bitwise_or(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_or";
    return static_api_bitwise_or(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_or";
    return eager_api_bitwise_or(self, args, kwargs);
  }
}
static PyObject *bitwise_or_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_or_";
    return static_api_bitwise_or_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_or_";
    return eager_api_bitwise_or_(self, args, kwargs);
  }
}
static PyObject *bitwise_right_shift(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_right_shift";
    return static_api_bitwise_right_shift(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_right_shift";
    return eager_api_bitwise_right_shift(self, args, kwargs);
  }
}
static PyObject *bitwise_right_shift_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_right_shift_";
    return static_api_bitwise_right_shift_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_right_shift_";
    return eager_api_bitwise_right_shift_(self, args, kwargs);
  }
}
static PyObject *bitwise_xor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_xor";
    return static_api_bitwise_xor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_xor";
    return eager_api_bitwise_xor(self, args, kwargs);
  }
}
static PyObject *bitwise_xor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bitwise_xor_";
    return static_api_bitwise_xor_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bitwise_xor_";
    return eager_api_bitwise_xor_(self, args, kwargs);
  }
}
static PyObject *bmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bmm";
    return static_api_bmm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bmm";
    return eager_api_bmm(self, args, kwargs);
  }
}
static PyObject *box_coder(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_box_coder";
    return static_api_box_coder(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_box_coder";
    return eager_api_box_coder(self, args, kwargs);
  }
}
static PyObject *broadcast_tensors(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_broadcast_tensors";
    return static_api_broadcast_tensors(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_broadcast_tensors";
    return eager_api_broadcast_tensors(self, args, kwargs);
  }
}
static PyObject *ceil(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ceil";
    return static_api_ceil(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ceil";
    return eager_api_ceil(self, args, kwargs);
  }
}
static PyObject *ceil_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ceil_";
    return static_api_ceil_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ceil_";
    return eager_api_ceil_(self, args, kwargs);
  }
}
static PyObject *celu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_celu";
    return static_api_celu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_celu";
    return eager_api_celu(self, args, kwargs);
  }
}
static PyObject *check_finite_and_unscale_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_check_finite_and_unscale_";
    return static_api_check_finite_and_unscale_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_check_finite_and_unscale_";
    return eager_api_check_finite_and_unscale_(self, args, kwargs);
  }
}
static PyObject *check_numerics(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_check_numerics";
    return static_api_check_numerics(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_check_numerics";
    return eager_api_check_numerics(self, args, kwargs);
  }
}
static PyObject *cholesky(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cholesky";
    return static_api_cholesky(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cholesky";
    return eager_api_cholesky(self, args, kwargs);
  }
}
static PyObject *cholesky_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cholesky_solve";
    return static_api_cholesky_solve(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cholesky_solve";
    return eager_api_cholesky_solve(self, args, kwargs);
  }
}
static PyObject *class_center_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_class_center_sample";
    return static_api_class_center_sample(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_class_center_sample";
    return eager_api_class_center_sample(self, args, kwargs);
  }
}
static PyObject *clip(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_clip";
    return static_api_clip(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_clip";
    return eager_api_clip(self, args, kwargs);
  }
}
static PyObject *clip_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_clip_";
    return static_api_clip_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_clip_";
    return eager_api_clip_(self, args, kwargs);
  }
}
static PyObject *clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_clip_by_norm";
    return static_api_clip_by_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_clip_by_norm";
    return eager_api_clip_by_norm(self, args, kwargs);
  }
}
static PyObject *coalesce_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_coalesce_tensor";
    return static_api_coalesce_tensor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_coalesce_tensor";
    return eager_api_coalesce_tensor(self, args, kwargs);
  }
}
static PyObject *complex(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_complex";
    return static_api_complex(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_complex";
    return eager_api_complex(self, args, kwargs);
  }
}
static PyObject *concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_concat";
    return static_api_concat(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_concat";
    return eager_api_concat(self, args, kwargs);
  }
}
static PyObject *conj(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conj";
    return static_api_conj(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conj";
    return eager_api_conj(self, args, kwargs);
  }
}
static PyObject *conv2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv2d";
    return static_api_conv2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv2d";
    return eager_api_conv2d(self, args, kwargs);
  }
}
static PyObject *conv3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d";
    return static_api_conv3d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d";
    return eager_api_conv3d(self, args, kwargs);
  }
}
static PyObject *conv3d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d_transpose";
    return static_api_conv3d_transpose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d_transpose";
    return eager_api_conv3d_transpose(self, args, kwargs);
  }
}
static PyObject *copysign(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_copysign";
    return static_api_copysign(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_copysign";
    return eager_api_copysign(self, args, kwargs);
  }
}
static PyObject *copysign_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_copysign_";
    return static_api_copysign_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_copysign_";
    return eager_api_copysign_(self, args, kwargs);
  }
}
static PyObject *cos(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cos";
    return static_api_cos(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cos";
    return eager_api_cos(self, args, kwargs);
  }
}
static PyObject *cos_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cos_";
    return static_api_cos_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cos_";
    return eager_api_cos_(self, args, kwargs);
  }
}
static PyObject *cosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cosh";
    return static_api_cosh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cosh";
    return eager_api_cosh(self, args, kwargs);
  }
}
static PyObject *cosh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cosh_";
    return static_api_cosh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cosh_";
    return eager_api_cosh_(self, args, kwargs);
  }
}
static PyObject *crop(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_crop";
    return static_api_crop(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_crop";
    return eager_api_crop(self, args, kwargs);
  }
}
static PyObject *cross(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross";
    return static_api_cross(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross";
    return eager_api_cross(self, args, kwargs);
  }
}
static PyObject *cross_entropy_with_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross_entropy_with_softmax";
    return static_api_cross_entropy_with_softmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross_entropy_with_softmax";
    return eager_api_cross_entropy_with_softmax(self, args, kwargs);
  }
}
static PyObject *cross_entropy_with_softmax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross_entropy_with_softmax_";
    return static_api_cross_entropy_with_softmax_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross_entropy_with_softmax_";
    return eager_api_cross_entropy_with_softmax_(self, args, kwargs);
  }
}
static PyObject *cummax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cummax";
    return static_api_cummax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cummax";
    return eager_api_cummax(self, args, kwargs);
  }
}
static PyObject *cummin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cummin";
    return static_api_cummin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cummin";
    return eager_api_cummin(self, args, kwargs);
  }
}
static PyObject *cumprod(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumprod";
    return static_api_cumprod(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumprod";
    return eager_api_cumprod(self, args, kwargs);
  }
}
static PyObject *cumprod_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumprod_";
    return static_api_cumprod_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumprod_";
    return eager_api_cumprod_(self, args, kwargs);
  }
}
static PyObject *cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumsum";
    return static_api_cumsum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumsum";
    return eager_api_cumsum(self, args, kwargs);
  }
}
static PyObject *cumsum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumsum_";
    return static_api_cumsum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumsum_";
    return eager_api_cumsum_(self, args, kwargs);
  }
}
static PyObject *data(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_data";
    return static_api_data(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_data";
    return eager_api_data(self, args, kwargs);
  }
}
static PyObject *depthwise_conv2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depthwise_conv2d";
    return static_api_depthwise_conv2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depthwise_conv2d";
    return eager_api_depthwise_conv2d(self, args, kwargs);
  }
}
static PyObject *det(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_det";
    return static_api_det(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_det";
    return eager_api_det(self, args, kwargs);
  }
}
static PyObject *diag(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_diag";
    return static_api_diag(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_diag";
    return eager_api_diag(self, args, kwargs);
  }
}
static PyObject *diag_embed(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_diag_embed";
    return static_api_diag_embed(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_diag_embed";
    return eager_api_diag_embed(self, args, kwargs);
  }
}
static PyObject *diagonal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_diagonal";
    return static_api_diagonal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_diagonal";
    return eager_api_diagonal(self, args, kwargs);
  }
}
static PyObject *digamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_digamma";
    return static_api_digamma(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_digamma";
    return eager_api_digamma(self, args, kwargs);
  }
}
static PyObject *digamma_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_digamma_";
    return static_api_digamma_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_digamma_";
    return eager_api_digamma_(self, args, kwargs);
  }
}
static PyObject *dirichlet(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dirichlet";
    return static_api_dirichlet(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dirichlet";
    return eager_api_dirichlet(self, args, kwargs);
  }
}
static PyObject *dist(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dist";
    return static_api_dist(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dist";
    return eager_api_dist(self, args, kwargs);
  }
}
static PyObject *dot(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dot";
    return static_api_dot(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dot";
    return eager_api_dot(self, args, kwargs);
  }
}
static PyObject *edit_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_edit_distance";
    return static_api_edit_distance(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_edit_distance";
    return eager_api_edit_distance(self, args, kwargs);
  }
}
static PyObject *eig(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eig";
    return static_api_eig(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eig";
    return eager_api_eig(self, args, kwargs);
  }
}
static PyObject *eigh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eigh";
    return static_api_eigh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eigh";
    return eager_api_eigh(self, args, kwargs);
  }
}
static PyObject *eigvals(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eigvals";
    return static_api_eigvals(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eigvals";
    return eager_api_eigvals(self, args, kwargs);
  }
}
static PyObject *eigvalsh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eigvalsh";
    return static_api_eigvalsh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eigvalsh";
    return eager_api_eigvalsh(self, args, kwargs);
  }
}
static PyObject *elu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elu";
    return static_api_elu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elu";
    return eager_api_elu(self, args, kwargs);
  }
}
static PyObject *elu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elu_";
    return static_api_elu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elu_";
    return eager_api_elu_(self, args, kwargs);
  }
}
static PyObject *equal_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_equal_all";
    return static_api_equal_all(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_equal_all";
    return eager_api_equal_all(self, args, kwargs);
  }
}
static PyObject *erf(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erf";
    return static_api_erf(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erf";
    return eager_api_erf(self, args, kwargs);
  }
}
static PyObject *erf_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erf_";
    return static_api_erf_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erf_";
    return eager_api_erf_(self, args, kwargs);
  }
}
static PyObject *erfinv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erfinv";
    return static_api_erfinv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erfinv";
    return eager_api_erfinv(self, args, kwargs);
  }
}
static PyObject *erfinv_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erfinv_";
    return static_api_erfinv_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erfinv_";
    return eager_api_erfinv_(self, args, kwargs);
  }
}
static PyObject *exp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_exp";
    return static_api_exp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_exp";
    return eager_api_exp(self, args, kwargs);
  }
}
static PyObject *exp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_exp_";
    return static_api_exp_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_exp_";
    return eager_api_exp_(self, args, kwargs);
  }
}
static PyObject *expand(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expand";
    return static_api_expand(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expand";
    return eager_api_expand(self, args, kwargs);
  }
}
static PyObject *expand_as(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expand_as";
    return static_api_expand_as(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expand_as";
    return eager_api_expand_as(self, args, kwargs);
  }
}
static PyObject *expm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1";
    return static_api_expm1(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1";
    return eager_api_expm1(self, args, kwargs);
  }
}
static PyObject *expm1_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1_";
    return static_api_expm1_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1_";
    return eager_api_expm1_(self, args, kwargs);
  }
}
static PyObject *fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_c2c";
    return static_api_fft_c2c(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_c2c";
    return eager_api_fft_c2c(self, args, kwargs);
  }
}
static PyObject *fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_c2r";
    return static_api_fft_c2r(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_c2r";
    return eager_api_fft_c2r(self, args, kwargs);
  }
}
static PyObject *fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_r2c";
    return static_api_fft_r2c(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_r2c";
    return eager_api_fft_r2c(self, args, kwargs);
  }
}
static PyObject *fill(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill";
    return static_api_fill(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill";
    return eager_api_fill(self, args, kwargs);
  }
}
static PyObject *fill_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_";
    return static_api_fill_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_";
    return eager_api_fill_(self, args, kwargs);
  }
}
static PyObject *fill_diagonal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal";
    return static_api_fill_diagonal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal";
    return eager_api_fill_diagonal(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_";
    return static_api_fill_diagonal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_";
    return eager_api_fill_diagonal_(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_tensor";
    return static_api_fill_diagonal_tensor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_tensor";
    return eager_api_fill_diagonal_tensor(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_tensor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_tensor_";
    return static_api_fill_diagonal_tensor_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_tensor_";
    return eager_api_fill_diagonal_tensor_(self, args, kwargs);
  }
}
static PyObject *flash_attn(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn";
    return static_api_flash_attn(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn";
    return eager_api_flash_attn(self, args, kwargs);
  }
}
static PyObject *flash_attn_unpadded(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_unpadded";
    return static_api_flash_attn_unpadded(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_unpadded";
    return eager_api_flash_attn_unpadded(self, args, kwargs);
  }
}
static PyObject *flash_attn_with_sparse_mask(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_with_sparse_mask";
    return static_api_flash_attn_with_sparse_mask(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_with_sparse_mask";
    return eager_api_flash_attn_with_sparse_mask(self, args, kwargs);
  }
}
static PyObject *flatten(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flatten";
    return static_api_flatten(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flatten";
    return eager_api_flatten(self, args, kwargs);
  }
}
static PyObject *flatten_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flatten_";
    return static_api_flatten_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flatten_";
    return eager_api_flatten_(self, args, kwargs);
  }
}
static PyObject *flip(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flip";
    return static_api_flip(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flip";
    return eager_api_flip(self, args, kwargs);
  }
}
static PyObject *floor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor";
    return static_api_floor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor";
    return eager_api_floor(self, args, kwargs);
  }
}
static PyObject *floor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor_";
    return static_api_floor_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor_";
    return eager_api_floor_(self, args, kwargs);
  }
}
static PyObject *fmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fmax";
    return static_api_fmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fmax";
    return eager_api_fmax(self, args, kwargs);
  }
}
static PyObject *fmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fmin";
    return static_api_fmin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fmin";
    return eager_api_fmin(self, args, kwargs);
  }
}
static PyObject *fold(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fold";
    return static_api_fold(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fold";
    return eager_api_fold(self, args, kwargs);
  }
}
static PyObject *fractional_max_pool2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fractional_max_pool2d";
    return static_api_fractional_max_pool2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fractional_max_pool2d";
    return eager_api_fractional_max_pool2d(self, args, kwargs);
  }
}
static PyObject *fractional_max_pool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fractional_max_pool3d";
    return static_api_fractional_max_pool3d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fractional_max_pool3d";
    return eager_api_fractional_max_pool3d(self, args, kwargs);
  }
}
static PyObject *frame(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_frame";
    return static_api_frame(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_frame";
    return eager_api_frame(self, args, kwargs);
  }
}
static PyObject *full_int_array(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_int_array";
    return static_api_full_int_array(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_int_array";
    return eager_api_full_int_array(self, args, kwargs);
  }
}
static PyObject *gammaincc(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaincc";
    return static_api_gammaincc(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaincc";
    return eager_api_gammaincc(self, args, kwargs);
  }
}
static PyObject *gammaincc_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaincc_";
    return static_api_gammaincc_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaincc_";
    return eager_api_gammaincc_(self, args, kwargs);
  }
}
static PyObject *gammaln(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaln";
    return static_api_gammaln(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaln";
    return eager_api_gammaln(self, args, kwargs);
  }
}
static PyObject *gammaln_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaln_";
    return static_api_gammaln_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaln_";
    return eager_api_gammaln_(self, args, kwargs);
  }
}
static PyObject *gather(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gather";
    return static_api_gather(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gather";
    return eager_api_gather(self, args, kwargs);
  }
}
static PyObject *gather_nd(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gather_nd";
    return static_api_gather_nd(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gather_nd";
    return eager_api_gather_nd(self, args, kwargs);
  }
}
static PyObject *gather_tree(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gather_tree";
    return static_api_gather_tree(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gather_tree";
    return eager_api_gather_tree(self, args, kwargs);
  }
}
static PyObject *gaussian_inplace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian_inplace";
    return static_api_gaussian_inplace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian_inplace";
    return eager_api_gaussian_inplace(self, args, kwargs);
  }
}
static PyObject *gaussian_inplace_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian_inplace_";
    return static_api_gaussian_inplace_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian_inplace_";
    return eager_api_gaussian_inplace_(self, args, kwargs);
  }
}
static PyObject *gelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gelu";
    return static_api_gelu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gelu";
    return eager_api_gelu(self, args, kwargs);
  }
}
static PyObject *generate_proposals(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_generate_proposals";
    return static_api_generate_proposals(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_generate_proposals";
    return eager_api_generate_proposals(self, args, kwargs);
  }
}
static PyObject *graph_khop_sampler(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_graph_khop_sampler";
    return static_api_graph_khop_sampler(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_graph_khop_sampler";
    return eager_api_graph_khop_sampler(self, args, kwargs);
  }
}
static PyObject *graph_sample_neighbors(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_graph_sample_neighbors";
    return static_api_graph_sample_neighbors(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_graph_sample_neighbors";
    return eager_api_graph_sample_neighbors(self, args, kwargs);
  }
}
static PyObject *grid_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_grid_sample";
    return static_api_grid_sample(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_grid_sample";
    return eager_api_grid_sample(self, args, kwargs);
  }
}
static PyObject *group_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_group_norm";
    return static_api_group_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_group_norm";
    return eager_api_group_norm(self, args, kwargs);
  }
}
static PyObject *gumbel_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gumbel_softmax";
    return static_api_gumbel_softmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gumbel_softmax";
    return eager_api_gumbel_softmax(self, args, kwargs);
  }
}
static PyObject *hardshrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardshrink";
    return static_api_hardshrink(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardshrink";
    return eager_api_hardshrink(self, args, kwargs);
  }
}
static PyObject *hardsigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardsigmoid";
    return static_api_hardsigmoid(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardsigmoid";
    return eager_api_hardsigmoid(self, args, kwargs);
  }
}
static PyObject *hardtanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardtanh";
    return static_api_hardtanh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardtanh";
    return eager_api_hardtanh(self, args, kwargs);
  }
}
static PyObject *hardtanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardtanh_";
    return static_api_hardtanh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardtanh_";
    return eager_api_hardtanh_(self, args, kwargs);
  }
}
static PyObject *heaviside(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_heaviside";
    return static_api_heaviside(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_heaviside";
    return eager_api_heaviside(self, args, kwargs);
  }
}
static PyObject *histogram(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_histogram";
    return static_api_histogram(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_histogram";
    return eager_api_histogram(self, args, kwargs);
  }
}
static PyObject *huber_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_huber_loss";
    return static_api_huber_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_huber_loss";
    return eager_api_huber_loss(self, args, kwargs);
  }
}
static PyObject *i0(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i0";
    return static_api_i0(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i0";
    return eager_api_i0(self, args, kwargs);
  }
}
static PyObject *i0_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i0_";
    return static_api_i0_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i0_";
    return eager_api_i0_(self, args, kwargs);
  }
}
static PyObject *i0e(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i0e";
    return static_api_i0e(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i0e";
    return eager_api_i0e(self, args, kwargs);
  }
}
static PyObject *i1(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i1";
    return static_api_i1(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i1";
    return eager_api_i1(self, args, kwargs);
  }
}
static PyObject *i1e(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i1e";
    return static_api_i1e(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i1e";
    return eager_api_i1e(self, args, kwargs);
  }
}
static PyObject *identity_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_identity_loss";
    return static_api_identity_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_identity_loss";
    return eager_api_identity_loss(self, args, kwargs);
  }
}
static PyObject *identity_loss_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_identity_loss_";
    return static_api_identity_loss_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_identity_loss_";
    return eager_api_identity_loss_(self, args, kwargs);
  }
}
static PyObject *imag(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_imag";
    return static_api_imag(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_imag";
    return eager_api_imag(self, args, kwargs);
  }
}
static PyObject *index_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_add";
    return static_api_index_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_add";
    return eager_api_index_add(self, args, kwargs);
  }
}
static PyObject *index_add_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_add_";
    return static_api_index_add_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_add_";
    return eager_api_index_add_(self, args, kwargs);
  }
}
static PyObject *index_put(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_put";
    return static_api_index_put(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_put";
    return eager_api_index_put(self, args, kwargs);
  }
}
static PyObject *index_put_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_put_";
    return static_api_index_put_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_put_";
    return eager_api_index_put_(self, args, kwargs);
  }
}
static PyObject *index_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_sample";
    return static_api_index_sample(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_sample";
    return eager_api_index_sample(self, args, kwargs);
  }
}
static PyObject *index_select(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_select";
    return static_api_index_select(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_select";
    return eager_api_index_select(self, args, kwargs);
  }
}
static PyObject *index_select_strided(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_select_strided";
    return static_api_index_select_strided(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_select_strided";
    return eager_api_index_select_strided(self, args, kwargs);
  }
}
static PyObject *instance_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_instance_norm";
    return static_api_instance_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_instance_norm";
    return eager_api_instance_norm(self, args, kwargs);
  }
}
static PyObject *inverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_inverse";
    return static_api_inverse(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_inverse";
    return eager_api_inverse(self, args, kwargs);
  }
}
static PyObject *is_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_is_empty";
    return static_api_is_empty(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_is_empty";
    return eager_api_is_empty(self, args, kwargs);
  }
}
static PyObject *isclose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_isclose";
    return static_api_isclose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_isclose";
    return eager_api_isclose(self, args, kwargs);
  }
}
static PyObject *isfinite(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_isfinite";
    return static_api_isfinite(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_isfinite";
    return eager_api_isfinite(self, args, kwargs);
  }
}
static PyObject *isinf(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_isinf";
    return static_api_isinf(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_isinf";
    return eager_api_isinf(self, args, kwargs);
  }
}
static PyObject *isnan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_isnan";
    return static_api_isnan(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_isnan";
    return eager_api_isnan(self, args, kwargs);
  }
}
static PyObject *kldiv_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kldiv_loss";
    return static_api_kldiv_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kldiv_loss";
    return eager_api_kldiv_loss(self, args, kwargs);
  }
}
static PyObject *kron(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kron";
    return static_api_kron(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kron";
    return eager_api_kron(self, args, kwargs);
  }
}
static PyObject *kthvalue(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kthvalue";
    return static_api_kthvalue(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kthvalue";
    return eager_api_kthvalue(self, args, kwargs);
  }
}
static PyObject *label_smooth(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_label_smooth";
    return static_api_label_smooth(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_label_smooth";
    return eager_api_label_smooth(self, args, kwargs);
  }
}
static PyObject *lamb_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lamb_";
    return static_api_lamb_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lamb_";
    return eager_api_lamb_(self, args, kwargs);
  }
}
static PyObject *layer_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_layer_norm";
    return static_api_layer_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_layer_norm";
    return eager_api_layer_norm(self, args, kwargs);
  }
}
static PyObject *leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu";
    return static_api_leaky_relu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu";
    return eager_api_leaky_relu(self, args, kwargs);
  }
}
static PyObject *leaky_relu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu_";
    return static_api_leaky_relu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu_";
    return eager_api_leaky_relu_(self, args, kwargs);
  }
}
static PyObject *lerp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lerp";
    return static_api_lerp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lerp";
    return eager_api_lerp(self, args, kwargs);
  }
}
static PyObject *lerp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lerp_";
    return static_api_lerp_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lerp_";
    return eager_api_lerp_(self, args, kwargs);
  }
}
static PyObject *lgamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lgamma";
    return static_api_lgamma(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lgamma";
    return eager_api_lgamma(self, args, kwargs);
  }
}
static PyObject *lgamma_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lgamma_";
    return static_api_lgamma_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lgamma_";
    return eager_api_lgamma_(self, args, kwargs);
  }
}
static PyObject *linear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_linear_interp";
    return static_api_linear_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_linear_interp";
    return eager_api_linear_interp(self, args, kwargs);
  }
}
static PyObject *llm_int8_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_llm_int8_linear";
    return static_api_llm_int8_linear(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_llm_int8_linear";
    return eager_api_llm_int8_linear(self, args, kwargs);
  }
}
static PyObject *log(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log";
    return static_api_log(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log";
    return eager_api_log(self, args, kwargs);
  }
}
static PyObject *log_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_";
    return static_api_log_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_";
    return eager_api_log_(self, args, kwargs);
  }
}
static PyObject *log10(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log10";
    return static_api_log10(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log10";
    return eager_api_log10(self, args, kwargs);
  }
}
static PyObject *log10_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log10_";
    return static_api_log10_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log10_";
    return eager_api_log10_(self, args, kwargs);
  }
}
static PyObject *log1p(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p";
    return static_api_log1p(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p";
    return eager_api_log1p(self, args, kwargs);
  }
}
static PyObject *log1p_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p_";
    return static_api_log1p_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p_";
    return eager_api_log1p_(self, args, kwargs);
  }
}
static PyObject *log2(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log2";
    return static_api_log2(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log2";
    return eager_api_log2(self, args, kwargs);
  }
}
static PyObject *log2_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log2_";
    return static_api_log2_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log2_";
    return eager_api_log2_(self, args, kwargs);
  }
}
static PyObject *log_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_loss";
    return static_api_log_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_loss";
    return eager_api_log_loss(self, args, kwargs);
  }
}
static PyObject *log_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_softmax";
    return static_api_log_softmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_softmax";
    return eager_api_log_softmax(self, args, kwargs);
  }
}
static PyObject *logcumsumexp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logcumsumexp";
    return static_api_logcumsumexp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logcumsumexp";
    return eager_api_logcumsumexp(self, args, kwargs);
  }
}
static PyObject *logical_and(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_and";
    return static_api_logical_and(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_and";
    return eager_api_logical_and(self, args, kwargs);
  }
}
static PyObject *logical_and_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_and_";
    return static_api_logical_and_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_and_";
    return eager_api_logical_and_(self, args, kwargs);
  }
}
static PyObject *logical_not(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_not";
    return static_api_logical_not(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_not";
    return eager_api_logical_not(self, args, kwargs);
  }
}
static PyObject *logical_not_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_not_";
    return static_api_logical_not_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_not_";
    return eager_api_logical_not_(self, args, kwargs);
  }
}
static PyObject *logical_or(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_or";
    return static_api_logical_or(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_or";
    return eager_api_logical_or(self, args, kwargs);
  }
}
static PyObject *logical_or_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_or_";
    return static_api_logical_or_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_or_";
    return eager_api_logical_or_(self, args, kwargs);
  }
}
static PyObject *logical_xor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_xor";
    return static_api_logical_xor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_xor";
    return eager_api_logical_xor(self, args, kwargs);
  }
}
static PyObject *logical_xor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logical_xor_";
    return static_api_logical_xor_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logical_xor_";
    return eager_api_logical_xor_(self, args, kwargs);
  }
}
static PyObject *logit(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logit";
    return static_api_logit(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logit";
    return eager_api_logit(self, args, kwargs);
  }
}
static PyObject *logit_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logit_";
    return static_api_logit_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logit_";
    return eager_api_logit_(self, args, kwargs);
  }
}
static PyObject *logsigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logsigmoid";
    return static_api_logsigmoid(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logsigmoid";
    return eager_api_logsigmoid(self, args, kwargs);
  }
}
static PyObject *lstsq(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lstsq";
    return static_api_lstsq(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lstsq";
    return eager_api_lstsq(self, args, kwargs);
  }
}
static PyObject *lu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu";
    return static_api_lu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu";
    return eager_api_lu(self, args, kwargs);
  }
}
static PyObject *lu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_";
    return static_api_lu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_";
    return eager_api_lu_(self, args, kwargs);
  }
}
static PyObject *lu_unpack(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_unpack";
    return static_api_lu_unpack(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_unpack";
    return eager_api_lu_unpack(self, args, kwargs);
  }
}
static PyObject *margin_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_margin_cross_entropy";
    return static_api_margin_cross_entropy(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_margin_cross_entropy";
    return eager_api_margin_cross_entropy(self, args, kwargs);
  }
}
static PyObject *masked_multihead_attention_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_masked_multihead_attention_";
    return static_api_masked_multihead_attention_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_masked_multihead_attention_";
    return eager_api_masked_multihead_attention_(self, args, kwargs);
  }
}
static PyObject *masked_select(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_masked_select";
    return static_api_masked_select(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_masked_select";
    return eager_api_masked_select(self, args, kwargs);
  }
}
static PyObject *matrix_nms(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_nms";
    return static_api_matrix_nms(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_nms";
    return eager_api_matrix_nms(self, args, kwargs);
  }
}
static PyObject *matrix_power(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_power";
    return static_api_matrix_power(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_power";
    return eager_api_matrix_power(self, args, kwargs);
  }
}
static PyObject *max_pool2d_with_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max_pool2d_with_index";
    return static_api_max_pool2d_with_index(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max_pool2d_with_index";
    return eager_api_max_pool2d_with_index(self, args, kwargs);
  }
}
static PyObject *max_pool3d_with_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max_pool3d_with_index";
    return static_api_max_pool3d_with_index(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max_pool3d_with_index";
    return eager_api_max_pool3d_with_index(self, args, kwargs);
  }
}
static PyObject *maxout(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maxout";
    return static_api_maxout(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maxout";
    return eager_api_maxout(self, args, kwargs);
  }
}
static PyObject *mean_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mean_all";
    return static_api_mean_all(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mean_all";
    return eager_api_mean_all(self, args, kwargs);
  }
}
static PyObject *memory_efficient_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_memory_efficient_attention";
    return static_api_memory_efficient_attention(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_memory_efficient_attention";
    return eager_api_memory_efficient_attention(self, args, kwargs);
  }
}
static PyObject *merge_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_merge_selected_rows";
    return static_api_merge_selected_rows(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_merge_selected_rows";
    return eager_api_merge_selected_rows(self, args, kwargs);
  }
}
static PyObject *merged_adam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_merged_adam_";
    return static_api_merged_adam_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_merged_adam_";
    return eager_api_merged_adam_(self, args, kwargs);
  }
}
static PyObject *merged_momentum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_merged_momentum_";
    return static_api_merged_momentum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_merged_momentum_";
    return eager_api_merged_momentum_(self, args, kwargs);
  }
}
static PyObject *meshgrid(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_meshgrid";
    return static_api_meshgrid(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_meshgrid";
    return eager_api_meshgrid(self, args, kwargs);
  }
}
static PyObject *mode(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mode";
    return static_api_mode(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mode";
    return eager_api_mode(self, args, kwargs);
  }
}
static PyObject *momentum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_momentum_";
    return static_api_momentum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_momentum_";
    return eager_api_momentum_(self, args, kwargs);
  }
}
static PyObject *multi_dot(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multi_dot";
    return static_api_multi_dot(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multi_dot";
    return eager_api_multi_dot(self, args, kwargs);
  }
}
static PyObject *multiclass_nms3(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiclass_nms3";
    return static_api_multiclass_nms3(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiclass_nms3";
    return eager_api_multiclass_nms3(self, args, kwargs);
  }
}
static PyObject *multinomial(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multinomial";
    return static_api_multinomial(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multinomial";
    return eager_api_multinomial(self, args, kwargs);
  }
}
static PyObject *multiplex(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiplex";
    return static_api_multiplex(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiplex";
    return eager_api_multiplex(self, args, kwargs);
  }
}
static PyObject *mv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mv";
    return static_api_mv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mv";
    return eager_api_mv(self, args, kwargs);
  }
}
static PyObject *nanmedian(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nanmedian";
    return static_api_nanmedian(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nanmedian";
    return eager_api_nanmedian(self, args, kwargs);
  }
}
static PyObject *nearest_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nearest_interp";
    return static_api_nearest_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nearest_interp";
    return eager_api_nearest_interp(self, args, kwargs);
  }
}
static PyObject *nextafter(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nextafter";
    return static_api_nextafter(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nextafter";
    return eager_api_nextafter(self, args, kwargs);
  }
}
static PyObject *nll_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nll_loss";
    return static_api_nll_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nll_loss";
    return eager_api_nll_loss(self, args, kwargs);
  }
}
static PyObject *nms(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nms";
    return static_api_nms(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nms";
    return eager_api_nms(self, args, kwargs);
  }
}
static PyObject *nonzero(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nonzero";
    return static_api_nonzero(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nonzero";
    return eager_api_nonzero(self, args, kwargs);
  }
}
static PyObject *npu_identity(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_npu_identity";
    return static_api_npu_identity(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_npu_identity";
    return eager_api_npu_identity(self, args, kwargs);
  }
}
static PyObject *numel(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_numel";
    return static_api_numel(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_numel";
    return eager_api_numel(self, args, kwargs);
  }
}
static PyObject *overlap_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_overlap_add";
    return static_api_overlap_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_overlap_add";
    return eager_api_overlap_add(self, args, kwargs);
  }
}
static PyObject *p_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_p_norm";
    return static_api_p_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_p_norm";
    return eager_api_p_norm(self, args, kwargs);
  }
}
static PyObject *pad3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pad3d";
    return static_api_pad3d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pad3d";
    return eager_api_pad3d(self, args, kwargs);
  }
}
static PyObject *pixel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pixel_shuffle";
    return static_api_pixel_shuffle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pixel_shuffle";
    return eager_api_pixel_shuffle(self, args, kwargs);
  }
}
static PyObject *pixel_unshuffle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pixel_unshuffle";
    return static_api_pixel_unshuffle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pixel_unshuffle";
    return eager_api_pixel_unshuffle(self, args, kwargs);
  }
}
static PyObject *poisson(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_poisson";
    return static_api_poisson(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_poisson";
    return eager_api_poisson(self, args, kwargs);
  }
}
static PyObject *polygamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_polygamma";
    return static_api_polygamma(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_polygamma";
    return eager_api_polygamma(self, args, kwargs);
  }
}
static PyObject *polygamma_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_polygamma_";
    return static_api_polygamma_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_polygamma_";
    return eager_api_polygamma_(self, args, kwargs);
  }
}
static PyObject *pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow";
    return static_api_pow(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow";
    return eager_api_pow(self, args, kwargs);
  }
}
static PyObject *pow_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow_";
    return static_api_pow_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow_";
    return eager_api_pow_(self, args, kwargs);
  }
}
static PyObject *prelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prelu";
    return static_api_prelu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prelu";
    return eager_api_prelu(self, args, kwargs);
  }
}
static PyObject *prior_box(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prior_box";
    return static_api_prior_box(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prior_box";
    return eager_api_prior_box(self, args, kwargs);
  }
}
static PyObject *psroi_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_psroi_pool";
    return static_api_psroi_pool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_psroi_pool";
    return eager_api_psroi_pool(self, args, kwargs);
  }
}
static PyObject *put_along_axis(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_put_along_axis";
    return static_api_put_along_axis(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_put_along_axis";
    return eager_api_put_along_axis(self, args, kwargs);
  }
}
static PyObject *put_along_axis_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_put_along_axis_";
    return static_api_put_along_axis_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_put_along_axis_";
    return eager_api_put_along_axis_(self, args, kwargs);
  }
}
static PyObject *qr(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_qr";
    return static_api_qr(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_qr";
    return eager_api_qr(self, args, kwargs);
  }
}
static PyObject *real(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_real";
    return static_api_real(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_real";
    return eager_api_real(self, args, kwargs);
  }
}
static PyObject *reciprocal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reciprocal";
    return static_api_reciprocal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reciprocal";
    return eager_api_reciprocal(self, args, kwargs);
  }
}
static PyObject *reciprocal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reciprocal_";
    return static_api_reciprocal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reciprocal_";
    return eager_api_reciprocal_(self, args, kwargs);
  }
}
static PyObject *reindex_graph(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reindex_graph";
    return static_api_reindex_graph(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reindex_graph";
    return eager_api_reindex_graph(self, args, kwargs);
  }
}
static PyObject *relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu";
    return static_api_relu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu";
    return eager_api_relu(self, args, kwargs);
  }
}
static PyObject *relu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu_";
    return static_api_relu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu_";
    return eager_api_relu_(self, args, kwargs);
  }
}
static PyObject *relu6(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu6";
    return static_api_relu6(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu6";
    return eager_api_relu6(self, args, kwargs);
  }
}
static PyObject *renorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_renorm";
    return static_api_renorm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_renorm";
    return eager_api_renorm(self, args, kwargs);
  }
}
static PyObject *renorm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_renorm_";
    return static_api_renorm_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_renorm_";
    return eager_api_renorm_(self, args, kwargs);
  }
}
static PyObject *reverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reverse";
    return static_api_reverse(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reverse";
    return eager_api_reverse(self, args, kwargs);
  }
}
static PyObject *rms_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rms_norm";
    return static_api_rms_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rms_norm";
    return eager_api_rms_norm(self, args, kwargs);
  }
}
static PyObject *rmsprop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rmsprop_";
    return static_api_rmsprop_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rmsprop_";
    return eager_api_rmsprop_(self, args, kwargs);
  }
}
static PyObject *roi_align(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roi_align";
    return static_api_roi_align(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roi_align";
    return eager_api_roi_align(self, args, kwargs);
  }
}
static PyObject *roi_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roi_pool";
    return static_api_roi_pool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roi_pool";
    return eager_api_roi_pool(self, args, kwargs);
  }
}
static PyObject *roll(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roll";
    return static_api_roll(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roll";
    return eager_api_roll(self, args, kwargs);
  }
}
static PyObject *round(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_round";
    return static_api_round(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_round";
    return eager_api_round(self, args, kwargs);
  }
}
static PyObject *round_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_round_";
    return static_api_round_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_round_";
    return eager_api_round_(self, args, kwargs);
  }
}
static PyObject *rprop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rprop_";
    return static_api_rprop_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rprop_";
    return eager_api_rprop_(self, args, kwargs);
  }
}
static PyObject *rsqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rsqrt";
    return static_api_rsqrt(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rsqrt";
    return eager_api_rsqrt(self, args, kwargs);
  }
}
static PyObject *rsqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rsqrt_";
    return static_api_rsqrt_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rsqrt_";
    return eager_api_rsqrt_(self, args, kwargs);
  }
}
static PyObject *scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scale";
    return static_api_scale(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scale";
    return eager_api_scale(self, args, kwargs);
  }
}
static PyObject *scale_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scale_";
    return static_api_scale_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scale_";
    return eager_api_scale_(self, args, kwargs);
  }
}
static PyObject *scatter(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scatter";
    return static_api_scatter(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scatter";
    return eager_api_scatter(self, args, kwargs);
  }
}
static PyObject *scatter_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scatter_";
    return static_api_scatter_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scatter_";
    return eager_api_scatter_(self, args, kwargs);
  }
}
static PyObject *scatter_nd_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scatter_nd_add";
    return static_api_scatter_nd_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scatter_nd_add";
    return eager_api_scatter_nd_add(self, args, kwargs);
  }
}
static PyObject *searchsorted(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_searchsorted";
    return static_api_searchsorted(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_searchsorted";
    return eager_api_searchsorted(self, args, kwargs);
  }
}
static PyObject *segment_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_segment_pool";
    return static_api_segment_pool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_segment_pool";
    return eager_api_segment_pool(self, args, kwargs);
  }
}
static PyObject *selu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_selu";
    return static_api_selu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_selu";
    return eager_api_selu(self, args, kwargs);
  }
}
static PyObject *send_u_recv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_u_recv";
    return static_api_send_u_recv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_u_recv";
    return eager_api_send_u_recv(self, args, kwargs);
  }
}
static PyObject *send_ue_recv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_ue_recv";
    return static_api_send_ue_recv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_ue_recv";
    return eager_api_send_ue_recv(self, args, kwargs);
  }
}
static PyObject *send_uv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_uv";
    return static_api_send_uv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_uv";
    return eager_api_send_uv(self, args, kwargs);
  }
}
static PyObject *sgd_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sgd_";
    return static_api_sgd_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sgd_";
    return eager_api_sgd_(self, args, kwargs);
  }
}
static PyObject *shape(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shape";
    return static_api_shape(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shape";
    return eager_api_shape(self, args, kwargs);
  }
}
static PyObject *shard_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shard_index";
    return static_api_shard_index(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shard_index";
    return eager_api_shard_index(self, args, kwargs);
  }
}
static PyObject *sigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid";
    return static_api_sigmoid(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid";
    return eager_api_sigmoid(self, args, kwargs);
  }
}
static PyObject *sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_";
    return static_api_sigmoid_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_";
    return eager_api_sigmoid_(self, args, kwargs);
  }
}
static PyObject *sigmoid_cross_entropy_with_logits(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_cross_entropy_with_logits";
    return static_api_sigmoid_cross_entropy_with_logits(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_cross_entropy_with_logits";
    return eager_api_sigmoid_cross_entropy_with_logits(self, args, kwargs);
  }
}
static PyObject *sigmoid_cross_entropy_with_logits_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_cross_entropy_with_logits_";
    return static_api_sigmoid_cross_entropy_with_logits_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_cross_entropy_with_logits_";
    return eager_api_sigmoid_cross_entropy_with_logits_(self, args, kwargs);
  }
}
static PyObject *sign(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sign";
    return static_api_sign(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sign";
    return eager_api_sign(self, args, kwargs);
  }
}
static PyObject *silu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_silu";
    return static_api_silu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_silu";
    return eager_api_silu(self, args, kwargs);
  }
}
static PyObject *sin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin";
    return static_api_sin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin";
    return eager_api_sin(self, args, kwargs);
  }
}
static PyObject *sin_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin_";
    return static_api_sin_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin_";
    return eager_api_sin_(self, args, kwargs);
  }
}
static PyObject *sinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh";
    return static_api_sinh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh";
    return eager_api_sinh(self, args, kwargs);
  }
}
static PyObject *sinh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh_";
    return static_api_sinh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh_";
    return eager_api_sinh_(self, args, kwargs);
  }
}
static PyObject *slogdet(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slogdet";
    return static_api_slogdet(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slogdet";
    return eager_api_slogdet(self, args, kwargs);
  }
}
static PyObject *softplus(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softplus";
    return static_api_softplus(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softplus";
    return eager_api_softplus(self, args, kwargs);
  }
}
static PyObject *softshrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softshrink";
    return static_api_softshrink(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softshrink";
    return eager_api_softshrink(self, args, kwargs);
  }
}
static PyObject *softsign(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softsign";
    return static_api_softsign(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softsign";
    return eager_api_softsign(self, args, kwargs);
  }
}
static PyObject *solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_solve";
    return static_api_solve(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_solve";
    return eager_api_solve(self, args, kwargs);
  }
}
static PyObject *spectral_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_spectral_norm";
    return static_api_spectral_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_spectral_norm";
    return eager_api_spectral_norm(self, args, kwargs);
  }
}
static PyObject *sqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt";
    return static_api_sqrt(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt";
    return eager_api_sqrt(self, args, kwargs);
  }
}
static PyObject *sqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt_";
    return static_api_sqrt_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt_";
    return eager_api_sqrt_(self, args, kwargs);
  }
}
static PyObject *square(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square";
    return static_api_square(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square";
    return eager_api_square(self, args, kwargs);
  }
}
static PyObject *squared_l2_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squared_l2_norm";
    return static_api_squared_l2_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squared_l2_norm";
    return eager_api_squared_l2_norm(self, args, kwargs);
  }
}
static PyObject *squeeze(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squeeze";
    return static_api_squeeze(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squeeze";
    return eager_api_squeeze(self, args, kwargs);
  }
}
static PyObject *squeeze_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squeeze_";
    return static_api_squeeze_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squeeze_";
    return eager_api_squeeze_(self, args, kwargs);
  }
}
static PyObject *stack(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stack";
    return static_api_stack(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stack";
    return eager_api_stack(self, args, kwargs);
  }
}
static PyObject *standard_gamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_standard_gamma";
    return static_api_standard_gamma(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_standard_gamma";
    return eager_api_standard_gamma(self, args, kwargs);
  }
}
static PyObject *stanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stanh";
    return static_api_stanh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stanh";
    return eager_api_stanh(self, args, kwargs);
  }
}
static PyObject *svd(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_svd";
    return static_api_svd(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_svd";
    return eager_api_svd(self, args, kwargs);
  }
}
static PyObject *swiglu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_swiglu";
    return static_api_swiglu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_swiglu";
    return eager_api_swiglu(self, args, kwargs);
  }
}
static PyObject *take_along_axis(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_take_along_axis";
    return static_api_take_along_axis(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_take_along_axis";
    return eager_api_take_along_axis(self, args, kwargs);
  }
}
static PyObject *tan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan";
    return static_api_tan(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan";
    return eager_api_tan(self, args, kwargs);
  }
}
static PyObject *tan_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan_";
    return static_api_tan_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan_";
    return eager_api_tan_(self, args, kwargs);
  }
}
static PyObject *tanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh";
    return static_api_tanh(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh";
    return eager_api_tanh(self, args, kwargs);
  }
}
static PyObject *tanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_";
    return static_api_tanh_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_";
    return eager_api_tanh_(self, args, kwargs);
  }
}
static PyObject *tanh_shrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_shrink";
    return static_api_tanh_shrink(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_shrink";
    return eager_api_tanh_shrink(self, args, kwargs);
  }
}
static PyObject *temporal_shift(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_temporal_shift";
    return static_api_temporal_shift(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_temporal_shift";
    return eager_api_temporal_shift(self, args, kwargs);
  }
}
static PyObject *tensor_unfold(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tensor_unfold";
    return static_api_tensor_unfold(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tensor_unfold";
    return eager_api_tensor_unfold(self, args, kwargs);
  }
}
static PyObject *thresholded_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_thresholded_relu";
    return static_api_thresholded_relu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_thresholded_relu";
    return eager_api_thresholded_relu(self, args, kwargs);
  }
}
static PyObject *thresholded_relu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_thresholded_relu_";
    return static_api_thresholded_relu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_thresholded_relu_";
    return eager_api_thresholded_relu_(self, args, kwargs);
  }
}
static PyObject *top_p_sampling(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_top_p_sampling";
    return static_api_top_p_sampling(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_top_p_sampling";
    return eager_api_top_p_sampling(self, args, kwargs);
  }
}
static PyObject *topk(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_topk";
    return static_api_topk(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_topk";
    return eager_api_topk(self, args, kwargs);
  }
}
static PyObject *trace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trace";
    return static_api_trace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trace";
    return eager_api_trace(self, args, kwargs);
  }
}
static PyObject *triangular_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triangular_solve";
    return static_api_triangular_solve(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triangular_solve";
    return eager_api_triangular_solve(self, args, kwargs);
  }
}
static PyObject *trilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trilinear_interp";
    return static_api_trilinear_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trilinear_interp";
    return eager_api_trilinear_interp(self, args, kwargs);
  }
}
static PyObject *trunc(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trunc";
    return static_api_trunc(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trunc";
    return eager_api_trunc(self, args, kwargs);
  }
}
static PyObject *trunc_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trunc_";
    return static_api_trunc_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trunc_";
    return eager_api_trunc_(self, args, kwargs);
  }
}
static PyObject *unbind(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unbind";
    return static_api_unbind(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unbind";
    return eager_api_unbind(self, args, kwargs);
  }
}
static PyObject *unfold(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unfold";
    return static_api_unfold(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unfold";
    return eager_api_unfold(self, args, kwargs);
  }
}
static PyObject *uniform_inplace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform_inplace";
    return static_api_uniform_inplace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform_inplace";
    return eager_api_uniform_inplace(self, args, kwargs);
  }
}
static PyObject *uniform_inplace_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform_inplace_";
    return static_api_uniform_inplace_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform_inplace_";
    return eager_api_uniform_inplace_(self, args, kwargs);
  }
}
static PyObject *unique_consecutive(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unique_consecutive";
    return static_api_unique_consecutive(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unique_consecutive";
    return eager_api_unique_consecutive(self, args, kwargs);
  }
}
static PyObject *unpool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unpool3d";
    return static_api_unpool3d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unpool3d";
    return eager_api_unpool3d(self, args, kwargs);
  }
}
static PyObject *unsqueeze(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unsqueeze";
    return static_api_unsqueeze(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unsqueeze";
    return eager_api_unsqueeze(self, args, kwargs);
  }
}
static PyObject *unsqueeze_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unsqueeze_";
    return static_api_unsqueeze_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unsqueeze_";
    return eager_api_unsqueeze_(self, args, kwargs);
  }
}
static PyObject *unstack(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unstack";
    return static_api_unstack(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unstack";
    return eager_api_unstack(self, args, kwargs);
  }
}
static PyObject *update_loss_scaling_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_update_loss_scaling_";
    return static_api_update_loss_scaling_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_update_loss_scaling_";
    return eager_api_update_loss_scaling_(self, args, kwargs);
  }
}
static PyObject *view_dtype(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_view_dtype";
    return static_api_view_dtype(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_view_dtype";
    return eager_api_view_dtype(self, args, kwargs);
  }
}
static PyObject *view_shape(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_view_shape";
    return static_api_view_shape(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_view_shape";
    return eager_api_view_shape(self, args, kwargs);
  }
}
static PyObject *viterbi_decode(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_viterbi_decode";
    return static_api_viterbi_decode(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_viterbi_decode";
    return eager_api_viterbi_decode(self, args, kwargs);
  }
}
static PyObject *warpctc(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_warpctc";
    return static_api_warpctc(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_warpctc";
    return eager_api_warpctc(self, args, kwargs);
  }
}
static PyObject *warprnnt(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_warprnnt";
    return static_api_warprnnt(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_warprnnt";
    return eager_api_warprnnt(self, args, kwargs);
  }
}
static PyObject *weight_dequantize(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_weight_dequantize";
    return static_api_weight_dequantize(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_weight_dequantize";
    return eager_api_weight_dequantize(self, args, kwargs);
  }
}
static PyObject *weight_only_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_weight_only_linear";
    return static_api_weight_only_linear(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_weight_only_linear";
    return eager_api_weight_only_linear(self, args, kwargs);
  }
}
static PyObject *weight_quantize(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_weight_quantize";
    return static_api_weight_quantize(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_weight_quantize";
    return eager_api_weight_quantize(self, args, kwargs);
  }
}
static PyObject *weighted_sample_neighbors(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_weighted_sample_neighbors";
    return static_api_weighted_sample_neighbors(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_weighted_sample_neighbors";
    return eager_api_weighted_sample_neighbors(self, args, kwargs);
  }
}
static PyObject *where(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_where";
    return static_api_where(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_where";
    return eager_api_where(self, args, kwargs);
  }
}
static PyObject *where_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_where_";
    return static_api_where_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_where_";
    return eager_api_where_(self, args, kwargs);
  }
}
static PyObject *yolo_box(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_yolo_box";
    return static_api_yolo_box(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_yolo_box";
    return eager_api_yolo_box(self, args, kwargs);
  }
}
static PyObject *yolo_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_yolo_loss";
    return static_api_yolo_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_yolo_loss";
    return eager_api_yolo_loss(self, args, kwargs);
  }
}
static PyObject *block_multihead_attention_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_block_multihead_attention_";
    return static_api_block_multihead_attention_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_block_multihead_attention_";
    return eager_api_block_multihead_attention_(self, args, kwargs);
  }
}
static PyObject *fc(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fc";
  return static_api_fc(self, args, kwargs);
}
static PyObject *fused_bias_act(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bias_act";
    return static_api_fused_bias_act(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bias_act";
    return eager_api_fused_bias_act(self, args, kwargs);
  }
}
static PyObject *fused_bias_dropout_residual_layer_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bias_dropout_residual_layer_norm";
    return static_api_fused_bias_dropout_residual_layer_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bias_dropout_residual_layer_norm";
    return eager_api_fused_bias_dropout_residual_layer_norm(self, args, kwargs);
  }
}
static PyObject *fused_bias_residual_layernorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bias_residual_layernorm";
    return static_api_fused_bias_residual_layernorm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bias_residual_layernorm";
    return eager_api_fused_bias_residual_layernorm(self, args, kwargs);
  }
}
static PyObject *fused_conv2d_add_act(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_conv2d_add_act";
  return static_api_fused_conv2d_add_act(self, args, kwargs);
}
static PyObject *fused_dropout_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_dropout_add";
    return static_api_fused_dropout_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_dropout_add";
    return eager_api_fused_dropout_add(self, args, kwargs);
  }
}
static PyObject *fused_embedding_eltwise_layernorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_embedding_eltwise_layernorm";
  return static_api_fused_embedding_eltwise_layernorm(self, args, kwargs);
}
static PyObject *fused_fc_elementwise_layernorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_fc_elementwise_layernorm";
  return static_api_fused_fc_elementwise_layernorm(self, args, kwargs);
}
static PyObject *fused_linear_param_grad_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_linear_param_grad_add";
    return static_api_fused_linear_param_grad_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_linear_param_grad_add";
    return eager_api_fused_linear_param_grad_add(self, args, kwargs);
  }
}
static PyObject *fused_rotary_position_embedding(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_rotary_position_embedding";
    return static_api_fused_rotary_position_embedding(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_rotary_position_embedding";
    return eager_api_fused_rotary_position_embedding(self, args, kwargs);
  }
}
static PyObject *fusion_gru(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_gru";
  return static_api_fusion_gru(self, args, kwargs);
}
static PyObject *fusion_repeated_fc_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_repeated_fc_relu";
  return static_api_fusion_repeated_fc_relu(self, args, kwargs);
}
static PyObject *fusion_seqconv_eltadd_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_seqconv_eltadd_relu";
  return static_api_fusion_seqconv_eltadd_relu(self, args, kwargs);
}
static PyObject *fusion_seqexpand_concat_fc(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_seqexpand_concat_fc";
  return static_api_fusion_seqexpand_concat_fc(self, args, kwargs);
}
static PyObject *fusion_squared_mat_sub(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_squared_mat_sub";
  return static_api_fusion_squared_mat_sub(self, args, kwargs);
}
static PyObject *fusion_transpose_flatten_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fusion_transpose_flatten_concat";
  return static_api_fusion_transpose_flatten_concat(self, args, kwargs);
}
static PyObject *multihead_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_multihead_matmul";
  return static_api_multihead_matmul(self, args, kwargs);
}
static PyObject *self_dp_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_self_dp_attention";
  return static_api_self_dp_attention(self, args, kwargs);
}
static PyObject *skip_layernorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_skip_layernorm";
  return static_api_skip_layernorm(self, args, kwargs);
}
static PyObject *squeeze_excitation_block(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_squeeze_excitation_block";
  return static_api_squeeze_excitation_block(self, args, kwargs);
}
static PyObject *variable_length_memory_efficient_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_variable_length_memory_efficient_attention";
    return static_api_variable_length_memory_efficient_attention(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_variable_length_memory_efficient_attention";
    return eager_api_variable_length_memory_efficient_attention(self, args, kwargs);
  }
}
static PyObject *adadelta_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adadelta_";
    return static_api_adadelta_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adadelta_";
    return eager_api_adadelta_(self, args, kwargs);
  }
}
static PyObject *add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add";
    return static_api_add(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add";
    return eager_api_add(self, args, kwargs);
  }
}
static PyObject *add_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_";
    return static_api_add_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_";
    return eager_api_add_(self, args, kwargs);
  }
}
static PyObject *add_n(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_n";
    return static_api_add_n(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_n";
    return eager_api_add_n(self, args, kwargs);
  }
}
static PyObject *all(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all";
    return static_api_all(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all";
    return eager_api_all(self, args, kwargs);
  }
}
static PyObject *amax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_amax";
    return static_api_amax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_amax";
    return eager_api_amax(self, args, kwargs);
  }
}
static PyObject *amin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_amin";
    return static_api_amin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_amin";
    return eager_api_amin(self, args, kwargs);
  }
}
static PyObject *any(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_any";
    return static_api_any(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_any";
    return eager_api_any(self, args, kwargs);
  }
}
static PyObject *assign(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign";
    return static_api_assign(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign";
    return eager_api_assign(self, args, kwargs);
  }
}
static PyObject *assign_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_";
    return static_api_assign_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_";
    return eager_api_assign_(self, args, kwargs);
  }
}
static PyObject *assign_out_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_out_";
    return static_api_assign_out_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_out_";
    return eager_api_assign_out_(self, args, kwargs);
  }
}
static PyObject *assign_value(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_assign_value";
  return static_api_assign_value(self, args, kwargs);
}
static PyObject *assign_value_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_value_";
    return static_api_assign_value_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_value_";
    return eager_api_assign_value_(self, args, kwargs);
  }
}
static PyObject *batch_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_batch_norm";
    return static_api_batch_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_batch_norm";
    return eager_api_batch_norm(self, args, kwargs);
  }
}
static PyObject *batch_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_batch_norm_";
  return static_api_batch_norm_(self, args, kwargs);
}
static PyObject *c_allreduce_avg_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_c_allreduce_avg_";
  return static_api_c_allreduce_avg_(self, args, kwargs);
}
static PyObject *c_allreduce_max_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_allreduce_max_";
    return static_api_c_allreduce_max_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_allreduce_max_";
    return eager_api_c_allreduce_max_(self, args, kwargs);
  }
}
static PyObject *c_allreduce_min_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_c_allreduce_min_";
  return static_api_c_allreduce_min_(self, args, kwargs);
}
static PyObject *c_allreduce_prod_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_c_allreduce_prod_";
  return static_api_c_allreduce_prod_(self, args, kwargs);
}
static PyObject *c_allreduce_sum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_allreduce_sum_";
    return static_api_c_allreduce_sum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_allreduce_sum_";
    return eager_api_c_allreduce_sum_(self, args, kwargs);
  }
}
static PyObject *c_broadcast(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_broadcast";
    return static_api_c_broadcast(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_broadcast";
    return eager_api_c_broadcast(self, args, kwargs);
  }
}
static PyObject *c_broadcast_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_broadcast_";
    return static_api_c_broadcast_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_broadcast_";
    return eager_api_c_broadcast_(self, args, kwargs);
  }
}
static PyObject *c_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_concat";
    return static_api_c_concat(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_concat";
    return eager_api_c_concat(self, args, kwargs);
  }
}
static PyObject *c_identity_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_identity_";
    return static_api_c_identity_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_identity_";
    return eager_api_c_identity_(self, args, kwargs);
  }
}
static PyObject *c_reduce_sum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_reduce_sum_";
    return static_api_c_reduce_sum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_reduce_sum_";
    return eager_api_c_reduce_sum_(self, args, kwargs);
  }
}
static PyObject *c_sync_calc_stream(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_sync_calc_stream";
    return static_api_c_sync_calc_stream(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_sync_calc_stream";
    return eager_api_c_sync_calc_stream(self, args, kwargs);
  }
}
static PyObject *c_sync_calc_stream_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_sync_calc_stream_";
    return static_api_c_sync_calc_stream_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_sync_calc_stream_";
    return eager_api_c_sync_calc_stream_(self, args, kwargs);
  }
}
static PyObject *c_sync_comm_stream(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_sync_comm_stream";
    return static_api_c_sync_comm_stream(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_sync_comm_stream";
    return eager_api_c_sync_comm_stream(self, args, kwargs);
  }
}
static PyObject *c_sync_comm_stream_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_sync_comm_stream_";
    return static_api_c_sync_comm_stream_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_sync_comm_stream_";
    return eager_api_c_sync_comm_stream_(self, args, kwargs);
  }
}
static PyObject *cast(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cast";
    return static_api_cast(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cast";
    return eager_api_cast(self, args, kwargs);
  }
}
static PyObject *cast_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cast_";
    return static_api_cast_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cast_";
    return eager_api_cast_(self, args, kwargs);
  }
}
static PyObject *channel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_channel_shuffle";
    return static_api_channel_shuffle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_channel_shuffle";
    return eager_api_channel_shuffle(self, args, kwargs);
  }
}
static PyObject *coalesce_tensor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_coalesce_tensor_";
  return static_api_coalesce_tensor_(self, args, kwargs);
}
static PyObject *conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv2d_transpose";
    return static_api_conv2d_transpose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv2d_transpose";
    return eager_api_conv2d_transpose(self, args, kwargs);
  }
}
static PyObject *conv2d_transpose_bias(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv2d_transpose_bias";
    return static_api_conv2d_transpose_bias(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv2d_transpose_bias";
    return eager_api_conv2d_transpose_bias(self, args, kwargs);
  }
}
static PyObject *decode_jpeg(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_decode_jpeg";
    return static_api_decode_jpeg(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_decode_jpeg";
    return eager_api_decode_jpeg(self, args, kwargs);
  }
}
static PyObject *deformable_conv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_deformable_conv";
    return static_api_deformable_conv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_deformable_conv";
    return eager_api_deformable_conv(self, args, kwargs);
  }
}
static PyObject *depthwise_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depthwise_conv2d_transpose";
    return static_api_depthwise_conv2d_transpose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depthwise_conv2d_transpose";
    return eager_api_depthwise_conv2d_transpose(self, args, kwargs);
  }
}
static PyObject *dequantize_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_dequantize_linear";
  return static_api_dequantize_linear(self, args, kwargs);
}
static PyObject *dequantize_linear_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_dequantize_linear_";
  return static_api_dequantize_linear_(self, args, kwargs);
}
static PyObject *disable_check_model_nan_inf(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_disable_check_model_nan_inf";
    return static_api_disable_check_model_nan_inf(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_disable_check_model_nan_inf";
    return eager_api_disable_check_model_nan_inf(self, args, kwargs);
  }
}
static PyObject *distribute_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_distribute_fpn_proposals";
    return static_api_distribute_fpn_proposals(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_distribute_fpn_proposals";
    return eager_api_distribute_fpn_proposals(self, args, kwargs);
  }
}
static PyObject *distributed_fused_lamb_init(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_distributed_fused_lamb_init";
  return static_api_distributed_fused_lamb_init(self, args, kwargs);
}
static PyObject *distributed_fused_lamb_init_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_distributed_fused_lamb_init_";
  return static_api_distributed_fused_lamb_init_(self, args, kwargs);
}
static PyObject *divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide";
    return static_api_divide(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide";
    return eager_api_divide(self, args, kwargs);
  }
}
static PyObject *divide_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide_";
    return static_api_divide_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide_";
    return eager_api_divide_(self, args, kwargs);
  }
}
static PyObject *dropout(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dropout";
    return static_api_dropout(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dropout";
    return eager_api_dropout(self, args, kwargs);
  }
}
static PyObject *einsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_einsum";
    return static_api_einsum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_einsum";
    return eager_api_einsum(self, args, kwargs);
  }
}
static PyObject *elementwise_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elementwise_pow";
    return static_api_elementwise_pow(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elementwise_pow";
    return eager_api_elementwise_pow(self, args, kwargs);
  }
}
static PyObject *embedding(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_embedding";
    return static_api_embedding(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_embedding";
    return eager_api_embedding(self, args, kwargs);
  }
}
static PyObject *empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_empty";
    return static_api_empty(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_empty";
    return eager_api_empty(self, args, kwargs);
  }
}
static PyObject *empty_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_empty_like";
    return static_api_empty_like(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_empty_like";
    return eager_api_empty_like(self, args, kwargs);
  }
}
static PyObject *enable_check_model_nan_inf(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_enable_check_model_nan_inf";
    return static_api_enable_check_model_nan_inf(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_enable_check_model_nan_inf";
    return eager_api_enable_check_model_nan_inf(self, args, kwargs);
  }
}
static PyObject *equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_equal";
    return static_api_equal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_equal";
    return eager_api_equal(self, args, kwargs);
  }
}
static PyObject *equal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_equal_";
    return static_api_equal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_equal_";
    return eager_api_equal_(self, args, kwargs);
  }
}
static PyObject *exponential_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_exponential_";
    return static_api_exponential_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_exponential_";
    return eager_api_exponential_(self, args, kwargs);
  }
}
static PyObject *eye(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eye";
    return static_api_eye(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eye";
    return eager_api_eye(self, args, kwargs);
  }
}
static PyObject *fetch(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fetch";
  return static_api_fetch(self, args, kwargs);
}
static PyObject *floor_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor_divide";
    return static_api_floor_divide(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor_divide";
    return eager_api_floor_divide(self, args, kwargs);
  }
}
static PyObject *floor_divide_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor_divide_";
    return static_api_floor_divide_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor_divide_";
    return eager_api_floor_divide_(self, args, kwargs);
  }
}
static PyObject *frobenius_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_frobenius_norm";
    return static_api_frobenius_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_frobenius_norm";
    return eager_api_frobenius_norm(self, args, kwargs);
  }
}
static PyObject *full(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full";
    return static_api_full(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full";
    return eager_api_full(self, args, kwargs);
  }
}
static PyObject *full_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_";
    return static_api_full_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_";
    return eager_api_full_(self, args, kwargs);
  }
}
static PyObject *full_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_batch_size_like";
    return static_api_full_batch_size_like(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_batch_size_like";
    return eager_api_full_batch_size_like(self, args, kwargs);
  }
}
static PyObject *full_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_like";
    return static_api_full_like(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_like";
    return eager_api_full_like(self, args, kwargs);
  }
}
static PyObject *full_with_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_with_tensor";
    return static_api_full_with_tensor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_with_tensor";
    return eager_api_full_with_tensor(self, args, kwargs);
  }
}
static PyObject *fused_batch_norm_act(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_batch_norm_act";
    return static_api_fused_batch_norm_act(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_batch_norm_act";
    return eager_api_fused_batch_norm_act(self, args, kwargs);
  }
}
static PyObject *fused_bn_add_activation(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bn_add_activation";
    return static_api_fused_bn_add_activation(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bn_add_activation";
    return eager_api_fused_bn_add_activation(self, args, kwargs);
  }
}
static PyObject *fused_multi_transformer(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_multi_transformer";
    return static_api_fused_multi_transformer(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_multi_transformer";
    return eager_api_fused_multi_transformer(self, args, kwargs);
  }
}
static PyObject *fused_softmax_mask(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_softmax_mask";
    return static_api_fused_softmax_mask(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_softmax_mask";
    return eager_api_fused_softmax_mask(self, args, kwargs);
  }
}
static PyObject *fused_softmax_mask_upper_triangle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_softmax_mask_upper_triangle";
    return static_api_fused_softmax_mask_upper_triangle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_softmax_mask_upper_triangle";
    return eager_api_fused_softmax_mask_upper_triangle(self, args, kwargs);
  }
}
static PyObject *gaussian(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian";
    return static_api_gaussian(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian";
    return eager_api_gaussian(self, args, kwargs);
  }
}
static PyObject *get_tensor_from_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_get_tensor_from_selected_rows";
  return static_api_get_tensor_from_selected_rows(self, args, kwargs);
}
static PyObject *greater_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_greater_equal";
    return static_api_greater_equal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_greater_equal";
    return eager_api_greater_equal(self, args, kwargs);
  }
}
static PyObject *greater_equal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_greater_equal_";
    return static_api_greater_equal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_greater_equal_";
    return eager_api_greater_equal_(self, args, kwargs);
  }
}
static PyObject *greater_than(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_greater_than";
    return static_api_greater_than(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_greater_than";
    return eager_api_greater_than(self, args, kwargs);
  }
}
static PyObject *greater_than_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_greater_than_";
    return static_api_greater_than_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_greater_than_";
    return eager_api_greater_than_(self, args, kwargs);
  }
}
static PyObject *hardswish(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardswish";
    return static_api_hardswish(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardswish";
    return eager_api_hardswish(self, args, kwargs);
  }
}
static PyObject *hsigmoid_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hsigmoid_loss";
    return static_api_hsigmoid_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hsigmoid_loss";
    return eager_api_hsigmoid_loss(self, args, kwargs);
  }
}
static PyObject *increment(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_increment";
    return static_api_increment(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_increment";
    return eager_api_increment(self, args, kwargs);
  }
}
static PyObject *increment_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_increment_";
    return static_api_increment_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_increment_";
    return eager_api_increment_(self, args, kwargs);
  }
}
static PyObject *less_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_less_equal";
    return static_api_less_equal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_less_equal";
    return eager_api_less_equal(self, args, kwargs);
  }
}
static PyObject *less_equal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_less_equal_";
    return static_api_less_equal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_less_equal_";
    return eager_api_less_equal_(self, args, kwargs);
  }
}
static PyObject *less_than(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_less_than";
    return static_api_less_than(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_less_than";
    return eager_api_less_than(self, args, kwargs);
  }
}
static PyObject *less_than_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_less_than_";
    return static_api_less_than_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_less_than_";
    return eager_api_less_than_(self, args, kwargs);
  }
}
static PyObject *linspace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_linspace";
    return static_api_linspace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_linspace";
    return eager_api_linspace(self, args, kwargs);
  }
}
static PyObject *logspace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logspace";
    return static_api_logspace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logspace";
    return eager_api_logspace(self, args, kwargs);
  }
}
static PyObject *logsumexp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logsumexp";
    return static_api_logsumexp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logsumexp";
    return eager_api_logsumexp(self, args, kwargs);
  }
}
static PyObject *lrn(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_lrn";
  return static_api_lrn(self, args, kwargs);
}
static PyObject *matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matmul";
    return static_api_matmul(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matmul";
    return eager_api_matmul(self, args, kwargs);
  }
}
static PyObject *matmul_with_flatten(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_matmul_with_flatten";
  return static_api_matmul_with_flatten(self, args, kwargs);
}
static PyObject *matrix_rank(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_rank";
    return static_api_matrix_rank(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_rank";
    return eager_api_matrix_rank(self, args, kwargs);
  }
}
static PyObject *matrix_rank_tol(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_rank_tol";
    return static_api_matrix_rank_tol(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_rank_tol";
    return eager_api_matrix_rank_tol(self, args, kwargs);
  }
}
static PyObject *max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max";
    return static_api_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max";
    return eager_api_max(self, args, kwargs);
  }
}
static PyObject *maximum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maximum";
    return static_api_maximum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maximum";
    return eager_api_maximum(self, args, kwargs);
  }
}
static PyObject *mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mean";
    return static_api_mean(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mean";
    return eager_api_mean(self, args, kwargs);
  }
}
static PyObject *memcpy(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_memcpy";
  return static_api_memcpy(self, args, kwargs);
}
static PyObject *memcpy_d2h(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_memcpy_d2h";
    return static_api_memcpy_d2h(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_memcpy_d2h";
    return eager_api_memcpy_d2h(self, args, kwargs);
  }
}
static PyObject *memcpy_h2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_memcpy_h2d";
    return static_api_memcpy_h2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_memcpy_h2d";
    return eager_api_memcpy_h2d(self, args, kwargs);
  }
}
static PyObject *min(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_min";
    return static_api_min(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_min";
    return eager_api_min(self, args, kwargs);
  }
}
static PyObject *minimum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_minimum";
    return static_api_minimum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_minimum";
    return eager_api_minimum(self, args, kwargs);
  }
}
static PyObject *mish(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mish";
    return static_api_mish(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mish";
    return eager_api_mish(self, args, kwargs);
  }
}
static PyObject *multiply(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiply";
    return static_api_multiply(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiply";
    return eager_api_multiply(self, args, kwargs);
  }
}
static PyObject *multiply_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiply_";
    return static_api_multiply_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiply_";
    return eager_api_multiply_(self, args, kwargs);
  }
}
static PyObject *norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_norm";
    return static_api_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_norm";
    return eager_api_norm(self, args, kwargs);
  }
}
static PyObject *not_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_not_equal";
    return static_api_not_equal(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_not_equal";
    return eager_api_not_equal(self, args, kwargs);
  }
}
static PyObject *not_equal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_not_equal_";
    return static_api_not_equal_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_not_equal_";
    return eager_api_not_equal_(self, args, kwargs);
  }
}
static PyObject *one_hot(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_one_hot";
    return static_api_one_hot(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_one_hot";
    return eager_api_one_hot(self, args, kwargs);
  }
}
static PyObject *pad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pad";
    return static_api_pad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pad";
    return eager_api_pad(self, args, kwargs);
  }
}
static PyObject *pool2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pool2d";
    return static_api_pool2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pool2d";
    return eager_api_pool2d(self, args, kwargs);
  }
}
static PyObject *pool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pool3d";
    return static_api_pool3d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pool3d";
    return eager_api_pool3d(self, args, kwargs);
  }
}
static PyObject *print(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_print";
  return static_api_print(self, args, kwargs);
}
static PyObject *prod(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prod";
    return static_api_prod(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prod";
    return eager_api_prod(self, args, kwargs);
  }
}
static PyObject *quantize_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_quantize_linear";
  return static_api_quantize_linear(self, args, kwargs);
}
static PyObject *quantize_linear_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_quantize_linear_";
  return static_api_quantize_linear_(self, args, kwargs);
}
static PyObject *randint(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_randint";
    return static_api_randint(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_randint";
    return eager_api_randint(self, args, kwargs);
  }
}
static PyObject *randperm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_randperm";
    return static_api_randperm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_randperm";
    return eager_api_randperm(self, args, kwargs);
  }
}
static PyObject *read_file(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_read_file";
    return static_api_read_file(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_read_file";
    return eager_api_read_file(self, args, kwargs);
  }
}
static PyObject *remainder(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_remainder";
    return static_api_remainder(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_remainder";
    return eager_api_remainder(self, args, kwargs);
  }
}
static PyObject *remainder_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_remainder_";
    return static_api_remainder_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_remainder_";
    return eager_api_remainder_(self, args, kwargs);
  }
}
static PyObject *repeat_interleave(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_repeat_interleave";
    return static_api_repeat_interleave(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_repeat_interleave";
    return eager_api_repeat_interleave(self, args, kwargs);
  }
}
static PyObject *repeat_interleave_with_tensor_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_repeat_interleave_with_tensor_index";
    return static_api_repeat_interleave_with_tensor_index(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_repeat_interleave_with_tensor_index";
    return eager_api_repeat_interleave_with_tensor_index(self, args, kwargs);
  }
}
static PyObject *reshape(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape";
    return static_api_reshape(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape";
    return eager_api_reshape(self, args, kwargs);
  }
}
static PyObject *reshape_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape_";
    return static_api_reshape_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape_";
    return eager_api_reshape_(self, args, kwargs);
  }
}
static PyObject *rnn(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rnn";
    return static_api_rnn(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rnn";
    return eager_api_rnn(self, args, kwargs);
  }
}
static PyObject *rrelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rrelu";
    return static_api_rrelu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rrelu";
    return eager_api_rrelu(self, args, kwargs);
  }
}
static PyObject *set_value(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value";
    return static_api_set_value(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value";
    return eager_api_set_value(self, args, kwargs);
  }
}
static PyObject *set_value_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value_";
    return static_api_set_value_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value_";
    return eager_api_set_value_(self, args, kwargs);
  }
}
static PyObject *set_value_with_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value_with_tensor";
    return static_api_set_value_with_tensor(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value_with_tensor";
    return eager_api_set_value_with_tensor(self, args, kwargs);
  }
}
static PyObject *set_value_with_tensor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value_with_tensor_";
    return static_api_set_value_with_tensor_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value_with_tensor_";
    return eager_api_set_value_with_tensor_(self, args, kwargs);
  }
}
static PyObject *share_data(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_share_data";
  return static_api_share_data(self, args, kwargs);
}
static PyObject *slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slice";
    return static_api_slice(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slice";
    return eager_api_slice(self, args, kwargs);
  }
}
static PyObject *softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softmax";
    return static_api_softmax(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softmax";
    return eager_api_softmax(self, args, kwargs);
  }
}
static PyObject *softmax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softmax_";
    return static_api_softmax_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softmax_";
    return eager_api_softmax_(self, args, kwargs);
  }
}
static PyObject *split(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_split";
    return static_api_split(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_split";
    return eager_api_split(self, args, kwargs);
  }
}
static PyObject *split_with_num(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_split_with_num";
    return static_api_split_with_num(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_split_with_num";
    return eager_api_split_with_num(self, args, kwargs);
  }
}
static PyObject *strided_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_strided_slice";
    return static_api_strided_slice(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_strided_slice";
    return eager_api_strided_slice(self, args, kwargs);
  }
}
static PyObject *subtract(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract";
    return static_api_subtract(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract";
    return eager_api_subtract(self, args, kwargs);
  }
}
static PyObject *subtract_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract_";
    return static_api_subtract_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract_";
    return eager_api_subtract_(self, args, kwargs);
  }
}
static PyObject *sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sum";
    return static_api_sum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sum";
    return eager_api_sum(self, args, kwargs);
  }
}
static PyObject *swish(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_swish";
    return static_api_swish(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_swish";
    return eager_api_swish(self, args, kwargs);
  }
}
static PyObject *sync_batch_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_batch_norm_";
    return static_api_sync_batch_norm_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_batch_norm_";
    return eager_api_sync_batch_norm_(self, args, kwargs);
  }
}
static PyObject *tile(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tile";
    return static_api_tile(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tile";
    return eager_api_tile(self, args, kwargs);
  }
}
static PyObject *trans_layout(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trans_layout";
    return static_api_trans_layout(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trans_layout";
    return eager_api_trans_layout(self, args, kwargs);
  }
}
static PyObject *transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_transpose";
    return static_api_transpose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_transpose";
    return eager_api_transpose(self, args, kwargs);
  }
}
static PyObject *transpose_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_transpose_";
    return static_api_transpose_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_transpose_";
    return eager_api_transpose_(self, args, kwargs);
  }
}
static PyObject *tril(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tril";
    return static_api_tril(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tril";
    return eager_api_tril(self, args, kwargs);
  }
}
static PyObject *tril_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tril_";
    return static_api_tril_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tril_";
    return eager_api_tril_(self, args, kwargs);
  }
}
static PyObject *tril_indices(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tril_indices";
    return static_api_tril_indices(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tril_indices";
    return eager_api_tril_indices(self, args, kwargs);
  }
}
static PyObject *triu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triu";
    return static_api_triu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triu";
    return eager_api_triu(self, args, kwargs);
  }
}
static PyObject *triu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triu_";
    return static_api_triu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triu_";
    return eager_api_triu_(self, args, kwargs);
  }
}
static PyObject *triu_indices(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triu_indices";
    return static_api_triu_indices(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triu_indices";
    return eager_api_triu_indices(self, args, kwargs);
  }
}
static PyObject *truncated_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_truncated_gaussian_random";
    return static_api_truncated_gaussian_random(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_truncated_gaussian_random";
    return eager_api_truncated_gaussian_random(self, args, kwargs);
  }
}
static PyObject *uniform(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform";
    return static_api_uniform(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform";
    return eager_api_uniform(self, args, kwargs);
  }
}
static PyObject *unique(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unique";
    return static_api_unique(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unique";
    return eager_api_unique(self, args, kwargs);
  }
}
static PyObject *unpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unpool";
    return static_api_unpool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unpool";
    return eager_api_unpool(self, args, kwargs);
  }
}
static PyObject *fused_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_attention";
  return static_api_fused_attention(self, args, kwargs);
}
static PyObject *fused_feedforward(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_feedforward";
  return static_api_fused_feedforward(self, args, kwargs);
}
static PyObject *moving_average_abs_max_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_moving_average_abs_max_scale";
  return static_api_moving_average_abs_max_scale(self, args, kwargs);
}
static PyObject *moving_average_abs_max_scale_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_moving_average_abs_max_scale_";
  return static_api_moving_average_abs_max_scale_(self, args, kwargs);
}
static PyObject *number_count(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_number_count";
  return static_api_number_count(self, args, kwargs);
}
static PyObject *onednn_to_paddle_layout(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_onednn_to_paddle_layout";
  return static_api_onednn_to_paddle_layout(self, args, kwargs);
}
static PyObject *arange(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_arange";
    return static_api_arange(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_arange";
    return eager_api_arange(self, args, kwargs);
  }
}
static PyObject *sequence_mask(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sequence_mask";
    return static_api_sequence_mask(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sequence_mask";
    return eager_api_sequence_mask(self, args, kwargs);
  }
}

static PyMethodDef OpsAPI[] = {

{"abs", (PyCFunction)(void (*)(void))abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs."},
{"abs_", (PyCFunction)(void (*)(void))abs_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs_."},
{"accuracy", (PyCFunction)(void (*)(void))accuracy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for accuracy."},
{"acos", (PyCFunction)(void (*)(void))acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos."},
{"acos_", (PyCFunction)(void (*)(void))acos_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos_."},
{"acosh", (PyCFunction)(void (*)(void))acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh."},
{"acosh_", (PyCFunction)(void (*)(void))acosh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh_."},
{"adagrad_", (PyCFunction)(void (*)(void))adagrad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adagrad_."},
{"adam_", (PyCFunction)(void (*)(void))adam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adam_."},
{"adamax_", (PyCFunction)(void (*)(void))adamax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamax_."},
{"adamw_", (PyCFunction)(void (*)(void))adamw_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamw_."},
{"addmm", (PyCFunction)(void (*)(void))addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm."},
{"addmm_", (PyCFunction)(void (*)(void))addmm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm_."},
{"affine_grid", (PyCFunction)(void (*)(void))affine_grid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid."},
{"allclose", (PyCFunction)(void (*)(void))allclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allclose."},
{"angle", (PyCFunction)(void (*)(void))angle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle."},
{"apply_per_channel_scale", (PyCFunction)(void (*)(void))apply_per_channel_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for apply_per_channel_scale."},
{"argmax", (PyCFunction)(void (*)(void))argmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argmax."},
{"argmin", (PyCFunction)(void (*)(void))argmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argmin."},
{"argsort", (PyCFunction)(void (*)(void))argsort, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argsort."},
{"as_complex", (PyCFunction)(void (*)(void))as_complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_complex."},
{"as_real", (PyCFunction)(void (*)(void))as_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_real."},
{"as_strided", (PyCFunction)(void (*)(void))as_strided, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_strided."},
{"asgd_", (PyCFunction)(void (*)(void))asgd_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asgd_."},
{"asin", (PyCFunction)(void (*)(void))asin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin."},
{"asin_", (PyCFunction)(void (*)(void))asin_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin_."},
{"asinh", (PyCFunction)(void (*)(void))asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh."},
{"asinh_", (PyCFunction)(void (*)(void))asinh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh_."},
{"atan", (PyCFunction)(void (*)(void))atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan."},
{"atan_", (PyCFunction)(void (*)(void))atan_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan_."},
{"atan2", (PyCFunction)(void (*)(void))atan2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2."},
{"atanh", (PyCFunction)(void (*)(void))atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh."},
{"atanh_", (PyCFunction)(void (*)(void))atanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh_."},
{"auc", (PyCFunction)(void (*)(void))auc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for auc."},
{"average_accumulates_", (PyCFunction)(void (*)(void))average_accumulates_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for average_accumulates_."},
{"bce_loss", (PyCFunction)(void (*)(void))bce_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss."},
{"bce_loss_", (PyCFunction)(void (*)(void))bce_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_."},
{"bernoulli", (PyCFunction)(void (*)(void))bernoulli, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bernoulli."},
{"bicubic_interp", (PyCFunction)(void (*)(void))bicubic_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp."},
{"bilinear", (PyCFunction)(void (*)(void))bilinear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear."},
{"bilinear_interp", (PyCFunction)(void (*)(void))bilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp."},
{"bincount", (PyCFunction)(void (*)(void))bincount, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bincount."},
{"binomial", (PyCFunction)(void (*)(void))binomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for binomial."},
{"bitwise_and", (PyCFunction)(void (*)(void))bitwise_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_and."},
{"bitwise_and_", (PyCFunction)(void (*)(void))bitwise_and_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_and_."},
{"bitwise_left_shift", (PyCFunction)(void (*)(void))bitwise_left_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_left_shift."},
{"bitwise_left_shift_", (PyCFunction)(void (*)(void))bitwise_left_shift_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_left_shift_."},
{"bitwise_not", (PyCFunction)(void (*)(void))bitwise_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_not."},
{"bitwise_not_", (PyCFunction)(void (*)(void))bitwise_not_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_not_."},
{"bitwise_or", (PyCFunction)(void (*)(void))bitwise_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_or."},
{"bitwise_or_", (PyCFunction)(void (*)(void))bitwise_or_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_or_."},
{"bitwise_right_shift", (PyCFunction)(void (*)(void))bitwise_right_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_right_shift."},
{"bitwise_right_shift_", (PyCFunction)(void (*)(void))bitwise_right_shift_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_right_shift_."},
{"bitwise_xor", (PyCFunction)(void (*)(void))bitwise_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_xor."},
{"bitwise_xor_", (PyCFunction)(void (*)(void))bitwise_xor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_xor_."},
{"bmm", (PyCFunction)(void (*)(void))bmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bmm."},
{"box_coder", (PyCFunction)(void (*)(void))box_coder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_coder."},
{"broadcast_tensors", (PyCFunction)(void (*)(void))broadcast_tensors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors."},
{"ceil", (PyCFunction)(void (*)(void))ceil, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil."},
{"ceil_", (PyCFunction)(void (*)(void))ceil_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_."},
{"celu", (PyCFunction)(void (*)(void))celu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu."},
{"check_finite_and_unscale_", (PyCFunction)(void (*)(void))check_finite_and_unscale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_finite_and_unscale_."},
{"check_numerics", (PyCFunction)(void (*)(void))check_numerics, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_numerics."},
{"cholesky", (PyCFunction)(void (*)(void))cholesky, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky."},
{"cholesky_solve", (PyCFunction)(void (*)(void))cholesky_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve."},
{"class_center_sample", (PyCFunction)(void (*)(void))class_center_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for class_center_sample."},
{"clip", (PyCFunction)(void (*)(void))clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip."},
{"clip_", (PyCFunction)(void (*)(void))clip_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_."},
{"clip_by_norm", (PyCFunction)(void (*)(void))clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_by_norm."},
{"coalesce_tensor", (PyCFunction)(void (*)(void))coalesce_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor."},
{"complex", (PyCFunction)(void (*)(void))complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for complex."},
{"concat", (PyCFunction)(void (*)(void))concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat."},
{"conj", (PyCFunction)(void (*)(void))conj, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conj."},
{"conv2d", (PyCFunction)(void (*)(void))conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d."},
{"conv3d", (PyCFunction)(void (*)(void))conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d."},
{"conv3d_transpose", (PyCFunction)(void (*)(void))conv3d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose."},
{"copysign", (PyCFunction)(void (*)(void))copysign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign."},
{"copysign_", (PyCFunction)(void (*)(void))copysign_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign_."},
{"cos", (PyCFunction)(void (*)(void))cos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos."},
{"cos_", (PyCFunction)(void (*)(void))cos_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_."},
{"cosh", (PyCFunction)(void (*)(void))cosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh."},
{"cosh_", (PyCFunction)(void (*)(void))cosh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh_."},
{"crop", (PyCFunction)(void (*)(void))crop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop."},
{"cross", (PyCFunction)(void (*)(void))cross, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross."},
{"cross_entropy_with_softmax", (PyCFunction)(void (*)(void))cross_entropy_with_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax."},
{"cross_entropy_with_softmax_", (PyCFunction)(void (*)(void))cross_entropy_with_softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax_."},
{"cummax", (PyCFunction)(void (*)(void))cummax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummax."},
{"cummin", (PyCFunction)(void (*)(void))cummin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummin."},
{"cumprod", (PyCFunction)(void (*)(void))cumprod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod."},
{"cumprod_", (PyCFunction)(void (*)(void))cumprod_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod_."},
{"cumsum", (PyCFunction)(void (*)(void))cumsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum."},
{"cumsum_", (PyCFunction)(void (*)(void))cumsum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum_."},
{"data", (PyCFunction)(void (*)(void))data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for data."},
{"depthwise_conv2d", (PyCFunction)(void (*)(void))depthwise_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d."},
{"det", (PyCFunction)(void (*)(void))det, METH_VARARGS | METH_KEYWORDS, "C++ interface function for det."},
{"diag", (PyCFunction)(void (*)(void))diag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag."},
{"diag_embed", (PyCFunction)(void (*)(void))diag_embed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_embed."},
{"diagonal", (PyCFunction)(void (*)(void))diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diagonal."},
{"digamma", (PyCFunction)(void (*)(void))digamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma."},
{"digamma_", (PyCFunction)(void (*)(void))digamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma_."},
{"dirichlet", (PyCFunction)(void (*)(void))dirichlet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dirichlet."},
{"dist", (PyCFunction)(void (*)(void))dist, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist."},
{"dot", (PyCFunction)(void (*)(void))dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot."},
{"edit_distance", (PyCFunction)(void (*)(void))edit_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for edit_distance."},
{"eig", (PyCFunction)(void (*)(void))eig, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eig."},
{"eigh", (PyCFunction)(void (*)(void))eigh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh."},
{"eigvals", (PyCFunction)(void (*)(void))eigvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvals."},
{"eigvalsh", (PyCFunction)(void (*)(void))eigvalsh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvalsh."},
{"elu", (PyCFunction)(void (*)(void))elu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu."},
{"elu_", (PyCFunction)(void (*)(void))elu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_."},
{"equal_all", (PyCFunction)(void (*)(void))equal_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal_all."},
{"erf", (PyCFunction)(void (*)(void))erf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf."},
{"erf_", (PyCFunction)(void (*)(void))erf_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf_."},
{"erfinv", (PyCFunction)(void (*)(void))erfinv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv."},
{"erfinv_", (PyCFunction)(void (*)(void))erfinv_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv_."},
{"exp", (PyCFunction)(void (*)(void))exp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp."},
{"exp_", (PyCFunction)(void (*)(void))exp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp_."},
{"expand", (PyCFunction)(void (*)(void))expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand."},
{"expand_as", (PyCFunction)(void (*)(void))expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as."},
{"expm1", (PyCFunction)(void (*)(void))expm1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1."},
{"expm1_", (PyCFunction)(void (*)(void))expm1_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1_."},
{"fft_c2c", (PyCFunction)(void (*)(void))fft_c2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2c."},
{"fft_c2r", (PyCFunction)(void (*)(void))fft_c2r, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2r."},
{"fft_r2c", (PyCFunction)(void (*)(void))fft_r2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_r2c."},
{"fill", (PyCFunction)(void (*)(void))fill, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill."},
{"fill_", (PyCFunction)(void (*)(void))fill_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_."},
{"fill_diagonal", (PyCFunction)(void (*)(void))fill_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal."},
{"fill_diagonal_", (PyCFunction)(void (*)(void))fill_diagonal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_."},
{"fill_diagonal_tensor", (PyCFunction)(void (*)(void))fill_diagonal_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor."},
{"fill_diagonal_tensor_", (PyCFunction)(void (*)(void))fill_diagonal_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_."},
{"flash_attn", (PyCFunction)(void (*)(void))flash_attn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn."},
{"flash_attn_unpadded", (PyCFunction)(void (*)(void))flash_attn_unpadded, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_unpadded."},
{"flash_attn_with_sparse_mask", (PyCFunction)(void (*)(void))flash_attn_with_sparse_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_with_sparse_mask."},
{"flatten", (PyCFunction)(void (*)(void))flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten."},
{"flatten_", (PyCFunction)(void (*)(void))flatten_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_."},
{"flip", (PyCFunction)(void (*)(void))flip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flip."},
{"floor", (PyCFunction)(void (*)(void))floor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor."},
{"floor_", (PyCFunction)(void (*)(void))floor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_."},
{"fmax", (PyCFunction)(void (*)(void))fmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmax."},
{"fmin", (PyCFunction)(void (*)(void))fmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmin."},
{"fold", (PyCFunction)(void (*)(void))fold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fold."},
{"fractional_max_pool2d", (PyCFunction)(void (*)(void))fractional_max_pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fractional_max_pool2d."},
{"fractional_max_pool3d", (PyCFunction)(void (*)(void))fractional_max_pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fractional_max_pool3d."},
{"frame", (PyCFunction)(void (*)(void))frame, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frame."},
{"full_int_array", (PyCFunction)(void (*)(void))full_int_array, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_int_array."},
{"gammaincc", (PyCFunction)(void (*)(void))gammaincc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaincc."},
{"gammaincc_", (PyCFunction)(void (*)(void))gammaincc_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaincc_."},
{"gammaln", (PyCFunction)(void (*)(void))gammaln, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaln."},
{"gammaln_", (PyCFunction)(void (*)(void))gammaln_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaln_."},
{"gather", (PyCFunction)(void (*)(void))gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather."},
{"gather_nd", (PyCFunction)(void (*)(void))gather_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_nd."},
{"gather_tree", (PyCFunction)(void (*)(void))gather_tree, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_tree."},
{"gaussian_inplace", (PyCFunction)(void (*)(void))gaussian_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace."},
{"gaussian_inplace_", (PyCFunction)(void (*)(void))gaussian_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace_."},
{"gelu", (PyCFunction)(void (*)(void))gelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu."},
{"generate_proposals", (PyCFunction)(void (*)(void))generate_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals."},
{"graph_khop_sampler", (PyCFunction)(void (*)(void))graph_khop_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_khop_sampler."},
{"graph_sample_neighbors", (PyCFunction)(void (*)(void))graph_sample_neighbors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_sample_neighbors."},
{"grid_sample", (PyCFunction)(void (*)(void))grid_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sample."},
{"group_norm", (PyCFunction)(void (*)(void))group_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm."},
{"gumbel_softmax", (PyCFunction)(void (*)(void))gumbel_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gumbel_softmax."},
{"hardshrink", (PyCFunction)(void (*)(void))hardshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardshrink."},
{"hardsigmoid", (PyCFunction)(void (*)(void))hardsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardsigmoid."},
{"hardtanh", (PyCFunction)(void (*)(void))hardtanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh."},
{"hardtanh_", (PyCFunction)(void (*)(void))hardtanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh_."},
{"heaviside", (PyCFunction)(void (*)(void))heaviside, METH_VARARGS | METH_KEYWORDS, "C++ interface function for heaviside."},
{"histogram", (PyCFunction)(void (*)(void))histogram, METH_VARARGS | METH_KEYWORDS, "C++ interface function for histogram."},
{"huber_loss", (PyCFunction)(void (*)(void))huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for huber_loss."},
{"i0", (PyCFunction)(void (*)(void))i0, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0."},
{"i0_", (PyCFunction)(void (*)(void))i0_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0_."},
{"i0e", (PyCFunction)(void (*)(void))i0e, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0e."},
{"i1", (PyCFunction)(void (*)(void))i1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1."},
{"i1e", (PyCFunction)(void (*)(void))i1e, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1e."},
{"identity_loss", (PyCFunction)(void (*)(void))identity_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss."},
{"identity_loss_", (PyCFunction)(void (*)(void))identity_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss_."},
{"imag", (PyCFunction)(void (*)(void))imag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag."},
{"index_add", (PyCFunction)(void (*)(void))index_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add."},
{"index_add_", (PyCFunction)(void (*)(void))index_add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add_."},
{"index_put", (PyCFunction)(void (*)(void))index_put, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_put."},
{"index_put_", (PyCFunction)(void (*)(void))index_put_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_put_."},
{"index_sample", (PyCFunction)(void (*)(void))index_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_sample."},
{"index_select", (PyCFunction)(void (*)(void))index_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select."},
{"index_select_strided", (PyCFunction)(void (*)(void))index_select_strided, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select_strided."},
{"instance_norm", (PyCFunction)(void (*)(void))instance_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for instance_norm."},
{"inverse", (PyCFunction)(void (*)(void))inverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inverse."},
{"is_empty", (PyCFunction)(void (*)(void))is_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for is_empty."},
{"isclose", (PyCFunction)(void (*)(void))isclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isclose."},
{"isfinite", (PyCFunction)(void (*)(void))isfinite, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite."},
{"isinf", (PyCFunction)(void (*)(void))isinf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf."},
{"isnan", (PyCFunction)(void (*)(void))isnan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isnan."},
{"kldiv_loss", (PyCFunction)(void (*)(void))kldiv_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kldiv_loss."},
{"kron", (PyCFunction)(void (*)(void))kron, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kron."},
{"kthvalue", (PyCFunction)(void (*)(void))kthvalue, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kthvalue."},
{"label_smooth", (PyCFunction)(void (*)(void))label_smooth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth."},
{"lamb_", (PyCFunction)(void (*)(void))lamb_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lamb_."},
{"layer_norm", (PyCFunction)(void (*)(void))layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm."},
{"leaky_relu", (PyCFunction)(void (*)(void))leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu."},
{"leaky_relu_", (PyCFunction)(void (*)(void))leaky_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_."},
{"lerp", (PyCFunction)(void (*)(void))lerp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp."},
{"lerp_", (PyCFunction)(void (*)(void))lerp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp_."},
{"lgamma", (PyCFunction)(void (*)(void))lgamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma."},
{"lgamma_", (PyCFunction)(void (*)(void))lgamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma_."},
{"linear_interp", (PyCFunction)(void (*)(void))linear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp."},
{"llm_int8_linear", (PyCFunction)(void (*)(void))llm_int8_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for llm_int8_linear."},
{"log", (PyCFunction)(void (*)(void))log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log."},
{"log_", (PyCFunction)(void (*)(void))log_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_."},
{"log10", (PyCFunction)(void (*)(void))log10, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10."},
{"log10_", (PyCFunction)(void (*)(void))log10_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10_."},
{"log1p", (PyCFunction)(void (*)(void))log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p."},
{"log1p_", (PyCFunction)(void (*)(void))log1p_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p_."},
{"log2", (PyCFunction)(void (*)(void))log2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2."},
{"log2_", (PyCFunction)(void (*)(void))log2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2_."},
{"log_loss", (PyCFunction)(void (*)(void))log_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_loss."},
{"log_softmax", (PyCFunction)(void (*)(void))log_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_softmax."},
{"logcumsumexp", (PyCFunction)(void (*)(void))logcumsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logcumsumexp."},
{"logical_and", (PyCFunction)(void (*)(void))logical_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_and."},
{"logical_and_", (PyCFunction)(void (*)(void))logical_and_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_and_."},
{"logical_not", (PyCFunction)(void (*)(void))logical_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_not."},
{"logical_not_", (PyCFunction)(void (*)(void))logical_not_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_not_."},
{"logical_or", (PyCFunction)(void (*)(void))logical_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_or."},
{"logical_or_", (PyCFunction)(void (*)(void))logical_or_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_or_."},
{"logical_xor", (PyCFunction)(void (*)(void))logical_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_xor."},
{"logical_xor_", (PyCFunction)(void (*)(void))logical_xor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_xor_."},
{"logit", (PyCFunction)(void (*)(void))logit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logit."},
{"logit_", (PyCFunction)(void (*)(void))logit_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logit_."},
{"logsigmoid", (PyCFunction)(void (*)(void))logsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid."},
{"lstsq", (PyCFunction)(void (*)(void))lstsq, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstsq."},
{"lu", (PyCFunction)(void (*)(void))lu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu."},
{"lu_", (PyCFunction)(void (*)(void))lu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_."},
{"lu_unpack", (PyCFunction)(void (*)(void))lu_unpack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack."},
{"margin_cross_entropy", (PyCFunction)(void (*)(void))margin_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy."},
{"masked_multihead_attention_", (PyCFunction)(void (*)(void))masked_multihead_attention_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_multihead_attention_."},
{"masked_select", (PyCFunction)(void (*)(void))masked_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select."},
{"matrix_nms", (PyCFunction)(void (*)(void))matrix_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_nms."},
{"matrix_power", (PyCFunction)(void (*)(void))matrix_power, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power."},
{"max_pool2d_with_index", (PyCFunction)(void (*)(void))max_pool2d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index."},
{"max_pool3d_with_index", (PyCFunction)(void (*)(void))max_pool3d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index."},
{"maxout", (PyCFunction)(void (*)(void))maxout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout."},
{"mean_all", (PyCFunction)(void (*)(void))mean_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_all."},
{"memory_efficient_attention", (PyCFunction)(void (*)(void))memory_efficient_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memory_efficient_attention."},
{"merge_selected_rows", (PyCFunction)(void (*)(void))merge_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merge_selected_rows."},
{"merged_adam_", (PyCFunction)(void (*)(void))merged_adam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_adam_."},
{"merged_momentum_", (PyCFunction)(void (*)(void))merged_momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_momentum_."},
{"meshgrid", (PyCFunction)(void (*)(void))meshgrid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for meshgrid."},
{"mode", (PyCFunction)(void (*)(void))mode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode."},
{"momentum_", (PyCFunction)(void (*)(void))momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for momentum_."},
{"multi_dot", (PyCFunction)(void (*)(void))multi_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot."},
{"multiclass_nms3", (PyCFunction)(void (*)(void))multiclass_nms3, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms3."},
{"multinomial", (PyCFunction)(void (*)(void))multinomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multinomial."},
{"multiplex", (PyCFunction)(void (*)(void))multiplex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex."},
{"mv", (PyCFunction)(void (*)(void))mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv."},
{"nanmedian", (PyCFunction)(void (*)(void))nanmedian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nanmedian."},
{"nearest_interp", (PyCFunction)(void (*)(void))nearest_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp."},
{"nextafter", (PyCFunction)(void (*)(void))nextafter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nextafter."},
{"nll_loss", (PyCFunction)(void (*)(void))nll_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss."},
{"nms", (PyCFunction)(void (*)(void))nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nms."},
{"nonzero", (PyCFunction)(void (*)(void))nonzero, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nonzero."},
{"npu_identity", (PyCFunction)(void (*)(void))npu_identity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for npu_identity."},
{"numel", (PyCFunction)(void (*)(void))numel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for numel."},
{"overlap_add", (PyCFunction)(void (*)(void))overlap_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add."},
{"p_norm", (PyCFunction)(void (*)(void))p_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm."},
{"pad3d", (PyCFunction)(void (*)(void))pad3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d."},
{"pixel_shuffle", (PyCFunction)(void (*)(void))pixel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_shuffle."},
{"pixel_unshuffle", (PyCFunction)(void (*)(void))pixel_unshuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_unshuffle."},
{"poisson", (PyCFunction)(void (*)(void))poisson, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson."},
{"polygamma", (PyCFunction)(void (*)(void))polygamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygamma."},
{"polygamma_", (PyCFunction)(void (*)(void))polygamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygamma_."},
{"pow", (PyCFunction)(void (*)(void))pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow."},
{"pow_", (PyCFunction)(void (*)(void))pow_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow_."},
{"prelu", (PyCFunction)(void (*)(void))prelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu."},
{"prior_box", (PyCFunction)(void (*)(void))prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prior_box."},
{"psroi_pool", (PyCFunction)(void (*)(void))psroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for psroi_pool."},
{"put_along_axis", (PyCFunction)(void (*)(void))put_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis."},
{"put_along_axis_", (PyCFunction)(void (*)(void))put_along_axis_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_."},
{"qr", (PyCFunction)(void (*)(void))qr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr."},
{"real", (PyCFunction)(void (*)(void))real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real."},
{"reciprocal", (PyCFunction)(void (*)(void))reciprocal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal."},
{"reciprocal_", (PyCFunction)(void (*)(void))reciprocal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_."},
{"reindex_graph", (PyCFunction)(void (*)(void))reindex_graph, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reindex_graph."},
{"relu", (PyCFunction)(void (*)(void))relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu."},
{"relu_", (PyCFunction)(void (*)(void))relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_."},
{"relu6", (PyCFunction)(void (*)(void))relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6."},
{"renorm", (PyCFunction)(void (*)(void))renorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm."},
{"renorm_", (PyCFunction)(void (*)(void))renorm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm_."},
{"reverse", (PyCFunction)(void (*)(void))reverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reverse."},
{"rms_norm", (PyCFunction)(void (*)(void))rms_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rms_norm."},
{"rmsprop_", (PyCFunction)(void (*)(void))rmsprop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rmsprop_."},
{"roi_align", (PyCFunction)(void (*)(void))roi_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align."},
{"roi_pool", (PyCFunction)(void (*)(void))roi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool."},
{"roll", (PyCFunction)(void (*)(void))roll, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll."},
{"round", (PyCFunction)(void (*)(void))round, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round."},
{"round_", (PyCFunction)(void (*)(void))round_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_."},
{"rprop_", (PyCFunction)(void (*)(void))rprop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rprop_."},
{"rsqrt", (PyCFunction)(void (*)(void))rsqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt."},
{"rsqrt_", (PyCFunction)(void (*)(void))rsqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt_."},
{"scale", (PyCFunction)(void (*)(void))scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale."},
{"scale_", (PyCFunction)(void (*)(void))scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale_."},
{"scatter", (PyCFunction)(void (*)(void))scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter."},
{"scatter_", (PyCFunction)(void (*)(void))scatter_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_."},
{"scatter_nd_add", (PyCFunction)(void (*)(void))scatter_nd_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_nd_add."},
{"searchsorted", (PyCFunction)(void (*)(void))searchsorted, METH_VARARGS | METH_KEYWORDS, "C++ interface function for searchsorted."},
{"segment_pool", (PyCFunction)(void (*)(void))segment_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for segment_pool."},
{"selu", (PyCFunction)(void (*)(void))selu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for selu."},
{"send_u_recv", (PyCFunction)(void (*)(void))send_u_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_u_recv."},
{"send_ue_recv", (PyCFunction)(void (*)(void))send_ue_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_ue_recv."},
{"send_uv", (PyCFunction)(void (*)(void))send_uv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_uv."},
{"sgd_", (PyCFunction)(void (*)(void))sgd_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sgd_."},
{"shape", (PyCFunction)(void (*)(void))shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape."},
{"shard_index", (PyCFunction)(void (*)(void))shard_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shard_index."},
{"sigmoid", (PyCFunction)(void (*)(void))sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid."},
{"sigmoid_", (PyCFunction)(void (*)(void))sigmoid_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_."},
{"sigmoid_cross_entropy_with_logits", (PyCFunction)(void (*)(void))sigmoid_cross_entropy_with_logits, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits."},
{"sigmoid_cross_entropy_with_logits_", (PyCFunction)(void (*)(void))sigmoid_cross_entropy_with_logits_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits_."},
{"sign", (PyCFunction)(void (*)(void))sign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sign."},
{"silu", (PyCFunction)(void (*)(void))silu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu."},
{"sin", (PyCFunction)(void (*)(void))sin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin."},
{"sin_", (PyCFunction)(void (*)(void))sin_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin_."},
{"sinh", (PyCFunction)(void (*)(void))sinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh."},
{"sinh_", (PyCFunction)(void (*)(void))sinh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh_."},
{"slogdet", (PyCFunction)(void (*)(void))slogdet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdet."},
{"softplus", (PyCFunction)(void (*)(void))softplus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus."},
{"softshrink", (PyCFunction)(void (*)(void))softshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink."},
{"softsign", (PyCFunction)(void (*)(void))softsign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign."},
{"solve", (PyCFunction)(void (*)(void))solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for solve."},
{"spectral_norm", (PyCFunction)(void (*)(void))spectral_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spectral_norm."},
{"sqrt", (PyCFunction)(void (*)(void))sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt."},
{"sqrt_", (PyCFunction)(void (*)(void))sqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_."},
{"square", (PyCFunction)(void (*)(void))square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square."},
{"squared_l2_norm", (PyCFunction)(void (*)(void))squared_l2_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_norm."},
{"squeeze", (PyCFunction)(void (*)(void))squeeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze."},
{"squeeze_", (PyCFunction)(void (*)(void))squeeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_."},
{"stack", (PyCFunction)(void (*)(void))stack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack."},
{"standard_gamma", (PyCFunction)(void (*)(void))standard_gamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for standard_gamma."},
{"stanh", (PyCFunction)(void (*)(void))stanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stanh."},
{"svd", (PyCFunction)(void (*)(void))svd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svd."},
{"swiglu", (PyCFunction)(void (*)(void))swiglu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swiglu."},
{"take_along_axis", (PyCFunction)(void (*)(void))take_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis."},
{"tan", (PyCFunction)(void (*)(void))tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan."},
{"tan_", (PyCFunction)(void (*)(void))tan_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan_."},
{"tanh", (PyCFunction)(void (*)(void))tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh."},
{"tanh_", (PyCFunction)(void (*)(void))tanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_."},
{"tanh_shrink", (PyCFunction)(void (*)(void))tanh_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink."},
{"temporal_shift", (PyCFunction)(void (*)(void))temporal_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift."},
{"tensor_unfold", (PyCFunction)(void (*)(void))tensor_unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tensor_unfold."},
{"thresholded_relu", (PyCFunction)(void (*)(void))thresholded_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu."},
{"thresholded_relu_", (PyCFunction)(void (*)(void))thresholded_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu_."},
{"top_p_sampling", (PyCFunction)(void (*)(void))top_p_sampling, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_p_sampling."},
{"topk", (PyCFunction)(void (*)(void))topk, METH_VARARGS | METH_KEYWORDS, "C++ interface function for topk."},
{"trace", (PyCFunction)(void (*)(void))trace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace."},
{"triangular_solve", (PyCFunction)(void (*)(void))triangular_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triangular_solve."},
{"trilinear_interp", (PyCFunction)(void (*)(void))trilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp."},
{"trunc", (PyCFunction)(void (*)(void))trunc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc."},
{"trunc_", (PyCFunction)(void (*)(void))trunc_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc_."},
{"unbind", (PyCFunction)(void (*)(void))unbind, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unbind."},
{"unfold", (PyCFunction)(void (*)(void))unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold."},
{"uniform_inplace", (PyCFunction)(void (*)(void))uniform_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace."},
{"uniform_inplace_", (PyCFunction)(void (*)(void))uniform_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace_."},
{"unique_consecutive", (PyCFunction)(void (*)(void))unique_consecutive, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_consecutive."},
{"unpool3d", (PyCFunction)(void (*)(void))unpool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d."},
{"unsqueeze", (PyCFunction)(void (*)(void))unsqueeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze."},
{"unsqueeze_", (PyCFunction)(void (*)(void))unsqueeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze_."},
{"unstack", (PyCFunction)(void (*)(void))unstack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack."},
{"update_loss_scaling_", (PyCFunction)(void (*)(void))update_loss_scaling_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for update_loss_scaling_."},
{"view_dtype", (PyCFunction)(void (*)(void))view_dtype, METH_VARARGS | METH_KEYWORDS, "C++ interface function for view_dtype."},
{"view_shape", (PyCFunction)(void (*)(void))view_shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for view_shape."},
{"viterbi_decode", (PyCFunction)(void (*)(void))viterbi_decode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for viterbi_decode."},
{"warpctc", (PyCFunction)(void (*)(void))warpctc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warpctc."},
{"warprnnt", (PyCFunction)(void (*)(void))warprnnt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warprnnt."},
{"weight_dequantize", (PyCFunction)(void (*)(void))weight_dequantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for weight_dequantize."},
{"weight_only_linear", (PyCFunction)(void (*)(void))weight_only_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for weight_only_linear."},
{"weight_quantize", (PyCFunction)(void (*)(void))weight_quantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for weight_quantize."},
{"weighted_sample_neighbors", (PyCFunction)(void (*)(void))weighted_sample_neighbors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for weighted_sample_neighbors."},
{"where", (PyCFunction)(void (*)(void))where, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where."},
{"where_", (PyCFunction)(void (*)(void))where_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where_."},
{"yolo_box", (PyCFunction)(void (*)(void))yolo_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box."},
{"yolo_loss", (PyCFunction)(void (*)(void))yolo_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_loss."},
{"block_multihead_attention_", (PyCFunction)(void (*)(void))block_multihead_attention_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for block_multihead_attention_."},
{"fc", (PyCFunction)(void (*)(void))fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fc."},
{"fused_bias_act", (PyCFunction)(void (*)(void))fused_bias_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_act."},
{"fused_bias_dropout_residual_layer_norm", (PyCFunction)(void (*)(void))fused_bias_dropout_residual_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_dropout_residual_layer_norm."},
{"fused_bias_residual_layernorm", (PyCFunction)(void (*)(void))fused_bias_residual_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_residual_layernorm."},
{"fused_conv2d_add_act", (PyCFunction)(void (*)(void))fused_conv2d_add_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_conv2d_add_act."},
{"fused_dropout_add", (PyCFunction)(void (*)(void))fused_dropout_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_dropout_add."},
{"fused_embedding_eltwise_layernorm", (PyCFunction)(void (*)(void))fused_embedding_eltwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_eltwise_layernorm."},
{"fused_fc_elementwise_layernorm", (PyCFunction)(void (*)(void))fused_fc_elementwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_fc_elementwise_layernorm."},
{"fused_linear_param_grad_add", (PyCFunction)(void (*)(void))fused_linear_param_grad_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_linear_param_grad_add."},
{"fused_rotary_position_embedding", (PyCFunction)(void (*)(void))fused_rotary_position_embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_rotary_position_embedding."},
{"fusion_gru", (PyCFunction)(void (*)(void))fusion_gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_gru."},
{"fusion_repeated_fc_relu", (PyCFunction)(void (*)(void))fusion_repeated_fc_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_repeated_fc_relu."},
{"fusion_seqconv_eltadd_relu", (PyCFunction)(void (*)(void))fusion_seqconv_eltadd_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqconv_eltadd_relu."},
{"fusion_seqexpand_concat_fc", (PyCFunction)(void (*)(void))fusion_seqexpand_concat_fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqexpand_concat_fc."},
{"fusion_squared_mat_sub", (PyCFunction)(void (*)(void))fusion_squared_mat_sub, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_squared_mat_sub."},
{"fusion_transpose_flatten_concat", (PyCFunction)(void (*)(void))fusion_transpose_flatten_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_transpose_flatten_concat."},
{"multihead_matmul", (PyCFunction)(void (*)(void))multihead_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multihead_matmul."},
{"self_dp_attention", (PyCFunction)(void (*)(void))self_dp_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for self_dp_attention."},
{"skip_layernorm", (PyCFunction)(void (*)(void))skip_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for skip_layernorm."},
{"squeeze_excitation_block", (PyCFunction)(void (*)(void))squeeze_excitation_block, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_excitation_block."},
{"variable_length_memory_efficient_attention", (PyCFunction)(void (*)(void))variable_length_memory_efficient_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for variable_length_memory_efficient_attention."},
{"adadelta_", (PyCFunction)(void (*)(void))adadelta_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adadelta_."},
{"add", (PyCFunction)(void (*)(void))add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add."},
{"add_", (PyCFunction)(void (*)(void))add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_."},
{"add_n", (PyCFunction)(void (*)(void))add_n, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_n."},
{"all", (PyCFunction)(void (*)(void))all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all."},
{"amax", (PyCFunction)(void (*)(void))amax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amax."},
{"amin", (PyCFunction)(void (*)(void))amin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amin."},
{"any", (PyCFunction)(void (*)(void))any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for any."},
{"assign", (PyCFunction)(void (*)(void))assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign."},
{"assign_", (PyCFunction)(void (*)(void))assign_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_."},
{"assign_out_", (PyCFunction)(void (*)(void))assign_out_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_out_."},
{"assign_value", (PyCFunction)(void (*)(void))assign_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value."},
{"assign_value_", (PyCFunction)(void (*)(void))assign_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value_."},
{"batch_norm", (PyCFunction)(void (*)(void))batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm."},
{"batch_norm_", (PyCFunction)(void (*)(void))batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm_."},
{"c_allreduce_avg_", (PyCFunction)(void (*)(void))c_allreduce_avg_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_avg_."},
{"c_allreduce_max_", (PyCFunction)(void (*)(void))c_allreduce_max_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_max_."},
{"c_allreduce_min_", (PyCFunction)(void (*)(void))c_allreduce_min_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_min_."},
{"c_allreduce_prod_", (PyCFunction)(void (*)(void))c_allreduce_prod_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_prod_."},
{"c_allreduce_sum_", (PyCFunction)(void (*)(void))c_allreduce_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_sum_."},
{"c_broadcast", (PyCFunction)(void (*)(void))c_broadcast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_broadcast."},
{"c_broadcast_", (PyCFunction)(void (*)(void))c_broadcast_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_broadcast_."},
{"c_concat", (PyCFunction)(void (*)(void))c_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_concat."},
{"c_identity_", (PyCFunction)(void (*)(void))c_identity_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_identity_."},
{"c_reduce_sum_", (PyCFunction)(void (*)(void))c_reduce_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reduce_sum_."},
{"c_sync_calc_stream", (PyCFunction)(void (*)(void))c_sync_calc_stream, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_calc_stream."},
{"c_sync_calc_stream_", (PyCFunction)(void (*)(void))c_sync_calc_stream_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_calc_stream_."},
{"c_sync_comm_stream", (PyCFunction)(void (*)(void))c_sync_comm_stream, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_comm_stream."},
{"c_sync_comm_stream_", (PyCFunction)(void (*)(void))c_sync_comm_stream_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_comm_stream_."},
{"cast", (PyCFunction)(void (*)(void))cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast."},
{"cast_", (PyCFunction)(void (*)(void))cast_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast_."},
{"channel_shuffle", (PyCFunction)(void (*)(void))channel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for channel_shuffle."},
{"coalesce_tensor_", (PyCFunction)(void (*)(void))coalesce_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor_."},
{"conv2d_transpose", (PyCFunction)(void (*)(void))conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose."},
{"conv2d_transpose_bias", (PyCFunction)(void (*)(void))conv2d_transpose_bias, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose_bias."},
{"decode_jpeg", (PyCFunction)(void (*)(void))decode_jpeg, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decode_jpeg."},
{"deformable_conv", (PyCFunction)(void (*)(void))deformable_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv."},
{"depthwise_conv2d_transpose", (PyCFunction)(void (*)(void))depthwise_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose."},
{"dequantize_linear", (PyCFunction)(void (*)(void))dequantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_linear."},
{"dequantize_linear_", (PyCFunction)(void (*)(void))dequantize_linear_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_linear_."},
{"disable_check_model_nan_inf", (PyCFunction)(void (*)(void))disable_check_model_nan_inf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for disable_check_model_nan_inf."},
{"distribute_fpn_proposals", (PyCFunction)(void (*)(void))distribute_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distribute_fpn_proposals."},
{"distributed_fused_lamb_init", (PyCFunction)(void (*)(void))distributed_fused_lamb_init, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb_init."},
{"distributed_fused_lamb_init_", (PyCFunction)(void (*)(void))distributed_fused_lamb_init_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb_init_."},
{"divide", (PyCFunction)(void (*)(void))divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide."},
{"divide_", (PyCFunction)(void (*)(void))divide_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide_."},
{"dropout", (PyCFunction)(void (*)(void))dropout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout."},
{"einsum", (PyCFunction)(void (*)(void))einsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for einsum."},
{"elementwise_pow", (PyCFunction)(void (*)(void))elementwise_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow."},
{"embedding", (PyCFunction)(void (*)(void))embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding."},
{"empty", (PyCFunction)(void (*)(void))empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty."},
{"empty_like", (PyCFunction)(void (*)(void))empty_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty_like."},
{"enable_check_model_nan_inf", (PyCFunction)(void (*)(void))enable_check_model_nan_inf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for enable_check_model_nan_inf."},
{"equal", (PyCFunction)(void (*)(void))equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal."},
{"equal_", (PyCFunction)(void (*)(void))equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal_."},
{"exponential_", (PyCFunction)(void (*)(void))exponential_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential_."},
{"eye", (PyCFunction)(void (*)(void))eye, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eye."},
{"fetch", (PyCFunction)(void (*)(void))fetch, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fetch."},
{"floor_divide", (PyCFunction)(void (*)(void))floor_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_divide."},
{"floor_divide_", (PyCFunction)(void (*)(void))floor_divide_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_divide_."},
{"frobenius_norm", (PyCFunction)(void (*)(void))frobenius_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frobenius_norm."},
{"full", (PyCFunction)(void (*)(void))full, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full."},
{"full_", (PyCFunction)(void (*)(void))full_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_."},
{"full_batch_size_like", (PyCFunction)(void (*)(void))full_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_batch_size_like."},
{"full_like", (PyCFunction)(void (*)(void))full_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_like."},
{"full_with_tensor", (PyCFunction)(void (*)(void))full_with_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_with_tensor."},
{"fused_batch_norm_act", (PyCFunction)(void (*)(void))fused_batch_norm_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_batch_norm_act."},
{"fused_bn_add_activation", (PyCFunction)(void (*)(void))fused_bn_add_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bn_add_activation."},
{"fused_multi_transformer", (PyCFunction)(void (*)(void))fused_multi_transformer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer."},
{"fused_softmax_mask", (PyCFunction)(void (*)(void))fused_softmax_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask."},
{"fused_softmax_mask_upper_triangle", (PyCFunction)(void (*)(void))fused_softmax_mask_upper_triangle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_upper_triangle."},
{"gaussian", (PyCFunction)(void (*)(void))gaussian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian."},
{"get_tensor_from_selected_rows", (PyCFunction)(void (*)(void))get_tensor_from_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for get_tensor_from_selected_rows."},
{"greater_equal", (PyCFunction)(void (*)(void))greater_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal."},
{"greater_equal_", (PyCFunction)(void (*)(void))greater_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal_."},
{"greater_than", (PyCFunction)(void (*)(void))greater_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than."},
{"greater_than_", (PyCFunction)(void (*)(void))greater_than_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than_."},
{"hardswish", (PyCFunction)(void (*)(void))hardswish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardswish."},
{"hsigmoid_loss", (PyCFunction)(void (*)(void))hsigmoid_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hsigmoid_loss."},
{"increment", (PyCFunction)(void (*)(void))increment, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment."},
{"increment_", (PyCFunction)(void (*)(void))increment_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment_."},
{"less_equal", (PyCFunction)(void (*)(void))less_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal."},
{"less_equal_", (PyCFunction)(void (*)(void))less_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal_."},
{"less_than", (PyCFunction)(void (*)(void))less_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than."},
{"less_than_", (PyCFunction)(void (*)(void))less_than_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than_."},
{"linspace", (PyCFunction)(void (*)(void))linspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linspace."},
{"logspace", (PyCFunction)(void (*)(void))logspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logspace."},
{"logsumexp", (PyCFunction)(void (*)(void))logsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp."},
{"lrn", (PyCFunction)(void (*)(void))lrn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lrn."},
{"matmul", (PyCFunction)(void (*)(void))matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul."},
{"matmul_with_flatten", (PyCFunction)(void (*)(void))matmul_with_flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul_with_flatten."},
{"matrix_rank", (PyCFunction)(void (*)(void))matrix_rank, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank."},
{"matrix_rank_tol", (PyCFunction)(void (*)(void))matrix_rank_tol, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank_tol."},
{"max", (PyCFunction)(void (*)(void))max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max."},
{"maximum", (PyCFunction)(void (*)(void))maximum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maximum."},
{"mean", (PyCFunction)(void (*)(void))mean, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean."},
{"memcpy", (PyCFunction)(void (*)(void))memcpy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy."},
{"memcpy_d2h", (PyCFunction)(void (*)(void))memcpy_d2h, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_d2h."},
{"memcpy_h2d", (PyCFunction)(void (*)(void))memcpy_h2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_h2d."},
{"min", (PyCFunction)(void (*)(void))min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for min."},
{"minimum", (PyCFunction)(void (*)(void))minimum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minimum."},
{"mish", (PyCFunction)(void (*)(void))mish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish."},
{"multiply", (PyCFunction)(void (*)(void))multiply, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply."},
{"multiply_", (PyCFunction)(void (*)(void))multiply_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply_."},
{"norm", (PyCFunction)(void (*)(void))norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm."},
{"not_equal", (PyCFunction)(void (*)(void))not_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal."},
{"not_equal_", (PyCFunction)(void (*)(void))not_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal_."},
{"one_hot", (PyCFunction)(void (*)(void))one_hot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for one_hot."},
{"pad", (PyCFunction)(void (*)(void))pad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad."},
{"pool2d", (PyCFunction)(void (*)(void))pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d."},
{"pool3d", (PyCFunction)(void (*)(void))pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d."},
{"print", (PyCFunction)(void (*)(void))print, METH_VARARGS | METH_KEYWORDS, "C++ interface function for print."},
{"prod", (PyCFunction)(void (*)(void))prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prod."},
{"quantize_linear", (PyCFunction)(void (*)(void))quantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize_linear."},
{"quantize_linear_", (PyCFunction)(void (*)(void))quantize_linear_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize_linear_."},
{"randint", (PyCFunction)(void (*)(void))randint, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randint."},
{"randperm", (PyCFunction)(void (*)(void))randperm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randperm."},
{"read_file", (PyCFunction)(void (*)(void))read_file, METH_VARARGS | METH_KEYWORDS, "C++ interface function for read_file."},
{"remainder", (PyCFunction)(void (*)(void))remainder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for remainder."},
{"remainder_", (PyCFunction)(void (*)(void))remainder_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for remainder_."},
{"repeat_interleave", (PyCFunction)(void (*)(void))repeat_interleave, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave."},
{"repeat_interleave_with_tensor_index", (PyCFunction)(void (*)(void))repeat_interleave_with_tensor_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave_with_tensor_index."},
{"reshape", (PyCFunction)(void (*)(void))reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape."},
{"reshape_", (PyCFunction)(void (*)(void))reshape_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_."},
{"rnn", (PyCFunction)(void (*)(void))rnn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rnn."},
{"rrelu", (PyCFunction)(void (*)(void))rrelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rrelu."},
{"set_value", (PyCFunction)(void (*)(void))set_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value."},
{"set_value_", (PyCFunction)(void (*)(void))set_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_."},
{"set_value_with_tensor", (PyCFunction)(void (*)(void))set_value_with_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_with_tensor."},
{"set_value_with_tensor_", (PyCFunction)(void (*)(void))set_value_with_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_with_tensor_."},
{"share_data", (PyCFunction)(void (*)(void))share_data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_data."},
{"slice", (PyCFunction)(void (*)(void))slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slice."},
{"softmax", (PyCFunction)(void (*)(void))softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax."},
{"softmax_", (PyCFunction)(void (*)(void))softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_."},
{"split", (PyCFunction)(void (*)(void))split, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split."},
{"split_with_num", (PyCFunction)(void (*)(void))split_with_num, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split_with_num."},
{"strided_slice", (PyCFunction)(void (*)(void))strided_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice."},
{"subtract", (PyCFunction)(void (*)(void))subtract, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract."},
{"subtract_", (PyCFunction)(void (*)(void))subtract_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract_."},
{"sum", (PyCFunction)(void (*)(void))sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum."},
{"swish", (PyCFunction)(void (*)(void))swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish."},
{"sync_batch_norm_", (PyCFunction)(void (*)(void))sync_batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm_."},
{"tile", (PyCFunction)(void (*)(void))tile, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tile."},
{"trans_layout", (PyCFunction)(void (*)(void))trans_layout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trans_layout."},
{"transpose", (PyCFunction)(void (*)(void))transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose."},
{"transpose_", (PyCFunction)(void (*)(void))transpose_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose_."},
{"tril", (PyCFunction)(void (*)(void))tril, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril."},
{"tril_", (PyCFunction)(void (*)(void))tril_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_."},
{"tril_indices", (PyCFunction)(void (*)(void))tril_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_indices."},
{"triu", (PyCFunction)(void (*)(void))triu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu."},
{"triu_", (PyCFunction)(void (*)(void))triu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_."},
{"triu_indices", (PyCFunction)(void (*)(void))triu_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_indices."},
{"truncated_gaussian_random", (PyCFunction)(void (*)(void))truncated_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for truncated_gaussian_random."},
{"uniform", (PyCFunction)(void (*)(void))uniform, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform."},
{"unique", (PyCFunction)(void (*)(void))unique, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique."},
{"unpool", (PyCFunction)(void (*)(void))unpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool."},
{"fused_attention", (PyCFunction)(void (*)(void))fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention."},
{"fused_feedforward", (PyCFunction)(void (*)(void))fused_feedforward, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_feedforward."},
{"moving_average_abs_max_scale", (PyCFunction)(void (*)(void))moving_average_abs_max_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale."},
{"moving_average_abs_max_scale_", (PyCFunction)(void (*)(void))moving_average_abs_max_scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale_."},
{"number_count", (PyCFunction)(void (*)(void))number_count, METH_VARARGS | METH_KEYWORDS, "C++ interface function for number_count."},
{"onednn_to_paddle_layout", (PyCFunction)(void (*)(void))onednn_to_paddle_layout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for onednn_to_paddle_layout."},
{"arange", (PyCFunction)(void (*)(void))arange, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arange."},
{"sequence_mask", (PyCFunction)(void (*)(void))sequence_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_mask."},
{nullptr, nullptr, 0, nullptr}
};

void BindOpsAPI(pybind11::module *module) {
  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {
    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));
  }
  if (PyModule_AddFunctions(module->ptr(), ManualOpsAPI) < 0) {
    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));
  }
}

} // namespace pybind

} // namespace paddle


