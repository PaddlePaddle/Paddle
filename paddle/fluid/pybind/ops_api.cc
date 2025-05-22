
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
static PyObject *accuracy_check(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_accuracy_check";
    return static_api_accuracy_check(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_accuracy_check";
    return eager_api_accuracy_check(self, args, kwargs);
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
static PyObject *adadelta_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_adadelta_";
    return static_api_adadelta_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_adadelta_";
    return eager_api_adadelta_(self, args, kwargs);
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
static PyObject *add_position_encoding(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_position_encoding";
    return static_api_add_position_encoding(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_position_encoding";
    return eager_api_add_position_encoding(self, args, kwargs);
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
static PyObject *affine_channel(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_channel";
    return static_api_affine_channel(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_channel";
    return eager_api_affine_channel(self, args, kwargs);
  }
}
static PyObject *affine_channel_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_channel_";
    return static_api_affine_channel_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_channel_";
    return eager_api_affine_channel_(self, args, kwargs);
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
static PyObject *all(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all";
    return static_api_all(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all";
    return eager_api_all(self, args, kwargs);
  }
}
static PyObject *all_gather(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all_gather";
    return static_api_all_gather(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all_gather";
    return eager_api_all_gather(self, args, kwargs);
  }
}
static PyObject *all_reduce(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all_reduce";
    return static_api_all_reduce(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all_reduce";
    return eager_api_all_reduce(self, args, kwargs);
  }
}
static PyObject *all_reduce_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all_reduce_";
    return static_api_all_reduce_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all_reduce_";
    return eager_api_all_reduce_(self, args, kwargs);
  }
}
static PyObject *all_to_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_all_to_all";
    return static_api_all_to_all(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_all_to_all";
    return eager_api_all_to_all(self, args, kwargs);
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
static PyObject *angle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_angle";
    return static_api_angle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_angle";
    return eager_api_angle(self, args, kwargs);
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
static PyObject *ap_facade(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ap_facade";
    return static_api_ap_facade(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ap_facade";
    return eager_api_ap_facade(self, args, kwargs);
  }
}
static PyObject *ap_trivial_fusion_begin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ap_trivial_fusion_begin";
    return static_api_ap_trivial_fusion_begin(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ap_trivial_fusion_begin";
    return eager_api_ap_trivial_fusion_begin(self, args, kwargs);
  }
}
static PyObject *ap_trivial_fusion_end(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ap_trivial_fusion_end";
    return static_api_ap_trivial_fusion_end(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ap_trivial_fusion_end";
    return eager_api_ap_trivial_fusion_end(self, args, kwargs);
  }
}
static PyObject *ap_variadic(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ap_variadic";
    return static_api_ap_variadic(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ap_variadic";
    return eager_api_ap_variadic(self, args, kwargs);
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
static PyObject *assign_out_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_out_";
    return static_api_assign_out_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_out_";
    return eager_api_assign_out_(self, args, kwargs);
  }
}
static PyObject *assign_pos(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_pos";
    return static_api_assign_pos(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_pos";
    return eager_api_assign_pos(self, args, kwargs);
  }
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
static PyObject *attention_lstm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_attention_lstm";
    return static_api_attention_lstm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_attention_lstm";
    return eager_api_attention_lstm(self, args, kwargs);
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
static PyObject *baddbmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_baddbmm";
    return static_api_baddbmm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_baddbmm";
    return eager_api_baddbmm(self, args, kwargs);
  }
}
static PyObject *baddbmm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_baddbmm_";
    return static_api_baddbmm_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_baddbmm_";
    return eager_api_baddbmm_(self, args, kwargs);
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
static PyObject *beam_search(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_beam_search";
    return static_api_beam_search(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_beam_search";
    return eager_api_beam_search(self, args, kwargs);
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
static PyObject *bipartite_match(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bipartite_match";
    return static_api_bipartite_match(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bipartite_match";
    return eager_api_bipartite_match(self, args, kwargs);
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
static PyObject *box_clip(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_box_clip";
    return static_api_box_clip(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_box_clip";
    return eager_api_box_clip(self, args, kwargs);
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
static PyObject *broadcast(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_broadcast";
    return static_api_broadcast(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_broadcast";
    return eager_api_broadcast(self, args, kwargs);
  }
}
static PyObject *broadcast_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_broadcast_";
    return static_api_broadcast_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_broadcast_";
    return eager_api_broadcast_(self, args, kwargs);
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
static PyObject *c_allreduce_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_allreduce_sum";
    return static_api_c_allreduce_sum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_allreduce_sum";
    return eager_api_c_allreduce_sum(self, args, kwargs);
  }
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
static PyObject *c_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_concat";
    return static_api_c_concat(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_concat";
    return eager_api_c_concat(self, args, kwargs);
  }
}
static PyObject *c_identity(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_identity";
    return static_api_c_identity(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_identity";
    return eager_api_c_identity(self, args, kwargs);
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
static PyObject *c_softmax_with_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_softmax_with_cross_entropy";
    return static_api_c_softmax_with_cross_entropy(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_softmax_with_cross_entropy";
    return eager_api_c_softmax_with_cross_entropy(self, args, kwargs);
  }
}
static PyObject *calc_reduced_attn_scores(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_calc_reduced_attn_scores";
    return static_api_calc_reduced_attn_scores(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_calc_reduced_attn_scores";
    return eager_api_calc_reduced_attn_scores(self, args, kwargs);
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
static PyObject *channel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_channel_shuffle";
    return static_api_channel_shuffle(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_channel_shuffle";
    return eager_api_channel_shuffle(self, args, kwargs);
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
static PyObject *collect_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_collect_fpn_proposals";
    return static_api_collect_fpn_proposals(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_collect_fpn_proposals";
    return eager_api_collect_fpn_proposals(self, args, kwargs);
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
static PyObject *correlation(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_correlation";
    return static_api_correlation(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_correlation";
    return eager_api_correlation(self, args, kwargs);
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
static PyObject *crf_decoding(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_crf_decoding";
    return static_api_crf_decoding(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_crf_decoding";
    return eager_api_crf_decoding(self, args, kwargs);
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
static PyObject *ctc_align(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ctc_align";
    return static_api_ctc_align(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ctc_align";
    return eager_api_ctc_align(self, args, kwargs);
  }
}
static PyObject *cudnn_lstm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cudnn_lstm";
    return static_api_cudnn_lstm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cudnn_lstm";
    return eager_api_cudnn_lstm(self, args, kwargs);
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
static PyObject *cvm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cvm";
    return static_api_cvm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cvm";
    return eager_api_cvm(self, args, kwargs);
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
static PyObject *depend(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depend";
    return static_api_depend(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depend";
    return eager_api_depend(self, args, kwargs);
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
static PyObject *depthwise_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depthwise_conv2d_transpose";
    return static_api_depthwise_conv2d_transpose(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depthwise_conv2d_transpose";
    return eager_api_depthwise_conv2d_transpose(self, args, kwargs);
  }
}
static PyObject *dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dequantize_abs_max";
    return static_api_dequantize_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dequantize_abs_max";
    return eager_api_dequantize_abs_max(self, args, kwargs);
  }
}
static PyObject *dequantize_log(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dequantize_log";
    return static_api_dequantize_log(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dequantize_log";
    return eager_api_dequantize_log(self, args, kwargs);
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
static PyObject *dgc_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dgc_clip_by_norm";
    return static_api_dgc_clip_by_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dgc_clip_by_norm";
    return eager_api_dgc_clip_by_norm(self, args, kwargs);
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
static PyObject *disable_check_model_nan_inf(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_disable_check_model_nan_inf";
    return static_api_disable_check_model_nan_inf(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_disable_check_model_nan_inf";
    return eager_api_disable_check_model_nan_inf(self, args, kwargs);
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
static PyObject *dropout(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dropout";
    return static_api_dropout(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dropout";
    return eager_api_dropout(self, args, kwargs);
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
static PyObject *embedding_with_scaled_gradient(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_embedding_with_scaled_gradient";
    return static_api_embedding_with_scaled_gradient(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_embedding_with_scaled_gradient";
    return eager_api_embedding_with_scaled_gradient(self, args, kwargs);
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
static PyObject *fake_channel_wise_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_channel_wise_dequantize_max_abs";
    return static_api_fake_channel_wise_dequantize_max_abs(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_channel_wise_dequantize_max_abs";
    return eager_api_fake_channel_wise_dequantize_max_abs(self, args, kwargs);
  }
}
static PyObject *fake_channel_wise_quantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_channel_wise_quantize_abs_max";
    return static_api_fake_channel_wise_quantize_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_channel_wise_quantize_abs_max";
    return eager_api_fake_channel_wise_quantize_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_channel_wise_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_channel_wise_quantize_dequantize_abs_max";
    return static_api_fake_channel_wise_quantize_dequantize_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_channel_wise_quantize_dequantize_abs_max";
    return eager_api_fake_channel_wise_quantize_dequantize_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_dequantize_max_abs";
    return static_api_fake_dequantize_max_abs(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_dequantize_max_abs";
    return eager_api_fake_dequantize_max_abs(self, args, kwargs);
  }
}
static PyObject *fake_quantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_abs_max";
    return static_api_fake_quantize_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_abs_max";
    return eager_api_fake_quantize_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_dequantize_abs_max";
    return static_api_fake_quantize_dequantize_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_dequantize_abs_max";
    return eager_api_fake_quantize_dequantize_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_quantize_dequantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_dequantize_moving_average_abs_max";
    return static_api_fake_quantize_dequantize_moving_average_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_dequantize_moving_average_abs_max";
    return eager_api_fake_quantize_dequantize_moving_average_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_quantize_dequantize_moving_average_abs_max_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_dequantize_moving_average_abs_max_";
    return static_api_fake_quantize_dequantize_moving_average_abs_max_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_dequantize_moving_average_abs_max_";
    return eager_api_fake_quantize_dequantize_moving_average_abs_max_(self, args, kwargs);
  }
}
static PyObject *fake_quantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_moving_average_abs_max";
    return static_api_fake_quantize_moving_average_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_moving_average_abs_max";
    return eager_api_fake_quantize_moving_average_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_quantize_moving_average_abs_max_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_moving_average_abs_max_";
    return static_api_fake_quantize_moving_average_abs_max_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_moving_average_abs_max_";
    return eager_api_fake_quantize_moving_average_abs_max_(self, args, kwargs);
  }
}
static PyObject *fake_quantize_range_abs_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_range_abs_max";
    return static_api_fake_quantize_range_abs_max(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_range_abs_max";
    return eager_api_fake_quantize_range_abs_max(self, args, kwargs);
  }
}
static PyObject *fake_quantize_range_abs_max_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_range_abs_max_";
    return static_api_fake_quantize_range_abs_max_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_range_abs_max_";
    return eager_api_fake_quantize_range_abs_max_(self, args, kwargs);
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
static PyObject *flash_attn_qkvpacked(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_qkvpacked";
    return static_api_flash_attn_qkvpacked(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_qkvpacked";
    return eager_api_flash_attn_qkvpacked(self, args, kwargs);
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
static PyObject *flash_attn_v3(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_v3";
    return static_api_flash_attn_v3(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_v3";
    return eager_api_flash_attn_v3(self, args, kwargs);
  }
}
static PyObject *flash_attn_varlen_qkvpacked(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_varlen_qkvpacked";
    return static_api_flash_attn_varlen_qkvpacked(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_varlen_qkvpacked";
    return eager_api_flash_attn_varlen_qkvpacked(self, args, kwargs);
  }
}
static PyObject *flashmask_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flashmask_attention";
    return static_api_flashmask_attention(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flashmask_attention";
    return eager_api_flashmask_attention(self, args, kwargs);
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
static PyObject *full_int_array(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_int_array";
    return static_api_full_int_array(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_int_array";
    return eager_api_full_int_array(self, args, kwargs);
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
static PyObject *gaussian(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian";
    return static_api_gaussian(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian";
    return eager_api_gaussian(self, args, kwargs);
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
static PyObject *gru(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gru";
    return static_api_gru(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gru";
    return eager_api_gru(self, args, kwargs);
  }
}
static PyObject *gru_unit(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gru_unit";
    return static_api_gru_unit(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gru_unit";
    return eager_api_gru_unit(self, args, kwargs);
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
static PyObject *hinge_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hinge_loss";
    return static_api_hinge_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hinge_loss";
    return eager_api_hinge_loss(self, args, kwargs);
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
static PyObject *hsigmoid_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hsigmoid_loss";
    return static_api_hsigmoid_loss(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hsigmoid_loss";
    return eager_api_hsigmoid_loss(self, args, kwargs);
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
static PyObject *im2sequence(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_im2sequence";
    return static_api_im2sequence(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_im2sequence";
    return eager_api_im2sequence(self, args, kwargs);
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
static PyObject *l1_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_l1_norm";
    return static_api_l1_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_l1_norm";
    return eager_api_l1_norm(self, args, kwargs);
  }
}
static PyObject *l1_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_l1_norm_";
    return static_api_l1_norm_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_l1_norm_";
    return eager_api_l1_norm_(self, args, kwargs);
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
static PyObject *limit_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_limit_by_capacity";
    return static_api_limit_by_capacity(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_limit_by_capacity";
    return eager_api_limit_by_capacity(self, args, kwargs);
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
static PyObject *linspace(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_linspace";
    return static_api_linspace(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_linspace";
    return eager_api_linspace(self, args, kwargs);
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
static PyObject *lookup_table_dequant(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lookup_table_dequant";
    return static_api_lookup_table_dequant(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lookup_table_dequant";
    return eager_api_lookup_table_dequant(self, args, kwargs);
  }
}
static PyObject *lp_pool2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lp_pool2d";
    return static_api_lp_pool2d(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lp_pool2d";
    return eager_api_lp_pool2d(self, args, kwargs);
  }
}
static PyObject *lstm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lstm";
    return static_api_lstm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lstm";
    return eager_api_lstm(self, args, kwargs);
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
static PyObject *lu_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_solve";
    return static_api_lu_solve(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_solve";
    return eager_api_lu_solve(self, args, kwargs);
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
static PyObject *matrix_rank(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_rank";
    return static_api_matrix_rank(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_rank";
    return eager_api_matrix_rank(self, args, kwargs);
  }
}
static PyObject *matrix_rank_atol_rtol(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_rank_atol_rtol";
    return static_api_matrix_rank_atol_rtol(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_rank_atol_rtol";
    return eager_api_matrix_rank_atol_rtol(self, args, kwargs);
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
static PyObject *mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mean";
    return static_api_mean(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mean";
    return eager_api_mean(self, args, kwargs);
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
static PyObject *mish(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mish";
    return static_api_mish(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mish";
    return eager_api_mish(self, args, kwargs);
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
static PyObject *mp_allreduce_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mp_allreduce_sum";
    return static_api_mp_allreduce_sum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mp_allreduce_sum";
    return eager_api_mp_allreduce_sum(self, args, kwargs);
  }
}
static PyObject *mp_allreduce_sum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mp_allreduce_sum_";
    return static_api_mp_allreduce_sum_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mp_allreduce_sum_";
    return eager_api_mp_allreduce_sum_(self, args, kwargs);
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
static PyObject *nadam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nadam_";
    return static_api_nadam_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nadam_";
    return eager_api_nadam_(self, args, kwargs);
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
static PyObject *norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_norm";
    return static_api_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_norm";
    return eager_api_norm(self, args, kwargs);
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
static PyObject *one_hot(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_one_hot";
    return static_api_one_hot(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_one_hot";
    return eager_api_one_hot(self, args, kwargs);
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
static PyObject *pad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pad";
    return static_api_pad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pad";
    return eager_api_pad(self, args, kwargs);
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
static PyObject *prod(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prod";
    return static_api_prod(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prod";
    return eager_api_prod(self, args, kwargs);
  }
}
static PyObject *prune_gate_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prune_gate_by_capacity";
    return static_api_prune_gate_by_capacity(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prune_gate_by_capacity";
    return eager_api_prune_gate_by_capacity(self, args, kwargs);
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
static PyObject *pyramid_hash(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pyramid_hash";
    return static_api_pyramid_hash(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pyramid_hash";
    return eager_api_pyramid_hash(self, args, kwargs);
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
static PyObject *radam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_radam_";
    return static_api_radam_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_radam_";
    return eager_api_radam_(self, args, kwargs);
  }
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
static PyObject *random_routing_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_random_routing_";
    return static_api_random_routing_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_random_routing_";
    return eager_api_random_routing_(self, args, kwargs);
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
static PyObject *rank_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rank_attention";
    return static_api_rank_attention(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rank_attention";
    return eager_api_rank_attention(self, args, kwargs);
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
static PyObject *reduce(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reduce";
    return static_api_reduce(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reduce";
    return eager_api_reduce(self, args, kwargs);
  }
}
static PyObject *reduce_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reduce_";
    return static_api_reduce_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reduce_";
    return eager_api_reduce_(self, args, kwargs);
  }
}
static PyObject *reduce_as(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reduce_as";
    return static_api_reduce_as(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reduce_as";
    return eager_api_reduce_as(self, args, kwargs);
  }
}
static PyObject *reduce_scatter(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reduce_scatter";
    return static_api_reduce_scatter(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reduce_scatter";
    return eager_api_reduce_scatter(self, args, kwargs);
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
static PyObject *restrict_nonzero(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_restrict_nonzero";
    return static_api_restrict_nonzero(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_restrict_nonzero";
    return eager_api_restrict_nonzero(self, args, kwargs);
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
static PyObject *rnn(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rnn";
    return static_api_rnn(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rnn";
    return eager_api_rnn(self, args, kwargs);
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
static PyObject *rrelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rrelu";
    return static_api_rrelu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rrelu";
    return eager_api_rrelu(self, args, kwargs);
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
static PyObject *sequence_conv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sequence_conv";
    return static_api_sequence_conv(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sequence_conv";
    return eager_api_sequence_conv(self, args, kwargs);
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
static PyObject *sequence_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sequence_pool";
    return static_api_sequence_pool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sequence_pool";
    return eager_api_sequence_pool(self, args, kwargs);
  }
}
static PyObject *set(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set";
    return static_api_set(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set";
    return eager_api_set(self, args, kwargs);
  }
}
static PyObject *set_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_";
    return static_api_set_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_";
    return eager_api_set_(self, args, kwargs);
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
static PyObject *shape64(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shape64";
    return static_api_shape64(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shape64";
    return eager_api_shape64(self, args, kwargs);
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
static PyObject *share_data(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_share_data";
    return static_api_share_data(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_share_data";
    return eager_api_share_data(self, args, kwargs);
  }
}
static PyObject *shuffle_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shuffle_batch";
    return static_api_shuffle_batch(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shuffle_batch";
    return eager_api_shuffle_batch(self, args, kwargs);
  }
}
static PyObject *shuffle_channel(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shuffle_channel";
    return static_api_shuffle_channel(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shuffle_channel";
    return eager_api_shuffle_channel(self, args, kwargs);
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
static PyObject *slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slice";
    return static_api_slice(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slice";
    return eager_api_slice(self, args, kwargs);
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
static PyObject *sparse_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sparse_attention";
    return static_api_sparse_attention(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sparse_attention";
    return eager_api_sparse_attention(self, args, kwargs);
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
static PyObject *square_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square_";
    return static_api_square_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square_";
    return eager_api_square_(self, args, kwargs);
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
static PyObject *stft(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stft";
    return static_api_stft(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stft";
    return eager_api_stft(self, args, kwargs);
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
static PyObject *sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sum";
    return static_api_sum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sum";
    return eager_api_sum(self, args, kwargs);
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
static PyObject *svdvals(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_svdvals";
    return static_api_svdvals(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_svdvals";
    return eager_api_svdvals(self, args, kwargs);
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
static PyObject *sync_calc_stream(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_calc_stream";
    return static_api_sync_calc_stream(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_calc_stream";
    return eager_api_sync_calc_stream(self, args, kwargs);
  }
}
static PyObject *sync_calc_stream_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_calc_stream_";
    return static_api_sync_calc_stream_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_calc_stream_";
    return eager_api_sync_calc_stream_(self, args, kwargs);
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
static PyObject *tdm_child(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tdm_child";
    return static_api_tdm_child(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tdm_child";
    return eager_api_tdm_child(self, args, kwargs);
  }
}
static PyObject *tdm_sampler(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tdm_sampler";
    return static_api_tdm_sampler(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tdm_sampler";
    return eager_api_tdm_sampler(self, args, kwargs);
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
static PyObject *triangular_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triangular_solve";
    return static_api_triangular_solve(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triangular_solve";
    return eager_api_triangular_solve(self, args, kwargs);
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
static PyObject *trilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trilinear_interp";
    return static_api_trilinear_interp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trilinear_interp";
    return eager_api_trilinear_interp(self, args, kwargs);
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
static PyObject *truncated_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_truncated_gaussian_random";
    return static_api_truncated_gaussian_random(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_truncated_gaussian_random";
    return eager_api_truncated_gaussian_random(self, args, kwargs);
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
static PyObject *uniform(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform";
    return static_api_uniform(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform";
    return eager_api_uniform(self, args, kwargs);
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
static PyObject *uniform_random_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform_random_batch_size_like";
    return static_api_uniform_random_batch_size_like(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform_random_batch_size_like";
    return eager_api_uniform_random_batch_size_like(self, args, kwargs);
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
static PyObject *unpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unpool";
    return static_api_unpool(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unpool";
    return eager_api_unpool(self, args, kwargs);
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
static PyObject *variance(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_variance";
    return static_api_variance(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_variance";
    return eager_api_variance(self, args, kwargs);
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
static PyObject *view_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_view_slice";
    return static_api_view_slice(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_view_slice";
    return eager_api_view_slice(self, args, kwargs);
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
static PyObject *yolo_box_head(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_yolo_box_head";
    return static_api_yolo_box_head(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_yolo_box_head";
    return eager_api_yolo_box_head(self, args, kwargs);
  }
}
static PyObject *yolo_box_post(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_yolo_box_post";
    return static_api_yolo_box_post(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_yolo_box_post";
    return eager_api_yolo_box_post(self, args, kwargs);
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
static PyObject *chunk_eval(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_chunk_eval";
    return static_api_chunk_eval(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_chunk_eval";
    return eager_api_chunk_eval(self, args, kwargs);
  }
}
static PyObject *number_count(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_number_count";
    return static_api_number_count(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_number_count";
    return eager_api_number_count(self, args, kwargs);
  }
}
static PyObject *fused_rms_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_rms_norm";
    return static_api_fused_rms_norm(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_rms_norm";
    return eager_api_fused_rms_norm(self, args, kwargs);
  }
}
static PyObject *int_bincount(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_int_bincount";
    return static_api_int_bincount(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_int_bincount";
    return eager_api_int_bincount(self, args, kwargs);
  }
}
static PyObject *abs_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs_grad";
    return static_api_abs_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs_grad";
    return eager_api_abs_grad(self, args, kwargs);
  }
}
static PyObject *acos_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos_grad";
    return static_api_acos_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos_grad";
    return eager_api_acos_grad(self, args, kwargs);
  }
}
static PyObject *acos_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos_grad_";
    return static_api_acos_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos_grad_";
    return eager_api_acos_grad_(self, args, kwargs);
  }
}
static PyObject *acosh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh_grad";
    return static_api_acosh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh_grad";
    return eager_api_acosh_grad(self, args, kwargs);
  }
}
static PyObject *acosh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh_grad_";
    return static_api_acosh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh_grad_";
    return eager_api_acosh_grad_(self, args, kwargs);
  }
}
static PyObject *add_position_encoding_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_position_encoding_grad";
    return static_api_add_position_encoding_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_position_encoding_grad";
    return eager_api_add_position_encoding_grad(self, args, kwargs);
  }
}
static PyObject *addmm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_addmm_grad";
    return static_api_addmm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_addmm_grad";
    return eager_api_addmm_grad(self, args, kwargs);
  }
}
static PyObject *affine_channel_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_channel_grad";
    return static_api_affine_channel_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_channel_grad";
    return eager_api_affine_channel_grad(self, args, kwargs);
  }
}
static PyObject *affine_channel_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_channel_grad_";
    return static_api_affine_channel_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_channel_grad_";
    return eager_api_affine_channel_grad_(self, args, kwargs);
  }
}
static PyObject *affine_grid_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_affine_grid_grad";
    return static_api_affine_grid_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_affine_grid_grad";
    return eager_api_affine_grid_grad(self, args, kwargs);
  }
}
static PyObject *amax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_amax_grad";
    return static_api_amax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_amax_grad";
    return eager_api_amax_grad(self, args, kwargs);
  }
}
static PyObject *amin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_amin_grad";
    return static_api_amin_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_amin_grad";
    return eager_api_amin_grad(self, args, kwargs);
  }
}
static PyObject *angle_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_angle_grad";
    return static_api_angle_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_angle_grad";
    return eager_api_angle_grad(self, args, kwargs);
  }
}
static PyObject *argsort_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_argsort_grad";
    return static_api_argsort_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_argsort_grad";
    return eager_api_argsort_grad(self, args, kwargs);
  }
}
static PyObject *as_strided_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_as_strided_grad";
    return static_api_as_strided_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_as_strided_grad";
    return eager_api_as_strided_grad(self, args, kwargs);
  }
}
static PyObject *asin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin_grad";
    return static_api_asin_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin_grad";
    return eager_api_asin_grad(self, args, kwargs);
  }
}
static PyObject *asin_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin_grad_";
    return static_api_asin_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin_grad_";
    return eager_api_asin_grad_(self, args, kwargs);
  }
}
static PyObject *asinh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh_grad";
    return static_api_asinh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh_grad";
    return eager_api_asinh_grad(self, args, kwargs);
  }
}
static PyObject *asinh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh_grad_";
    return static_api_asinh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh_grad_";
    return eager_api_asinh_grad_(self, args, kwargs);
  }
}
static PyObject *atan2_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan2_grad";
    return static_api_atan2_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan2_grad";
    return eager_api_atan2_grad(self, args, kwargs);
  }
}
static PyObject *atan_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan_grad";
    return static_api_atan_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan_grad";
    return eager_api_atan_grad(self, args, kwargs);
  }
}
static PyObject *atan_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan_grad_";
    return static_api_atan_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan_grad_";
    return eager_api_atan_grad_(self, args, kwargs);
  }
}
static PyObject *atanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh_grad";
    return static_api_atanh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh_grad";
    return eager_api_atanh_grad(self, args, kwargs);
  }
}
static PyObject *atanh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh_grad_";
    return static_api_atanh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh_grad_";
    return eager_api_atanh_grad_(self, args, kwargs);
  }
}
static PyObject *baddbmm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_baddbmm_grad";
    return static_api_baddbmm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_baddbmm_grad";
    return eager_api_baddbmm_grad(self, args, kwargs);
  }
}
static PyObject *bce_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bce_loss_grad";
    return static_api_bce_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bce_loss_grad";
    return eager_api_bce_loss_grad(self, args, kwargs);
  }
}
static PyObject *bce_loss_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bce_loss_grad_";
    return static_api_bce_loss_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bce_loss_grad_";
    return eager_api_bce_loss_grad_(self, args, kwargs);
  }
}
static PyObject *bicubic_interp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bicubic_interp_grad";
    return static_api_bicubic_interp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bicubic_interp_grad";
    return eager_api_bicubic_interp_grad(self, args, kwargs);
  }
}
static PyObject *bilinear_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bilinear_grad";
    return static_api_bilinear_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bilinear_grad";
    return eager_api_bilinear_grad(self, args, kwargs);
  }
}
static PyObject *bilinear_interp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bilinear_interp_grad";
    return static_api_bilinear_interp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bilinear_interp_grad";
    return eager_api_bilinear_interp_grad(self, args, kwargs);
  }
}
static PyObject *bmm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_bmm_grad";
    return static_api_bmm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_bmm_grad";
    return eager_api_bmm_grad(self, args, kwargs);
  }
}
static PyObject *broadcast_tensors_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_broadcast_tensors_grad";
    return static_api_broadcast_tensors_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_broadcast_tensors_grad";
    return eager_api_broadcast_tensors_grad(self, args, kwargs);
  }
}
static PyObject *c_softmax_with_cross_entropy_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_softmax_with_cross_entropy_grad";
    return static_api_c_softmax_with_cross_entropy_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_softmax_with_cross_entropy_grad";
    return eager_api_c_softmax_with_cross_entropy_grad(self, args, kwargs);
  }
}
static PyObject *c_softmax_with_cross_entropy_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_softmax_with_cross_entropy_grad_";
    return static_api_c_softmax_with_cross_entropy_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_softmax_with_cross_entropy_grad_";
    return eager_api_c_softmax_with_cross_entropy_grad_(self, args, kwargs);
  }
}
static PyObject *ceil_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ceil_grad";
    return static_api_ceil_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ceil_grad";
    return eager_api_ceil_grad(self, args, kwargs);
  }
}
static PyObject *ceil_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_ceil_grad_";
    return static_api_ceil_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_ceil_grad_";
    return eager_api_ceil_grad_(self, args, kwargs);
  }
}
static PyObject *celu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_celu_grad";
    return static_api_celu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_celu_grad";
    return eager_api_celu_grad(self, args, kwargs);
  }
}
static PyObject *celu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_celu_grad_";
    return static_api_celu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_celu_grad_";
    return eager_api_celu_grad_(self, args, kwargs);
  }
}
static PyObject *channel_shuffle_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_channel_shuffle_grad";
    return static_api_channel_shuffle_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_channel_shuffle_grad";
    return eager_api_channel_shuffle_grad(self, args, kwargs);
  }
}
static PyObject *cholesky_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cholesky_grad";
    return static_api_cholesky_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cholesky_grad";
    return eager_api_cholesky_grad(self, args, kwargs);
  }
}
static PyObject *cholesky_solve_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cholesky_solve_grad";
    return static_api_cholesky_solve_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cholesky_solve_grad";
    return eager_api_cholesky_solve_grad(self, args, kwargs);
  }
}
static PyObject *clip_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_clip_grad";
    return static_api_clip_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_clip_grad";
    return eager_api_clip_grad(self, args, kwargs);
  }
}
static PyObject *clip_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_clip_grad_";
    return static_api_clip_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_clip_grad_";
    return eager_api_clip_grad_(self, args, kwargs);
  }
}
static PyObject *complex_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_complex_grad";
    return static_api_complex_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_complex_grad";
    return eager_api_complex_grad(self, args, kwargs);
  }
}
static PyObject *concat_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_concat_grad";
    return static_api_concat_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_concat_grad";
    return eager_api_concat_grad(self, args, kwargs);
  }
}
static PyObject *conv2d_transpose_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv2d_transpose_grad";
    return static_api_conv2d_transpose_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv2d_transpose_grad";
    return eager_api_conv2d_transpose_grad(self, args, kwargs);
  }
}
static PyObject *conv3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d_grad";
    return static_api_conv3d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d_grad";
    return eager_api_conv3d_grad(self, args, kwargs);
  }
}
static PyObject *conv3d_transpose_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d_transpose_grad";
    return static_api_conv3d_transpose_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d_transpose_grad";
    return eager_api_conv3d_transpose_grad(self, args, kwargs);
  }
}
static PyObject *copysign_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_copysign_grad";
    return static_api_copysign_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_copysign_grad";
    return eager_api_copysign_grad(self, args, kwargs);
  }
}
static PyObject *copysign_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_copysign_grad_";
    return static_api_copysign_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_copysign_grad_";
    return eager_api_copysign_grad_(self, args, kwargs);
  }
}
static PyObject *correlation_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_correlation_grad";
    return static_api_correlation_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_correlation_grad";
    return eager_api_correlation_grad(self, args, kwargs);
  }
}
static PyObject *cos_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cos_grad";
    return static_api_cos_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cos_grad";
    return eager_api_cos_grad(self, args, kwargs);
  }
}
static PyObject *cos_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cos_grad_";
    return static_api_cos_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cos_grad_";
    return eager_api_cos_grad_(self, args, kwargs);
  }
}
static PyObject *cosh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cosh_grad";
    return static_api_cosh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cosh_grad";
    return eager_api_cosh_grad(self, args, kwargs);
  }
}
static PyObject *cosh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cosh_grad_";
    return static_api_cosh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cosh_grad_";
    return eager_api_cosh_grad_(self, args, kwargs);
  }
}
static PyObject *crop_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_crop_grad";
    return static_api_crop_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_crop_grad";
    return eager_api_crop_grad(self, args, kwargs);
  }
}
static PyObject *cross_entropy_with_softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross_entropy_with_softmax_grad";
    return static_api_cross_entropy_with_softmax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross_entropy_with_softmax_grad";
    return eager_api_cross_entropy_with_softmax_grad(self, args, kwargs);
  }
}
static PyObject *cross_entropy_with_softmax_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross_entropy_with_softmax_grad_";
    return static_api_cross_entropy_with_softmax_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross_entropy_with_softmax_grad_";
    return eager_api_cross_entropy_with_softmax_grad_(self, args, kwargs);
  }
}
static PyObject *cross_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cross_grad";
    return static_api_cross_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cross_grad";
    return eager_api_cross_grad(self, args, kwargs);
  }
}
static PyObject *cudnn_lstm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cudnn_lstm_grad";
    return static_api_cudnn_lstm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cudnn_lstm_grad";
    return eager_api_cudnn_lstm_grad(self, args, kwargs);
  }
}
static PyObject *cummax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cummax_grad";
    return static_api_cummax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cummax_grad";
    return eager_api_cummax_grad(self, args, kwargs);
  }
}
static PyObject *cummin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cummin_grad";
    return static_api_cummin_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cummin_grad";
    return eager_api_cummin_grad(self, args, kwargs);
  }
}
static PyObject *cumprod_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumprod_grad";
    return static_api_cumprod_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumprod_grad";
    return eager_api_cumprod_grad(self, args, kwargs);
  }
}
static PyObject *cumsum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cumsum_grad";
    return static_api_cumsum_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cumsum_grad";
    return eager_api_cumsum_grad(self, args, kwargs);
  }
}
static PyObject *cvm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cvm_grad";
    return static_api_cvm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cvm_grad";
    return eager_api_cvm_grad(self, args, kwargs);
  }
}
static PyObject *deformable_conv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_deformable_conv_grad";
    return static_api_deformable_conv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_deformable_conv_grad";
    return eager_api_deformable_conv_grad(self, args, kwargs);
  }
}
static PyObject *depthwise_conv2d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depthwise_conv2d_grad";
    return static_api_depthwise_conv2d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depthwise_conv2d_grad";
    return eager_api_depthwise_conv2d_grad(self, args, kwargs);
  }
}
static PyObject *depthwise_conv2d_transpose_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_depthwise_conv2d_transpose_grad";
    return static_api_depthwise_conv2d_transpose_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_depthwise_conv2d_transpose_grad";
    return eager_api_depthwise_conv2d_transpose_grad(self, args, kwargs);
  }
}
static PyObject *det_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_det_grad";
    return static_api_det_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_det_grad";
    return eager_api_det_grad(self, args, kwargs);
  }
}
static PyObject *diag_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_diag_grad";
    return static_api_diag_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_diag_grad";
    return eager_api_diag_grad(self, args, kwargs);
  }
}
static PyObject *diagonal_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_diagonal_grad";
    return static_api_diagonal_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_diagonal_grad";
    return eager_api_diagonal_grad(self, args, kwargs);
  }
}
static PyObject *digamma_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_digamma_grad";
    return static_api_digamma_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_digamma_grad";
    return eager_api_digamma_grad(self, args, kwargs);
  }
}
static PyObject *dist_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dist_grad";
    return static_api_dist_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dist_grad";
    return eager_api_dist_grad(self, args, kwargs);
  }
}
static PyObject *dot_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dot_grad";
    return static_api_dot_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dot_grad";
    return eager_api_dot_grad(self, args, kwargs);
  }
}
static PyObject *dropout_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_dropout_grad";
    return static_api_dropout_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_dropout_grad";
    return eager_api_dropout_grad(self, args, kwargs);
  }
}
static PyObject *eig_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eig_grad";
    return static_api_eig_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eig_grad";
    return eager_api_eig_grad(self, args, kwargs);
  }
}
static PyObject *eigh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eigh_grad";
    return static_api_eigh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eigh_grad";
    return eager_api_eigh_grad(self, args, kwargs);
  }
}
static PyObject *eigvalsh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_eigvalsh_grad";
    return static_api_eigvalsh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_eigvalsh_grad";
    return eager_api_eigvalsh_grad(self, args, kwargs);
  }
}
static PyObject *elu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elu_grad";
    return static_api_elu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elu_grad";
    return eager_api_elu_grad(self, args, kwargs);
  }
}
static PyObject *elu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elu_grad_";
    return static_api_elu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elu_grad_";
    return eager_api_elu_grad_(self, args, kwargs);
  }
}
static PyObject *embedding_with_scaled_gradient_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_embedding_with_scaled_gradient_grad";
    return static_api_embedding_with_scaled_gradient_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_embedding_with_scaled_gradient_grad";
    return eager_api_embedding_with_scaled_gradient_grad(self, args, kwargs);
  }
}
static PyObject *erf_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erf_grad";
    return static_api_erf_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erf_grad";
    return eager_api_erf_grad(self, args, kwargs);
  }
}
static PyObject *erfinv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_erfinv_grad";
    return static_api_erfinv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_erfinv_grad";
    return eager_api_erfinv_grad(self, args, kwargs);
  }
}
static PyObject *exp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_exp_grad";
    return static_api_exp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_exp_grad";
    return eager_api_exp_grad(self, args, kwargs);
  }
}
static PyObject *exp_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_exp_grad_";
    return static_api_exp_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_exp_grad_";
    return eager_api_exp_grad_(self, args, kwargs);
  }
}
static PyObject *expand_as_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expand_as_grad";
    return static_api_expand_as_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expand_as_grad";
    return eager_api_expand_as_grad(self, args, kwargs);
  }
}
static PyObject *expand_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expand_grad";
    return static_api_expand_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expand_grad";
    return eager_api_expand_grad(self, args, kwargs);
  }
}
static PyObject *expm1_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1_grad";
    return static_api_expm1_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1_grad";
    return eager_api_expm1_grad(self, args, kwargs);
  }
}
static PyObject *expm1_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1_grad_";
    return static_api_expm1_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1_grad_";
    return eager_api_expm1_grad_(self, args, kwargs);
  }
}
static PyObject *fake_channel_wise_quantize_dequantize_abs_max_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_channel_wise_quantize_dequantize_abs_max_grad";
    return static_api_fake_channel_wise_quantize_dequantize_abs_max_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_channel_wise_quantize_dequantize_abs_max_grad";
    return eager_api_fake_channel_wise_quantize_dequantize_abs_max_grad(self, args, kwargs);
  }
}
static PyObject *fake_quantize_dequantize_abs_max_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_dequantize_abs_max_grad";
    return static_api_fake_quantize_dequantize_abs_max_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_dequantize_abs_max_grad";
    return eager_api_fake_quantize_dequantize_abs_max_grad(self, args, kwargs);
  }
}
static PyObject *fake_quantize_dequantize_moving_average_abs_max_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fake_quantize_dequantize_moving_average_abs_max_grad";
    return static_api_fake_quantize_dequantize_moving_average_abs_max_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fake_quantize_dequantize_moving_average_abs_max_grad";
    return eager_api_fake_quantize_dequantize_moving_average_abs_max_grad(self, args, kwargs);
  }
}
static PyObject *fft_c2c_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_c2c_grad";
    return static_api_fft_c2c_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_c2c_grad";
    return eager_api_fft_c2c_grad(self, args, kwargs);
  }
}
static PyObject *fft_c2r_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_c2r_grad";
    return static_api_fft_c2r_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_c2r_grad";
    return eager_api_fft_c2r_grad(self, args, kwargs);
  }
}
static PyObject *fft_r2c_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fft_r2c_grad";
    return static_api_fft_r2c_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fft_r2c_grad";
    return eager_api_fft_r2c_grad(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_grad";
    return static_api_fill_diagonal_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_grad";
    return eager_api_fill_diagonal_grad(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_tensor_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_tensor_grad";
    return static_api_fill_diagonal_tensor_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_tensor_grad";
    return eager_api_fill_diagonal_tensor_grad(self, args, kwargs);
  }
}
static PyObject *fill_diagonal_tensor_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_diagonal_tensor_grad_";
    return static_api_fill_diagonal_tensor_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_diagonal_tensor_grad_";
    return eager_api_fill_diagonal_tensor_grad_(self, args, kwargs);
  }
}
static PyObject *fill_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_grad";
    return static_api_fill_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_grad";
    return eager_api_fill_grad(self, args, kwargs);
  }
}
static PyObject *fill_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fill_grad_";
    return static_api_fill_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fill_grad_";
    return eager_api_fill_grad_(self, args, kwargs);
  }
}
static PyObject *flash_attn_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_grad";
    return static_api_flash_attn_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_grad";
    return eager_api_flash_attn_grad(self, args, kwargs);
  }
}
static PyObject *flash_attn_qkvpacked_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_qkvpacked_grad";
    return static_api_flash_attn_qkvpacked_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_qkvpacked_grad";
    return eager_api_flash_attn_qkvpacked_grad(self, args, kwargs);
  }
}
static PyObject *flash_attn_unpadded_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_unpadded_grad";
    return static_api_flash_attn_unpadded_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_unpadded_grad";
    return eager_api_flash_attn_unpadded_grad(self, args, kwargs);
  }
}
static PyObject *flash_attn_v3_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_v3_grad";
    return static_api_flash_attn_v3_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_v3_grad";
    return eager_api_flash_attn_v3_grad(self, args, kwargs);
  }
}
static PyObject *flash_attn_varlen_qkvpacked_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flash_attn_varlen_qkvpacked_grad";
    return static_api_flash_attn_varlen_qkvpacked_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flash_attn_varlen_qkvpacked_grad";
    return eager_api_flash_attn_varlen_qkvpacked_grad(self, args, kwargs);
  }
}
static PyObject *flashmask_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flashmask_attention_grad";
    return static_api_flashmask_attention_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flashmask_attention_grad";
    return eager_api_flashmask_attention_grad(self, args, kwargs);
  }
}
static PyObject *flatten_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flatten_grad";
    return static_api_flatten_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flatten_grad";
    return eager_api_flatten_grad(self, args, kwargs);
  }
}
static PyObject *flatten_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_flatten_grad_";
    return static_api_flatten_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_flatten_grad_";
    return eager_api_flatten_grad_(self, args, kwargs);
  }
}
static PyObject *floor_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor_grad";
    return static_api_floor_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor_grad";
    return eager_api_floor_grad(self, args, kwargs);
  }
}
static PyObject *floor_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_floor_grad_";
    return static_api_floor_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_floor_grad_";
    return eager_api_floor_grad_(self, args, kwargs);
  }
}
static PyObject *fmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fmax_grad";
    return static_api_fmax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fmax_grad";
    return eager_api_fmax_grad(self, args, kwargs);
  }
}
static PyObject *fmin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fmin_grad";
    return static_api_fmin_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fmin_grad";
    return eager_api_fmin_grad(self, args, kwargs);
  }
}
static PyObject *fold_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fold_grad";
    return static_api_fold_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fold_grad";
    return eager_api_fold_grad(self, args, kwargs);
  }
}
static PyObject *fractional_max_pool2d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fractional_max_pool2d_grad";
    return static_api_fractional_max_pool2d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fractional_max_pool2d_grad";
    return eager_api_fractional_max_pool2d_grad(self, args, kwargs);
  }
}
static PyObject *fractional_max_pool3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fractional_max_pool3d_grad";
    return static_api_fractional_max_pool3d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fractional_max_pool3d_grad";
    return eager_api_fractional_max_pool3d_grad(self, args, kwargs);
  }
}
static PyObject *frame_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_frame_grad";
    return static_api_frame_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_frame_grad";
    return eager_api_frame_grad(self, args, kwargs);
  }
}
static PyObject *frobenius_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_frobenius_norm_grad";
    return static_api_frobenius_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_frobenius_norm_grad";
    return eager_api_frobenius_norm_grad(self, args, kwargs);
  }
}
static PyObject *fused_batch_norm_act_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_batch_norm_act_grad";
    return static_api_fused_batch_norm_act_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_batch_norm_act_grad";
    return eager_api_fused_batch_norm_act_grad(self, args, kwargs);
  }
}
static PyObject *fused_bn_add_activation_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bn_add_activation_grad";
    return static_api_fused_bn_add_activation_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bn_add_activation_grad";
    return eager_api_fused_bn_add_activation_grad(self, args, kwargs);
  }
}
static PyObject *fused_softmax_mask_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_softmax_mask_grad";
    return static_api_fused_softmax_mask_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_softmax_mask_grad";
    return eager_api_fused_softmax_mask_grad(self, args, kwargs);
  }
}
static PyObject *fused_softmax_mask_upper_triangle_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_softmax_mask_upper_triangle_grad";
    return static_api_fused_softmax_mask_upper_triangle_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_softmax_mask_upper_triangle_grad";
    return eager_api_fused_softmax_mask_upper_triangle_grad(self, args, kwargs);
  }
}
static PyObject *gammaincc_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaincc_grad";
    return static_api_gammaincc_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaincc_grad";
    return eager_api_gammaincc_grad(self, args, kwargs);
  }
}
static PyObject *gammaln_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gammaln_grad";
    return static_api_gammaln_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gammaln_grad";
    return eager_api_gammaln_grad(self, args, kwargs);
  }
}
static PyObject *gather_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gather_grad";
    return static_api_gather_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gather_grad";
    return eager_api_gather_grad(self, args, kwargs);
  }
}
static PyObject *gather_nd_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gather_nd_grad";
    return static_api_gather_nd_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gather_nd_grad";
    return eager_api_gather_nd_grad(self, args, kwargs);
  }
}
static PyObject *gaussian_inplace_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian_inplace_grad";
    return static_api_gaussian_inplace_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian_inplace_grad";
    return eager_api_gaussian_inplace_grad(self, args, kwargs);
  }
}
static PyObject *gaussian_inplace_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gaussian_inplace_grad_";
    return static_api_gaussian_inplace_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gaussian_inplace_grad_";
    return eager_api_gaussian_inplace_grad_(self, args, kwargs);
  }
}
static PyObject *gelu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gelu_grad";
    return static_api_gelu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gelu_grad";
    return eager_api_gelu_grad(self, args, kwargs);
  }
}
static PyObject *grid_sample_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_grid_sample_grad";
    return static_api_grid_sample_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_grid_sample_grad";
    return eager_api_grid_sample_grad(self, args, kwargs);
  }
}
static PyObject *group_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_group_norm_grad";
    return static_api_group_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_group_norm_grad";
    return eager_api_group_norm_grad(self, args, kwargs);
  }
}
static PyObject *group_norm_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_group_norm_grad_";
    return static_api_group_norm_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_group_norm_grad_";
    return eager_api_group_norm_grad_(self, args, kwargs);
  }
}
static PyObject *gru_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gru_grad";
    return static_api_gru_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gru_grad";
    return eager_api_gru_grad(self, args, kwargs);
  }
}
static PyObject *gru_unit_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gru_unit_grad";
    return static_api_gru_unit_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gru_unit_grad";
    return eager_api_gru_unit_grad(self, args, kwargs);
  }
}
static PyObject *gumbel_softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_gumbel_softmax_grad";
    return static_api_gumbel_softmax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_gumbel_softmax_grad";
    return eager_api_gumbel_softmax_grad(self, args, kwargs);
  }
}
static PyObject *hardshrink_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardshrink_grad";
    return static_api_hardshrink_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardshrink_grad";
    return eager_api_hardshrink_grad(self, args, kwargs);
  }
}
static PyObject *hardshrink_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardshrink_grad_";
    return static_api_hardshrink_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardshrink_grad_";
    return eager_api_hardshrink_grad_(self, args, kwargs);
  }
}
static PyObject *hardsigmoid_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardsigmoid_grad";
    return static_api_hardsigmoid_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardsigmoid_grad";
    return eager_api_hardsigmoid_grad(self, args, kwargs);
  }
}
static PyObject *hardsigmoid_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardsigmoid_grad_";
    return static_api_hardsigmoid_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardsigmoid_grad_";
    return eager_api_hardsigmoid_grad_(self, args, kwargs);
  }
}
static PyObject *hardtanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardtanh_grad";
    return static_api_hardtanh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardtanh_grad";
    return eager_api_hardtanh_grad(self, args, kwargs);
  }
}
static PyObject *hardtanh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardtanh_grad_";
    return static_api_hardtanh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardtanh_grad_";
    return eager_api_hardtanh_grad_(self, args, kwargs);
  }
}
static PyObject *heaviside_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_heaviside_grad";
    return static_api_heaviside_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_heaviside_grad";
    return eager_api_heaviside_grad(self, args, kwargs);
  }
}
static PyObject *hinge_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hinge_loss_grad";
    return static_api_hinge_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hinge_loss_grad";
    return eager_api_hinge_loss_grad(self, args, kwargs);
  }
}
static PyObject *hsigmoid_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hsigmoid_loss_grad";
    return static_api_hsigmoid_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hsigmoid_loss_grad";
    return eager_api_hsigmoid_loss_grad(self, args, kwargs);
  }
}
static PyObject *huber_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_huber_loss_grad";
    return static_api_huber_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_huber_loss_grad";
    return eager_api_huber_loss_grad(self, args, kwargs);
  }
}
static PyObject *i0_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i0_grad";
    return static_api_i0_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i0_grad";
    return eager_api_i0_grad(self, args, kwargs);
  }
}
static PyObject *i0e_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i0e_grad";
    return static_api_i0e_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i0e_grad";
    return eager_api_i0e_grad(self, args, kwargs);
  }
}
static PyObject *i1_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i1_grad";
    return static_api_i1_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i1_grad";
    return eager_api_i1_grad(self, args, kwargs);
  }
}
static PyObject *i1e_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_i1e_grad";
    return static_api_i1e_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_i1e_grad";
    return eager_api_i1e_grad(self, args, kwargs);
  }
}
static PyObject *identity_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_identity_loss_grad";
    return static_api_identity_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_identity_loss_grad";
    return eager_api_identity_loss_grad(self, args, kwargs);
  }
}
static PyObject *identity_loss_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_identity_loss_grad_";
    return static_api_identity_loss_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_identity_loss_grad_";
    return eager_api_identity_loss_grad_(self, args, kwargs);
  }
}
static PyObject *imag_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_imag_grad";
    return static_api_imag_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_imag_grad";
    return eager_api_imag_grad(self, args, kwargs);
  }
}
static PyObject *index_add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_add_grad";
    return static_api_index_add_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_add_grad";
    return eager_api_index_add_grad(self, args, kwargs);
  }
}
static PyObject *index_add_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_add_grad_";
    return static_api_index_add_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_add_grad_";
    return eager_api_index_add_grad_(self, args, kwargs);
  }
}
static PyObject *index_put_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_put_grad";
    return static_api_index_put_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_put_grad";
    return eager_api_index_put_grad(self, args, kwargs);
  }
}
static PyObject *index_sample_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_sample_grad";
    return static_api_index_sample_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_sample_grad";
    return eager_api_index_sample_grad(self, args, kwargs);
  }
}
static PyObject *index_select_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_select_grad";
    return static_api_index_select_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_select_grad";
    return eager_api_index_select_grad(self, args, kwargs);
  }
}
static PyObject *index_select_strided_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_index_select_strided_grad";
    return static_api_index_select_strided_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_index_select_strided_grad";
    return eager_api_index_select_strided_grad(self, args, kwargs);
  }
}
static PyObject *instance_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_instance_norm_grad";
    return static_api_instance_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_instance_norm_grad";
    return eager_api_instance_norm_grad(self, args, kwargs);
  }
}
static PyObject *inverse_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_inverse_grad";
    return static_api_inverse_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_inverse_grad";
    return eager_api_inverse_grad(self, args, kwargs);
  }
}
static PyObject *kldiv_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kldiv_loss_grad";
    return static_api_kldiv_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kldiv_loss_grad";
    return eager_api_kldiv_loss_grad(self, args, kwargs);
  }
}
static PyObject *kron_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kron_grad";
    return static_api_kron_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kron_grad";
    return eager_api_kron_grad(self, args, kwargs);
  }
}
static PyObject *kthvalue_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_kthvalue_grad";
    return static_api_kthvalue_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_kthvalue_grad";
    return eager_api_kthvalue_grad(self, args, kwargs);
  }
}
static PyObject *l1_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_l1_norm_grad";
    return static_api_l1_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_l1_norm_grad";
    return eager_api_l1_norm_grad(self, args, kwargs);
  }
}
static PyObject *label_smooth_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_label_smooth_grad";
    return static_api_label_smooth_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_label_smooth_grad";
    return eager_api_label_smooth_grad(self, args, kwargs);
  }
}
static PyObject *layer_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_layer_norm_grad";
    return static_api_layer_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_layer_norm_grad";
    return eager_api_layer_norm_grad(self, args, kwargs);
  }
}
static PyObject *leaky_relu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu_grad";
    return static_api_leaky_relu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu_grad";
    return eager_api_leaky_relu_grad(self, args, kwargs);
  }
}
static PyObject *leaky_relu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu_grad_";
    return static_api_leaky_relu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu_grad_";
    return eager_api_leaky_relu_grad_(self, args, kwargs);
  }
}
static PyObject *lerp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lerp_grad";
    return static_api_lerp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lerp_grad";
    return eager_api_lerp_grad(self, args, kwargs);
  }
}
static PyObject *lgamma_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lgamma_grad";
    return static_api_lgamma_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lgamma_grad";
    return eager_api_lgamma_grad(self, args, kwargs);
  }
}
static PyObject *linear_interp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_linear_interp_grad";
    return static_api_linear_interp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_linear_interp_grad";
    return eager_api_linear_interp_grad(self, args, kwargs);
  }
}
static PyObject *log10_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log10_grad";
    return static_api_log10_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log10_grad";
    return eager_api_log10_grad(self, args, kwargs);
  }
}
static PyObject *log10_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log10_grad_";
    return static_api_log10_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log10_grad_";
    return eager_api_log10_grad_(self, args, kwargs);
  }
}
static PyObject *log1p_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p_grad";
    return static_api_log1p_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p_grad";
    return eager_api_log1p_grad(self, args, kwargs);
  }
}
static PyObject *log1p_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p_grad_";
    return static_api_log1p_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p_grad_";
    return eager_api_log1p_grad_(self, args, kwargs);
  }
}
static PyObject *log2_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log2_grad";
    return static_api_log2_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log2_grad";
    return eager_api_log2_grad(self, args, kwargs);
  }
}
static PyObject *log2_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log2_grad_";
    return static_api_log2_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log2_grad_";
    return eager_api_log2_grad_(self, args, kwargs);
  }
}
static PyObject *log_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_grad";
    return static_api_log_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_grad";
    return eager_api_log_grad(self, args, kwargs);
  }
}
static PyObject *log_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_grad_";
    return static_api_log_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_grad_";
    return eager_api_log_grad_(self, args, kwargs);
  }
}
static PyObject *log_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_loss_grad";
    return static_api_log_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_loss_grad";
    return eager_api_log_loss_grad(self, args, kwargs);
  }
}
static PyObject *log_softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log_softmax_grad";
    return static_api_log_softmax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log_softmax_grad";
    return eager_api_log_softmax_grad(self, args, kwargs);
  }
}
static PyObject *logcumsumexp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logcumsumexp_grad";
    return static_api_logcumsumexp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logcumsumexp_grad";
    return eager_api_logcumsumexp_grad(self, args, kwargs);
  }
}
static PyObject *logit_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logit_grad";
    return static_api_logit_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logit_grad";
    return eager_api_logit_grad(self, args, kwargs);
  }
}
static PyObject *logsigmoid_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logsigmoid_grad";
    return static_api_logsigmoid_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logsigmoid_grad";
    return eager_api_logsigmoid_grad(self, args, kwargs);
  }
}
static PyObject *logsigmoid_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logsigmoid_grad_";
    return static_api_logsigmoid_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logsigmoid_grad_";
    return eager_api_logsigmoid_grad_(self, args, kwargs);
  }
}
static PyObject *logsumexp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_logsumexp_grad";
    return static_api_logsumexp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_logsumexp_grad";
    return eager_api_logsumexp_grad(self, args, kwargs);
  }
}
static PyObject *lp_pool2d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lp_pool2d_grad";
    return static_api_lp_pool2d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lp_pool2d_grad";
    return eager_api_lp_pool2d_grad(self, args, kwargs);
  }
}
static PyObject *lstm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lstm_grad";
    return static_api_lstm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lstm_grad";
    return eager_api_lstm_grad(self, args, kwargs);
  }
}
static PyObject *lu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_grad";
    return static_api_lu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_grad";
    return eager_api_lu_grad(self, args, kwargs);
  }
}
static PyObject *lu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_grad_";
    return static_api_lu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_grad_";
    return eager_api_lu_grad_(self, args, kwargs);
  }
}
static PyObject *lu_solve_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_solve_grad";
    return static_api_lu_solve_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_solve_grad";
    return eager_api_lu_solve_grad(self, args, kwargs);
  }
}
static PyObject *lu_unpack_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_lu_unpack_grad";
    return static_api_lu_unpack_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_lu_unpack_grad";
    return eager_api_lu_unpack_grad(self, args, kwargs);
  }
}
static PyObject *margin_cross_entropy_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_margin_cross_entropy_grad";
    return static_api_margin_cross_entropy_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_margin_cross_entropy_grad";
    return eager_api_margin_cross_entropy_grad(self, args, kwargs);
  }
}
static PyObject *margin_cross_entropy_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_margin_cross_entropy_grad_";
    return static_api_margin_cross_entropy_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_margin_cross_entropy_grad_";
    return eager_api_margin_cross_entropy_grad_(self, args, kwargs);
  }
}
static PyObject *masked_select_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_masked_select_grad";
    return static_api_masked_select_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_masked_select_grad";
    return eager_api_masked_select_grad(self, args, kwargs);
  }
}
static PyObject *matrix_power_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matrix_power_grad";
    return static_api_matrix_power_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matrix_power_grad";
    return eager_api_matrix_power_grad(self, args, kwargs);
  }
}
static PyObject *max_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max_grad";
    return static_api_max_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max_grad";
    return eager_api_max_grad(self, args, kwargs);
  }
}
static PyObject *max_pool2d_with_index_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max_pool2d_with_index_grad";
    return static_api_max_pool2d_with_index_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max_pool2d_with_index_grad";
    return eager_api_max_pool2d_with_index_grad(self, args, kwargs);
  }
}
static PyObject *max_pool3d_with_index_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_max_pool3d_with_index_grad";
    return static_api_max_pool3d_with_index_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_max_pool3d_with_index_grad";
    return eager_api_max_pool3d_with_index_grad(self, args, kwargs);
  }
}
static PyObject *maxout_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maxout_grad";
    return static_api_maxout_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maxout_grad";
    return eager_api_maxout_grad(self, args, kwargs);
  }
}
static PyObject *mean_all_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mean_all_grad";
    return static_api_mean_all_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mean_all_grad";
    return eager_api_mean_all_grad(self, args, kwargs);
  }
}
static PyObject *mean_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mean_grad";
    return static_api_mean_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mean_grad";
    return eager_api_mean_grad(self, args, kwargs);
  }
}
static PyObject *memory_efficient_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_memory_efficient_attention_grad";
    return static_api_memory_efficient_attention_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_memory_efficient_attention_grad";
    return eager_api_memory_efficient_attention_grad(self, args, kwargs);
  }
}
static PyObject *meshgrid_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_meshgrid_grad";
    return static_api_meshgrid_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_meshgrid_grad";
    return eager_api_meshgrid_grad(self, args, kwargs);
  }
}
static PyObject *mish_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mish_grad";
    return static_api_mish_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mish_grad";
    return eager_api_mish_grad(self, args, kwargs);
  }
}
static PyObject *mish_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mish_grad_";
    return static_api_mish_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mish_grad_";
    return eager_api_mish_grad_(self, args, kwargs);
  }
}
static PyObject *mode_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mode_grad";
    return static_api_mode_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mode_grad";
    return eager_api_mode_grad(self, args, kwargs);
  }
}
static PyObject *multi_dot_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multi_dot_grad";
    return static_api_multi_dot_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multi_dot_grad";
    return eager_api_multi_dot_grad(self, args, kwargs);
  }
}
static PyObject *multiplex_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiplex_grad";
    return static_api_multiplex_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiplex_grad";
    return eager_api_multiplex_grad(self, args, kwargs);
  }
}
static PyObject *mv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mv_grad";
    return static_api_mv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mv_grad";
    return eager_api_mv_grad(self, args, kwargs);
  }
}
static PyObject *nanmedian_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nanmedian_grad";
    return static_api_nanmedian_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nanmedian_grad";
    return eager_api_nanmedian_grad(self, args, kwargs);
  }
}
static PyObject *nearest_interp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nearest_interp_grad";
    return static_api_nearest_interp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nearest_interp_grad";
    return eager_api_nearest_interp_grad(self, args, kwargs);
  }
}
static PyObject *nll_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_nll_loss_grad";
    return static_api_nll_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_nll_loss_grad";
    return eager_api_nll_loss_grad(self, args, kwargs);
  }
}
static PyObject *norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_norm_grad";
    return static_api_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_norm_grad";
    return eager_api_norm_grad(self, args, kwargs);
  }
}
static PyObject *overlap_add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_overlap_add_grad";
    return static_api_overlap_add_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_overlap_add_grad";
    return eager_api_overlap_add_grad(self, args, kwargs);
  }
}
static PyObject *p_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_p_norm_grad";
    return static_api_p_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_p_norm_grad";
    return eager_api_p_norm_grad(self, args, kwargs);
  }
}
static PyObject *pad3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pad3d_grad";
    return static_api_pad3d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pad3d_grad";
    return eager_api_pad3d_grad(self, args, kwargs);
  }
}
static PyObject *pad_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pad_grad";
    return static_api_pad_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pad_grad";
    return eager_api_pad_grad(self, args, kwargs);
  }
}
static PyObject *pixel_shuffle_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pixel_shuffle_grad";
    return static_api_pixel_shuffle_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pixel_shuffle_grad";
    return eager_api_pixel_shuffle_grad(self, args, kwargs);
  }
}
static PyObject *pixel_unshuffle_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pixel_unshuffle_grad";
    return static_api_pixel_unshuffle_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pixel_unshuffle_grad";
    return eager_api_pixel_unshuffle_grad(self, args, kwargs);
  }
}
static PyObject *poisson_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_poisson_grad";
    return static_api_poisson_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_poisson_grad";
    return eager_api_poisson_grad(self, args, kwargs);
  }
}
static PyObject *polygamma_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_polygamma_grad";
    return static_api_polygamma_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_polygamma_grad";
    return eager_api_polygamma_grad(self, args, kwargs);
  }
}
static PyObject *pool2d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pool2d_grad";
    return static_api_pool2d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pool2d_grad";
    return eager_api_pool2d_grad(self, args, kwargs);
  }
}
static PyObject *pool3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pool3d_grad";
    return static_api_pool3d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pool3d_grad";
    return eager_api_pool3d_grad(self, args, kwargs);
  }
}
static PyObject *pow_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow_grad";
    return static_api_pow_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow_grad";
    return eager_api_pow_grad(self, args, kwargs);
  }
}
static PyObject *pow_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow_grad_";
    return static_api_pow_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow_grad_";
    return eager_api_pow_grad_(self, args, kwargs);
  }
}
static PyObject *prelu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prelu_grad";
    return static_api_prelu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prelu_grad";
    return eager_api_prelu_grad(self, args, kwargs);
  }
}
static PyObject *prod_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_prod_grad";
    return static_api_prod_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_prod_grad";
    return eager_api_prod_grad(self, args, kwargs);
  }
}
static PyObject *psroi_pool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_psroi_pool_grad";
    return static_api_psroi_pool_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_psroi_pool_grad";
    return eager_api_psroi_pool_grad(self, args, kwargs);
  }
}
static PyObject *put_along_axis_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_put_along_axis_grad";
    return static_api_put_along_axis_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_put_along_axis_grad";
    return eager_api_put_along_axis_grad(self, args, kwargs);
  }
}
static PyObject *qr_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_qr_grad";
    return static_api_qr_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_qr_grad";
    return eager_api_qr_grad(self, args, kwargs);
  }
}
static PyObject *rank_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rank_attention_grad";
    return static_api_rank_attention_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rank_attention_grad";
    return eager_api_rank_attention_grad(self, args, kwargs);
  }
}
static PyObject *real_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_real_grad";
    return static_api_real_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_real_grad";
    return eager_api_real_grad(self, args, kwargs);
  }
}
static PyObject *reciprocal_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reciprocal_grad";
    return static_api_reciprocal_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reciprocal_grad";
    return eager_api_reciprocal_grad(self, args, kwargs);
  }
}
static PyObject *reciprocal_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reciprocal_grad_";
    return static_api_reciprocal_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reciprocal_grad_";
    return eager_api_reciprocal_grad_(self, args, kwargs);
  }
}
static PyObject *reduce_as_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reduce_as_grad";
    return static_api_reduce_as_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reduce_as_grad";
    return eager_api_reduce_as_grad(self, args, kwargs);
  }
}
static PyObject *relu6_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu6_grad";
    return static_api_relu6_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu6_grad";
    return eager_api_relu6_grad(self, args, kwargs);
  }
}
static PyObject *relu6_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu6_grad_";
    return static_api_relu6_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu6_grad_";
    return eager_api_relu6_grad_(self, args, kwargs);
  }
}
static PyObject *relu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu_grad";
    return static_api_relu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu_grad";
    return eager_api_relu_grad(self, args, kwargs);
  }
}
static PyObject *relu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu_grad_";
    return static_api_relu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu_grad_";
    return eager_api_relu_grad_(self, args, kwargs);
  }
}
static PyObject *renorm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_renorm_grad";
    return static_api_renorm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_renorm_grad";
    return eager_api_renorm_grad(self, args, kwargs);
  }
}
static PyObject *repeat_interleave_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_repeat_interleave_grad";
    return static_api_repeat_interleave_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_repeat_interleave_grad";
    return eager_api_repeat_interleave_grad(self, args, kwargs);
  }
}
static PyObject *repeat_interleave_with_tensor_index_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_repeat_interleave_with_tensor_index_grad";
    return static_api_repeat_interleave_with_tensor_index_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_repeat_interleave_with_tensor_index_grad";
    return eager_api_repeat_interleave_with_tensor_index_grad(self, args, kwargs);
  }
}
static PyObject *reshape_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape_grad";
    return static_api_reshape_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape_grad";
    return eager_api_reshape_grad(self, args, kwargs);
  }
}
static PyObject *reshape_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape_grad_";
    return static_api_reshape_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape_grad_";
    return eager_api_reshape_grad_(self, args, kwargs);
  }
}
static PyObject *rms_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rms_norm_grad";
    return static_api_rms_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rms_norm_grad";
    return eager_api_rms_norm_grad(self, args, kwargs);
  }
}
static PyObject *rnn_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rnn_grad";
    return static_api_rnn_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rnn_grad";
    return eager_api_rnn_grad(self, args, kwargs);
  }
}
static PyObject *roi_align_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roi_align_grad";
    return static_api_roi_align_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roi_align_grad";
    return eager_api_roi_align_grad(self, args, kwargs);
  }
}
static PyObject *roi_pool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roi_pool_grad";
    return static_api_roi_pool_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roi_pool_grad";
    return eager_api_roi_pool_grad(self, args, kwargs);
  }
}
static PyObject *roll_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_roll_grad";
    return static_api_roll_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_roll_grad";
    return eager_api_roll_grad(self, args, kwargs);
  }
}
static PyObject *round_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_round_grad";
    return static_api_round_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_round_grad";
    return eager_api_round_grad(self, args, kwargs);
  }
}
static PyObject *round_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_round_grad_";
    return static_api_round_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_round_grad_";
    return eager_api_round_grad_(self, args, kwargs);
  }
}
static PyObject *rrelu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rrelu_grad";
    return static_api_rrelu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rrelu_grad";
    return eager_api_rrelu_grad(self, args, kwargs);
  }
}
static PyObject *rsqrt_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rsqrt_grad";
    return static_api_rsqrt_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rsqrt_grad";
    return eager_api_rsqrt_grad(self, args, kwargs);
  }
}
static PyObject *rsqrt_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_rsqrt_grad_";
    return static_api_rsqrt_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_rsqrt_grad_";
    return eager_api_rsqrt_grad_(self, args, kwargs);
  }
}
static PyObject *scatter_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scatter_grad";
    return static_api_scatter_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scatter_grad";
    return eager_api_scatter_grad(self, args, kwargs);
  }
}
static PyObject *scatter_nd_add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scatter_nd_add_grad";
    return static_api_scatter_nd_add_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scatter_nd_add_grad";
    return eager_api_scatter_nd_add_grad(self, args, kwargs);
  }
}
static PyObject *segment_pool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_segment_pool_grad";
    return static_api_segment_pool_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_segment_pool_grad";
    return eager_api_segment_pool_grad(self, args, kwargs);
  }
}
static PyObject *selu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_selu_grad";
    return static_api_selu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_selu_grad";
    return eager_api_selu_grad(self, args, kwargs);
  }
}
static PyObject *send_u_recv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_u_recv_grad";
    return static_api_send_u_recv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_u_recv_grad";
    return eager_api_send_u_recv_grad(self, args, kwargs);
  }
}
static PyObject *send_ue_recv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_ue_recv_grad";
    return static_api_send_ue_recv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_ue_recv_grad";
    return eager_api_send_ue_recv_grad(self, args, kwargs);
  }
}
static PyObject *send_uv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_send_uv_grad";
    return static_api_send_uv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_send_uv_grad";
    return eager_api_send_uv_grad(self, args, kwargs);
  }
}
static PyObject *sequence_conv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sequence_conv_grad";
    return static_api_sequence_conv_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sequence_conv_grad";
    return eager_api_sequence_conv_grad(self, args, kwargs);
  }
}
static PyObject *sequence_pool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sequence_pool_grad";
    return static_api_sequence_pool_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sequence_pool_grad";
    return eager_api_sequence_pool_grad(self, args, kwargs);
  }
}
static PyObject *set_value_with_tensor_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value_with_tensor_grad";
    return static_api_set_value_with_tensor_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value_with_tensor_grad";
    return eager_api_set_value_with_tensor_grad(self, args, kwargs);
  }
}
static PyObject *shuffle_channel_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shuffle_channel_grad";
    return static_api_shuffle_channel_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shuffle_channel_grad";
    return eager_api_shuffle_channel_grad(self, args, kwargs);
  }
}
static PyObject *sigmoid_cross_entropy_with_logits_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_cross_entropy_with_logits_grad";
    return static_api_sigmoid_cross_entropy_with_logits_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_cross_entropy_with_logits_grad";
    return eager_api_sigmoid_cross_entropy_with_logits_grad(self, args, kwargs);
  }
}
static PyObject *sigmoid_cross_entropy_with_logits_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_cross_entropy_with_logits_grad_";
    return static_api_sigmoid_cross_entropy_with_logits_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_cross_entropy_with_logits_grad_";
    return eager_api_sigmoid_cross_entropy_with_logits_grad_(self, args, kwargs);
  }
}
static PyObject *sigmoid_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_grad";
    return static_api_sigmoid_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_grad";
    return eager_api_sigmoid_grad(self, args, kwargs);
  }
}
static PyObject *sigmoid_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sigmoid_grad_";
    return static_api_sigmoid_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sigmoid_grad_";
    return eager_api_sigmoid_grad_(self, args, kwargs);
  }
}
static PyObject *silu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_silu_grad";
    return static_api_silu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_silu_grad";
    return eager_api_silu_grad(self, args, kwargs);
  }
}
static PyObject *silu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_silu_grad_";
    return static_api_silu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_silu_grad_";
    return eager_api_silu_grad_(self, args, kwargs);
  }
}
static PyObject *sin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin_grad";
    return static_api_sin_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin_grad";
    return eager_api_sin_grad(self, args, kwargs);
  }
}
static PyObject *sin_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin_grad_";
    return static_api_sin_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin_grad_";
    return eager_api_sin_grad_(self, args, kwargs);
  }
}
static PyObject *sinh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh_grad";
    return static_api_sinh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh_grad";
    return eager_api_sinh_grad(self, args, kwargs);
  }
}
static PyObject *sinh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh_grad_";
    return static_api_sinh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh_grad_";
    return eager_api_sinh_grad_(self, args, kwargs);
  }
}
static PyObject *slice_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slice_grad";
    return static_api_slice_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slice_grad";
    return eager_api_slice_grad(self, args, kwargs);
  }
}
static PyObject *slogdet_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slogdet_grad";
    return static_api_slogdet_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slogdet_grad";
    return eager_api_slogdet_grad(self, args, kwargs);
  }
}
static PyObject *softplus_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softplus_grad";
    return static_api_softplus_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softplus_grad";
    return eager_api_softplus_grad(self, args, kwargs);
  }
}
static PyObject *softplus_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softplus_grad_";
    return static_api_softplus_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softplus_grad_";
    return eager_api_softplus_grad_(self, args, kwargs);
  }
}
static PyObject *softshrink_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softshrink_grad";
    return static_api_softshrink_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softshrink_grad";
    return eager_api_softshrink_grad(self, args, kwargs);
  }
}
static PyObject *softshrink_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softshrink_grad_";
    return static_api_softshrink_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softshrink_grad_";
    return eager_api_softshrink_grad_(self, args, kwargs);
  }
}
static PyObject *softsign_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softsign_grad";
    return static_api_softsign_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softsign_grad";
    return eager_api_softsign_grad(self, args, kwargs);
  }
}
static PyObject *softsign_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softsign_grad_";
    return static_api_softsign_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softsign_grad_";
    return eager_api_softsign_grad_(self, args, kwargs);
  }
}
static PyObject *solve_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_solve_grad";
    return static_api_solve_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_solve_grad";
    return eager_api_solve_grad(self, args, kwargs);
  }
}
static PyObject *spectral_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_spectral_norm_grad";
    return static_api_spectral_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_spectral_norm_grad";
    return eager_api_spectral_norm_grad(self, args, kwargs);
  }
}
static PyObject *split_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_split_grad";
    return static_api_split_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_split_grad";
    return eager_api_split_grad(self, args, kwargs);
  }
}
static PyObject *sqrt_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt_grad";
    return static_api_sqrt_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt_grad";
    return eager_api_sqrt_grad(self, args, kwargs);
  }
}
static PyObject *sqrt_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt_grad_";
    return static_api_sqrt_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt_grad_";
    return eager_api_sqrt_grad_(self, args, kwargs);
  }
}
static PyObject *square_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square_grad";
    return static_api_square_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square_grad";
    return eager_api_square_grad(self, args, kwargs);
  }
}
static PyObject *square_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square_grad_";
    return static_api_square_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square_grad_";
    return eager_api_square_grad_(self, args, kwargs);
  }
}
static PyObject *squared_l2_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squared_l2_norm_grad";
    return static_api_squared_l2_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squared_l2_norm_grad";
    return eager_api_squared_l2_norm_grad(self, args, kwargs);
  }
}
static PyObject *squeeze_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squeeze_grad";
    return static_api_squeeze_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squeeze_grad";
    return eager_api_squeeze_grad(self, args, kwargs);
  }
}
static PyObject *squeeze_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_squeeze_grad_";
    return static_api_squeeze_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_squeeze_grad_";
    return eager_api_squeeze_grad_(self, args, kwargs);
  }
}
static PyObject *stack_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stack_grad";
    return static_api_stack_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stack_grad";
    return eager_api_stack_grad(self, args, kwargs);
  }
}
static PyObject *stanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stanh_grad";
    return static_api_stanh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stanh_grad";
    return eager_api_stanh_grad(self, args, kwargs);
  }
}
static PyObject *strided_slice_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_strided_slice_grad";
    return static_api_strided_slice_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_strided_slice_grad";
    return eager_api_strided_slice_grad(self, args, kwargs);
  }
}
static PyObject *sum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sum_grad";
    return static_api_sum_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sum_grad";
    return eager_api_sum_grad(self, args, kwargs);
  }
}
static PyObject *svd_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_svd_grad";
    return static_api_svd_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_svd_grad";
    return eager_api_svd_grad(self, args, kwargs);
  }
}
static PyObject *svdvals_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_svdvals_grad";
    return static_api_svdvals_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_svdvals_grad";
    return eager_api_svdvals_grad(self, args, kwargs);
  }
}
static PyObject *swiglu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_swiglu_grad";
    return static_api_swiglu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_swiglu_grad";
    return eager_api_swiglu_grad(self, args, kwargs);
  }
}
static PyObject *swish_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_swish_grad";
    return static_api_swish_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_swish_grad";
    return eager_api_swish_grad(self, args, kwargs);
  }
}
static PyObject *swish_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_swish_grad_";
    return static_api_swish_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_swish_grad_";
    return eager_api_swish_grad_(self, args, kwargs);
  }
}
static PyObject *sync_batch_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_batch_norm_grad";
    return static_api_sync_batch_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_batch_norm_grad";
    return eager_api_sync_batch_norm_grad(self, args, kwargs);
  }
}
static PyObject *take_along_axis_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_take_along_axis_grad";
    return static_api_take_along_axis_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_take_along_axis_grad";
    return eager_api_take_along_axis_grad(self, args, kwargs);
  }
}
static PyObject *tan_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan_grad";
    return static_api_tan_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan_grad";
    return eager_api_tan_grad(self, args, kwargs);
  }
}
static PyObject *tan_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan_grad_";
    return static_api_tan_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan_grad_";
    return eager_api_tan_grad_(self, args, kwargs);
  }
}
static PyObject *tanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_grad";
    return static_api_tanh_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_grad";
    return eager_api_tanh_grad(self, args, kwargs);
  }
}
static PyObject *tanh_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_grad_";
    return static_api_tanh_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_grad_";
    return eager_api_tanh_grad_(self, args, kwargs);
  }
}
static PyObject *tanh_shrink_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_shrink_grad";
    return static_api_tanh_shrink_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_shrink_grad";
    return eager_api_tanh_shrink_grad(self, args, kwargs);
  }
}
static PyObject *tanh_shrink_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_shrink_grad_";
    return static_api_tanh_shrink_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_shrink_grad_";
    return eager_api_tanh_shrink_grad_(self, args, kwargs);
  }
}
static PyObject *temporal_shift_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_temporal_shift_grad";
    return static_api_temporal_shift_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_temporal_shift_grad";
    return eager_api_temporal_shift_grad(self, args, kwargs);
  }
}
static PyObject *thresholded_relu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_thresholded_relu_grad";
    return static_api_thresholded_relu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_thresholded_relu_grad";
    return eager_api_thresholded_relu_grad(self, args, kwargs);
  }
}
static PyObject *thresholded_relu_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_thresholded_relu_grad_";
    return static_api_thresholded_relu_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_thresholded_relu_grad_";
    return eager_api_thresholded_relu_grad_(self, args, kwargs);
  }
}
static PyObject *topk_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_topk_grad";
    return static_api_topk_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_topk_grad";
    return eager_api_topk_grad(self, args, kwargs);
  }
}
static PyObject *trace_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trace_grad";
    return static_api_trace_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trace_grad";
    return eager_api_trace_grad(self, args, kwargs);
  }
}
static PyObject *trans_layout_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trans_layout_grad";
    return static_api_trans_layout_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trans_layout_grad";
    return eager_api_trans_layout_grad(self, args, kwargs);
  }
}
static PyObject *transpose_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_transpose_grad";
    return static_api_transpose_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_transpose_grad";
    return eager_api_transpose_grad(self, args, kwargs);
  }
}
static PyObject *triangular_solve_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triangular_solve_grad";
    return static_api_triangular_solve_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triangular_solve_grad";
    return eager_api_triangular_solve_grad(self, args, kwargs);
  }
}
static PyObject *tril_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tril_grad";
    return static_api_tril_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tril_grad";
    return eager_api_tril_grad(self, args, kwargs);
  }
}
static PyObject *trilinear_interp_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trilinear_interp_grad";
    return static_api_trilinear_interp_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trilinear_interp_grad";
    return eager_api_trilinear_interp_grad(self, args, kwargs);
  }
}
static PyObject *triu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_triu_grad";
    return static_api_triu_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_triu_grad";
    return eager_api_triu_grad(self, args, kwargs);
  }
}
static PyObject *trunc_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_trunc_grad";
    return static_api_trunc_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_trunc_grad";
    return eager_api_trunc_grad(self, args, kwargs);
  }
}
static PyObject *unfold_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unfold_grad";
    return static_api_unfold_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unfold_grad";
    return eager_api_unfold_grad(self, args, kwargs);
  }
}
static PyObject *uniform_inplace_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform_inplace_grad";
    return static_api_uniform_inplace_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform_inplace_grad";
    return eager_api_uniform_inplace_grad(self, args, kwargs);
  }
}
static PyObject *uniform_inplace_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_uniform_inplace_grad_";
    return static_api_uniform_inplace_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_uniform_inplace_grad_";
    return eager_api_uniform_inplace_grad_(self, args, kwargs);
  }
}
static PyObject *unsqueeze_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unsqueeze_grad";
    return static_api_unsqueeze_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unsqueeze_grad";
    return eager_api_unsqueeze_grad(self, args, kwargs);
  }
}
static PyObject *unsqueeze_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unsqueeze_grad_";
    return static_api_unsqueeze_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unsqueeze_grad_";
    return eager_api_unsqueeze_grad_(self, args, kwargs);
  }
}
static PyObject *unstack_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unstack_grad";
    return static_api_unstack_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unstack_grad";
    return eager_api_unstack_grad(self, args, kwargs);
  }
}
static PyObject *view_dtype_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_view_dtype_grad";
    return static_api_view_dtype_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_view_dtype_grad";
    return eager_api_view_dtype_grad(self, args, kwargs);
  }
}
static PyObject *warpctc_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_warpctc_grad";
    return static_api_warpctc_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_warpctc_grad";
    return eager_api_warpctc_grad(self, args, kwargs);
  }
}
static PyObject *warprnnt_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_warprnnt_grad";
    return static_api_warprnnt_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_warprnnt_grad";
    return eager_api_warprnnt_grad(self, args, kwargs);
  }
}
static PyObject *weight_only_linear_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_weight_only_linear_grad";
    return static_api_weight_only_linear_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_weight_only_linear_grad";
    return eager_api_weight_only_linear_grad(self, args, kwargs);
  }
}
static PyObject *where_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_where_grad";
    return static_api_where_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_where_grad";
    return eager_api_where_grad(self, args, kwargs);
  }
}
static PyObject *yolo_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_yolo_loss_grad";
    return static_api_yolo_loss_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_yolo_loss_grad";
    return eager_api_yolo_loss_grad(self, args, kwargs);
  }
}
static PyObject *disable_check_model_nan_inf_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_disable_check_model_nan_inf_grad";
    return static_api_disable_check_model_nan_inf_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_disable_check_model_nan_inf_grad";
    return eager_api_disable_check_model_nan_inf_grad(self, args, kwargs);
  }
}
static PyObject *enable_check_model_nan_inf_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_enable_check_model_nan_inf_grad";
    return static_api_enable_check_model_nan_inf_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_enable_check_model_nan_inf_grad";
    return eager_api_enable_check_model_nan_inf_grad(self, args, kwargs);
  }
}
static PyObject *im2sequence_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_im2sequence_grad";
    return static_api_im2sequence_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_im2sequence_grad";
    return eager_api_im2sequence_grad(self, args, kwargs);
  }
}
static PyObject *pyramid_hash_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pyramid_hash_grad";
    return static_api_pyramid_hash_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pyramid_hash_grad";
    return eager_api_pyramid_hash_grad(self, args, kwargs);
  }
}
static PyObject *shuffle_batch_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_shuffle_batch_grad";
    return static_api_shuffle_batch_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_shuffle_batch_grad";
    return eager_api_shuffle_batch_grad(self, args, kwargs);
  }
}
static PyObject *sparse_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sparse_attention_grad";
    return static_api_sparse_attention_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sparse_attention_grad";
    return eager_api_sparse_attention_grad(self, args, kwargs);
  }
}
static PyObject *stft_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_stft_grad";
    return static_api_stft_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_stft_grad";
    return eager_api_stft_grad(self, args, kwargs);
  }
}
static PyObject *unpool3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unpool3d_grad";
    return static_api_unpool3d_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unpool3d_grad";
    return eager_api_unpool3d_grad(self, args, kwargs);
  }
}
static PyObject *unpool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_unpool_grad";
    return static_api_unpool_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_unpool_grad";
    return eager_api_unpool_grad(self, args, kwargs);
  }
}
static PyObject *fused_rms_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_rms_norm_grad";
    return static_api_fused_rms_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_rms_norm_grad";
    return eager_api_fused_rms_norm_grad(self, args, kwargs);
  }
}
static PyObject *blha_get_max_len(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_blha_get_max_len";
    return static_api_blha_get_max_len(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_blha_get_max_len";
    return eager_api_blha_get_max_len(self, args, kwargs);
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
static PyObject *block_multihead_attention_xpu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_block_multihead_attention_xpu_";
    return static_api_block_multihead_attention_xpu_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_block_multihead_attention_xpu_";
    return eager_api_block_multihead_attention_xpu_(self, args, kwargs);
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
static PyObject *fc(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fc";
  return static_api_fc(self, args, kwargs);
}
static PyObject *fp8_fp8_half_gemm_fused(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fp8_fp8_half_gemm_fused";
    return static_api_fp8_fp8_half_gemm_fused(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fp8_fp8_half_gemm_fused";
    return eager_api_fp8_fp8_half_gemm_fused(self, args, kwargs);
  }
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
static PyObject *fused_multi_transformer_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_multi_transformer_";
    return static_api_fused_multi_transformer_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_multi_transformer_";
    return eager_api_fused_multi_transformer_(self, args, kwargs);
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
static PyObject *fusion_seqpool_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fusion_seqpool_concat";
    return static_api_fusion_seqpool_concat(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fusion_seqpool_concat";
    return eager_api_fusion_seqpool_concat(self, args, kwargs);
  }
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
static PyObject *qkv_unpack_mha(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_qkv_unpack_mha";
  return static_api_qkv_unpack_mha(self, args, kwargs);
}
static PyObject *resnet_basic_block(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_resnet_basic_block";
    return static_api_resnet_basic_block(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_resnet_basic_block";
    return eager_api_resnet_basic_block(self, args, kwargs);
  }
}
static PyObject *resnet_unit(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_resnet_unit";
    return static_api_resnet_unit(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_resnet_unit";
    return eager_api_resnet_unit(self, args, kwargs);
  }
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
static PyObject *add_group_norm_silu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_group_norm_silu";
    return static_api_add_group_norm_silu(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_group_norm_silu";
    return eager_api_add_group_norm_silu(self, args, kwargs);
  }
}
static PyObject *fused_bias_dropout_residual_layer_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_bias_dropout_residual_layer_norm_grad";
    return static_api_fused_bias_dropout_residual_layer_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_bias_dropout_residual_layer_norm_grad";
    return eager_api_fused_bias_dropout_residual_layer_norm_grad(self, args, kwargs);
  }
}
static PyObject *fused_dropout_add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_dropout_add_grad";
    return static_api_fused_dropout_add_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_dropout_add_grad";
    return eager_api_fused_dropout_add_grad(self, args, kwargs);
  }
}
static PyObject *fused_rotary_position_embedding_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_rotary_position_embedding_grad";
    return static_api_fused_rotary_position_embedding_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_rotary_position_embedding_grad";
    return eager_api_fused_rotary_position_embedding_grad(self, args, kwargs);
  }
}
static PyObject *resnet_basic_block_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_resnet_basic_block_grad";
    return static_api_resnet_basic_block_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_resnet_basic_block_grad";
    return eager_api_resnet_basic_block_grad(self, args, kwargs);
  }
}
static PyObject *resnet_unit_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_resnet_unit_grad";
    return static_api_resnet_unit_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_resnet_unit_grad";
    return eager_api_resnet_unit_grad(self, args, kwargs);
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
static PyObject *assign_value(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_assign_value";
  return static_api_assign_value(self, args, kwargs);
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
static PyObject *beam_search_decode(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_beam_search_decode";
  return static_api_beam_search_decode(self, args, kwargs);
}
static PyObject *c_embedding(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_embedding";
    return static_api_c_embedding(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_embedding";
    return eager_api_c_embedding(self, args, kwargs);
  }
}
static PyObject *coalesce_tensor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_coalesce_tensor_";
  return static_api_coalesce_tensor_(self, args, kwargs);
}
static PyObject *dequantize_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_dequantize_linear";
  return static_api_dequantize_linear(self, args, kwargs);
}
static PyObject *dequantize_linear_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_dequantize_linear_";
  return static_api_dequantize_linear_(self, args, kwargs);
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
static PyObject *hash(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_hash";
  return static_api_hash(self, args, kwargs);
}
static PyObject *lars_momentum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_lars_momentum_";
  return static_api_lars_momentum_(self, args, kwargs);
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
static PyObject *maximum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maximum";
    return static_api_maximum(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maximum";
    return eager_api_maximum(self, args, kwargs);
  }
}
static PyObject *memcpy(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_memcpy";
  return static_api_memcpy(self, args, kwargs);
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
static PyObject *nop(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_nop";
  return static_api_nop(self, args, kwargs);
}
static PyObject *nop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_nop_";
  return static_api_nop_(self, args, kwargs);
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
static PyObject *print(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_print";
  return static_api_print(self, args, kwargs);
}
static PyObject *quantize_linear(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_quantize_linear";
  return static_api_quantize_linear(self, args, kwargs);
}
static PyObject *quantize_linear_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_quantize_linear_";
  return static_api_quantize_linear_(self, args, kwargs);
}
static PyObject *recv_v2(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_recv_v2";
  return static_api_recv_v2(self, args, kwargs);
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
static PyObject *send_v2(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_send_v2";
  return static_api_send_v2(self, args, kwargs);
}
static PyObject *sequence_expand(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_sequence_expand";
  return static_api_sequence_expand(self, args, kwargs);
}
static PyObject *sequence_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_sequence_softmax";
  return static_api_sequence_softmax(self, args, kwargs);
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
static PyObject *share_data_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_share_data_";
  return static_api_share_data_(self, args, kwargs);
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
static PyObject *tile(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tile";
    return static_api_tile(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tile";
    return eager_api_tile(self, args, kwargs);
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
static PyObject *c_softmax_with_multi_label_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_c_softmax_with_multi_label_cross_entropy";
  return static_api_c_softmax_with_multi_label_cross_entropy(self, args, kwargs);
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
static PyObject *onednn_to_paddle_layout(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_onednn_to_paddle_layout";
  return static_api_onednn_to_paddle_layout(self, args, kwargs);
}
static PyObject *add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_grad";
    return static_api_add_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_grad";
    return eager_api_add_grad(self, args, kwargs);
  }
}
static PyObject *add_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_grad_";
    return static_api_add_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_grad_";
    return eager_api_add_grad_(self, args, kwargs);
  }
}
static PyObject *assign_out__grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_out__grad";
    return static_api_assign_out__grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_out__grad";
    return eager_api_assign_out__grad(self, args, kwargs);
  }
}
static PyObject *assign_out__grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_assign_out__grad_";
    return static_api_assign_out__grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_assign_out__grad_";
    return eager_api_assign_out__grad_(self, args, kwargs);
  }
}
static PyObject *batch_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_batch_norm_grad";
    return static_api_batch_norm_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_batch_norm_grad";
    return eager_api_batch_norm_grad(self, args, kwargs);
  }
}
static PyObject *c_embedding_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_c_embedding_grad";
    return static_api_c_embedding_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_c_embedding_grad";
    return eager_api_c_embedding_grad(self, args, kwargs);
  }
}
static PyObject *c_softmax_with_multi_label_cross_entropy_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_c_softmax_with_multi_label_cross_entropy_grad";
  return static_api_c_softmax_with_multi_label_cross_entropy_grad(self, args, kwargs);
}
static PyObject *divide_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide_grad";
    return static_api_divide_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide_grad";
    return eager_api_divide_grad(self, args, kwargs);
  }
}
static PyObject *einsum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_einsum_grad";
    return static_api_einsum_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_einsum_grad";
    return eager_api_einsum_grad(self, args, kwargs);
  }
}
static PyObject *elementwise_pow_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_elementwise_pow_grad";
    return static_api_elementwise_pow_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_elementwise_pow_grad";
    return eager_api_elementwise_pow_grad(self, args, kwargs);
  }
}
static PyObject *embedding_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_embedding_grad";
    return static_api_embedding_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_embedding_grad";
    return eager_api_embedding_grad(self, args, kwargs);
  }
}
static PyObject *fused_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_attention_grad";
  return static_api_fused_attention_grad(self, args, kwargs);
}
static PyObject *fused_feedforward_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_fused_feedforward_grad";
  return static_api_fused_feedforward_grad(self, args, kwargs);
}
static PyObject *hardswish_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardswish_grad";
    return static_api_hardswish_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardswish_grad";
    return eager_api_hardswish_grad(self, args, kwargs);
  }
}
static PyObject *hardswish_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_hardswish_grad_";
    return static_api_hardswish_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_hardswish_grad_";
    return eager_api_hardswish_grad_(self, args, kwargs);
  }
}
static PyObject *lod_reset_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_lod_reset_grad_";
  return static_api_lod_reset_grad_(self, args, kwargs);
}
static PyObject *matmul_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matmul_grad";
    return static_api_matmul_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matmul_grad";
    return eager_api_matmul_grad(self, args, kwargs);
  }
}
static PyObject *matmul_with_flatten_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_matmul_with_flatten_grad";
  return static_api_matmul_with_flatten_grad(self, args, kwargs);
}
static PyObject *maximum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maximum_grad";
    return static_api_maximum_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maximum_grad";
    return eager_api_maximum_grad(self, args, kwargs);
  }
}
static PyObject *min_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_min_grad";
    return static_api_min_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_min_grad";
    return eager_api_min_grad(self, args, kwargs);
  }
}
static PyObject *minimum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_minimum_grad";
    return static_api_minimum_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_minimum_grad";
    return eager_api_minimum_grad(self, args, kwargs);
  }
}
static PyObject *remainder_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_remainder_grad";
    return static_api_remainder_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_remainder_grad";
    return eager_api_remainder_grad(self, args, kwargs);
  }
}
static PyObject *sequence_expand_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_sequence_expand_grad";
  return static_api_sequence_expand_grad(self, args, kwargs);
}
static PyObject *sequence_softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  VLOG(6) << "Call static_api_sequence_softmax_grad";
  return static_api_sequence_softmax_grad(self, args, kwargs);
}
static PyObject *set_value_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_set_value_grad";
    return static_api_set_value_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_set_value_grad";
    return eager_api_set_value_grad(self, args, kwargs);
  }
}
static PyObject *softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softmax_grad";
    return static_api_softmax_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softmax_grad";
    return eager_api_softmax_grad(self, args, kwargs);
  }
}
static PyObject *subtract_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract_grad";
    return static_api_subtract_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract_grad";
    return eager_api_subtract_grad(self, args, kwargs);
  }
}
static PyObject *subtract_grad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract_grad_";
    return static_api_subtract_grad_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract_grad_";
    return eager_api_subtract_grad_(self, args, kwargs);
  }
}
static PyObject *tile_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tile_grad";
    return static_api_tile_grad(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tile_grad";
    return eager_api_tile_grad(self, args, kwargs);
  }
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
static PyObject *sparse_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs";
    return static_api_abs_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs";
    return sparse::eager_api_abs(self, args, kwargs);
  }
}
static PyObject *sparse_acos(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos";
    return static_api_acos_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos";
    return sparse::eager_api_acos(self, args, kwargs);
  }
}
static PyObject *sparse_acosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh";
    return static_api_acosh_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh";
    return sparse::eager_api_acosh(self, args, kwargs);
  }
}
static PyObject *sparse_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add";
    return static_api_add_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add";
    return sparse::eager_api_add(self, args, kwargs);
  }
}
static PyObject *sparse_asin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin";
    return static_api_asin_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin";
    return sparse::eager_api_asin(self, args, kwargs);
  }
}
static PyObject *sparse_asinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh";
    return static_api_asinh_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh";
    return sparse::eager_api_asinh(self, args, kwargs);
  }
}
static PyObject *sparse_atan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan";
    return static_api_atan_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan";
    return sparse::eager_api_atan(self, args, kwargs);
  }
}
static PyObject *sparse_atanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh";
    return static_api_atanh_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh";
    return sparse::eager_api_atanh(self, args, kwargs);
  }
}
static PyObject *sparse_batch_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_batch_norm_";
    return static_api_batch_norm_sp_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_batch_norm_";
    return sparse::eager_api_batch_norm_(self, args, kwargs);
  }
}
static PyObject *sparse_cast(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cast";
    return static_api_cast_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cast";
    return sparse::eager_api_cast(self, args, kwargs);
  }
}
static PyObject *sparse_conv3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d";
    return static_api_conv3d_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d";
    return sparse::eager_api_conv3d(self, args, kwargs);
  }
}
static PyObject *sparse_conv3d_implicit_gemm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d_implicit_gemm";
    return static_api_conv3d_implicit_gemm_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d_implicit_gemm";
    return sparse::eager_api_conv3d_implicit_gemm(self, args, kwargs);
  }
}
static PyObject *sparse_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide";
    return static_api_divide_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide";
    return sparse::eager_api_divide(self, args, kwargs);
  }
}
static PyObject *sparse_divide_scalar(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide_scalar";
    return static_api_divide_scalar_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide_scalar";
    return sparse::eager_api_divide_scalar(self, args, kwargs);
  }
}
static PyObject *sparse_expm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1";
    return static_api_expm1_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1";
    return sparse::eager_api_expm1(self, args, kwargs);
  }
}
static PyObject *sparse_isnan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_isnan";
    return static_api_isnan_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_isnan";
    return sparse::eager_api_isnan(self, args, kwargs);
  }
}
static PyObject *sparse_leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu";
    return static_api_leaky_relu_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu";
    return sparse::eager_api_leaky_relu(self, args, kwargs);
  }
}
static PyObject *sparse_log1p(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p";
    return static_api_log1p_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p";
    return sparse::eager_api_log1p(self, args, kwargs);
  }
}
static PyObject *sparse_multiply(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_multiply";
    return static_api_multiply_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_multiply";
    return sparse::eager_api_multiply(self, args, kwargs);
  }
}
static PyObject *sparse_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow";
    return static_api_pow_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow";
    return sparse::eager_api_pow(self, args, kwargs);
  }
}
static PyObject *sparse_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu";
    return static_api_relu_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu";
    return sparse::eager_api_relu(self, args, kwargs);
  }
}
static PyObject *sparse_relu6(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu6";
    return static_api_relu6_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu6";
    return sparse::eager_api_relu6(self, args, kwargs);
  }
}
static PyObject *sparse_reshape(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape";
    return static_api_reshape_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape";
    return sparse::eager_api_reshape(self, args, kwargs);
  }
}
static PyObject *sparse_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_scale";
    return static_api_scale_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_scale";
    return sparse::eager_api_scale(self, args, kwargs);
  }
}
static PyObject *sparse_sin(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin";
    return static_api_sin_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin";
    return sparse::eager_api_sin(self, args, kwargs);
  }
}
static PyObject *sparse_sinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh";
    return static_api_sinh_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh";
    return sparse::eager_api_sinh(self, args, kwargs);
  }
}
static PyObject *sparse_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softmax";
    return static_api_softmax_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softmax";
    return sparse::eager_api_softmax(self, args, kwargs);
  }
}
static PyObject *sparse_sparse_coo_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sparse_coo_tensor";
    return static_api_sparse_coo_tensor_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sparse_coo_tensor";
    return sparse::eager_api_sparse_coo_tensor(self, args, kwargs);
  }
}
static PyObject *sparse_sqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt";
    return static_api_sqrt_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt";
    return sparse::eager_api_sqrt(self, args, kwargs);
  }
}
static PyObject *sparse_square(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square";
    return static_api_square_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square";
    return sparse::eager_api_square(self, args, kwargs);
  }
}
static PyObject *sparse_subtract(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract";
    return static_api_subtract_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract";
    return sparse::eager_api_subtract(self, args, kwargs);
  }
}
static PyObject *sparse_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sum";
    return static_api_sum_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sum";
    return sparse::eager_api_sum(self, args, kwargs);
  }
}
static PyObject *sparse_sync_batch_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_batch_norm_";
    return static_api_sync_batch_norm_sp_(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_batch_norm_";
    return sparse::eager_api_sync_batch_norm_(self, args, kwargs);
  }
}
static PyObject *sparse_tan(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan";
    return static_api_tan_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan";
    return sparse::eager_api_tan(self, args, kwargs);
  }
}
static PyObject *sparse_tanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh";
    return static_api_tanh_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh";
    return sparse::eager_api_tanh(self, args, kwargs);
  }
}
static PyObject *sparse_to_dense(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_to_dense";
    return static_api_to_dense_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_to_dense";
    return sparse::eager_api_to_dense(self, args, kwargs);
  }
}
static PyObject *sparse_to_sparse_coo(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_to_sparse_coo";
    return static_api_to_sparse_coo_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_to_sparse_coo";
    return sparse::eager_api_to_sparse_coo(self, args, kwargs);
  }
}
static PyObject *sparse_to_sparse_csr(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_to_sparse_csr";
    return static_api_to_sparse_csr_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_to_sparse_csr";
    return sparse::eager_api_to_sparse_csr(self, args, kwargs);
  }
}
static PyObject *sparse_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_transpose";
    return static_api_transpose_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_transpose";
    return sparse::eager_api_transpose(self, args, kwargs);
  }
}
static PyObject *sparse_values(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_values";
    return static_api_values_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_values";
    return sparse::eager_api_values(self, args, kwargs);
  }
}
static PyObject *sparse_addmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_addmm";
    return static_api_addmm_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_addmm";
    return sparse::eager_api_addmm(self, args, kwargs);
  }
}
static PyObject *sparse_coalesce(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_coalesce";
    return static_api_coalesce_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_coalesce";
    return sparse::eager_api_coalesce(self, args, kwargs);
  }
}
static PyObject *sparse_full_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_full_like";
    return static_api_full_like_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_full_like";
    return sparse::eager_api_full_like(self, args, kwargs);
  }
}
static PyObject *sparse_fused_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_attention";
    return static_api_fused_attention_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_attention";
    return sparse::eager_api_fused_attention(self, args, kwargs);
  }
}
static PyObject *sparse_indices(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_indices";
    return static_api_indices_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_indices";
    return sparse::eager_api_indices(self, args, kwargs);
  }
}
static PyObject *sparse_mask_as(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mask_as";
    return static_api_mask_as_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mask_as";
    return sparse::eager_api_mask_as(self, args, kwargs);
  }
}
static PyObject *sparse_masked_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_masked_matmul";
    return static_api_masked_matmul_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_masked_matmul";
    return sparse::eager_api_masked_matmul(self, args, kwargs);
  }
}
static PyObject *sparse_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matmul";
    return static_api_matmul_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matmul";
    return sparse::eager_api_matmul(self, args, kwargs);
  }
}
static PyObject *sparse_maxpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maxpool";
    return static_api_maxpool_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maxpool";
    return sparse::eager_api_maxpool(self, args, kwargs);
  }
}
static PyObject *sparse_mv(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mv";
    return static_api_mv_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mv";
    return sparse::eager_api_mv(self, args, kwargs);
  }
}
static PyObject *sparse_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slice";
    return static_api_slice_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slice";
    return sparse::eager_api_slice(self, args, kwargs);
  }
}
static PyObject *sparse_abs_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_abs_grad";
    return static_api_abs_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_abs_grad";
    return sparse::eager_api_abs_grad(self, args, kwargs);
  }
}
static PyObject *sparse_acos_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acos_grad";
    return static_api_acos_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acos_grad";
    return sparse::eager_api_acos_grad(self, args, kwargs);
  }
}
static PyObject *sparse_acosh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_acosh_grad";
    return static_api_acosh_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_acosh_grad";
    return sparse::eager_api_acosh_grad(self, args, kwargs);
  }
}
static PyObject *sparse_add_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_add_grad";
    return static_api_add_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_add_grad";
    return sparse::eager_api_add_grad(self, args, kwargs);
  }
}
static PyObject *sparse_addmm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_addmm_grad";
    return static_api_addmm_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_addmm_grad";
    return sparse::eager_api_addmm_grad(self, args, kwargs);
  }
}
static PyObject *sparse_asin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asin_grad";
    return static_api_asin_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asin_grad";
    return sparse::eager_api_asin_grad(self, args, kwargs);
  }
}
static PyObject *sparse_asinh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_asinh_grad";
    return static_api_asinh_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_asinh_grad";
    return sparse::eager_api_asinh_grad(self, args, kwargs);
  }
}
static PyObject *sparse_atan_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atan_grad";
    return static_api_atan_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atan_grad";
    return sparse::eager_api_atan_grad(self, args, kwargs);
  }
}
static PyObject *sparse_atanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_atanh_grad";
    return static_api_atanh_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_atanh_grad";
    return sparse::eager_api_atanh_grad(self, args, kwargs);
  }
}
static PyObject *sparse_batch_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_batch_norm_grad";
    return static_api_batch_norm_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_batch_norm_grad";
    return sparse::eager_api_batch_norm_grad(self, args, kwargs);
  }
}
static PyObject *sparse_cast_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_cast_grad";
    return static_api_cast_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_cast_grad";
    return sparse::eager_api_cast_grad(self, args, kwargs);
  }
}
static PyObject *sparse_conv3d_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_conv3d_grad";
    return static_api_conv3d_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_conv3d_grad";
    return sparse::eager_api_conv3d_grad(self, args, kwargs);
  }
}
static PyObject *sparse_divide_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_divide_grad";
    return static_api_divide_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_divide_grad";
    return sparse::eager_api_divide_grad(self, args, kwargs);
  }
}
static PyObject *sparse_expm1_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_expm1_grad";
    return static_api_expm1_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_expm1_grad";
    return sparse::eager_api_expm1_grad(self, args, kwargs);
  }
}
static PyObject *sparse_leaky_relu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_leaky_relu_grad";
    return static_api_leaky_relu_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_leaky_relu_grad";
    return sparse::eager_api_leaky_relu_grad(self, args, kwargs);
  }
}
static PyObject *sparse_log1p_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_log1p_grad";
    return static_api_log1p_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_log1p_grad";
    return sparse::eager_api_log1p_grad(self, args, kwargs);
  }
}
static PyObject *sparse_mask_as_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mask_as_grad";
    return static_api_mask_as_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mask_as_grad";
    return sparse::eager_api_mask_as_grad(self, args, kwargs);
  }
}
static PyObject *sparse_masked_matmul_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_masked_matmul_grad";
    return static_api_masked_matmul_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_masked_matmul_grad";
    return sparse::eager_api_masked_matmul_grad(self, args, kwargs);
  }
}
static PyObject *sparse_matmul_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_matmul_grad";
    return static_api_matmul_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_matmul_grad";
    return sparse::eager_api_matmul_grad(self, args, kwargs);
  }
}
static PyObject *sparse_maxpool_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_maxpool_grad";
    return static_api_maxpool_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_maxpool_grad";
    return sparse::eager_api_maxpool_grad(self, args, kwargs);
  }
}
static PyObject *sparse_mv_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_mv_grad";
    return static_api_mv_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_mv_grad";
    return sparse::eager_api_mv_grad(self, args, kwargs);
  }
}
static PyObject *sparse_pow_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_pow_grad";
    return static_api_pow_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_pow_grad";
    return sparse::eager_api_pow_grad(self, args, kwargs);
  }
}
static PyObject *sparse_relu6_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu6_grad";
    return static_api_relu6_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu6_grad";
    return sparse::eager_api_relu6_grad(self, args, kwargs);
  }
}
static PyObject *sparse_relu_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_relu_grad";
    return static_api_relu_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_relu_grad";
    return sparse::eager_api_relu_grad(self, args, kwargs);
  }
}
static PyObject *sparse_reshape_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_reshape_grad";
    return static_api_reshape_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_reshape_grad";
    return sparse::eager_api_reshape_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sin_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sin_grad";
    return static_api_sin_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sin_grad";
    return sparse::eager_api_sin_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sinh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sinh_grad";
    return static_api_sinh_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sinh_grad";
    return sparse::eager_api_sinh_grad(self, args, kwargs);
  }
}
static PyObject *sparse_softmax_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_softmax_grad";
    return static_api_softmax_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_softmax_grad";
    return sparse::eager_api_softmax_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sparse_coo_tensor_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sparse_coo_tensor_grad";
    return static_api_sparse_coo_tensor_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sparse_coo_tensor_grad";
    return sparse::eager_api_sparse_coo_tensor_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sqrt_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sqrt_grad";
    return static_api_sqrt_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sqrt_grad";
    return sparse::eager_api_sqrt_grad(self, args, kwargs);
  }
}
static PyObject *sparse_square_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_square_grad";
    return static_api_square_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_square_grad";
    return sparse::eager_api_square_grad(self, args, kwargs);
  }
}
static PyObject *sparse_subtract_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_subtract_grad";
    return static_api_subtract_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_subtract_grad";
    return sparse::eager_api_subtract_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sum_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sum_grad";
    return static_api_sum_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sum_grad";
    return sparse::eager_api_sum_grad(self, args, kwargs);
  }
}
static PyObject *sparse_sync_batch_norm_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_sync_batch_norm_grad";
    return static_api_sync_batch_norm_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_sync_batch_norm_grad";
    return sparse::eager_api_sync_batch_norm_grad(self, args, kwargs);
  }
}
static PyObject *sparse_tan_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tan_grad";
    return static_api_tan_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tan_grad";
    return sparse::eager_api_tan_grad(self, args, kwargs);
  }
}
static PyObject *sparse_tanh_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_tanh_grad";
    return static_api_tanh_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_tanh_grad";
    return sparse::eager_api_tanh_grad(self, args, kwargs);
  }
}
static PyObject *sparse_to_dense_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_to_dense_grad";
    return static_api_to_dense_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_to_dense_grad";
    return sparse::eager_api_to_dense_grad(self, args, kwargs);
  }
}
static PyObject *sparse_to_sparse_coo_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_to_sparse_coo_grad";
    return static_api_to_sparse_coo_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_to_sparse_coo_grad";
    return sparse::eager_api_to_sparse_coo_grad(self, args, kwargs);
  }
}
static PyObject *sparse_transpose_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_transpose_grad";
    return static_api_transpose_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_transpose_grad";
    return sparse::eager_api_transpose_grad(self, args, kwargs);
  }
}
static PyObject *sparse_values_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_values_grad";
    return static_api_values_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_values_grad";
    return sparse::eager_api_values_grad(self, args, kwargs);
  }
}
static PyObject *sparse_fused_attention_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_fused_attention_grad";
    return static_api_fused_attention_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_fused_attention_grad";
    return sparse::eager_api_fused_attention_grad(self, args, kwargs);
  }
}
static PyObject *sparse_slice_grad(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {
    VLOG(6) << "Call static_api_slice_grad";
    return static_api_slice_grad_sp(self, args, kwargs);
  } else {
    VLOG(6) << "Call eager_api_slice_grad";
    return sparse::eager_api_slice_grad(self, args, kwargs);
  }
}

static PyMethodDef OpsAPI[] = {

{"abs", (PyCFunction)(void (*)(void))abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs."},
{"abs_", (PyCFunction)(void (*)(void))abs_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs_."},
{"accuracy", (PyCFunction)(void (*)(void))accuracy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for accuracy."},
{"accuracy_check", (PyCFunction)(void (*)(void))accuracy_check, METH_VARARGS | METH_KEYWORDS, "C++ interface function for accuracy_check."},
{"acos", (PyCFunction)(void (*)(void))acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos."},
{"acos_", (PyCFunction)(void (*)(void))acos_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos_."},
{"acosh", (PyCFunction)(void (*)(void))acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh."},
{"acosh_", (PyCFunction)(void (*)(void))acosh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh_."},
{"adadelta_", (PyCFunction)(void (*)(void))adadelta_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adadelta_."},
{"adagrad_", (PyCFunction)(void (*)(void))adagrad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adagrad_."},
{"adam_", (PyCFunction)(void (*)(void))adam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adam_."},
{"adamax_", (PyCFunction)(void (*)(void))adamax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamax_."},
{"adamw_", (PyCFunction)(void (*)(void))adamw_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamw_."},
{"add_position_encoding", (PyCFunction)(void (*)(void))add_position_encoding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_position_encoding."},
{"addmm", (PyCFunction)(void (*)(void))addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm."},
{"addmm_", (PyCFunction)(void (*)(void))addmm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm_."},
{"affine_channel", (PyCFunction)(void (*)(void))affine_channel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel."},
{"affine_channel_", (PyCFunction)(void (*)(void))affine_channel_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel_."},
{"affine_grid", (PyCFunction)(void (*)(void))affine_grid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid."},
{"all", (PyCFunction)(void (*)(void))all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all."},
{"all_gather", (PyCFunction)(void (*)(void))all_gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all_gather."},
{"all_reduce", (PyCFunction)(void (*)(void))all_reduce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all_reduce."},
{"all_reduce_", (PyCFunction)(void (*)(void))all_reduce_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all_reduce_."},
{"all_to_all", (PyCFunction)(void (*)(void))all_to_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all_to_all."},
{"allclose", (PyCFunction)(void (*)(void))allclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allclose."},
{"amax", (PyCFunction)(void (*)(void))amax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amax."},
{"amin", (PyCFunction)(void (*)(void))amin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amin."},
{"angle", (PyCFunction)(void (*)(void))angle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle."},
{"any", (PyCFunction)(void (*)(void))any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for any."},
{"ap_facade", (PyCFunction)(void (*)(void))ap_facade, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ap_facade."},
{"ap_trivial_fusion_begin", (PyCFunction)(void (*)(void))ap_trivial_fusion_begin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ap_trivial_fusion_begin."},
{"ap_trivial_fusion_end", (PyCFunction)(void (*)(void))ap_trivial_fusion_end, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ap_trivial_fusion_end."},
{"ap_variadic", (PyCFunction)(void (*)(void))ap_variadic, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ap_variadic."},
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
{"assign_out_", (PyCFunction)(void (*)(void))assign_out_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_out_."},
{"assign_pos", (PyCFunction)(void (*)(void))assign_pos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_pos."},
{"assign_value_", (PyCFunction)(void (*)(void))assign_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value_."},
{"atan", (PyCFunction)(void (*)(void))atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan."},
{"atan_", (PyCFunction)(void (*)(void))atan_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan_."},
{"atan2", (PyCFunction)(void (*)(void))atan2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2."},
{"atanh", (PyCFunction)(void (*)(void))atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh."},
{"atanh_", (PyCFunction)(void (*)(void))atanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh_."},
{"attention_lstm", (PyCFunction)(void (*)(void))attention_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for attention_lstm."},
{"auc", (PyCFunction)(void (*)(void))auc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for auc."},
{"average_accumulates_", (PyCFunction)(void (*)(void))average_accumulates_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for average_accumulates_."},
{"baddbmm", (PyCFunction)(void (*)(void))baddbmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for baddbmm."},
{"baddbmm_", (PyCFunction)(void (*)(void))baddbmm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for baddbmm_."},
{"bce_loss", (PyCFunction)(void (*)(void))bce_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss."},
{"bce_loss_", (PyCFunction)(void (*)(void))bce_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_."},
{"beam_search", (PyCFunction)(void (*)(void))beam_search, METH_VARARGS | METH_KEYWORDS, "C++ interface function for beam_search."},
{"bernoulli", (PyCFunction)(void (*)(void))bernoulli, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bernoulli."},
{"bicubic_interp", (PyCFunction)(void (*)(void))bicubic_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp."},
{"bilinear", (PyCFunction)(void (*)(void))bilinear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear."},
{"bilinear_interp", (PyCFunction)(void (*)(void))bilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp."},
{"bincount", (PyCFunction)(void (*)(void))bincount, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bincount."},
{"binomial", (PyCFunction)(void (*)(void))binomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for binomial."},
{"bipartite_match", (PyCFunction)(void (*)(void))bipartite_match, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bipartite_match."},
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
{"box_clip", (PyCFunction)(void (*)(void))box_clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_clip."},
{"box_coder", (PyCFunction)(void (*)(void))box_coder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_coder."},
{"broadcast", (PyCFunction)(void (*)(void))broadcast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast."},
{"broadcast_", (PyCFunction)(void (*)(void))broadcast_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_."},
{"broadcast_tensors", (PyCFunction)(void (*)(void))broadcast_tensors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors."},
{"c_allreduce_sum", (PyCFunction)(void (*)(void))c_allreduce_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_sum."},
{"c_allreduce_sum_", (PyCFunction)(void (*)(void))c_allreduce_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_sum_."},
{"c_concat", (PyCFunction)(void (*)(void))c_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_concat."},
{"c_identity", (PyCFunction)(void (*)(void))c_identity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_identity."},
{"c_identity_", (PyCFunction)(void (*)(void))c_identity_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_identity_."},
{"c_softmax_with_cross_entropy", (PyCFunction)(void (*)(void))c_softmax_with_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_cross_entropy."},
{"calc_reduced_attn_scores", (PyCFunction)(void (*)(void))calc_reduced_attn_scores, METH_VARARGS | METH_KEYWORDS, "C++ interface function for calc_reduced_attn_scores."},
{"cast", (PyCFunction)(void (*)(void))cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast."},
{"cast_", (PyCFunction)(void (*)(void))cast_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast_."},
{"ceil", (PyCFunction)(void (*)(void))ceil, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil."},
{"ceil_", (PyCFunction)(void (*)(void))ceil_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_."},
{"celu", (PyCFunction)(void (*)(void))celu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu."},
{"channel_shuffle", (PyCFunction)(void (*)(void))channel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for channel_shuffle."},
{"check_finite_and_unscale_", (PyCFunction)(void (*)(void))check_finite_and_unscale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_finite_and_unscale_."},
{"check_numerics", (PyCFunction)(void (*)(void))check_numerics, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_numerics."},
{"cholesky", (PyCFunction)(void (*)(void))cholesky, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky."},
{"cholesky_solve", (PyCFunction)(void (*)(void))cholesky_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve."},
{"class_center_sample", (PyCFunction)(void (*)(void))class_center_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for class_center_sample."},
{"clip", (PyCFunction)(void (*)(void))clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip."},
{"clip_", (PyCFunction)(void (*)(void))clip_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_."},
{"clip_by_norm", (PyCFunction)(void (*)(void))clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_by_norm."},
{"coalesce_tensor", (PyCFunction)(void (*)(void))coalesce_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor."},
{"collect_fpn_proposals", (PyCFunction)(void (*)(void))collect_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for collect_fpn_proposals."},
{"complex", (PyCFunction)(void (*)(void))complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for complex."},
{"concat", (PyCFunction)(void (*)(void))concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat."},
{"conj", (PyCFunction)(void (*)(void))conj, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conj."},
{"conv2d", (PyCFunction)(void (*)(void))conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d."},
{"conv2d_transpose", (PyCFunction)(void (*)(void))conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose."},
{"conv2d_transpose_bias", (PyCFunction)(void (*)(void))conv2d_transpose_bias, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose_bias."},
{"conv3d", (PyCFunction)(void (*)(void))conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d."},
{"conv3d_transpose", (PyCFunction)(void (*)(void))conv3d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose."},
{"copysign", (PyCFunction)(void (*)(void))copysign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign."},
{"copysign_", (PyCFunction)(void (*)(void))copysign_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign_."},
{"correlation", (PyCFunction)(void (*)(void))correlation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for correlation."},
{"cos", (PyCFunction)(void (*)(void))cos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos."},
{"cos_", (PyCFunction)(void (*)(void))cos_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_."},
{"cosh", (PyCFunction)(void (*)(void))cosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh."},
{"cosh_", (PyCFunction)(void (*)(void))cosh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh_."},
{"crf_decoding", (PyCFunction)(void (*)(void))crf_decoding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crf_decoding."},
{"crop", (PyCFunction)(void (*)(void))crop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop."},
{"cross", (PyCFunction)(void (*)(void))cross, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross."},
{"cross_entropy_with_softmax", (PyCFunction)(void (*)(void))cross_entropy_with_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax."},
{"cross_entropy_with_softmax_", (PyCFunction)(void (*)(void))cross_entropy_with_softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax_."},
{"ctc_align", (PyCFunction)(void (*)(void))ctc_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ctc_align."},
{"cudnn_lstm", (PyCFunction)(void (*)(void))cudnn_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cudnn_lstm."},
{"cummax", (PyCFunction)(void (*)(void))cummax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummax."},
{"cummin", (PyCFunction)(void (*)(void))cummin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummin."},
{"cumprod", (PyCFunction)(void (*)(void))cumprod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod."},
{"cumprod_", (PyCFunction)(void (*)(void))cumprod_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod_."},
{"cumsum", (PyCFunction)(void (*)(void))cumsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum."},
{"cumsum_", (PyCFunction)(void (*)(void))cumsum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum_."},
{"cvm", (PyCFunction)(void (*)(void))cvm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cvm."},
{"data", (PyCFunction)(void (*)(void))data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for data."},
{"decode_jpeg", (PyCFunction)(void (*)(void))decode_jpeg, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decode_jpeg."},
{"deformable_conv", (PyCFunction)(void (*)(void))deformable_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv."},
{"depend", (PyCFunction)(void (*)(void))depend, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depend."},
{"depthwise_conv2d", (PyCFunction)(void (*)(void))depthwise_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d."},
{"depthwise_conv2d_transpose", (PyCFunction)(void (*)(void))depthwise_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose."},
{"dequantize_abs_max", (PyCFunction)(void (*)(void))dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_abs_max."},
{"dequantize_log", (PyCFunction)(void (*)(void))dequantize_log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_log."},
{"det", (PyCFunction)(void (*)(void))det, METH_VARARGS | METH_KEYWORDS, "C++ interface function for det."},
{"dgc_clip_by_norm", (PyCFunction)(void (*)(void))dgc_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dgc_clip_by_norm."},
{"diag", (PyCFunction)(void (*)(void))diag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag."},
{"diag_embed", (PyCFunction)(void (*)(void))diag_embed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_embed."},
{"diagonal", (PyCFunction)(void (*)(void))diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diagonal."},
{"digamma", (PyCFunction)(void (*)(void))digamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma."},
{"digamma_", (PyCFunction)(void (*)(void))digamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma_."},
{"dirichlet", (PyCFunction)(void (*)(void))dirichlet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dirichlet."},
{"disable_check_model_nan_inf", (PyCFunction)(void (*)(void))disable_check_model_nan_inf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for disable_check_model_nan_inf."},
{"dist", (PyCFunction)(void (*)(void))dist, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist."},
{"dot", (PyCFunction)(void (*)(void))dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot."},
{"dropout", (PyCFunction)(void (*)(void))dropout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout."},
{"edit_distance", (PyCFunction)(void (*)(void))edit_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for edit_distance."},
{"eig", (PyCFunction)(void (*)(void))eig, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eig."},
{"eigh", (PyCFunction)(void (*)(void))eigh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh."},
{"eigvals", (PyCFunction)(void (*)(void))eigvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvals."},
{"eigvalsh", (PyCFunction)(void (*)(void))eigvalsh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvalsh."},
{"elu", (PyCFunction)(void (*)(void))elu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu."},
{"elu_", (PyCFunction)(void (*)(void))elu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_."},
{"embedding_with_scaled_gradient", (PyCFunction)(void (*)(void))embedding_with_scaled_gradient, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding_with_scaled_gradient."},
{"empty", (PyCFunction)(void (*)(void))empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty."},
{"empty_like", (PyCFunction)(void (*)(void))empty_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty_like."},
{"enable_check_model_nan_inf", (PyCFunction)(void (*)(void))enable_check_model_nan_inf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for enable_check_model_nan_inf."},
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
{"exponential_", (PyCFunction)(void (*)(void))exponential_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential_."},
{"eye", (PyCFunction)(void (*)(void))eye, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eye."},
{"fake_channel_wise_dequantize_max_abs", (PyCFunction)(void (*)(void))fake_channel_wise_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_dequantize_max_abs."},
{"fake_channel_wise_quantize_abs_max", (PyCFunction)(void (*)(void))fake_channel_wise_quantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_abs_max."},
{"fake_channel_wise_quantize_dequantize_abs_max", (PyCFunction)(void (*)(void))fake_channel_wise_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_dequantize_abs_max."},
{"fake_dequantize_max_abs", (PyCFunction)(void (*)(void))fake_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_dequantize_max_abs."},
{"fake_quantize_abs_max", (PyCFunction)(void (*)(void))fake_quantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_abs_max."},
{"fake_quantize_dequantize_abs_max", (PyCFunction)(void (*)(void))fake_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_abs_max."},
{"fake_quantize_dequantize_moving_average_abs_max", (PyCFunction)(void (*)(void))fake_quantize_dequantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_moving_average_abs_max."},
{"fake_quantize_dequantize_moving_average_abs_max_", (PyCFunction)(void (*)(void))fake_quantize_dequantize_moving_average_abs_max_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_moving_average_abs_max_."},
{"fake_quantize_moving_average_abs_max", (PyCFunction)(void (*)(void))fake_quantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_moving_average_abs_max."},
{"fake_quantize_moving_average_abs_max_", (PyCFunction)(void (*)(void))fake_quantize_moving_average_abs_max_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_moving_average_abs_max_."},
{"fake_quantize_range_abs_max", (PyCFunction)(void (*)(void))fake_quantize_range_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_range_abs_max."},
{"fake_quantize_range_abs_max_", (PyCFunction)(void (*)(void))fake_quantize_range_abs_max_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_range_abs_max_."},
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
{"flash_attn_qkvpacked", (PyCFunction)(void (*)(void))flash_attn_qkvpacked, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_qkvpacked."},
{"flash_attn_unpadded", (PyCFunction)(void (*)(void))flash_attn_unpadded, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_unpadded."},
{"flash_attn_v3", (PyCFunction)(void (*)(void))flash_attn_v3, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_v3."},
{"flash_attn_varlen_qkvpacked", (PyCFunction)(void (*)(void))flash_attn_varlen_qkvpacked, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_varlen_qkvpacked."},
{"flashmask_attention", (PyCFunction)(void (*)(void))flashmask_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flashmask_attention."},
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
{"frobenius_norm", (PyCFunction)(void (*)(void))frobenius_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frobenius_norm."},
{"full", (PyCFunction)(void (*)(void))full, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full."},
{"full_", (PyCFunction)(void (*)(void))full_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_."},
{"full_batch_size_like", (PyCFunction)(void (*)(void))full_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_batch_size_like."},
{"full_int_array", (PyCFunction)(void (*)(void))full_int_array, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_int_array."},
{"full_like", (PyCFunction)(void (*)(void))full_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_like."},
{"full_with_tensor", (PyCFunction)(void (*)(void))full_with_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_with_tensor."},
{"fused_batch_norm_act", (PyCFunction)(void (*)(void))fused_batch_norm_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_batch_norm_act."},
{"fused_bn_add_activation", (PyCFunction)(void (*)(void))fused_bn_add_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bn_add_activation."},
{"fused_softmax_mask", (PyCFunction)(void (*)(void))fused_softmax_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask."},
{"fused_softmax_mask_upper_triangle", (PyCFunction)(void (*)(void))fused_softmax_mask_upper_triangle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_upper_triangle."},
{"gammaincc", (PyCFunction)(void (*)(void))gammaincc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaincc."},
{"gammaincc_", (PyCFunction)(void (*)(void))gammaincc_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaincc_."},
{"gammaln", (PyCFunction)(void (*)(void))gammaln, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaln."},
{"gammaln_", (PyCFunction)(void (*)(void))gammaln_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaln_."},
{"gather", (PyCFunction)(void (*)(void))gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather."},
{"gather_nd", (PyCFunction)(void (*)(void))gather_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_nd."},
{"gather_tree", (PyCFunction)(void (*)(void))gather_tree, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_tree."},
{"gaussian", (PyCFunction)(void (*)(void))gaussian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian."},
{"gaussian_inplace", (PyCFunction)(void (*)(void))gaussian_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace."},
{"gaussian_inplace_", (PyCFunction)(void (*)(void))gaussian_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace_."},
{"gelu", (PyCFunction)(void (*)(void))gelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu."},
{"generate_proposals", (PyCFunction)(void (*)(void))generate_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals."},
{"graph_khop_sampler", (PyCFunction)(void (*)(void))graph_khop_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_khop_sampler."},
{"graph_sample_neighbors", (PyCFunction)(void (*)(void))graph_sample_neighbors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_sample_neighbors."},
{"grid_sample", (PyCFunction)(void (*)(void))grid_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sample."},
{"group_norm", (PyCFunction)(void (*)(void))group_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm."},
{"gru", (PyCFunction)(void (*)(void))gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru."},
{"gru_unit", (PyCFunction)(void (*)(void))gru_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru_unit."},
{"gumbel_softmax", (PyCFunction)(void (*)(void))gumbel_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gumbel_softmax."},
{"hardshrink", (PyCFunction)(void (*)(void))hardshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardshrink."},
{"hardsigmoid", (PyCFunction)(void (*)(void))hardsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardsigmoid."},
{"hardtanh", (PyCFunction)(void (*)(void))hardtanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh."},
{"hardtanh_", (PyCFunction)(void (*)(void))hardtanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh_."},
{"heaviside", (PyCFunction)(void (*)(void))heaviside, METH_VARARGS | METH_KEYWORDS, "C++ interface function for heaviside."},
{"hinge_loss", (PyCFunction)(void (*)(void))hinge_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hinge_loss."},
{"histogram", (PyCFunction)(void (*)(void))histogram, METH_VARARGS | METH_KEYWORDS, "C++ interface function for histogram."},
{"hsigmoid_loss", (PyCFunction)(void (*)(void))hsigmoid_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hsigmoid_loss."},
{"huber_loss", (PyCFunction)(void (*)(void))huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for huber_loss."},
{"i0", (PyCFunction)(void (*)(void))i0, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0."},
{"i0_", (PyCFunction)(void (*)(void))i0_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0_."},
{"i0e", (PyCFunction)(void (*)(void))i0e, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0e."},
{"i1", (PyCFunction)(void (*)(void))i1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1."},
{"i1e", (PyCFunction)(void (*)(void))i1e, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1e."},
{"identity_loss", (PyCFunction)(void (*)(void))identity_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss."},
{"identity_loss_", (PyCFunction)(void (*)(void))identity_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss_."},
{"im2sequence", (PyCFunction)(void (*)(void))im2sequence, METH_VARARGS | METH_KEYWORDS, "C++ interface function for im2sequence."},
{"imag", (PyCFunction)(void (*)(void))imag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag."},
{"increment", (PyCFunction)(void (*)(void))increment, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment."},
{"increment_", (PyCFunction)(void (*)(void))increment_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment_."},
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
{"l1_norm", (PyCFunction)(void (*)(void))l1_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for l1_norm."},
{"l1_norm_", (PyCFunction)(void (*)(void))l1_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for l1_norm_."},
{"label_smooth", (PyCFunction)(void (*)(void))label_smooth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth."},
{"lamb_", (PyCFunction)(void (*)(void))lamb_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lamb_."},
{"layer_norm", (PyCFunction)(void (*)(void))layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm."},
{"leaky_relu", (PyCFunction)(void (*)(void))leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu."},
{"leaky_relu_", (PyCFunction)(void (*)(void))leaky_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_."},
{"lerp", (PyCFunction)(void (*)(void))lerp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp."},
{"lerp_", (PyCFunction)(void (*)(void))lerp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp_."},
{"lgamma", (PyCFunction)(void (*)(void))lgamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma."},
{"lgamma_", (PyCFunction)(void (*)(void))lgamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma_."},
{"limit_by_capacity", (PyCFunction)(void (*)(void))limit_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for limit_by_capacity."},
{"linear_interp", (PyCFunction)(void (*)(void))linear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp."},
{"linspace", (PyCFunction)(void (*)(void))linspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linspace."},
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
{"logspace", (PyCFunction)(void (*)(void))logspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logspace."},
{"logsumexp", (PyCFunction)(void (*)(void))logsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp."},
{"lookup_table_dequant", (PyCFunction)(void (*)(void))lookup_table_dequant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lookup_table_dequant."},
{"lp_pool2d", (PyCFunction)(void (*)(void))lp_pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lp_pool2d."},
{"lstm", (PyCFunction)(void (*)(void))lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm."},
{"lstsq", (PyCFunction)(void (*)(void))lstsq, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstsq."},
{"lu", (PyCFunction)(void (*)(void))lu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu."},
{"lu_", (PyCFunction)(void (*)(void))lu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_."},
{"lu_solve", (PyCFunction)(void (*)(void))lu_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_solve."},
{"lu_unpack", (PyCFunction)(void (*)(void))lu_unpack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack."},
{"margin_cross_entropy", (PyCFunction)(void (*)(void))margin_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy."},
{"masked_multihead_attention_", (PyCFunction)(void (*)(void))masked_multihead_attention_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_multihead_attention_."},
{"masked_select", (PyCFunction)(void (*)(void))masked_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select."},
{"matrix_nms", (PyCFunction)(void (*)(void))matrix_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_nms."},
{"matrix_power", (PyCFunction)(void (*)(void))matrix_power, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power."},
{"matrix_rank", (PyCFunction)(void (*)(void))matrix_rank, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank."},
{"matrix_rank_atol_rtol", (PyCFunction)(void (*)(void))matrix_rank_atol_rtol, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank_atol_rtol."},
{"matrix_rank_tol", (PyCFunction)(void (*)(void))matrix_rank_tol, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank_tol."},
{"max", (PyCFunction)(void (*)(void))max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max."},
{"max_pool2d_with_index", (PyCFunction)(void (*)(void))max_pool2d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index."},
{"max_pool3d_with_index", (PyCFunction)(void (*)(void))max_pool3d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index."},
{"maxout", (PyCFunction)(void (*)(void))maxout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout."},
{"mean", (PyCFunction)(void (*)(void))mean, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean."},
{"mean_all", (PyCFunction)(void (*)(void))mean_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_all."},
{"memcpy_d2h", (PyCFunction)(void (*)(void))memcpy_d2h, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_d2h."},
{"memcpy_h2d", (PyCFunction)(void (*)(void))memcpy_h2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_h2d."},
{"memory_efficient_attention", (PyCFunction)(void (*)(void))memory_efficient_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memory_efficient_attention."},
{"merge_selected_rows", (PyCFunction)(void (*)(void))merge_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merge_selected_rows."},
{"merged_adam_", (PyCFunction)(void (*)(void))merged_adam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_adam_."},
{"merged_momentum_", (PyCFunction)(void (*)(void))merged_momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_momentum_."},
{"meshgrid", (PyCFunction)(void (*)(void))meshgrid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for meshgrid."},
{"mish", (PyCFunction)(void (*)(void))mish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish."},
{"mode", (PyCFunction)(void (*)(void))mode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode."},
{"momentum_", (PyCFunction)(void (*)(void))momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for momentum_."},
{"mp_allreduce_sum", (PyCFunction)(void (*)(void))mp_allreduce_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mp_allreduce_sum."},
{"mp_allreduce_sum_", (PyCFunction)(void (*)(void))mp_allreduce_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mp_allreduce_sum_."},
{"multi_dot", (PyCFunction)(void (*)(void))multi_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot."},
{"multiclass_nms3", (PyCFunction)(void (*)(void))multiclass_nms3, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms3."},
{"multinomial", (PyCFunction)(void (*)(void))multinomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multinomial."},
{"multiplex", (PyCFunction)(void (*)(void))multiplex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex."},
{"mv", (PyCFunction)(void (*)(void))mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv."},
{"nadam_", (PyCFunction)(void (*)(void))nadam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nadam_."},
{"nanmedian", (PyCFunction)(void (*)(void))nanmedian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nanmedian."},
{"nearest_interp", (PyCFunction)(void (*)(void))nearest_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp."},
{"nextafter", (PyCFunction)(void (*)(void))nextafter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nextafter."},
{"nll_loss", (PyCFunction)(void (*)(void))nll_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss."},
{"nms", (PyCFunction)(void (*)(void))nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nms."},
{"nonzero", (PyCFunction)(void (*)(void))nonzero, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nonzero."},
{"norm", (PyCFunction)(void (*)(void))norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm."},
{"npu_identity", (PyCFunction)(void (*)(void))npu_identity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for npu_identity."},
{"numel", (PyCFunction)(void (*)(void))numel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for numel."},
{"one_hot", (PyCFunction)(void (*)(void))one_hot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for one_hot."},
{"overlap_add", (PyCFunction)(void (*)(void))overlap_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add."},
{"p_norm", (PyCFunction)(void (*)(void))p_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm."},
{"pad", (PyCFunction)(void (*)(void))pad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad."},
{"pad3d", (PyCFunction)(void (*)(void))pad3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d."},
{"pixel_shuffle", (PyCFunction)(void (*)(void))pixel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_shuffle."},
{"pixel_unshuffle", (PyCFunction)(void (*)(void))pixel_unshuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_unshuffle."},
{"poisson", (PyCFunction)(void (*)(void))poisson, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson."},
{"polygamma", (PyCFunction)(void (*)(void))polygamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygamma."},
{"polygamma_", (PyCFunction)(void (*)(void))polygamma_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygamma_."},
{"pool2d", (PyCFunction)(void (*)(void))pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d."},
{"pool3d", (PyCFunction)(void (*)(void))pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d."},
{"pow", (PyCFunction)(void (*)(void))pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow."},
{"pow_", (PyCFunction)(void (*)(void))pow_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow_."},
{"prelu", (PyCFunction)(void (*)(void))prelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu."},
{"prior_box", (PyCFunction)(void (*)(void))prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prior_box."},
{"prod", (PyCFunction)(void (*)(void))prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prod."},
{"prune_gate_by_capacity", (PyCFunction)(void (*)(void))prune_gate_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prune_gate_by_capacity."},
{"psroi_pool", (PyCFunction)(void (*)(void))psroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for psroi_pool."},
{"put_along_axis", (PyCFunction)(void (*)(void))put_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis."},
{"put_along_axis_", (PyCFunction)(void (*)(void))put_along_axis_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_."},
{"pyramid_hash", (PyCFunction)(void (*)(void))pyramid_hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pyramid_hash."},
{"qr", (PyCFunction)(void (*)(void))qr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr."},
{"radam_", (PyCFunction)(void (*)(void))radam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for radam_."},
{"randint", (PyCFunction)(void (*)(void))randint, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randint."},
{"random_routing_", (PyCFunction)(void (*)(void))random_routing_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for random_routing_."},
{"randperm", (PyCFunction)(void (*)(void))randperm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randperm."},
{"rank_attention", (PyCFunction)(void (*)(void))rank_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rank_attention."},
{"read_file", (PyCFunction)(void (*)(void))read_file, METH_VARARGS | METH_KEYWORDS, "C++ interface function for read_file."},
{"real", (PyCFunction)(void (*)(void))real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real."},
{"reciprocal", (PyCFunction)(void (*)(void))reciprocal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal."},
{"reciprocal_", (PyCFunction)(void (*)(void))reciprocal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_."},
{"reduce", (PyCFunction)(void (*)(void))reduce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce."},
{"reduce_", (PyCFunction)(void (*)(void))reduce_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_."},
{"reduce_as", (PyCFunction)(void (*)(void))reduce_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_as."},
{"reduce_scatter", (PyCFunction)(void (*)(void))reduce_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_scatter."},
{"reindex_graph", (PyCFunction)(void (*)(void))reindex_graph, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reindex_graph."},
{"relu", (PyCFunction)(void (*)(void))relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu."},
{"relu_", (PyCFunction)(void (*)(void))relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_."},
{"relu6", (PyCFunction)(void (*)(void))relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6."},
{"renorm", (PyCFunction)(void (*)(void))renorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm."},
{"renorm_", (PyCFunction)(void (*)(void))renorm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm_."},
{"repeat_interleave", (PyCFunction)(void (*)(void))repeat_interleave, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave."},
{"repeat_interleave_with_tensor_index", (PyCFunction)(void (*)(void))repeat_interleave_with_tensor_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave_with_tensor_index."},
{"reshape", (PyCFunction)(void (*)(void))reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape."},
{"reshape_", (PyCFunction)(void (*)(void))reshape_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_."},
{"restrict_nonzero", (PyCFunction)(void (*)(void))restrict_nonzero, METH_VARARGS | METH_KEYWORDS, "C++ interface function for restrict_nonzero."},
{"reverse", (PyCFunction)(void (*)(void))reverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reverse."},
{"rms_norm", (PyCFunction)(void (*)(void))rms_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rms_norm."},
{"rmsprop_", (PyCFunction)(void (*)(void))rmsprop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rmsprop_."},
{"rnn", (PyCFunction)(void (*)(void))rnn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rnn."},
{"roi_align", (PyCFunction)(void (*)(void))roi_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align."},
{"roi_pool", (PyCFunction)(void (*)(void))roi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool."},
{"roll", (PyCFunction)(void (*)(void))roll, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll."},
{"round", (PyCFunction)(void (*)(void))round, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round."},
{"round_", (PyCFunction)(void (*)(void))round_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_."},
{"rprop_", (PyCFunction)(void (*)(void))rprop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rprop_."},
{"rrelu", (PyCFunction)(void (*)(void))rrelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rrelu."},
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
{"sequence_conv", (PyCFunction)(void (*)(void))sequence_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_conv."},
{"sequence_mask", (PyCFunction)(void (*)(void))sequence_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_mask."},
{"sequence_pool", (PyCFunction)(void (*)(void))sequence_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_pool."},
{"set", (PyCFunction)(void (*)(void))set, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set."},
{"set_", (PyCFunction)(void (*)(void))set_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_."},
{"set_value_with_tensor", (PyCFunction)(void (*)(void))set_value_with_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_with_tensor."},
{"set_value_with_tensor_", (PyCFunction)(void (*)(void))set_value_with_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_with_tensor_."},
{"sgd_", (PyCFunction)(void (*)(void))sgd_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sgd_."},
{"shape", (PyCFunction)(void (*)(void))shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape."},
{"shape64", (PyCFunction)(void (*)(void))shape64, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape64."},
{"shard_index", (PyCFunction)(void (*)(void))shard_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shard_index."},
{"share_data", (PyCFunction)(void (*)(void))share_data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_data."},
{"shuffle_batch", (PyCFunction)(void (*)(void))shuffle_batch, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_batch."},
{"shuffle_channel", (PyCFunction)(void (*)(void))shuffle_channel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_channel."},
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
{"slice", (PyCFunction)(void (*)(void))slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slice."},
{"slogdet", (PyCFunction)(void (*)(void))slogdet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdet."},
{"softplus", (PyCFunction)(void (*)(void))softplus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus."},
{"softshrink", (PyCFunction)(void (*)(void))softshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink."},
{"softsign", (PyCFunction)(void (*)(void))softsign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign."},
{"solve", (PyCFunction)(void (*)(void))solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for solve."},
{"sparse_attention", (PyCFunction)(void (*)(void))sparse_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_attention."},
{"spectral_norm", (PyCFunction)(void (*)(void))spectral_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spectral_norm."},
{"split", (PyCFunction)(void (*)(void))split, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split."},
{"split_with_num", (PyCFunction)(void (*)(void))split_with_num, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split_with_num."},
{"sqrt", (PyCFunction)(void (*)(void))sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt."},
{"sqrt_", (PyCFunction)(void (*)(void))sqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_."},
{"square", (PyCFunction)(void (*)(void))square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square."},
{"square_", (PyCFunction)(void (*)(void))square_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square_."},
{"squared_l2_norm", (PyCFunction)(void (*)(void))squared_l2_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_norm."},
{"squeeze", (PyCFunction)(void (*)(void))squeeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze."},
{"squeeze_", (PyCFunction)(void (*)(void))squeeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_."},
{"stack", (PyCFunction)(void (*)(void))stack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack."},
{"standard_gamma", (PyCFunction)(void (*)(void))standard_gamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for standard_gamma."},
{"stanh", (PyCFunction)(void (*)(void))stanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stanh."},
{"stft", (PyCFunction)(void (*)(void))stft, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stft."},
{"strided_slice", (PyCFunction)(void (*)(void))strided_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice."},
{"sum", (PyCFunction)(void (*)(void))sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum."},
{"svd", (PyCFunction)(void (*)(void))svd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svd."},
{"svdvals", (PyCFunction)(void (*)(void))svdvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svdvals."},
{"swiglu", (PyCFunction)(void (*)(void))swiglu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swiglu."},
{"swish", (PyCFunction)(void (*)(void))swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish."},
{"sync_batch_norm_", (PyCFunction)(void (*)(void))sync_batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm_."},
{"sync_calc_stream", (PyCFunction)(void (*)(void))sync_calc_stream, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_calc_stream."},
{"sync_calc_stream_", (PyCFunction)(void (*)(void))sync_calc_stream_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_calc_stream_."},
{"take_along_axis", (PyCFunction)(void (*)(void))take_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis."},
{"tan", (PyCFunction)(void (*)(void))tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan."},
{"tan_", (PyCFunction)(void (*)(void))tan_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan_."},
{"tanh", (PyCFunction)(void (*)(void))tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh."},
{"tanh_", (PyCFunction)(void (*)(void))tanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_."},
{"tanh_shrink", (PyCFunction)(void (*)(void))tanh_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink."},
{"tdm_child", (PyCFunction)(void (*)(void))tdm_child, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tdm_child."},
{"tdm_sampler", (PyCFunction)(void (*)(void))tdm_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tdm_sampler."},
{"temporal_shift", (PyCFunction)(void (*)(void))temporal_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift."},
{"thresholded_relu", (PyCFunction)(void (*)(void))thresholded_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu."},
{"thresholded_relu_", (PyCFunction)(void (*)(void))thresholded_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu_."},
{"top_p_sampling", (PyCFunction)(void (*)(void))top_p_sampling, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_p_sampling."},
{"topk", (PyCFunction)(void (*)(void))topk, METH_VARARGS | METH_KEYWORDS, "C++ interface function for topk."},
{"trace", (PyCFunction)(void (*)(void))trace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace."},
{"trans_layout", (PyCFunction)(void (*)(void))trans_layout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trans_layout."},
{"transpose", (PyCFunction)(void (*)(void))transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose."},
{"transpose_", (PyCFunction)(void (*)(void))transpose_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose_."},
{"triangular_solve", (PyCFunction)(void (*)(void))triangular_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triangular_solve."},
{"tril", (PyCFunction)(void (*)(void))tril, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril."},
{"tril_", (PyCFunction)(void (*)(void))tril_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_."},
{"tril_indices", (PyCFunction)(void (*)(void))tril_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_indices."},
{"trilinear_interp", (PyCFunction)(void (*)(void))trilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp."},
{"triu", (PyCFunction)(void (*)(void))triu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu."},
{"triu_", (PyCFunction)(void (*)(void))triu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_."},
{"triu_indices", (PyCFunction)(void (*)(void))triu_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_indices."},
{"trunc", (PyCFunction)(void (*)(void))trunc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc."},
{"trunc_", (PyCFunction)(void (*)(void))trunc_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc_."},
{"truncated_gaussian_random", (PyCFunction)(void (*)(void))truncated_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for truncated_gaussian_random."},
{"unbind", (PyCFunction)(void (*)(void))unbind, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unbind."},
{"unfold", (PyCFunction)(void (*)(void))unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold."},
{"uniform", (PyCFunction)(void (*)(void))uniform, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform."},
{"uniform_inplace", (PyCFunction)(void (*)(void))uniform_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace."},
{"uniform_inplace_", (PyCFunction)(void (*)(void))uniform_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace_."},
{"uniform_random_batch_size_like", (PyCFunction)(void (*)(void))uniform_random_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_batch_size_like."},
{"unique_consecutive", (PyCFunction)(void (*)(void))unique_consecutive, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_consecutive."},
{"unpool", (PyCFunction)(void (*)(void))unpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool."},
{"unpool3d", (PyCFunction)(void (*)(void))unpool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d."},
{"unsqueeze", (PyCFunction)(void (*)(void))unsqueeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze."},
{"unsqueeze_", (PyCFunction)(void (*)(void))unsqueeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze_."},
{"unstack", (PyCFunction)(void (*)(void))unstack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack."},
{"update_loss_scaling_", (PyCFunction)(void (*)(void))update_loss_scaling_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for update_loss_scaling_."},
{"variance", (PyCFunction)(void (*)(void))variance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for variance."},
{"view_dtype", (PyCFunction)(void (*)(void))view_dtype, METH_VARARGS | METH_KEYWORDS, "C++ interface function for view_dtype."},
{"view_slice", (PyCFunction)(void (*)(void))view_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for view_slice."},
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
{"yolo_box_head", (PyCFunction)(void (*)(void))yolo_box_head, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_head."},
{"yolo_box_post", (PyCFunction)(void (*)(void))yolo_box_post, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_post."},
{"yolo_loss", (PyCFunction)(void (*)(void))yolo_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_loss."},
{"chunk_eval", (PyCFunction)(void (*)(void))chunk_eval, METH_VARARGS | METH_KEYWORDS, "C++ interface function for chunk_eval."},
{"number_count", (PyCFunction)(void (*)(void))number_count, METH_VARARGS | METH_KEYWORDS, "C++ interface function for number_count."},
{"fused_rms_norm", (PyCFunction)(void (*)(void))fused_rms_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_rms_norm."},
{"int_bincount", (PyCFunction)(void (*)(void))int_bincount, METH_VARARGS | METH_KEYWORDS, "C++ interface function for int_bincount."},
{"abs_grad", (PyCFunction)(void (*)(void))abs_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs_grad."},
{"acos_grad", (PyCFunction)(void (*)(void))acos_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos_grad."},
{"acos_grad_", (PyCFunction)(void (*)(void))acos_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos_grad_."},
{"acosh_grad", (PyCFunction)(void (*)(void))acosh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh_grad."},
{"acosh_grad_", (PyCFunction)(void (*)(void))acosh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh_grad_."},
{"add_position_encoding_grad", (PyCFunction)(void (*)(void))add_position_encoding_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_position_encoding_grad."},
{"addmm_grad", (PyCFunction)(void (*)(void))addmm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm_grad."},
{"affine_channel_grad", (PyCFunction)(void (*)(void))affine_channel_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel_grad."},
{"affine_channel_grad_", (PyCFunction)(void (*)(void))affine_channel_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel_grad_."},
{"affine_grid_grad", (PyCFunction)(void (*)(void))affine_grid_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid_grad."},
{"amax_grad", (PyCFunction)(void (*)(void))amax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amax_grad."},
{"amin_grad", (PyCFunction)(void (*)(void))amin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amin_grad."},
{"angle_grad", (PyCFunction)(void (*)(void))angle_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle_grad."},
{"argsort_grad", (PyCFunction)(void (*)(void))argsort_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argsort_grad."},
{"as_strided_grad", (PyCFunction)(void (*)(void))as_strided_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_strided_grad."},
{"asin_grad", (PyCFunction)(void (*)(void))asin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin_grad."},
{"asin_grad_", (PyCFunction)(void (*)(void))asin_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin_grad_."},
{"asinh_grad", (PyCFunction)(void (*)(void))asinh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh_grad."},
{"asinh_grad_", (PyCFunction)(void (*)(void))asinh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh_grad_."},
{"atan2_grad", (PyCFunction)(void (*)(void))atan2_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2_grad."},
{"atan_grad", (PyCFunction)(void (*)(void))atan_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan_grad."},
{"atan_grad_", (PyCFunction)(void (*)(void))atan_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan_grad_."},
{"atanh_grad", (PyCFunction)(void (*)(void))atanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh_grad."},
{"atanh_grad_", (PyCFunction)(void (*)(void))atanh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh_grad_."},
{"baddbmm_grad", (PyCFunction)(void (*)(void))baddbmm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for baddbmm_grad."},
{"bce_loss_grad", (PyCFunction)(void (*)(void))bce_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_grad."},
{"bce_loss_grad_", (PyCFunction)(void (*)(void))bce_loss_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_grad_."},
{"bicubic_interp_grad", (PyCFunction)(void (*)(void))bicubic_interp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp_grad."},
{"bilinear_grad", (PyCFunction)(void (*)(void))bilinear_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_grad."},
{"bilinear_interp_grad", (PyCFunction)(void (*)(void))bilinear_interp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp_grad."},
{"bmm_grad", (PyCFunction)(void (*)(void))bmm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bmm_grad."},
{"broadcast_tensors_grad", (PyCFunction)(void (*)(void))broadcast_tensors_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors_grad."},
{"c_softmax_with_cross_entropy_grad", (PyCFunction)(void (*)(void))c_softmax_with_cross_entropy_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_cross_entropy_grad."},
{"c_softmax_with_cross_entropy_grad_", (PyCFunction)(void (*)(void))c_softmax_with_cross_entropy_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_cross_entropy_grad_."},
{"ceil_grad", (PyCFunction)(void (*)(void))ceil_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_grad."},
{"ceil_grad_", (PyCFunction)(void (*)(void))ceil_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_grad_."},
{"celu_grad", (PyCFunction)(void (*)(void))celu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu_grad."},
{"celu_grad_", (PyCFunction)(void (*)(void))celu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu_grad_."},
{"channel_shuffle_grad", (PyCFunction)(void (*)(void))channel_shuffle_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for channel_shuffle_grad."},
{"cholesky_grad", (PyCFunction)(void (*)(void))cholesky_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_grad."},
{"cholesky_solve_grad", (PyCFunction)(void (*)(void))cholesky_solve_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve_grad."},
{"clip_grad", (PyCFunction)(void (*)(void))clip_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_grad."},
{"clip_grad_", (PyCFunction)(void (*)(void))clip_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_grad_."},
{"complex_grad", (PyCFunction)(void (*)(void))complex_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for complex_grad."},
{"concat_grad", (PyCFunction)(void (*)(void))concat_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat_grad."},
{"conv2d_transpose_grad", (PyCFunction)(void (*)(void))conv2d_transpose_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose_grad."},
{"conv3d_grad", (PyCFunction)(void (*)(void))conv3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_grad."},
{"conv3d_transpose_grad", (PyCFunction)(void (*)(void))conv3d_transpose_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose_grad."},
{"copysign_grad", (PyCFunction)(void (*)(void))copysign_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign_grad."},
{"copysign_grad_", (PyCFunction)(void (*)(void))copysign_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copysign_grad_."},
{"correlation_grad", (PyCFunction)(void (*)(void))correlation_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for correlation_grad."},
{"cos_grad", (PyCFunction)(void (*)(void))cos_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_grad."},
{"cos_grad_", (PyCFunction)(void (*)(void))cos_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_grad_."},
{"cosh_grad", (PyCFunction)(void (*)(void))cosh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh_grad."},
{"cosh_grad_", (PyCFunction)(void (*)(void))cosh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh_grad_."},
{"crop_grad", (PyCFunction)(void (*)(void))crop_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop_grad."},
{"cross_entropy_with_softmax_grad", (PyCFunction)(void (*)(void))cross_entropy_with_softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax_grad."},
{"cross_entropy_with_softmax_grad_", (PyCFunction)(void (*)(void))cross_entropy_with_softmax_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax_grad_."},
{"cross_grad", (PyCFunction)(void (*)(void))cross_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_grad."},
{"cudnn_lstm_grad", (PyCFunction)(void (*)(void))cudnn_lstm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cudnn_lstm_grad."},
{"cummax_grad", (PyCFunction)(void (*)(void))cummax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummax_grad."},
{"cummin_grad", (PyCFunction)(void (*)(void))cummin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cummin_grad."},
{"cumprod_grad", (PyCFunction)(void (*)(void))cumprod_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod_grad."},
{"cumsum_grad", (PyCFunction)(void (*)(void))cumsum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum_grad."},
{"cvm_grad", (PyCFunction)(void (*)(void))cvm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cvm_grad."},
{"deformable_conv_grad", (PyCFunction)(void (*)(void))deformable_conv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv_grad."},
{"depthwise_conv2d_grad", (PyCFunction)(void (*)(void))depthwise_conv2d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_grad."},
{"depthwise_conv2d_transpose_grad", (PyCFunction)(void (*)(void))depthwise_conv2d_transpose_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose_grad."},
{"det_grad", (PyCFunction)(void (*)(void))det_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for det_grad."},
{"diag_grad", (PyCFunction)(void (*)(void))diag_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_grad."},
{"diagonal_grad", (PyCFunction)(void (*)(void))diagonal_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diagonal_grad."},
{"digamma_grad", (PyCFunction)(void (*)(void))digamma_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma_grad."},
{"dist_grad", (PyCFunction)(void (*)(void))dist_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist_grad."},
{"dot_grad", (PyCFunction)(void (*)(void))dot_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot_grad."},
{"dropout_grad", (PyCFunction)(void (*)(void))dropout_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout_grad."},
{"eig_grad", (PyCFunction)(void (*)(void))eig_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eig_grad."},
{"eigh_grad", (PyCFunction)(void (*)(void))eigh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh_grad."},
{"eigvalsh_grad", (PyCFunction)(void (*)(void))eigvalsh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvalsh_grad."},
{"elu_grad", (PyCFunction)(void (*)(void))elu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_grad."},
{"elu_grad_", (PyCFunction)(void (*)(void))elu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_grad_."},
{"embedding_with_scaled_gradient_grad", (PyCFunction)(void (*)(void))embedding_with_scaled_gradient_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding_with_scaled_gradient_grad."},
{"erf_grad", (PyCFunction)(void (*)(void))erf_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf_grad."},
{"erfinv_grad", (PyCFunction)(void (*)(void))erfinv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv_grad."},
{"exp_grad", (PyCFunction)(void (*)(void))exp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp_grad."},
{"exp_grad_", (PyCFunction)(void (*)(void))exp_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp_grad_."},
{"expand_as_grad", (PyCFunction)(void (*)(void))expand_as_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as_grad."},
{"expand_grad", (PyCFunction)(void (*)(void))expand_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_grad."},
{"expm1_grad", (PyCFunction)(void (*)(void))expm1_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1_grad."},
{"expm1_grad_", (PyCFunction)(void (*)(void))expm1_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1_grad_."},
{"fake_channel_wise_quantize_dequantize_abs_max_grad", (PyCFunction)(void (*)(void))fake_channel_wise_quantize_dequantize_abs_max_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_dequantize_abs_max_grad."},
{"fake_quantize_dequantize_abs_max_grad", (PyCFunction)(void (*)(void))fake_quantize_dequantize_abs_max_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_abs_max_grad."},
{"fake_quantize_dequantize_moving_average_abs_max_grad", (PyCFunction)(void (*)(void))fake_quantize_dequantize_moving_average_abs_max_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_moving_average_abs_max_grad."},
{"fft_c2c_grad", (PyCFunction)(void (*)(void))fft_c2c_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2c_grad."},
{"fft_c2r_grad", (PyCFunction)(void (*)(void))fft_c2r_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2r_grad."},
{"fft_r2c_grad", (PyCFunction)(void (*)(void))fft_r2c_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_r2c_grad."},
{"fill_diagonal_grad", (PyCFunction)(void (*)(void))fill_diagonal_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_grad."},
{"fill_diagonal_tensor_grad", (PyCFunction)(void (*)(void))fill_diagonal_tensor_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_grad."},
{"fill_diagonal_tensor_grad_", (PyCFunction)(void (*)(void))fill_diagonal_tensor_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_grad_."},
{"fill_grad", (PyCFunction)(void (*)(void))fill_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_grad."},
{"fill_grad_", (PyCFunction)(void (*)(void))fill_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_grad_."},
{"flash_attn_grad", (PyCFunction)(void (*)(void))flash_attn_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_grad."},
{"flash_attn_qkvpacked_grad", (PyCFunction)(void (*)(void))flash_attn_qkvpacked_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_qkvpacked_grad."},
{"flash_attn_unpadded_grad", (PyCFunction)(void (*)(void))flash_attn_unpadded_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_unpadded_grad."},
{"flash_attn_v3_grad", (PyCFunction)(void (*)(void))flash_attn_v3_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_v3_grad."},
{"flash_attn_varlen_qkvpacked_grad", (PyCFunction)(void (*)(void))flash_attn_varlen_qkvpacked_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flash_attn_varlen_qkvpacked_grad."},
{"flashmask_attention_grad", (PyCFunction)(void (*)(void))flashmask_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flashmask_attention_grad."},
{"flatten_grad", (PyCFunction)(void (*)(void))flatten_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_grad."},
{"flatten_grad_", (PyCFunction)(void (*)(void))flatten_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_grad_."},
{"floor_grad", (PyCFunction)(void (*)(void))floor_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_grad."},
{"floor_grad_", (PyCFunction)(void (*)(void))floor_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_grad_."},
{"fmax_grad", (PyCFunction)(void (*)(void))fmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmax_grad."},
{"fmin_grad", (PyCFunction)(void (*)(void))fmin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmin_grad."},
{"fold_grad", (PyCFunction)(void (*)(void))fold_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fold_grad."},
{"fractional_max_pool2d_grad", (PyCFunction)(void (*)(void))fractional_max_pool2d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fractional_max_pool2d_grad."},
{"fractional_max_pool3d_grad", (PyCFunction)(void (*)(void))fractional_max_pool3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fractional_max_pool3d_grad."},
{"frame_grad", (PyCFunction)(void (*)(void))frame_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frame_grad."},
{"frobenius_norm_grad", (PyCFunction)(void (*)(void))frobenius_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frobenius_norm_grad."},
{"fused_batch_norm_act_grad", (PyCFunction)(void (*)(void))fused_batch_norm_act_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_batch_norm_act_grad."},
{"fused_bn_add_activation_grad", (PyCFunction)(void (*)(void))fused_bn_add_activation_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bn_add_activation_grad."},
{"fused_softmax_mask_grad", (PyCFunction)(void (*)(void))fused_softmax_mask_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_grad."},
{"fused_softmax_mask_upper_triangle_grad", (PyCFunction)(void (*)(void))fused_softmax_mask_upper_triangle_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_upper_triangle_grad."},
{"gammaincc_grad", (PyCFunction)(void (*)(void))gammaincc_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaincc_grad."},
{"gammaln_grad", (PyCFunction)(void (*)(void))gammaln_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gammaln_grad."},
{"gather_grad", (PyCFunction)(void (*)(void))gather_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_grad."},
{"gather_nd_grad", (PyCFunction)(void (*)(void))gather_nd_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_nd_grad."},
{"gaussian_inplace_grad", (PyCFunction)(void (*)(void))gaussian_inplace_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace_grad."},
{"gaussian_inplace_grad_", (PyCFunction)(void (*)(void))gaussian_inplace_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_inplace_grad_."},
{"gelu_grad", (PyCFunction)(void (*)(void))gelu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu_grad."},
{"grid_sample_grad", (PyCFunction)(void (*)(void))grid_sample_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sample_grad."},
{"group_norm_grad", (PyCFunction)(void (*)(void))group_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm_grad."},
{"group_norm_grad_", (PyCFunction)(void (*)(void))group_norm_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm_grad_."},
{"gru_grad", (PyCFunction)(void (*)(void))gru_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru_grad."},
{"gru_unit_grad", (PyCFunction)(void (*)(void))gru_unit_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru_unit_grad."},
{"gumbel_softmax_grad", (PyCFunction)(void (*)(void))gumbel_softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gumbel_softmax_grad."},
{"hardshrink_grad", (PyCFunction)(void (*)(void))hardshrink_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardshrink_grad."},
{"hardshrink_grad_", (PyCFunction)(void (*)(void))hardshrink_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardshrink_grad_."},
{"hardsigmoid_grad", (PyCFunction)(void (*)(void))hardsigmoid_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardsigmoid_grad."},
{"hardsigmoid_grad_", (PyCFunction)(void (*)(void))hardsigmoid_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardsigmoid_grad_."},
{"hardtanh_grad", (PyCFunction)(void (*)(void))hardtanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh_grad."},
{"hardtanh_grad_", (PyCFunction)(void (*)(void))hardtanh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardtanh_grad_."},
{"heaviside_grad", (PyCFunction)(void (*)(void))heaviside_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for heaviside_grad."},
{"hinge_loss_grad", (PyCFunction)(void (*)(void))hinge_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hinge_loss_grad."},
{"hsigmoid_loss_grad", (PyCFunction)(void (*)(void))hsigmoid_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hsigmoid_loss_grad."},
{"huber_loss_grad", (PyCFunction)(void (*)(void))huber_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for huber_loss_grad."},
{"i0_grad", (PyCFunction)(void (*)(void))i0_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0_grad."},
{"i0e_grad", (PyCFunction)(void (*)(void))i0e_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i0e_grad."},
{"i1_grad", (PyCFunction)(void (*)(void))i1_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1_grad."},
{"i1e_grad", (PyCFunction)(void (*)(void))i1e_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for i1e_grad."},
{"identity_loss_grad", (PyCFunction)(void (*)(void))identity_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss_grad."},
{"identity_loss_grad_", (PyCFunction)(void (*)(void))identity_loss_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for identity_loss_grad_."},
{"imag_grad", (PyCFunction)(void (*)(void))imag_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag_grad."},
{"index_add_grad", (PyCFunction)(void (*)(void))index_add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add_grad."},
{"index_add_grad_", (PyCFunction)(void (*)(void))index_add_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add_grad_."},
{"index_put_grad", (PyCFunction)(void (*)(void))index_put_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_put_grad."},
{"index_sample_grad", (PyCFunction)(void (*)(void))index_sample_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_sample_grad."},
{"index_select_grad", (PyCFunction)(void (*)(void))index_select_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select_grad."},
{"index_select_strided_grad", (PyCFunction)(void (*)(void))index_select_strided_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select_strided_grad."},
{"instance_norm_grad", (PyCFunction)(void (*)(void))instance_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for instance_norm_grad."},
{"inverse_grad", (PyCFunction)(void (*)(void))inverse_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inverse_grad."},
{"kldiv_loss_grad", (PyCFunction)(void (*)(void))kldiv_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kldiv_loss_grad."},
{"kron_grad", (PyCFunction)(void (*)(void))kron_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kron_grad."},
{"kthvalue_grad", (PyCFunction)(void (*)(void))kthvalue_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kthvalue_grad."},
{"l1_norm_grad", (PyCFunction)(void (*)(void))l1_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for l1_norm_grad."},
{"label_smooth_grad", (PyCFunction)(void (*)(void))label_smooth_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth_grad."},
{"layer_norm_grad", (PyCFunction)(void (*)(void))layer_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm_grad."},
{"leaky_relu_grad", (PyCFunction)(void (*)(void))leaky_relu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_grad."},
{"leaky_relu_grad_", (PyCFunction)(void (*)(void))leaky_relu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_grad_."},
{"lerp_grad", (PyCFunction)(void (*)(void))lerp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp_grad."},
{"lgamma_grad", (PyCFunction)(void (*)(void))lgamma_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma_grad."},
{"linear_interp_grad", (PyCFunction)(void (*)(void))linear_interp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp_grad."},
{"log10_grad", (PyCFunction)(void (*)(void))log10_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10_grad."},
{"log10_grad_", (PyCFunction)(void (*)(void))log10_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10_grad_."},
{"log1p_grad", (PyCFunction)(void (*)(void))log1p_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p_grad."},
{"log1p_grad_", (PyCFunction)(void (*)(void))log1p_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p_grad_."},
{"log2_grad", (PyCFunction)(void (*)(void))log2_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2_grad."},
{"log2_grad_", (PyCFunction)(void (*)(void))log2_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2_grad_."},
{"log_grad", (PyCFunction)(void (*)(void))log_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_grad."},
{"log_grad_", (PyCFunction)(void (*)(void))log_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_grad_."},
{"log_loss_grad", (PyCFunction)(void (*)(void))log_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_loss_grad."},
{"log_softmax_grad", (PyCFunction)(void (*)(void))log_softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_softmax_grad."},
{"logcumsumexp_grad", (PyCFunction)(void (*)(void))logcumsumexp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logcumsumexp_grad."},
{"logit_grad", (PyCFunction)(void (*)(void))logit_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logit_grad."},
{"logsigmoid_grad", (PyCFunction)(void (*)(void))logsigmoid_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid_grad."},
{"logsigmoid_grad_", (PyCFunction)(void (*)(void))logsigmoid_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid_grad_."},
{"logsumexp_grad", (PyCFunction)(void (*)(void))logsumexp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp_grad."},
{"lp_pool2d_grad", (PyCFunction)(void (*)(void))lp_pool2d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lp_pool2d_grad."},
{"lstm_grad", (PyCFunction)(void (*)(void))lstm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm_grad."},
{"lu_grad", (PyCFunction)(void (*)(void))lu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_grad."},
{"lu_grad_", (PyCFunction)(void (*)(void))lu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_grad_."},
{"lu_solve_grad", (PyCFunction)(void (*)(void))lu_solve_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_solve_grad."},
{"lu_unpack_grad", (PyCFunction)(void (*)(void))lu_unpack_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack_grad."},
{"margin_cross_entropy_grad", (PyCFunction)(void (*)(void))margin_cross_entropy_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy_grad."},
{"margin_cross_entropy_grad_", (PyCFunction)(void (*)(void))margin_cross_entropy_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy_grad_."},
{"masked_select_grad", (PyCFunction)(void (*)(void))masked_select_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select_grad."},
{"matrix_power_grad", (PyCFunction)(void (*)(void))matrix_power_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power_grad."},
{"max_grad", (PyCFunction)(void (*)(void))max_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_grad."},
{"max_pool2d_with_index_grad", (PyCFunction)(void (*)(void))max_pool2d_with_index_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index_grad."},
{"max_pool3d_with_index_grad", (PyCFunction)(void (*)(void))max_pool3d_with_index_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index_grad."},
{"maxout_grad", (PyCFunction)(void (*)(void))maxout_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout_grad."},
{"mean_all_grad", (PyCFunction)(void (*)(void))mean_all_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_all_grad."},
{"mean_grad", (PyCFunction)(void (*)(void))mean_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_grad."},
{"memory_efficient_attention_grad", (PyCFunction)(void (*)(void))memory_efficient_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memory_efficient_attention_grad."},
{"meshgrid_grad", (PyCFunction)(void (*)(void))meshgrid_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for meshgrid_grad."},
{"mish_grad", (PyCFunction)(void (*)(void))mish_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish_grad."},
{"mish_grad_", (PyCFunction)(void (*)(void))mish_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish_grad_."},
{"mode_grad", (PyCFunction)(void (*)(void))mode_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode_grad."},
{"multi_dot_grad", (PyCFunction)(void (*)(void))multi_dot_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot_grad."},
{"multiplex_grad", (PyCFunction)(void (*)(void))multiplex_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex_grad."},
{"mv_grad", (PyCFunction)(void (*)(void))mv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv_grad."},
{"nanmedian_grad", (PyCFunction)(void (*)(void))nanmedian_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nanmedian_grad."},
{"nearest_interp_grad", (PyCFunction)(void (*)(void))nearest_interp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp_grad."},
{"nll_loss_grad", (PyCFunction)(void (*)(void))nll_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss_grad."},
{"norm_grad", (PyCFunction)(void (*)(void))norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm_grad."},
{"overlap_add_grad", (PyCFunction)(void (*)(void))overlap_add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add_grad."},
{"p_norm_grad", (PyCFunction)(void (*)(void))p_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm_grad."},
{"pad3d_grad", (PyCFunction)(void (*)(void))pad3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d_grad."},
{"pad_grad", (PyCFunction)(void (*)(void))pad_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad_grad."},
{"pixel_shuffle_grad", (PyCFunction)(void (*)(void))pixel_shuffle_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_shuffle_grad."},
{"pixel_unshuffle_grad", (PyCFunction)(void (*)(void))pixel_unshuffle_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_unshuffle_grad."},
{"poisson_grad", (PyCFunction)(void (*)(void))poisson_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson_grad."},
{"polygamma_grad", (PyCFunction)(void (*)(void))polygamma_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygamma_grad."},
{"pool2d_grad", (PyCFunction)(void (*)(void))pool2d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d_grad."},
{"pool3d_grad", (PyCFunction)(void (*)(void))pool3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d_grad."},
{"pow_grad", (PyCFunction)(void (*)(void))pow_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow_grad."},
{"pow_grad_", (PyCFunction)(void (*)(void))pow_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow_grad_."},
{"prelu_grad", (PyCFunction)(void (*)(void))prelu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu_grad."},
{"prod_grad", (PyCFunction)(void (*)(void))prod_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prod_grad."},
{"psroi_pool_grad", (PyCFunction)(void (*)(void))psroi_pool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for psroi_pool_grad."},
{"put_along_axis_grad", (PyCFunction)(void (*)(void))put_along_axis_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_grad."},
{"qr_grad", (PyCFunction)(void (*)(void))qr_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr_grad."},
{"rank_attention_grad", (PyCFunction)(void (*)(void))rank_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rank_attention_grad."},
{"real_grad", (PyCFunction)(void (*)(void))real_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real_grad."},
{"reciprocal_grad", (PyCFunction)(void (*)(void))reciprocal_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_grad."},
{"reciprocal_grad_", (PyCFunction)(void (*)(void))reciprocal_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_grad_."},
{"reduce_as_grad", (PyCFunction)(void (*)(void))reduce_as_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_as_grad."},
{"relu6_grad", (PyCFunction)(void (*)(void))relu6_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6_grad."},
{"relu6_grad_", (PyCFunction)(void (*)(void))relu6_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6_grad_."},
{"relu_grad", (PyCFunction)(void (*)(void))relu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_grad."},
{"relu_grad_", (PyCFunction)(void (*)(void))relu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_grad_."},
{"renorm_grad", (PyCFunction)(void (*)(void))renorm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm_grad."},
{"repeat_interleave_grad", (PyCFunction)(void (*)(void))repeat_interleave_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave_grad."},
{"repeat_interleave_with_tensor_index_grad", (PyCFunction)(void (*)(void))repeat_interleave_with_tensor_index_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave_with_tensor_index_grad."},
{"reshape_grad", (PyCFunction)(void (*)(void))reshape_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_grad."},
{"reshape_grad_", (PyCFunction)(void (*)(void))reshape_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_grad_."},
{"rms_norm_grad", (PyCFunction)(void (*)(void))rms_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rms_norm_grad."},
{"rnn_grad", (PyCFunction)(void (*)(void))rnn_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rnn_grad."},
{"roi_align_grad", (PyCFunction)(void (*)(void))roi_align_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align_grad."},
{"roi_pool_grad", (PyCFunction)(void (*)(void))roi_pool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool_grad."},
{"roll_grad", (PyCFunction)(void (*)(void))roll_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll_grad."},
{"round_grad", (PyCFunction)(void (*)(void))round_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_grad."},
{"round_grad_", (PyCFunction)(void (*)(void))round_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_grad_."},
{"rrelu_grad", (PyCFunction)(void (*)(void))rrelu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rrelu_grad."},
{"rsqrt_grad", (PyCFunction)(void (*)(void))rsqrt_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt_grad."},
{"rsqrt_grad_", (PyCFunction)(void (*)(void))rsqrt_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt_grad_."},
{"scatter_grad", (PyCFunction)(void (*)(void))scatter_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_grad."},
{"scatter_nd_add_grad", (PyCFunction)(void (*)(void))scatter_nd_add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_nd_add_grad."},
{"segment_pool_grad", (PyCFunction)(void (*)(void))segment_pool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for segment_pool_grad."},
{"selu_grad", (PyCFunction)(void (*)(void))selu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for selu_grad."},
{"send_u_recv_grad", (PyCFunction)(void (*)(void))send_u_recv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_u_recv_grad."},
{"send_ue_recv_grad", (PyCFunction)(void (*)(void))send_ue_recv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_ue_recv_grad."},
{"send_uv_grad", (PyCFunction)(void (*)(void))send_uv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_uv_grad."},
{"sequence_conv_grad", (PyCFunction)(void (*)(void))sequence_conv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_conv_grad."},
{"sequence_pool_grad", (PyCFunction)(void (*)(void))sequence_pool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_pool_grad."},
{"set_value_with_tensor_grad", (PyCFunction)(void (*)(void))set_value_with_tensor_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_with_tensor_grad."},
{"shuffle_channel_grad", (PyCFunction)(void (*)(void))shuffle_channel_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_channel_grad."},
{"sigmoid_cross_entropy_with_logits_grad", (PyCFunction)(void (*)(void))sigmoid_cross_entropy_with_logits_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits_grad."},
{"sigmoid_cross_entropy_with_logits_grad_", (PyCFunction)(void (*)(void))sigmoid_cross_entropy_with_logits_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits_grad_."},
{"sigmoid_grad", (PyCFunction)(void (*)(void))sigmoid_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_grad."},
{"sigmoid_grad_", (PyCFunction)(void (*)(void))sigmoid_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_grad_."},
{"silu_grad", (PyCFunction)(void (*)(void))silu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu_grad."},
{"silu_grad_", (PyCFunction)(void (*)(void))silu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu_grad_."},
{"sin_grad", (PyCFunction)(void (*)(void))sin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin_grad."},
{"sin_grad_", (PyCFunction)(void (*)(void))sin_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin_grad_."},
{"sinh_grad", (PyCFunction)(void (*)(void))sinh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh_grad."},
{"sinh_grad_", (PyCFunction)(void (*)(void))sinh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh_grad_."},
{"slice_grad", (PyCFunction)(void (*)(void))slice_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slice_grad."},
{"slogdet_grad", (PyCFunction)(void (*)(void))slogdet_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdet_grad."},
{"softplus_grad", (PyCFunction)(void (*)(void))softplus_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus_grad."},
{"softplus_grad_", (PyCFunction)(void (*)(void))softplus_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus_grad_."},
{"softshrink_grad", (PyCFunction)(void (*)(void))softshrink_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink_grad."},
{"softshrink_grad_", (PyCFunction)(void (*)(void))softshrink_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink_grad_."},
{"softsign_grad", (PyCFunction)(void (*)(void))softsign_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign_grad."},
{"softsign_grad_", (PyCFunction)(void (*)(void))softsign_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign_grad_."},
{"solve_grad", (PyCFunction)(void (*)(void))solve_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for solve_grad."},
{"spectral_norm_grad", (PyCFunction)(void (*)(void))spectral_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spectral_norm_grad."},
{"split_grad", (PyCFunction)(void (*)(void))split_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split_grad."},
{"sqrt_grad", (PyCFunction)(void (*)(void))sqrt_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_grad."},
{"sqrt_grad_", (PyCFunction)(void (*)(void))sqrt_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_grad_."},
{"square_grad", (PyCFunction)(void (*)(void))square_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square_grad."},
{"square_grad_", (PyCFunction)(void (*)(void))square_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square_grad_."},
{"squared_l2_norm_grad", (PyCFunction)(void (*)(void))squared_l2_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_norm_grad."},
{"squeeze_grad", (PyCFunction)(void (*)(void))squeeze_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_grad."},
{"squeeze_grad_", (PyCFunction)(void (*)(void))squeeze_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_grad_."},
{"stack_grad", (PyCFunction)(void (*)(void))stack_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack_grad."},
{"stanh_grad", (PyCFunction)(void (*)(void))stanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stanh_grad."},
{"strided_slice_grad", (PyCFunction)(void (*)(void))strided_slice_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice_grad."},
{"sum_grad", (PyCFunction)(void (*)(void))sum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum_grad."},
{"svd_grad", (PyCFunction)(void (*)(void))svd_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svd_grad."},
{"svdvals_grad", (PyCFunction)(void (*)(void))svdvals_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svdvals_grad."},
{"swiglu_grad", (PyCFunction)(void (*)(void))swiglu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swiglu_grad."},
{"swish_grad", (PyCFunction)(void (*)(void))swish_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish_grad."},
{"swish_grad_", (PyCFunction)(void (*)(void))swish_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish_grad_."},
{"sync_batch_norm_grad", (PyCFunction)(void (*)(void))sync_batch_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm_grad."},
{"take_along_axis_grad", (PyCFunction)(void (*)(void))take_along_axis_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis_grad."},
{"tan_grad", (PyCFunction)(void (*)(void))tan_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan_grad."},
{"tan_grad_", (PyCFunction)(void (*)(void))tan_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan_grad_."},
{"tanh_grad", (PyCFunction)(void (*)(void))tanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_grad."},
{"tanh_grad_", (PyCFunction)(void (*)(void))tanh_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_grad_."},
{"tanh_shrink_grad", (PyCFunction)(void (*)(void))tanh_shrink_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink_grad."},
{"tanh_shrink_grad_", (PyCFunction)(void (*)(void))tanh_shrink_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink_grad_."},
{"temporal_shift_grad", (PyCFunction)(void (*)(void))temporal_shift_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift_grad."},
{"thresholded_relu_grad", (PyCFunction)(void (*)(void))thresholded_relu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu_grad."},
{"thresholded_relu_grad_", (PyCFunction)(void (*)(void))thresholded_relu_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu_grad_."},
{"topk_grad", (PyCFunction)(void (*)(void))topk_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for topk_grad."},
{"trace_grad", (PyCFunction)(void (*)(void))trace_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace_grad."},
{"trans_layout_grad", (PyCFunction)(void (*)(void))trans_layout_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trans_layout_grad."},
{"transpose_grad", (PyCFunction)(void (*)(void))transpose_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose_grad."},
{"triangular_solve_grad", (PyCFunction)(void (*)(void))triangular_solve_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triangular_solve_grad."},
{"tril_grad", (PyCFunction)(void (*)(void))tril_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_grad."},
{"trilinear_interp_grad", (PyCFunction)(void (*)(void))trilinear_interp_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp_grad."},
{"triu_grad", (PyCFunction)(void (*)(void))triu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_grad."},
{"trunc_grad", (PyCFunction)(void (*)(void))trunc_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc_grad."},
{"unfold_grad", (PyCFunction)(void (*)(void))unfold_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold_grad."},
{"uniform_inplace_grad", (PyCFunction)(void (*)(void))uniform_inplace_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace_grad."},
{"uniform_inplace_grad_", (PyCFunction)(void (*)(void))uniform_inplace_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_inplace_grad_."},
{"unsqueeze_grad", (PyCFunction)(void (*)(void))unsqueeze_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze_grad."},
{"unsqueeze_grad_", (PyCFunction)(void (*)(void))unsqueeze_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze_grad_."},
{"unstack_grad", (PyCFunction)(void (*)(void))unstack_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack_grad."},
{"view_dtype_grad", (PyCFunction)(void (*)(void))view_dtype_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for view_dtype_grad."},
{"warpctc_grad", (PyCFunction)(void (*)(void))warpctc_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warpctc_grad."},
{"warprnnt_grad", (PyCFunction)(void (*)(void))warprnnt_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warprnnt_grad."},
{"weight_only_linear_grad", (PyCFunction)(void (*)(void))weight_only_linear_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for weight_only_linear_grad."},
{"where_grad", (PyCFunction)(void (*)(void))where_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where_grad."},
{"yolo_loss_grad", (PyCFunction)(void (*)(void))yolo_loss_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_loss_grad."},
{"disable_check_model_nan_inf_grad", (PyCFunction)(void (*)(void))disable_check_model_nan_inf_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for disable_check_model_nan_inf_grad."},
{"enable_check_model_nan_inf_grad", (PyCFunction)(void (*)(void))enable_check_model_nan_inf_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for enable_check_model_nan_inf_grad."},
{"im2sequence_grad", (PyCFunction)(void (*)(void))im2sequence_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for im2sequence_grad."},
{"pyramid_hash_grad", (PyCFunction)(void (*)(void))pyramid_hash_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pyramid_hash_grad."},
{"shuffle_batch_grad", (PyCFunction)(void (*)(void))shuffle_batch_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_batch_grad."},
{"sparse_attention_grad", (PyCFunction)(void (*)(void))sparse_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_attention_grad."},
{"stft_grad", (PyCFunction)(void (*)(void))stft_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stft_grad."},
{"unpool3d_grad", (PyCFunction)(void (*)(void))unpool3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d_grad."},
{"unpool_grad", (PyCFunction)(void (*)(void))unpool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool_grad."},
{"fused_rms_norm_grad", (PyCFunction)(void (*)(void))fused_rms_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_rms_norm_grad."},
{"blha_get_max_len", (PyCFunction)(void (*)(void))blha_get_max_len, METH_VARARGS | METH_KEYWORDS, "C++ interface function for blha_get_max_len."},
{"block_multihead_attention_", (PyCFunction)(void (*)(void))block_multihead_attention_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for block_multihead_attention_."},
{"block_multihead_attention_xpu_", (PyCFunction)(void (*)(void))block_multihead_attention_xpu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for block_multihead_attention_xpu_."},
{"distributed_fused_lamb_init", (PyCFunction)(void (*)(void))distributed_fused_lamb_init, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb_init."},
{"distributed_fused_lamb_init_", (PyCFunction)(void (*)(void))distributed_fused_lamb_init_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb_init_."},
{"fc", (PyCFunction)(void (*)(void))fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fc."},
{"fp8_fp8_half_gemm_fused", (PyCFunction)(void (*)(void))fp8_fp8_half_gemm_fused, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fp8_fp8_half_gemm_fused."},
{"fused_bias_act", (PyCFunction)(void (*)(void))fused_bias_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_act."},
{"fused_bias_dropout_residual_layer_norm", (PyCFunction)(void (*)(void))fused_bias_dropout_residual_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_dropout_residual_layer_norm."},
{"fused_bias_residual_layernorm", (PyCFunction)(void (*)(void))fused_bias_residual_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_residual_layernorm."},
{"fused_conv2d_add_act", (PyCFunction)(void (*)(void))fused_conv2d_add_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_conv2d_add_act."},
{"fused_dropout_add", (PyCFunction)(void (*)(void))fused_dropout_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_dropout_add."},
{"fused_embedding_eltwise_layernorm", (PyCFunction)(void (*)(void))fused_embedding_eltwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_eltwise_layernorm."},
{"fused_fc_elementwise_layernorm", (PyCFunction)(void (*)(void))fused_fc_elementwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_fc_elementwise_layernorm."},
{"fused_linear_param_grad_add", (PyCFunction)(void (*)(void))fused_linear_param_grad_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_linear_param_grad_add."},
{"fused_multi_transformer_", (PyCFunction)(void (*)(void))fused_multi_transformer_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer_."},
{"fused_rotary_position_embedding", (PyCFunction)(void (*)(void))fused_rotary_position_embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_rotary_position_embedding."},
{"fusion_gru", (PyCFunction)(void (*)(void))fusion_gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_gru."},
{"fusion_repeated_fc_relu", (PyCFunction)(void (*)(void))fusion_repeated_fc_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_repeated_fc_relu."},
{"fusion_seqconv_eltadd_relu", (PyCFunction)(void (*)(void))fusion_seqconv_eltadd_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqconv_eltadd_relu."},
{"fusion_seqpool_concat", (PyCFunction)(void (*)(void))fusion_seqpool_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqpool_concat."},
{"fusion_squared_mat_sub", (PyCFunction)(void (*)(void))fusion_squared_mat_sub, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_squared_mat_sub."},
{"fusion_transpose_flatten_concat", (PyCFunction)(void (*)(void))fusion_transpose_flatten_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_transpose_flatten_concat."},
{"multihead_matmul", (PyCFunction)(void (*)(void))multihead_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multihead_matmul."},
{"qkv_unpack_mha", (PyCFunction)(void (*)(void))qkv_unpack_mha, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qkv_unpack_mha."},
{"resnet_basic_block", (PyCFunction)(void (*)(void))resnet_basic_block, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_basic_block."},
{"resnet_unit", (PyCFunction)(void (*)(void))resnet_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_unit."},
{"self_dp_attention", (PyCFunction)(void (*)(void))self_dp_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for self_dp_attention."},
{"skip_layernorm", (PyCFunction)(void (*)(void))skip_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for skip_layernorm."},
{"squeeze_excitation_block", (PyCFunction)(void (*)(void))squeeze_excitation_block, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_excitation_block."},
{"variable_length_memory_efficient_attention", (PyCFunction)(void (*)(void))variable_length_memory_efficient_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for variable_length_memory_efficient_attention."},
{"add_group_norm_silu", (PyCFunction)(void (*)(void))add_group_norm_silu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_group_norm_silu."},
{"fused_bias_dropout_residual_layer_norm_grad", (PyCFunction)(void (*)(void))fused_bias_dropout_residual_layer_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_dropout_residual_layer_norm_grad."},
{"fused_dropout_add_grad", (PyCFunction)(void (*)(void))fused_dropout_add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_dropout_add_grad."},
{"fused_rotary_position_embedding_grad", (PyCFunction)(void (*)(void))fused_rotary_position_embedding_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_rotary_position_embedding_grad."},
{"resnet_basic_block_grad", (PyCFunction)(void (*)(void))resnet_basic_block_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_basic_block_grad."},
{"resnet_unit_grad", (PyCFunction)(void (*)(void))resnet_unit_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_unit_grad."},
{"add", (PyCFunction)(void (*)(void))add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add."},
{"add_", (PyCFunction)(void (*)(void))add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_."},
{"add_n", (PyCFunction)(void (*)(void))add_n, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_n."},
{"assign", (PyCFunction)(void (*)(void))assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign."},
{"assign_", (PyCFunction)(void (*)(void))assign_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_."},
{"assign_value", (PyCFunction)(void (*)(void))assign_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value."},
{"batch_norm", (PyCFunction)(void (*)(void))batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm."},
{"batch_norm_", (PyCFunction)(void (*)(void))batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm_."},
{"beam_search_decode", (PyCFunction)(void (*)(void))beam_search_decode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for beam_search_decode."},
{"c_embedding", (PyCFunction)(void (*)(void))c_embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_embedding."},
{"coalesce_tensor_", (PyCFunction)(void (*)(void))coalesce_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor_."},
{"dequantize_linear", (PyCFunction)(void (*)(void))dequantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_linear."},
{"dequantize_linear_", (PyCFunction)(void (*)(void))dequantize_linear_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_linear_."},
{"distribute_fpn_proposals", (PyCFunction)(void (*)(void))distribute_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distribute_fpn_proposals."},
{"divide", (PyCFunction)(void (*)(void))divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide."},
{"divide_", (PyCFunction)(void (*)(void))divide_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide_."},
{"einsum", (PyCFunction)(void (*)(void))einsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for einsum."},
{"elementwise_pow", (PyCFunction)(void (*)(void))elementwise_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow."},
{"embedding", (PyCFunction)(void (*)(void))embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding."},
{"equal", (PyCFunction)(void (*)(void))equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal."},
{"equal_", (PyCFunction)(void (*)(void))equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal_."},
{"floor_divide", (PyCFunction)(void (*)(void))floor_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_divide."},
{"floor_divide_", (PyCFunction)(void (*)(void))floor_divide_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_divide_."},
{"get_tensor_from_selected_rows", (PyCFunction)(void (*)(void))get_tensor_from_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for get_tensor_from_selected_rows."},
{"greater_equal", (PyCFunction)(void (*)(void))greater_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal."},
{"greater_equal_", (PyCFunction)(void (*)(void))greater_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal_."},
{"greater_than", (PyCFunction)(void (*)(void))greater_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than."},
{"greater_than_", (PyCFunction)(void (*)(void))greater_than_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than_."},
{"hardswish", (PyCFunction)(void (*)(void))hardswish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardswish."},
{"hash", (PyCFunction)(void (*)(void))hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hash."},
{"lars_momentum_", (PyCFunction)(void (*)(void))lars_momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lars_momentum_."},
{"less_equal", (PyCFunction)(void (*)(void))less_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal."},
{"less_equal_", (PyCFunction)(void (*)(void))less_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal_."},
{"less_than", (PyCFunction)(void (*)(void))less_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than."},
{"less_than_", (PyCFunction)(void (*)(void))less_than_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than_."},
{"matmul", (PyCFunction)(void (*)(void))matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul."},
{"matmul_with_flatten", (PyCFunction)(void (*)(void))matmul_with_flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul_with_flatten."},
{"maximum", (PyCFunction)(void (*)(void))maximum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maximum."},
{"memcpy", (PyCFunction)(void (*)(void))memcpy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy."},
{"min", (PyCFunction)(void (*)(void))min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for min."},
{"minimum", (PyCFunction)(void (*)(void))minimum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minimum."},
{"multiply", (PyCFunction)(void (*)(void))multiply, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply."},
{"multiply_", (PyCFunction)(void (*)(void))multiply_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply_."},
{"nop", (PyCFunction)(void (*)(void))nop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nop."},
{"nop_", (PyCFunction)(void (*)(void))nop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nop_."},
{"not_equal", (PyCFunction)(void (*)(void))not_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal."},
{"not_equal_", (PyCFunction)(void (*)(void))not_equal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal_."},
{"print", (PyCFunction)(void (*)(void))print, METH_VARARGS | METH_KEYWORDS, "C++ interface function for print."},
{"quantize_linear", (PyCFunction)(void (*)(void))quantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize_linear."},
{"quantize_linear_", (PyCFunction)(void (*)(void))quantize_linear_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize_linear_."},
{"recv_v2", (PyCFunction)(void (*)(void))recv_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for recv_v2."},
{"remainder", (PyCFunction)(void (*)(void))remainder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for remainder."},
{"remainder_", (PyCFunction)(void (*)(void))remainder_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for remainder_."},
{"send_v2", (PyCFunction)(void (*)(void))send_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_v2."},
{"sequence_expand", (PyCFunction)(void (*)(void))sequence_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand."},
{"sequence_softmax", (PyCFunction)(void (*)(void))sequence_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_softmax."},
{"set_value", (PyCFunction)(void (*)(void))set_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value."},
{"set_value_", (PyCFunction)(void (*)(void))set_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_."},
{"share_data_", (PyCFunction)(void (*)(void))share_data_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_data_."},
{"softmax", (PyCFunction)(void (*)(void))softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax."},
{"softmax_", (PyCFunction)(void (*)(void))softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_."},
{"subtract", (PyCFunction)(void (*)(void))subtract, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract."},
{"subtract_", (PyCFunction)(void (*)(void))subtract_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract_."},
{"tile", (PyCFunction)(void (*)(void))tile, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tile."},
{"unique", (PyCFunction)(void (*)(void))unique, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique."},
{"c_softmax_with_multi_label_cross_entropy", (PyCFunction)(void (*)(void))c_softmax_with_multi_label_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_multi_label_cross_entropy."},
{"fused_attention", (PyCFunction)(void (*)(void))fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention."},
{"fused_feedforward", (PyCFunction)(void (*)(void))fused_feedforward, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_feedforward."},
{"moving_average_abs_max_scale", (PyCFunction)(void (*)(void))moving_average_abs_max_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale."},
{"moving_average_abs_max_scale_", (PyCFunction)(void (*)(void))moving_average_abs_max_scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale_."},
{"onednn_to_paddle_layout", (PyCFunction)(void (*)(void))onednn_to_paddle_layout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for onednn_to_paddle_layout."},
{"add_grad", (PyCFunction)(void (*)(void))add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_grad."},
{"add_grad_", (PyCFunction)(void (*)(void))add_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_grad_."},
{"assign_out__grad", (PyCFunction)(void (*)(void))assign_out__grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_out__grad."},
{"assign_out__grad_", (PyCFunction)(void (*)(void))assign_out__grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_out__grad_."},
{"batch_norm_grad", (PyCFunction)(void (*)(void))batch_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm_grad."},
{"c_embedding_grad", (PyCFunction)(void (*)(void))c_embedding_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_embedding_grad."},
{"c_softmax_with_multi_label_cross_entropy_grad", (PyCFunction)(void (*)(void))c_softmax_with_multi_label_cross_entropy_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_multi_label_cross_entropy_grad."},
{"divide_grad", (PyCFunction)(void (*)(void))divide_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide_grad."},
{"einsum_grad", (PyCFunction)(void (*)(void))einsum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for einsum_grad."},
{"elementwise_pow_grad", (PyCFunction)(void (*)(void))elementwise_pow_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow_grad."},
{"embedding_grad", (PyCFunction)(void (*)(void))embedding_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding_grad."},
{"fused_attention_grad", (PyCFunction)(void (*)(void))fused_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention_grad."},
{"fused_feedforward_grad", (PyCFunction)(void (*)(void))fused_feedforward_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_feedforward_grad."},
{"hardswish_grad", (PyCFunction)(void (*)(void))hardswish_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardswish_grad."},
{"hardswish_grad_", (PyCFunction)(void (*)(void))hardswish_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hardswish_grad_."},
{"lod_reset_grad_", (PyCFunction)(void (*)(void))lod_reset_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lod_reset_grad_."},
{"matmul_grad", (PyCFunction)(void (*)(void))matmul_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul_grad."},
{"matmul_with_flatten_grad", (PyCFunction)(void (*)(void))matmul_with_flatten_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul_with_flatten_grad."},
{"maximum_grad", (PyCFunction)(void (*)(void))maximum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maximum_grad."},
{"min_grad", (PyCFunction)(void (*)(void))min_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for min_grad."},
{"minimum_grad", (PyCFunction)(void (*)(void))minimum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minimum_grad."},
{"remainder_grad", (PyCFunction)(void (*)(void))remainder_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for remainder_grad."},
{"sequence_expand_grad", (PyCFunction)(void (*)(void))sequence_expand_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand_grad."},
{"sequence_softmax_grad", (PyCFunction)(void (*)(void))sequence_softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_softmax_grad."},
{"set_value_grad", (PyCFunction)(void (*)(void))set_value_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_grad."},
{"softmax_grad", (PyCFunction)(void (*)(void))softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_grad."},
{"subtract_grad", (PyCFunction)(void (*)(void))subtract_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract_grad."},
{"subtract_grad_", (PyCFunction)(void (*)(void))subtract_grad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract_grad_."},
{"tile_grad", (PyCFunction)(void (*)(void))tile_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tile_grad."},
{"arange", (PyCFunction)(void (*)(void))arange, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arange."},
{"sparse_abs", (PyCFunction)(void (*)(void))sparse_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_abs."},
{"sparse_acos", (PyCFunction)(void (*)(void))sparse_acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_acos."},
{"sparse_acosh", (PyCFunction)(void (*)(void))sparse_acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_acosh."},
{"sparse_add", (PyCFunction)(void (*)(void))sparse_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_add."},
{"sparse_asin", (PyCFunction)(void (*)(void))sparse_asin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_asin."},
{"sparse_asinh", (PyCFunction)(void (*)(void))sparse_asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_asinh."},
{"sparse_atan", (PyCFunction)(void (*)(void))sparse_atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_atan."},
{"sparse_atanh", (PyCFunction)(void (*)(void))sparse_atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_atanh."},
{"sparse_batch_norm_", (PyCFunction)(void (*)(void))sparse_batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_batch_norm_."},
{"sparse_cast", (PyCFunction)(void (*)(void))sparse_cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_cast."},
{"sparse_conv3d", (PyCFunction)(void (*)(void))sparse_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_conv3d."},
{"sparse_conv3d_implicit_gemm", (PyCFunction)(void (*)(void))sparse_conv3d_implicit_gemm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_conv3d_implicit_gemm."},
{"sparse_divide", (PyCFunction)(void (*)(void))sparse_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_divide."},
{"sparse_divide_scalar", (PyCFunction)(void (*)(void))sparse_divide_scalar, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_divide_scalar."},
{"sparse_expm1", (PyCFunction)(void (*)(void))sparse_expm1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_expm1."},
{"sparse_isnan", (PyCFunction)(void (*)(void))sparse_isnan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_isnan."},
{"sparse_leaky_relu", (PyCFunction)(void (*)(void))sparse_leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_leaky_relu."},
{"sparse_log1p", (PyCFunction)(void (*)(void))sparse_log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_log1p."},
{"sparse_multiply", (PyCFunction)(void (*)(void))sparse_multiply, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_multiply."},
{"sparse_pow", (PyCFunction)(void (*)(void))sparse_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_pow."},
{"sparse_relu", (PyCFunction)(void (*)(void))sparse_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_relu."},
{"sparse_relu6", (PyCFunction)(void (*)(void))sparse_relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_relu6."},
{"sparse_reshape", (PyCFunction)(void (*)(void))sparse_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_reshape."},
{"sparse_scale", (PyCFunction)(void (*)(void))sparse_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_scale."},
{"sparse_sin", (PyCFunction)(void (*)(void))sparse_sin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sin."},
{"sparse_sinh", (PyCFunction)(void (*)(void))sparse_sinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sinh."},
{"sparse_softmax", (PyCFunction)(void (*)(void))sparse_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_softmax."},
{"sparse_sparse_coo_tensor", (PyCFunction)(void (*)(void))sparse_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sparse_coo_tensor."},
{"sparse_sqrt", (PyCFunction)(void (*)(void))sparse_sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sqrt."},
{"sparse_square", (PyCFunction)(void (*)(void))sparse_square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_square."},
{"sparse_subtract", (PyCFunction)(void (*)(void))sparse_subtract, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_subtract."},
{"sparse_sum", (PyCFunction)(void (*)(void))sparse_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sum."},
{"sparse_sync_batch_norm_", (PyCFunction)(void (*)(void))sparse_sync_batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sync_batch_norm_."},
{"sparse_tan", (PyCFunction)(void (*)(void))sparse_tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_tan."},
{"sparse_tanh", (PyCFunction)(void (*)(void))sparse_tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_tanh."},
{"sparse_to_dense", (PyCFunction)(void (*)(void))sparse_to_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_dense."},
{"sparse_to_sparse_coo", (PyCFunction)(void (*)(void))sparse_to_sparse_coo, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_sparse_coo."},
{"sparse_to_sparse_csr", (PyCFunction)(void (*)(void))sparse_to_sparse_csr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_sparse_csr."},
{"sparse_transpose", (PyCFunction)(void (*)(void))sparse_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_transpose."},
{"sparse_values", (PyCFunction)(void (*)(void))sparse_values, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_values."},
{"sparse_addmm", (PyCFunction)(void (*)(void))sparse_addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_addmm."},
{"sparse_coalesce", (PyCFunction)(void (*)(void))sparse_coalesce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_coalesce."},
{"sparse_full_like", (PyCFunction)(void (*)(void))sparse_full_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_full_like."},
{"sparse_fused_attention", (PyCFunction)(void (*)(void))sparse_fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_fused_attention."},
{"sparse_indices", (PyCFunction)(void (*)(void))sparse_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_indices."},
{"sparse_mask_as", (PyCFunction)(void (*)(void))sparse_mask_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_mask_as."},
{"sparse_masked_matmul", (PyCFunction)(void (*)(void))sparse_masked_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_masked_matmul."},
{"sparse_matmul", (PyCFunction)(void (*)(void))sparse_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_matmul."},
{"sparse_maxpool", (PyCFunction)(void (*)(void))sparse_maxpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_maxpool."},
{"sparse_mv", (PyCFunction)(void (*)(void))sparse_mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_mv."},
{"sparse_slice", (PyCFunction)(void (*)(void))sparse_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_slice."},
{"sparse_abs_grad", (PyCFunction)(void (*)(void))sparse_abs_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_abs_grad."},
{"sparse_acos_grad", (PyCFunction)(void (*)(void))sparse_acos_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_acos_grad."},
{"sparse_acosh_grad", (PyCFunction)(void (*)(void))sparse_acosh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_acosh_grad."},
{"sparse_add_grad", (PyCFunction)(void (*)(void))sparse_add_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_add_grad."},
{"sparse_addmm_grad", (PyCFunction)(void (*)(void))sparse_addmm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_addmm_grad."},
{"sparse_asin_grad", (PyCFunction)(void (*)(void))sparse_asin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_asin_grad."},
{"sparse_asinh_grad", (PyCFunction)(void (*)(void))sparse_asinh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_asinh_grad."},
{"sparse_atan_grad", (PyCFunction)(void (*)(void))sparse_atan_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_atan_grad."},
{"sparse_atanh_grad", (PyCFunction)(void (*)(void))sparse_atanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_atanh_grad."},
{"sparse_batch_norm_grad", (PyCFunction)(void (*)(void))sparse_batch_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_batch_norm_grad."},
{"sparse_cast_grad", (PyCFunction)(void (*)(void))sparse_cast_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_cast_grad."},
{"sparse_conv3d_grad", (PyCFunction)(void (*)(void))sparse_conv3d_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_conv3d_grad."},
{"sparse_divide_grad", (PyCFunction)(void (*)(void))sparse_divide_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_divide_grad."},
{"sparse_expm1_grad", (PyCFunction)(void (*)(void))sparse_expm1_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_expm1_grad."},
{"sparse_leaky_relu_grad", (PyCFunction)(void (*)(void))sparse_leaky_relu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_leaky_relu_grad."},
{"sparse_log1p_grad", (PyCFunction)(void (*)(void))sparse_log1p_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_log1p_grad."},
{"sparse_mask_as_grad", (PyCFunction)(void (*)(void))sparse_mask_as_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_mask_as_grad."},
{"sparse_masked_matmul_grad", (PyCFunction)(void (*)(void))sparse_masked_matmul_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_masked_matmul_grad."},
{"sparse_matmul_grad", (PyCFunction)(void (*)(void))sparse_matmul_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_matmul_grad."},
{"sparse_maxpool_grad", (PyCFunction)(void (*)(void))sparse_maxpool_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_maxpool_grad."},
{"sparse_mv_grad", (PyCFunction)(void (*)(void))sparse_mv_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_mv_grad."},
{"sparse_pow_grad", (PyCFunction)(void (*)(void))sparse_pow_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_pow_grad."},
{"sparse_relu6_grad", (PyCFunction)(void (*)(void))sparse_relu6_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_relu6_grad."},
{"sparse_relu_grad", (PyCFunction)(void (*)(void))sparse_relu_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_relu_grad."},
{"sparse_reshape_grad", (PyCFunction)(void (*)(void))sparse_reshape_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_reshape_grad."},
{"sparse_sin_grad", (PyCFunction)(void (*)(void))sparse_sin_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sin_grad."},
{"sparse_sinh_grad", (PyCFunction)(void (*)(void))sparse_sinh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sinh_grad."},
{"sparse_softmax_grad", (PyCFunction)(void (*)(void))sparse_softmax_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_softmax_grad."},
{"sparse_sparse_coo_tensor_grad", (PyCFunction)(void (*)(void))sparse_sparse_coo_tensor_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sparse_coo_tensor_grad."},
{"sparse_sqrt_grad", (PyCFunction)(void (*)(void))sparse_sqrt_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sqrt_grad."},
{"sparse_square_grad", (PyCFunction)(void (*)(void))sparse_square_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_square_grad."},
{"sparse_subtract_grad", (PyCFunction)(void (*)(void))sparse_subtract_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_subtract_grad."},
{"sparse_sum_grad", (PyCFunction)(void (*)(void))sparse_sum_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sum_grad."},
{"sparse_sync_batch_norm_grad", (PyCFunction)(void (*)(void))sparse_sync_batch_norm_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sync_batch_norm_grad."},
{"sparse_tan_grad", (PyCFunction)(void (*)(void))sparse_tan_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_tan_grad."},
{"sparse_tanh_grad", (PyCFunction)(void (*)(void))sparse_tanh_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_tanh_grad."},
{"sparse_to_dense_grad", (PyCFunction)(void (*)(void))sparse_to_dense_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_dense_grad."},
{"sparse_to_sparse_coo_grad", (PyCFunction)(void (*)(void))sparse_to_sparse_coo_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_sparse_coo_grad."},
{"sparse_transpose_grad", (PyCFunction)(void (*)(void))sparse_transpose_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_transpose_grad."},
{"sparse_values_grad", (PyCFunction)(void (*)(void))sparse_values_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_values_grad."},
{"sparse_fused_attention_grad", (PyCFunction)(void (*)(void))sparse_fused_attention_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_fused_attention_grad."},
{"sparse_slice_grad", (PyCFunction)(void (*)(void))sparse_slice_grad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_slice_grad."},
{nullptr, nullptr, 0, nullptr}
};

void BindOpsAPI(pybind11::module *module) {
  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {
    PADDLE_THROW(common::errors::Fatal("Add C++ api to core.ops failed!"));
  }
  if (PyModule_AddFunctions(module->ptr(), ManualOpsAPI) < 0) {
    PADDLE_THROW(common::errors::Fatal("Add C++ api to core.ops failed!"));
  }
}

} // namespace pybind

} // namespace paddle


