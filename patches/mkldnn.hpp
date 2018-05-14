// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_HPP
#define MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "mkldnn.h"
#endif

namespace mkldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// A class that provides the destructor for an Intel(R) MKL-DNN C handle
template <typename T>
class handle_traits {};

/// A class for wrapping an Intel(R) MKL-DNN handle. It is used as the base
/// class for primitive (#mkldnn_primitive_t), engine (#mkldnn_engine_t), and
/// stream (#mkldnn_stream_t) handles. An object of the #mkldnn::handle class
/// can be passed by value. This class enables wrapping:
///  - Newly constructed handles.
///    @n In this case, the constructed handle uses reference counting provided
///    by @p std::shared_ptr with a proper deleter function specified through
///    the @p handle_traits class.
///  - Pre-existing handles returned by the Intel(R) MKL-DNN C API (for
///    example, through #mkldnn_primitive_get_output()).
///    @n In this case, an Intel(R) MKL-DNN C API handle is wrapped without a
///    deleter because it is assumed that the handle wrapper for the original
///    object deletes the handle (this model is similar to @p std::weak_ptr).
template <typename T, typename traits = handle_traits<T>>
class handle {
private:
  std::shared_ptr<typename std::remove_pointer<T>::type> _data;
  handle(const handle &&) = delete;
  handle &operator=(const handle &&other) = delete;

protected:
  /// Constructs a C handle wrapper.
  /// @param t The C handle to wrap.
  /// @param weak A flag to specify whether to construct a weak wrapper.
  handle(T t = 0, bool weak = false) : _data(0) { reset(t, weak); }

  bool operator==(const T other) const { return other == _data.get(); }
  bool operator!=(const T other) const { return !(*this == other); }

public:
  handle(const handle &other) : _data(other._data) {}
  handle &operator=(const handle &other) {
    _data = other._data;
    return *this;
  }
  /// Resets the value of a C handle.
  /// @param t The new value of the C handle.
  /// @param weak A flag to specify whether the wrapper should be weak.
  void reset(T t, bool weak = false) {
    auto dummy_destructor = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    _data.reset(t, weak ? dummy_destructor : traits::destructor);
  }

  /// Returns the value of the underlying C handle.
  T get() const { return _data.get(); }

  bool operator==(const handle &other) const {
    return other._data.get() == _data.get();
  }
  bool operator!=(const handle &other) const { return !(*this == other); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct handle_traits<mkldnn_primitive_desc_t> {
  static constexpr auto destructor = &mkldnn_primitive_desc_destroy;
};

template <>
struct handle_traits<mkldnn_primitive_t> {
  static constexpr auto destructor = &mkldnn_primitive_destroy;
};
#endif

/// Base class for all computational primitives.
class primitive : public handle<mkldnn_primitive_t> {
  friend struct error;
  friend struct stream;
  friend class primitive_at;
  using handle::handle;

public:
  /// A proxy to C primitive kind enum
  enum class kind {
    undefined_primitive = mkldnn_undefined_primitive,
    memory = mkldnn_memory,
    view = mkldnn_view,
    reorder = mkldnn_reorder,
    concat = mkldnn_concat,
    concat_inplace = mkldnn_concat_inplace,
    sum = mkldnn_sum,
    convolution = mkldnn_convolution,
    deconvolution = mkldnn_deconvolution,
    eltwise = mkldnn_eltwise,
    relu = mkldnn_relu,
    softmax = mkldnn_softmax,
    pooling = mkldnn_pooling,
    lrn = mkldnn_lrn,
    batch_normalization = mkldnn_batch_normalization,
    inner_product = mkldnn_inner_product,
    convolution_relu = mkldnn_convolution_relu,
    rnn = mkldnn_rnn,
  };

  /// A wrapper structure to specify a particular output of a primitive.
  struct at {
    /// The underlying C API structure.
    mkldnn_primitive_at_t data;
    /// Constructs a wrapper specifying @p aprimitive output with index @p
    /// at.
    ///
    /// @param aprimitive The target primitive.
    /// @param at The output index.

    at(const primitive &aprimitive, size_t at = 0)
        : data(mkldnn_primitive_at(aprimitive.get(), at)) {}
    /// Returns the specified output.
    inline operator primitive() const;
  };

  /// Returns the descriptor of the underlying C API primitive
  inline const_mkldnn_primitive_desc_t get_primitive_desc() const;
  // TODO: use the C++ API wrapper structure.
};

inline mkldnn_primitive_kind_t convert_to_c(primitive::kind akind) {
  return static_cast<mkldnn_primitive_kind_t>(akind);
}

/// Intel(R) MKL-DNN exception class.
///
/// This class captures the status returned by the failed C API function, error
/// message, and, optionally, handle of the primitive that caused the error.
struct error : public std::exception {
  mkldnn_status_t status;
  std::string message;
  primitive error_primitive;

  /// Constructs an error instance.
  ///
  /// @param astatus The error status returned by the C API.
  /// @param amessage The error message.
  /// @param aerror_primitive (optional) A C handle of the primitive that
  ///                         caused the error.

  error(mkldnn_status_t astatus,
        std::string amessage,
        mkldnn_primitive_t aerror_primitive = 0)
      : status(astatus),
        message(amessage),
        error_primitive(aerror_primitive, true) {}

  /// A convenience function for wrapping calls to the C API. Checks the
  /// return status and throws an #error in case of failure.
  ///
  /// @param status The error status returned by the C API.
  /// @param message The error message.
  /// @param error_primitive (optional) A C handle of the primitive that
  ///                        caused the error.

  static void wrap_c_api(mkldnn_status_t status,
                         std::string message,
                         mkldnn_primitive_t *error_primitive = 0) {
    if (status != mkldnn_success) {
      if (nullptr != error_primitive)
        throw error(status, message, *error_primitive);
      else
        throw error(status, message, nullptr);
    }
  }
};

inline primitive::at::operator primitive() const {
  const_mkldnn_primitive_t output;
  error::wrap_c_api(
      mkldnn_primitive_get_output(data.primitive, data.output_index, &output),
      "could not get an output primitive");
  return primitive(const_cast<mkldnn_primitive_t>(output), true);
}

const_mkldnn_primitive_desc_t primitive::get_primitive_desc() const {
  const_mkldnn_primitive_desc_t pd;
  error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &pd),
                    "could not get primitive descriptor by primitive");
  return pd;
}
/// @}

/// @addtogroup cpp_api_enums Common data types and enumerations
/// @{

enum round_mode {
  round_nearest = mkldnn_round_nearest,
  round_down = mkldnn_round_down,
};

inline mkldnn_round_mode_t convert_to_c(round_mode mode) {
  return static_cast<mkldnn_round_mode_t>(mode);
}

enum padding_kind { zero = mkldnn_padding_zero };

inline mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
  return static_cast<mkldnn_padding_kind_t>(kind);
}

enum prop_kind {
  forward_training = mkldnn_forward_training,
  forward_scoring = mkldnn_forward_scoring,
  forward_inference = mkldnn_forward_inference,
  forward = mkldnn_forward,
  backward = mkldnn_backward,
  backward_data = mkldnn_backward_data,
  backward_weights = mkldnn_backward_weights,
  backward_bias = mkldnn_backward_bias
};

inline mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
  return static_cast<mkldnn_prop_kind_t>(kind);
}

enum algorithm {
  algorithm_undef = mkldnn_alg_kind_undef,
  convolution_direct = mkldnn_convolution_direct,
  convolution_winograd = mkldnn_convolution_winograd,
  deconvolution_direct = mkldnn_deconvolution_direct,
  deconvolution_winograd = mkldnn_deconvolution_winograd,
  eltwise_relu = mkldnn_eltwise_relu,
  eltwise_tanh = mkldnn_eltwise_tanh,
  eltwise_elu = mkldnn_eltwise_elu,
  eltwise_square = mkldnn_eltwise_square,
  eltwise_abs = mkldnn_eltwise_abs,
  eltwise_sqrt = mkldnn_eltwise_sqrt,
  eltwise_linear = mkldnn_eltwise_linear,
  eltwise_bounded_relu = mkldnn_eltwise_bounded_relu,
  eltwise_soft_relu = mkldnn_eltwise_soft_relu,
  eltwise_logistic = mkldnn_eltwise_logistic,
  lrn_across_channels = mkldnn_lrn_across_channels,
  lrn_within_channel = mkldnn_lrn_within_channel,
  pooling_max = mkldnn_pooling_max,
  pooling_avg = mkldnn_pooling_avg,
  pooling_avg_include_padding = mkldnn_pooling_avg_include_padding,
  pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding,
  vanilla_rnn = mkldnn_vanilla_rnn,
  vanilla_lstm = mkldnn_vanilla_lstm,
  vanilla_gru = mkldnn_vanilla_gru,
};

inline mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
  return static_cast<mkldnn_alg_kind_t>(aalgorithm);
}

enum batch_normalization_flag {
  use_global_stats = mkldnn_use_global_stats,
  use_scale_shift = mkldnn_use_scaleshift,
  omit_stats = mkldnn_omit_stats,
  fuse_bn_relu = mkldnn_fuse_bn_relu
};

inline mkldnn_batch_normalization_flag_t convert_to_c(
    batch_normalization_flag aflag) {
  return static_cast<mkldnn_batch_normalization_flag_t>(aflag);
}

enum rnn_direction {
  unidirectional_left2right = mkldnn_unidirectional_left2right,
  unidirectional_right2left = mkldnn_unidirectional_right2left,
  unidirectional = mkldnn_unidirectional,
  bidirectional_concat = mkldnn_bidirectional_concat,
  bidirectional_sum = mkldnn_bidirectional_sum,
};

inline mkldnn_rnn_direction_t convert_to_c(rnn_direction adir) {
  return static_cast<mkldnn_rnn_direction_t>(adir);
}

enum query {
  undef = mkldnn_query_undef,

  eengine = mkldnn_query_engine,
  primitive_kind = mkldnn_query_primitive_kind,

  num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32,
  num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32,

  time_estimate_f64 = mkldnn_query_time_estimate_f64,
  memory_consumption_s64 = mkldnn_query_memory_consumption_s64,

  impl_info_str = mkldnn_query_impl_info_str,

  memory_d = mkldnn_query_memory_d,
  convolution_d = mkldnn_query_convolution_d,
  deconvolution_d = mkldnn_query_deconvolution_d,
  eltwise_d = mkldnn_query_eltwise_d,
  relu_d = mkldnn_query_relu_d,
  softmax_d = mkldnn_query_softmax_d,
  pooling_d = mkldnn_query_pooling_d,
  lrn_d = mkldnn_query_lrn_d,
  batch_normalization_d = mkldnn_query_batch_normalization_d,
  inner_product_d = mkldnn_query_inner_product_d,
  convolution_relu_d = mkldnn_query_convolution_relu_d,
  rnn_d = mkldnn_query_rnn_d,

  input_pd = mkldnn_query_input_pd,
  output_pd = mkldnn_query_output_pd,
  src_pd = mkldnn_query_src_pd,
  diff_src_pd = mkldnn_query_diff_src_pd,
  weights_pd = mkldnn_query_weights_pd,
  diff_weights_pd = mkldnn_query_diff_weights_pd,
  dst_pd = mkldnn_query_dst_pd,
  diff_dst_pd = mkldnn_query_diff_dst_pd,
  workspace_pd = mkldnn_query_workspace_pd,
};

inline mkldnn_query_t convert_to_c(query aquery) {
  return static_cast<mkldnn_query_t>(aquery);
}

/// @}

/// @addtogroup cpp_api_attr Attributes
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct handle_traits<mkldnn_post_ops_t> {
  static constexpr auto destructor = &mkldnn_post_ops_destroy;
};
#endif

struct post_ops : public handle<mkldnn_post_ops_t> {
  post_ops() {
    mkldnn_post_ops_t result;
    error::wrap_c_api(mkldnn_post_ops_create(&result),
                      "could not create post operation sequence");
    reset(result);
  }

  int len() const { return mkldnn_post_ops_len(get()); }

  primitive::kind kind(int index) const {
    error::wrap_c_api(index < len() ? mkldnn_success : mkldnn_invalid_arguments,
                      "post_ops index is out of range");
    return static_cast<primitive::kind>(mkldnn_post_ops_get_kind(get(), index));
  }

  void append_sum(float scale = 1.) {
    error::wrap_c_api(mkldnn_post_ops_append_sum(get(), scale),
                      "could not append sum");
  }

  void get_params_sum(int index, float &scale) const {
    error::wrap_c_api(mkldnn_post_ops_get_params_sum(get(), index, &scale),
                      "could not get sum params");
  }

  void append_eltwise(float scale, algorithm alg, float alpha, float beta) {
    error::wrap_c_api(mkldnn_post_ops_append_eltwise(
                          get(), scale, convert_to_c(alg), alpha, beta),
                      "could not append eltwise");
  }

  void get_params_eltwise(int index,
                          float &scale,
                          algorithm &alg,
                          float &alpha,
                          float &beta) const {
    mkldnn_alg_kind_t c_alg;
    error::wrap_c_api(mkldnn_post_ops_get_params_eltwise(
                          get(), index, &scale, &c_alg, &alpha, &beta),
                      "could not get eltwise params");
    alg = static_cast<algorithm>(c_alg);
  }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct handle_traits<mkldnn_primitive_attr_t> {
  static constexpr auto destructor = &mkldnn_primitive_attr_destroy;
};
#endif

struct primitive_attr : public handle<mkldnn_primitive_attr_t> {
  primitive_attr() {
    mkldnn_primitive_attr_t result;
    error::wrap_c_api(mkldnn_primitive_attr_create(&result),
                      "could not create a primitive attr");
    reset(result);
  }

  round_mode get_int_output_round_mode() const {
    mkldnn_round_mode_t result;
    error::wrap_c_api(
        mkldnn_primitive_attr_get_int_output_round_mode(get(), &result),
        "could not get int output round mode");
    return round_mode(result);
  }

  void set_int_output_round_mode(round_mode mode) {
    error::wrap_c_api(mkldnn_primitive_attr_set_int_output_round_mode(
                          get(), mkldnn::convert_to_c(mode)),
                      "could not set int output round mode");
  }

  void get_output_scales(int &mask, std::vector<float> &scales) const {
    int count, c_mask;
    const float *c_scales;
    error::wrap_c_api(mkldnn_primitive_attr_get_output_scales(
                          get(), &count, &c_mask, &c_scales),
                      "could not get int output scales");
    scales.resize(count);

    mask = c_mask;
    for (int c = 0; c < count; ++c) scales[c] = c_scales[c];
  }

  void set_output_scales(int mask, const std::vector<float> &scales) {
    error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(
                          get(), (int)scales.size(), mask, &scales[0]),
                      "could not set int output scales");
  }

  const post_ops get_post_ops() const {
    post_ops result;
    const_mkldnn_post_ops_t c_result;
    error::wrap_c_api(mkldnn_primitive_attr_get_post_ops(get(), &c_result),
                      "could not get post operation sequence");
    result.reset(const_cast<mkldnn_post_ops_t>(c_result), true);
    return result;
  }

  void set_post_ops(post_ops ops) {
    error::wrap_c_api(mkldnn_primitive_attr_set_post_ops(get(), ops.get()),
                      "could not set post operation sequence");
  }
};

/// @}

/// @addtogroup cpp_api_engine Engine
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct handle_traits<mkldnn_engine_t> {
  static constexpr auto destructor = &mkldnn_engine_destroy;
};
#endif

/// An execution engine.
struct engine : public handle<mkldnn_engine_t> {
  friend class primitive;
  // gcc bug??? using handle::handle;

  /// Kinds of engines
  enum kind {
    /// An unspecified engine
    any = mkldnn_any_engine,
    /// CPU engine
    cpu = mkldnn_cpu,
  };

  /// Returns the number of engines of a certain kind.
  ///
  /// @param akind The kind of engines to count.

  static size_t get_count(kind akind) {
    return mkldnn_engine_get_count(convert_to_c(akind));
  }

  /// Constructs an engine.
  ///
  /// @param akind The kind of engine to construct.
  /// @param index The index of the engine. Must be less than the value
  ///              returned by #get_count() for this particular kind of engine.

  engine(kind akind, size_t index) {
    mkldnn_engine_t aengine;
    error::wrap_c_api(
        mkldnn_engine_create(&aengine, convert_to_c(akind), index),
        "could not create an engine");
    reset(aengine);
  }

  explicit engine(const mkldnn_engine_t &aengine) : handle(aengine, true) {}

  engine(const handle<mkldnn_primitive_desc_t> &pd) {
    mkldnn_engine_t engine_q;
    error::wrap_c_api(
        mkldnn_primitive_desc_query(
            pd.get(), mkldnn::convert_to_c(eengine), 0, &engine_q),
        "could not get engine from primitive_desc");
    reset(engine_q, true);
  }

  template <class primitive_desc>
  static engine query(const primitive_desc &pd) {
    mkldnn_engine_t engine_q;
    error::wrap_c_api(
        mkldnn_primitive_desc_query(
            pd.get(), mkldnn::convert_to_c(eengine), 0, &engine_q),
        "could not get engine from primitive_desc");

    return engine(engine_q);
  }

private:
  static mkldnn_engine_kind_t convert_to_c(kind akind) {
    return static_cast<mkldnn_engine_kind_t>(akind);
  }
};

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

/// @addtogroup cpp_api_memory Memory
/// @{

/// Memory primitive that describes the data.
struct memory : public primitive {
private:
  std::shared_ptr<char> _handle;

public:
  typedef std::vector<std::remove_extent<mkldnn_dims_t>::type> dims;

  template <typename T>
  static void validate_dims(std::vector<T> v) {
    if (v.size() > TENSOR_MAX_DIMS)
      throw error(mkldnn_invalid_arguments, "invalid dimensions");
  }

  /// Data type specification. See #mkldnn_data_type_t for a detailed
  /// description.
  enum data_type {
    data_undef = mkldnn_data_type_undef,
    f32 = mkldnn_f32,
    s32 = mkldnn_s32,
    s16 = mkldnn_s16,
    s8 = mkldnn_s8,
    u8 = mkldnn_u8,
  };

  /// Memory format specification. See #mkldnn_memory_format_t
  /// for a detailed description.
  enum format {
    format_undef = mkldnn_format_undef,
    any = mkldnn_any,
    blocked = mkldnn_blocked,
    x = mkldnn_x,
    nc = mkldnn_nc,
    nchw = mkldnn_nchw,
    nhwc = mkldnn_nhwc,
    chwn = mkldnn_chwn,
    nChw8c = mkldnn_nChw8c,
    nChw16c = mkldnn_nChw16c,
    ncdhw = mkldnn_ncdhw,
    ndhwc = mkldnn_ndhwc,
    nCdhw16c = mkldnn_nCdhw16c,
    oi = mkldnn_oi,
    io = mkldnn_io,
    oihw = mkldnn_oihw,
    ihwo = mkldnn_ihwo,
    hwio = mkldnn_hwio,
    oidhw = mkldnn_oidhw,
    OIdhw16i16o = mkldnn_OIdhw16i16o,
    OIdhw16o16i = mkldnn_OIdhw16o16i,
    Oidhw16o = mkldnn_Oidhw16o,
    Odhwi16o = mkldnn_Odhwi16o,
    oIhw8i = mkldnn_oIhw8i,
    oIhw16i = mkldnn_oIhw16i,
    OIhw8i8o = mkldnn_OIhw8i8o,
    OIhw16i16o = mkldnn_OIhw16i16o,
    OIhw8o8i = mkldnn_OIhw8o8i,
    OIhw16o16i = mkldnn_OIhw16o16i,
    IOhw16o16i = mkldnn_IOhw16o16i,
    OIhw8i16o2i = mkldnn_OIhw8i16o2i,
    OIhw8o16i2o = mkldnn_OIhw8o16i2o,
    OIhw4i16o4i = mkldnn_OIhw4i16o4i,
    Oihw8o = mkldnn_Oihw8o,
    Oihw16o = mkldnn_Oihw16o,
    Ohwi8o = mkldnn_Ohwi8o,
    Ohwi16o = mkldnn_Ohwi16o,
    OhIw16o4i = mkldnn_OhIw16o4i,
    goihw = mkldnn_goihw,
    hwigo = mkldnn_hwigo,
    gOIhw8i8o = mkldnn_gOIhw8i8o,
    gOIhw16i16o = mkldnn_gOIhw16i16o,
    gOIhw8i16o2i = mkldnn_gOIhw8i16o2i,
    gOIhw8o16i2o = mkldnn_gOIhw8o16i2o,
    gOIhw4i16o4i = mkldnn_gOIhw4i16o4i,
    gOihw8o = mkldnn_gOihw8o,
    gOihw16o = mkldnn_gOihw16o,
    gOhwi8o = mkldnn_gOhwi8o,
    gOhwi16o = mkldnn_gOhwi16o,
    Goihw8g = mkldnn_Goihw8g,
    Goihw16g = mkldnn_Goihw16g,
    gOIhw8o8i = mkldnn_gOIhw8o8i,
    gOIhw16o16i = mkldnn_gOIhw16o16i,
    gIOhw16o16i = mkldnn_gIOhw16o16i,
    gOhIw16o4i = mkldnn_gOhIw16o4i,
    goidhw = mkldnn_goidhw,
    gOIdhw16i16o = mkldnn_gOIdhw16i16o,
    gOIdhw16o16i = mkldnn_gOIdhw16o16i,
    gOidhw16o = mkldnn_gOidhw16o,
    gOdhwi16o = mkldnn_gOdhwi16o,
    ntc = mkldnn_ntc,
    tnc = mkldnn_tnc,
    ldsnc = mkldnn_ldsnc,
    ldigo = mkldnn_ldigo,
    ldigo_p = mkldnn_ldigo_p,
    ldgoi = mkldnn_ldgoi,
    ldgoi_p = mkldnn_ldgoi_p,
    ldgo = mkldnn_ldgo,
    wino_fmt = mkldnn_wino_fmt,
    format_last = mkldnn_format_last,
  };

  /// A memory descriptor.
  struct desc {
    friend struct memory;
    /// The underlying C API data structure.
    mkldnn_memory_desc_t data;

    /// Constructs a memory descriptor.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param aformat Data layout format.
    desc(dims adims, data_type adata_type, format aformat) {
      validate_dims(adims);
      error::wrap_c_api(
          mkldnn_memory_desc_init(&data,
                                  (int)adims.size(),
                                  adims.size() == 0 ? nullptr : &adims[0],
                                  convert_to_c(adata_type),
                                  convert_to_c(aformat)),
          "could not initialize a memory descriptor");
    }

    /// Constructs a memory descriptor from a C API data structure.
    ///
    /// @param adata A C API #mkldnn_memory_desc_t structure.
    desc(const mkldnn_memory_desc_t &adata) : data(adata) {}
  };

  /// A memory primitive descriptor.
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    friend struct memory;

    // TODO: make private
    primitive_desc() {}

    /// Constructs a memory primitive descriptor.
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_memory_primitive_desc_create(
                            &result, &adesc.data, aengine.get()),
                        "could not initialize a memory primitive descriptor");
      reset(result);
    }

    /// Returns the memory primitive descriptor.
    memory::desc desc() {
      auto memory_d = mkldnn_primitive_desc_query_memory_d(get());
      return memory::desc(*memory_d);
    }

    /// Returns the number of bytes required to allocate the memory described
    /// including the padding area.
    size_t get_size() const {
      return mkldnn_memory_primitive_desc_get_size(get());
    }

    bool operator==(const primitive_desc &other) const {
      return mkldnn_memory_primitive_desc_equal(get(), other.get());
    }

    bool operator!=(const primitive_desc &other) const {
      return !operator==(other);
    }

    engine get_engine() { return engine::query(*this); }
  };

  /// Constructs a memory primitive from a generic primitive.
  ///
  /// @param aprimitive The primitive to treat as memory.
  memory(const primitive &aprimitive) : primitive(aprimitive) {}
  /// Constructs a memory primitive.
  ///
  /// @param adesc Memory primitive descriptor.
  memory(const primitive_desc &adesc) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
        mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
        "could not create a memory primitive");
    reset(result);
    auto _malloc = [](size_t size, int alignment) {
      void *ptr;
#ifdef _WIN32
      ptr = _aligned_malloc(size, alignment);
      int rc = ((ptr) ? 0 : errno);
#else
      int rc = ::posix_memalign(&ptr, alignment, size);
#endif /* _WIN32 */
      return (rc == 0) ? (char *)ptr : nullptr;
    };
    auto _free = [](char *p) {
#ifdef _WIN32
      _aligned_free((void *)p);
#else
      ::free((void *)p);
#endif /* _WIN32 */
    };
    _handle.reset(_malloc(adesc.get_size(), 4096), _free);
    set_data_handle(_handle.get());
  }

  memory(const primitive_desc &adesc, void *ahandle) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
        mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
        "could not create a memory primitive");
    reset(result);
    set_data_handle(ahandle);
  }

  /// Returns the descriptor of the memory primitive.
  primitive_desc get_primitive_desc() const {
    primitive_desc adesc;
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(
        mkldnn_primitive_get_primitive_desc(get(), &cdesc),
        "could not get primitive descriptor from a memory primitive");
    /* FIXME: no const_cast should be here */
    adesc.reset(const_cast<mkldnn_primitive_desc_t>(cdesc), true);
    return adesc;
  }

  /// Returns a handle of the data contained in the memory primitive. On
  /// the CPU engine, this is a pointer to the allocated memory.
  inline void *get_data_handle() const {
    void *handle;
    error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
                      "could not get native handle");
    return handle;
  }

  inline void set_data_handle(void *handle) const {
    error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
                      "could not set native handle");
  }

  // Must go away or be private:
  static mkldnn_data_type_t convert_to_c(data_type adata_type) {
    return static_cast<mkldnn_data_type_t>(adata_type);
  }
  static mkldnn_memory_format_t convert_to_c(format aformat) {
    return static_cast<mkldnn_memory_format_t>(aformat);
  }
};

inline memory::desc zero_md() {
  mkldnn_memory_desc_t zero;
  zero.primitive_kind = mkldnn_memory;
  return memory::desc(zero);
}

inline memory null_memory(engine eng) {
  mkldnn::memory::desc zero = zero_md();
  return memory({zero, eng}, nullptr);
}

inline bool is_null_memory(const const_mkldnn_primitive_t &aprimitive) {
  const_mkldnn_primitive_desc_t aprimitive_pd;
  mkldnn_primitive_get_primitive_desc(aprimitive, &aprimitive_pd);
  const mkldnn_memory_desc_t *aprimitive_md =
      mkldnn_primitive_desc_query_memory_d(aprimitive_pd);

  return ((aprimitive_md != nullptr) && (aprimitive_md->ndims == 0));
}

inline bool operator==(mkldnn_data_type_t a, memory::data_type b) {
  return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_data_type_t a, memory::data_type b) {
  return !(a == b);
}
inline bool operator==(memory::data_type a, mkldnn_data_type_t b) {
  return b == a;
}
inline bool operator!=(memory::data_type a, mkldnn_data_type_t b) {
  return !(a == b);
}

inline bool operator==(mkldnn_memory_format_t a, memory::format b) {
  return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_memory_format_t a, memory::format b) {
  return !(a == b);
}
inline bool operator==(memory::format a, mkldnn_memory_format_t b) {
  return b == a;
}
inline bool operator!=(memory::format a, mkldnn_memory_format_t b) {
  return !(a == b);
}

/// @}

/// @addtogroup cpp_api_reorder Reorder
/// @{

struct reorder : public primitive {
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const memory::primitive_desc &input,
                   const memory::primitive_desc &output) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
                            &result, input.get(), output.get()),
                        "could not create a reorder primitive descriptor");
      reset(result);
    }

    primitive_desc(const memory::primitive_desc &input,
                   const memory::primitive_desc &output,
                   const primitive_attr &aattr) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
                            &result, input.get(), output.get(), aattr.get()),
                        "could not create a reorder primitive descriptor");
      reset(result);
    }

    engine get_engine() { return engine::query(*this); }
  };

  reorder(const primitive_desc &aprimitive_desc,
          const primitive::at &input,
          const memory &output) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {input.data};
    const_mkldnn_primitive_t outputs[] = {output.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a reorder primitive");
    reset(result);
  }

  reorder(const primitive::at &input, const memory &output) {
    auto input_mpd = memory(input).get_primitive_desc();
    auto output_mpd = output.get_primitive_desc();

    auto reorder_d = primitive_desc(input_mpd, output_mpd);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {input.data};
    const_mkldnn_primitive_t outputs[] = {output.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(&result, reorder_d.get(), inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_view View
/// @{

struct view : public primitive {
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const memory::primitive_desc &input,
                   memory::dims dims,
                   memory::dims offsets) {
      mkldnn_primitive_desc_t result;

      error::wrap_c_api(mkldnn_view_primitive_desc_create(
                            &result, input.get(), &dims[0], &offsets[0]),
                        "could not create a view primitive descriptor");
      reset(result);
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  view(const primitive_desc &view_pd, primitive::at input) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {input.data};
    error::wrap_c_api(
        mkldnn_primitive_create(&result, view_pd.get(), inputs, nullptr),
        "could not create a view primitive");
    reset(result);
  }

  view(memory input, memory::dims dims, memory::dims offsets) {
    mkldnn_primitive_t result;
    primitive_desc view_pd(input.get_primitive_desc(), dims, offsets);
    mkldnn_primitive_at_t inputs[] = {primitive::at(input).data};
    error::wrap_c_api(
        mkldnn_primitive_create(&result, view_pd.get(), inputs, nullptr),
        "could not create a view primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_concat Concat
/// @{

struct concat : public primitive {
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
        std::vector<memory::primitive_desc> inputs) {
      std::vector<const_mkldnn_primitive_desc_t> c_api_inputs;
      c_api_inputs.reserve(inputs.size());
      auto convert_to_c = [](memory::primitive_desc d) { return d.get(); };
      std::transform(inputs.begin(),
                     inputs.end(),
                     std::back_inserter(c_api_inputs),
                     convert_to_c);
      return c_api_inputs;
    }

    primitive_desc(const memory::desc &output,
                   int concat_dimension,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);

      error::wrap_c_api(
          mkldnn_concat_primitive_desc_create(&result,
                                              &output.data,
                                              (int)c_api_inputs.size(),
                                              concat_dimension,
                                              &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }

    primitive_desc(int concat_dimension,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);

      error::wrap_c_api(
          mkldnn_concat_primitive_desc_create(&result,
                                              nullptr,
                                              (int)c_api_inputs.size(),
                                              concat_dimension,
                                              &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  concat(const primitive_desc &concat_pd,
         std::vector<primitive::at> &inputs,
         const memory &output) {
    mkldnn_primitive_t result;

    std::vector<mkldnn_primitive_at_t> p_inputs;
    for (size_t i = 0; i < inputs.size(); i++)
      p_inputs.push_back(inputs[i].data);
    const_mkldnn_primitive_t outputs[] = {output.get()};

    error::wrap_c_api(mkldnn_primitive_create(
                          &result, concat_pd.get(), &p_inputs[0], outputs),
                      "could not create a concat primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_sum Sum
/// @{

struct sum : public primitive {
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
        std::vector<memory::primitive_desc> inputs) {
      std::vector<const_mkldnn_primitive_desc_t> c_api_inputs;
      c_api_inputs.reserve(inputs.size());
      auto convert_to_c = [](memory::primitive_desc d) { return d.get(); };
      std::transform(inputs.begin(),
                     inputs.end(),
                     std::back_inserter(c_api_inputs),
                     convert_to_c);
      return c_api_inputs;
    }

    primitive_desc(const memory::desc &output,
                   const std::vector<float> &scales,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);

      error::wrap_c_api(
          mkldnn_sum_primitive_desc_create(&result,
                                           &output.data,
                                           (int)c_api_inputs.size(),
                                           &scales[0],
                                           &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    primitive_desc(const std::vector<float> &scales,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);

      error::wrap_c_api(
          mkldnn_sum_primitive_desc_create(&result,
                                           nullptr,
                                           (int)c_api_inputs.size(),
                                           &scales[0],
                                           &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    /** @deprecated: api backwards compatibility for double scales type */
    MKLDNN_DEPRECATED
    primitive_desc(const memory::desc &output,
                   std::vector<double> scale,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);
      auto scale_f = scale_to_float(scale);

      error::wrap_c_api(
          mkldnn_sum_primitive_desc_create(&result,
                                           &output.data,
                                           (int)c_api_inputs.size(),
                                           &scale_f[0],
                                           &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    /** @deprecated: api backwards compatibility for double scales type */
    MKLDNN_DEPRECATED
    primitive_desc(std::vector<double> scale,
                   std::vector<memory::primitive_desc> inputs) {
      mkldnn_primitive_desc_t result;

      auto c_api_inputs = cpp_to_c(inputs);
      auto scale_f = scale_to_float(scale);

      error::wrap_c_api(
          mkldnn_sum_primitive_desc_create(&result,
                                           nullptr,
                                           (int)c_api_inputs.size(),
                                           &scale_f[0],
                                           &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  sum(const primitive_desc &sum_pd,
      std::vector<primitive::at> &inputs,
      const memory &output) {
    mkldnn_primitive_t result;

    std::vector<mkldnn_primitive_at_t> p_inputs;
    for (size_t i = 0; i < inputs.size(); i++)
      p_inputs.push_back(inputs[i].data);
    const_mkldnn_primitive_t outputs[] = {output.get()};

    error::wrap_c_api(
        mkldnn_primitive_create(&result, sum_pd.get(), &p_inputs[0], outputs),
        "could not create a sum primitive");
    reset(result);
  }

private:
  static std::vector<float> scale_to_float(const std::vector<double> &vd) {
    std::vector<float> vf(vd.size());
    std::transform(
        vd.begin(), vd.end(), vf.begin(), [=](double x) { return (float)x; });
    return vf;
  }
};

/// @}

/// @addtogroup cpp_api_convolution Convolution
/// @{

struct convolution_forward : public primitive {
  struct desc {
    mkldnn_convolution_desc_t data;
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(mkldnn_convolution_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm),
                            &src_desc.data,
                            &weights_desc.data,
                            &bias_desc.data,
                            &dst_desc.data,
                            &strides[0],
                            &padding_l[0],
                            &padding_r[0],
                            mkldnn::convert_to_c(apadding_kind)),
                        "could not create a convolution forward descriptor");
    }
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(mkldnn_convolution_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm),
                            &src_desc.data,
                            &weights_desc.data,
                            nullptr,
                            &dst_desc.data,
                            &strides[0],
                            &padding_l[0],
                            &padding_r[0],
                            mkldnn::convert_to_c(apadding_kind)),
                        "could not create a convolution forward descriptor");
    }
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims dilates,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(dilates);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_dilated_convolution_forward_desc_init(
              &data,
              mkldnn::convert_to_c(aprop_kind),
              convert_to_c(aalgorithm),
              &src_desc.data,
              &weights_desc.data,
              &bias_desc.data,
              &dst_desc.data,
              &strides[0],
              &dilates[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a dilated convolution forward descriptor");
    }
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims dilates,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(dilates);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_dilated_convolution_forward_desc_init(
              &data,
              mkldnn::convert_to_c(aprop_kind),
              convert_to_c(aalgorithm),
              &src_desc.data,
              &weights_desc.data,
              nullptr,
              &dst_desc.data,
              &strides[0],
              &dilates[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a dilated convolution forward descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
    }

    primitive_desc(const desc &adesc,
                   const primitive_attr &aattr,
                   const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create_v2(
              &result, &adesc.data, aattr.get(), aengine.get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  convolution_forward(const primitive_desc &aprimitive_desc,
                      const primitive::at &src,
                      const primitive::at &weights,
                      const primitive::at &bias,
                      const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data, bias.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a convolution forward bias primitive");
    reset(result);
  }

  convolution_forward(const primitive_desc &aprimitive_desc,
                      const primitive::at &src,
                      const primitive::at &weights,
                      const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a convolution forward primitive");
    reset(result);
  }
};

struct convolution_backward_data : public primitive {
  struct desc {
    mkldnn_convolution_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &diff_src_desc,
         const memory::desc &weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_convolution_backward_data_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &diff_src_desc.data,
              &weights_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &diff_src_desc,
         const memory::desc &weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims dilates,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(dilates);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_dilated_convolution_backward_data_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &diff_src_desc.data,
              &weights_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &dilates[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const convolution_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a convolution backward data primitive descriptor");
      reset(result);
    }
    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  convolution_backward_data(const primitive_desc &aprimitive_desc,
                            const primitive::at &diff_dst,
                            const primitive::at &weights,
                            const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {diff_dst.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a convolution backward data primitive");
    reset(result);
  }
};

struct convolution_backward_weights : public primitive {
  struct desc {
    mkldnn_convolution_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_bias_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_convolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              &diff_bias_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_convolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              nullptr,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_bias_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims dilates,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(dilates);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_dilated_convolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              &diff_bias_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &dilates[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims dilates,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(dilates);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_dilated_convolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              nullptr,
              &diff_dst_desc.data,
              &strides[0],
              &dilates[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const convolution_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a convolution backward weights primitive "
          "descriptor");
      reset(result);
    }
    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  convolution_backward_weights(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &diff_dst,
                               const memory &diff_weights,
                               const memory &diff_bias) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get(), diff_bias.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a convolution backward weights primitive");
    reset(result);
  }
  convolution_backward_weights(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &diff_dst,
                               const memory &diff_weights) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a convolution backward weights primitive");
    reset(result);
  }
};

struct convolution_relu_forward : public primitive {
  struct desc {
    mkldnn_convolution_relu_desc_t data;
    desc(const convolution_forward::desc conv_desc,
         const float negative_slope) {
      error::wrap_c_api(
          mkldnn_convolution_relu_desc_init(
              &data, &conv_desc.data, negative_slope),
          "could not create a convolution_relu_forward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a convolution relu forward descriptor");
      reset(result);
    }

    engine get_engine() { return engine::query(*this); }
  };

  convolution_relu_forward(const primitive_desc &aprimitive_desc,
                           const primitive::at &src,
                           const primitive::at &weights,
                           const primitive::at &bias,
                           const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data, bias.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a convolution relu forward primitive");
    reset(result);
  }

  convolution_relu_forward(const primitive_desc &aprimitive_desc,
                           const primitive::at &src,
                           const primitive::at &weights,
                           const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a convolution relu forward primitive");
    reset(result);
  }
};

/// @}
//
/// @addtogroup cpp_api_deconvolution Deconvolution
/// @{

struct deconvolution_forward : public primitive {
  struct desc {
    mkldnn_deconvolution_desc_t data;
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm),
                            &src_desc.data,
                            &weights_desc.data,
                            &bias_desc.data,
                            &dst_desc.data,
                            &strides[0],
                            &padding_l[0],
                            &padding_r[0],
                            mkldnn::convert_to_c(apadding_kind)),
                        "could not create a deconvolution forward descriptor");
    }
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm),
                            &src_desc.data,
                            &weights_desc.data,
                            nullptr,
                            &dst_desc.data,
                            &strides[0],
                            &padding_l[0],
                            &padding_r[0],
                            mkldnn::convert_to_c(apadding_kind)),
                        "could not create a deconvolution forward descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a deconvolution forward primitive descriptor");
      reset(result);
    }

    primitive_desc(const desc &adesc,
                   const primitive_attr &aattr,
                   const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create_v2(
              &result, &adesc.data, aattr.get(), aengine.get(), nullptr),
          "could not create a deconvolution forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  deconvolution_forward(const primitive_desc &aprimitive_desc,
                        const primitive::at &src,
                        const primitive::at &weights,
                        const primitive::at &bias,
                        const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data, bias.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a deconvolution forward bias primitive");
    reset(result);
  }

  deconvolution_forward(const primitive_desc &aprimitive_desc,
                        const primitive::at &src,
                        const primitive::at &weights,
                        const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a deconvolution forward primitive");
    reset(result);
  }
};

struct deconvolution_backward_data : public primitive {
  struct desc {
    mkldnn_deconvolution_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &diff_src_desc,
         const memory::desc &weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_deconvolution_backward_data_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &diff_src_desc.data,
              &weights_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward data descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const deconvolution_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a deconvolution backward data primitive "
          "descriptor");
      reset(result);
    }
    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  deconvolution_backward_data(const primitive_desc &aprimitive_desc,
                              const primitive::at &diff_dst,
                              const primitive::at &weights,
                              const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {diff_dst.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a deconvolution backward data primitive");
    reset(result);
  }
};

struct deconvolution_backward_weights : public primitive {
  struct desc {
    mkldnn_deconvolution_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_bias_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_deconvolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              &diff_bias_desc.data,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward weights descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims strides,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_deconvolution_backward_weights_desc_init(
              &data,
              convert_to_c(aalgorithm),
              &src_desc.data,
              &diff_weights_desc.data,
              nullptr,
              &diff_dst_desc.data,
              &strides[0],
              &padding_l[0],
              &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward weights descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const deconvolution_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a deconvolution backward weights primitive "
          "descriptor");
      reset(result);
    }
    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  deconvolution_backward_weights(const primitive_desc &aprimitive_desc,
                                 const primitive::at &src,
                                 const primitive::at &diff_dst,
                                 const memory &diff_weights,
                                 const memory &diff_bias) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get(), diff_bias.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a deconvolution backward weights primitive");
    reset(result);
  }
  deconvolution_backward_weights(const primitive_desc &aprimitive_desc,
                                 const primitive::at &src,
                                 const primitive::at &diff_dst,
                                 const memory &diff_weights) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a deconvolution backward weights primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_lrn LRN
/// @{

struct lrn_forward : public primitive {
  struct desc {
    mkldnn_lrn_desc_t data;
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         int local_size,
         float alpha,
         float beta,
         float k) {
      error::wrap_c_api(
          mkldnn_lrn_forward_desc_init(&data,
                                       mkldnn::convert_to_c(aprop_kind),
                                       convert_to_c(aalgorithm),
                                       &src_desc.data,
                                       local_size,
                                       alpha,
                                       beta,
                                       k),
          "could not create a lrn forward descriptor");
    }
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         int local_size,
         float alpha,
         float beta) {
      error::wrap_c_api(
          mkldnn_lrn_forward_desc_init(&data,
                                       mkldnn::convert_to_c(aprop_kind),
                                       convert_to_c(aalgorithm),
                                       &src_desc.data,
                                       local_size,
                                       alpha,
                                       beta,
                                       float(1.0)),
          "could not create a lrn forward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                            &result, &adesc.data, aengine.get(), nullptr),
                        "could not create a lrn forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t ldesc;
      const_mkldnn_primitive_desc_t const_ldesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(ldesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  lrn_forward(const primitive_desc &aprimitive_desc,
              const primitive::at &src,
              const memory &workspace,
              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get(), workspace.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a lrn forward primitive");
    reset(result);
  }

  lrn_forward(const primitive_desc &aprimitive_desc,
              const primitive::at &src,
              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a lrn forward primitive");
    reset(result);
  }
};

struct lrn_backward : public primitive {
  struct desc {
    mkldnn_lrn_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &data_desc,
         const memory::desc &diff_data_desc,
         int local_size,
         float alpha,
         float beta,
         float k) {
      error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                                                      convert_to_c(aalgorithm),
                                                      &diff_data_desc.data,
                                                      &data_desc.data,
                                                      local_size,
                                                      alpha,
                                                      beta,
                                                      k),
                        "could not create a lrn backward descriptor");
    }
    desc(algorithm aalgorithm,
         const memory::desc &data_desc,
         const memory::desc &diff_data_desc,
         int local_size,
         float alpha,
         float beta) {
      error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                                                      convert_to_c(aalgorithm),
                                                      &diff_data_desc.data,
                                                      &data_desc.data,
                                                      local_size,
                                                      alpha,
                                                      beta,
                                                      float(1.0)),
                        "could not create a lrn backward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc,
                   const engine &aengine,
                   const lrn_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a backward lrn primitive descriptor");
      reset(result);
    }

    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t ldesc;
      const_mkldnn_primitive_desc_t const_ldesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(ldesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff_dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  lrn_backward(const primitive_desc &aprimitive_desc,
               const primitive::at &src,
               const primitive::at &diff_dst,
               const primitive::at &workspace,
               const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data, workspace.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a lrn backward primitive");
    reset(result);
  }

  lrn_backward(const primitive_desc &aprimitive_desc,
               const primitive::at &src,
               const primitive::at &diff_dst,
               const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a lrn backward primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_pooling Pooling
/// @{

struct pooling_forward : public primitive {
  struct desc {
    mkldnn_pooling_desc_t data;
    desc(prop_kind aprop_kind,
         algorithm aalgorithm,
         const memory::desc &src_desc,
         const memory::desc &dst_desc,
         const memory::dims strides,
         const memory::dims kernel,
         const memory::dims padding_l,
         const memory::dims padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(kernel);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(
          mkldnn_pooling_forward_desc_init(&data,
                                           mkldnn::convert_to_c(aprop_kind),
                                           convert_to_c(aalgorithm),
                                           &src_desc.data,
                                           &dst_desc.data,
                                           &strides[0],
                                           &kernel[0],
                                           &padding_l[0],
                                           &padding_r[0],
                                           mkldnn::convert_to_c(apadding_kind)),
          "could not init a forward pooling descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a forward pooling primitive descriptor");
      reset(result);
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a workspace primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  pooling_forward(const primitive_desc &aprimitive_desc,
                  const primitive::at &src,
                  const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get(), nullptr};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a pooling forward primitive");
    reset(result);
  }

  pooling_forward(const primitive_desc &aprimitive_desc,
                  const primitive::at &src,
                  const memory &dst,
                  const memory &workspace) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get(), workspace.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a pooling forward primitive");
    reset(result);
  }
};

struct pooling_backward : public primitive {
  struct desc {
    mkldnn_pooling_desc_t data;
    desc(algorithm aalgorithm,
         const memory::desc &diff_src_desc,
         const memory::desc &diff_dst_desc,
         const memory::dims &strides,
         const memory::dims &kernel,
         const memory::dims &padding_l,
         const memory::dims &padding_r,
         const padding_kind apadding_kind) {
      memory::validate_dims(strides);
      memory::validate_dims(kernel);
      memory::validate_dims(padding_l);
      memory::validate_dims(padding_r);
      error::wrap_c_api(mkldnn_pooling_backward_desc_init(
                            &data,
                            convert_to_c(aalgorithm),
                            &diff_src_desc.data,
                            &diff_dst_desc.data,
                            &strides[0],
                            &kernel[0],
                            &padding_l[0],
                            &padding_r[0],
                            mkldnn::convert_to_c(apadding_kind)),
                        "could not init a backward pooling descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const pooling_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a backward pooling primitive descriptor");
      reset(result);
    }

    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  pooling_backward(const primitive_desc &aprimitive_desc,
                   const primitive::at &diff_dst,
                   const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a pooling backward primitive");
    reset(result);
  }

  pooling_backward(const primitive_desc &aprimitive_desc,
                   const primitive::at &diff_dst,
                   const primitive::at &workspace,
                   const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {diff_dst.data, workspace.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a pooling backward primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_eltwise Eltwise
/// @{

struct eltwise_forward : public primitive {
  struct desc {
    mkldnn_eltwise_desc_t data;
    template <typename T>
    desc(prop_kind aprop_kind,
         algorithm alg_kind,
         const memory::desc &src_desc,
         T alpha = 0,
         T beta = 0) {
      error::wrap_c_api(
          mkldnn_eltwise_forward_desc_init(&data,
                                           mkldnn::convert_to_c(aprop_kind),
                                           mkldnn::convert_to_c(alg_kind),
                                           &src_desc.data,
                                           static_cast<float>(alpha),
                                           static_cast<float>(beta)),
          "could not create a eltwise forward descriptor");
    }

    /** @deprecated: api backward compatibility for relu */
    template <typename T>
    MKLDNN_DEPRECATED desc(prop_kind aprop_kind,
                           const memory::desc &src_desc,
                           T negative_slope)
        : desc(aprop_kind, eltwise_relu, src_desc, negative_slope) {}
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a eltwise forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  eltwise_forward(const primitive_desc &aprimitive_desc,
                  const primitive::at &src,
                  const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a eltwise forward primitive");
    reset(result);
  }
};

typedef eltwise_forward relu_forward;

struct eltwise_backward : public primitive {
  struct desc {
    mkldnn_eltwise_desc_t data;

    template <typename T>
    desc(algorithm alg_kind,
         const memory::desc &diff_data_desc,
         const memory::desc &data_desc,
         T alpha = 0,
         T beta = 0) {
      error::wrap_c_api(
          mkldnn_eltwise_backward_desc_init(&data,
                                            mkldnn::convert_to_c(alg_kind),
                                            &diff_data_desc.data,
                                            &data_desc.data,
                                            static_cast<float>(alpha),
                                            static_cast<float>(beta)),
          "could not create a eltwise backward descriptor");
    }

    /** @deprecated: api backward compatibility for relu */
    template <typename T>
    MKLDNN_DEPRECATED desc(const memory::desc &diff_data_desc,
                           const memory::desc &data_desc,
                           T negative_slope)
        : desc(eltwise_relu, diff_data_desc, data_desc, negative_slope) {}
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const eltwise_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a eltwise backward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  eltwise_backward(const primitive_desc &aprimitive_desc,
                   const primitive::at &src,
                   const primitive::at &diff_dst,
                   const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a eltwise backward primitive");
    reset(result);
  }
};

typedef eltwise_backward relu_backward;

/// @}

/// @addtogroup cpp_api_softmax Softmax
/// @{

struct softmax_forward : public primitive {
  struct desc {
    mkldnn_softmax_desc_t data;
    desc(prop_kind aprop_kind,
         const memory::desc &data_desc,
         int softmax_axis) {
      error::wrap_c_api(
          mkldnn_softmax_forward_desc_init(&data,
                                           mkldnn::convert_to_c(aprop_kind),
                                           &data_desc.data,
                                           softmax_axis),
          "could not create a softmax forward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a softmax forward primitive descriptor");
      reset(result);
    }

    engine get_engine() { return engine::query(*this); }
  };

  softmax_forward(const primitive_desc &aprimitive_desc,
                  const primitive::at &src,
                  const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a softmax forward primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_batch_norm Batch normalization
/// @{

struct batch_normalization_forward : public primitive {
  struct desc {
    mkldnn_batch_normalization_desc_t data;
    template <typename T>
    desc(prop_kind aprop_kind,
         const memory::desc &src_desc,
         T epsilon,
         unsigned flags) {
      error::wrap_c_api(
          mkldnn_batch_normalization_forward_desc_init(
              &data,
              mkldnn::convert_to_c(aprop_kind),
              &src_desc.data,
              static_cast<float>(epsilon),
              flags),
          "could not create a batch normalization forward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                            &result, &adesc.data, aengine.get(), nullptr),
                        "could not create a batch normalization forward "
                        "primitive descriptor");
      reset(result);
    }

    primitive_desc(const desc &adesc,
                   const primitive_attr &aattr,
                   const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create_v2(
              &result, &adesc.data, aattr.get(), aengine.get(), nullptr),
          "could not create a batch normalization forward "
          "primitive descriptor");
      reset(result);
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t bndesc;
      const_mkldnn_primitive_desc_t const_bndesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(bndesc);
      return adesc;
    }

    memory::primitive_desc mean_primitive_desc() const {
      memory::primitive_desc aprimitive_desc;
      mkldnn_primitive_desc_t bndesc;
      mkldnn_batch_normalization_desc_t *p;
      error::wrap_c_api(
          mkldnn_primitive_desc_query(
              get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
          "could not get a batch-normalization descriptor");
      const_mkldnn_primitive_desc_t const_bndesc =
          (p->flags & use_global_stats)
              ? mkldnn_primitive_desc_query_pd(
                    get(), mkldnn::convert_to_c(src_pd), 1)
              : mkldnn_primitive_desc_query_pd(
                    get(), mkldnn::convert_to_c(dst_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a mean primitive descriptor");
      aprimitive_desc.reset(bndesc);
      return aprimitive_desc;
    }

    memory::primitive_desc variance_primitive_desc() const {
      memory::primitive_desc aprimitive_desc;
      mkldnn_primitive_desc_t bndesc;
      mkldnn_batch_normalization_desc_t *p;
      error::wrap_c_api(
          mkldnn_primitive_desc_query(
              get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
          "could not get a batch-normalization descriptor");
      const_mkldnn_primitive_desc_t const_bndesc =
          (p->flags & use_global_stats)
              ? mkldnn_primitive_desc_query_pd(
                    get(), mkldnn::convert_to_c(src_pd), 2)
              : mkldnn_primitive_desc_query_pd(
                    get(), mkldnn::convert_to_c(dst_pd), 2);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a variance primitive descriptor");
      aprimitive_desc.reset(bndesc);
      return aprimitive_desc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const primitive::at &mean,
                              const primitive::at &variance,
                              const primitive::at &weights,
                              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {
        src.data, mean.data, variance.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const primitive::at &mean,
                              const primitive::at &variance,
                              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, mean.data, variance.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  /// @warning batch_normalization_forward has 2 constructors with very
  ///          similar signatures:
  ///           - (pd, src, weights, dst, mean, variance) // 2 in, 3 out
  ///           - (pd, src, dst, mean, variance, workspace) // 1 in, 4 out
  ///          The only way to distinguish between those is to explicitly
  ///          cast all input parameters to their type, i.e. to
  ///          const primitive:at &.
  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const primitive::at &weights,
                              const memory &dst,
                              const memory &mean,
                              const memory &variance) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {
        dst.get(), mean.get(), variance.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const primitive::at &weights,
                              const memory &dst,
                              const memory &mean,
                              const memory &variance,
                              const memory &workspace) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {
        dst.get(), mean.get(), variance.get(), workspace.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const memory &dst,
                              const memory &mean,
                              const memory &variance) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {
        dst.get(), mean.get(), variance.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  /// @warning batch_normalization_forward has 2 constructors with very
  ///          similar signatures:
  ///           - (pd, src, weights, dst, mean, variance) // 2 in, 3 out
  ///           - (pd, src, dst, mean, variance, workspace) // 1 in, 4 out
  ///          The only way to distinguish between those is to explicitly
  ///          cast all input parameters to their type, i.e. to
  ///          const primitive:at &.
  /// @note to make users' experience a little bit better this constructor
  ///       checks if whether parameters match corresponding primitive
  ///       descriptor, and if they are not -- call the other (proper)
  ///       constructor. Yeah, this is still very ugly...
  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const memory &dst,
                              const memory &mean,
                              const memory &variance,
                              const memory &workspace) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[2] = {src.data};
    const_mkldnn_primitive_t outputs[4] = {
        dst.get(), mean.get(), variance.get(), workspace.get()};

    if (1) {  // check whether this is the `wrong` constructor
      const int n_inputs_expected = mkldnn_primitive_desc_query_s32(
          aprimitive_desc.get(), mkldnn_query_num_of_inputs_s32, 0);
      const int n_outputs_expected = mkldnn_primitive_desc_query_s32(
          aprimitive_desc.get(), mkldnn_query_num_of_outputs_s32, 0);
      if (n_inputs_expected == 2 && n_outputs_expected == 3) {
        // shift parameters, get rid of workspace, and add weights...
        auto _weights = dst;
        inputs[1] = {_weights.get(), 0};

        auto _dst = mean, _mean = variance, _variance = workspace;
        outputs[0] = _dst.get();
        outputs[1] = _mean.get();
        outputs[2] = _variance.get();
        outputs[3] = nullptr;
      }
    }
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const primitive::at &weights,
                              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }

  batch_normalization_forward(const primitive_desc &aprimitive_desc,
                              const primitive::at &src,
                              const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization forward primitive");
    reset(result);
  }
};

struct batch_normalization_backward : public primitive {
  struct desc {
    mkldnn_batch_normalization_desc_t data;
    template <typename T>
    desc(prop_kind aprop_kind,
         const memory::desc &diff_data_desc,
         const memory::desc &data_desc,
         T epsilon,
         unsigned flags) {
      error::wrap_c_api(
          mkldnn_batch_normalization_backward_desc_init(
              &data,
              mkldnn::convert_to_c(aprop_kind),
              &diff_data_desc.data,
              &data_desc.data,
              static_cast<float>(epsilon),
              flags),
          "could not create a batch normalization backward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc,
                   const engine &aengine,
                   const batch_normalization_forward::primitive_desc
                       &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a batch normalization backward primitive "
          "descriptor");
      reset(result);
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t bndesc;
      const_mkldnn_primitive_desc_t const_bndesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(bndesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t bndesc;
      const_mkldnn_primitive_desc_t const_bndesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a diff_weights primitive descriptor");
      adesc.reset(bndesc);
      return adesc;
    }

    memory::primitive_desc mean_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t bndesc;
      const_mkldnn_primitive_desc_t const_bndesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a mean primitive descriptor");
      adesc.reset(bndesc);
      return adesc;
    }

    memory::primitive_desc variance_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t bndesc;
      const_mkldnn_primitive_desc_t const_bndesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 2);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc, const_bndesc),
                        "could not clone a variance primitive descriptor");
      adesc.reset(bndesc);
      return adesc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  // Prop_kind == backward
  batch_normalization_backward(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &mean,
                               const primitive::at &variance,
                               const primitive::at &diff_dst,
                               const primitive::at &weights,
                               const memory &diff_src,
                               const memory &diff_weights) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {
        src.data, mean.data, variance.data, diff_dst.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get(), diff_weights.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization backward primitive");
    reset(result);
  }

  // Prop_kind == backward (+ws)
  batch_normalization_backward(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &mean,
                               const primitive::at &variance,
                               const primitive::at &diff_dst,
                               const primitive::at &weights,
                               const primitive::at &workspace,
                               const memory &diff_src,
                               const memory &diff_weights) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data,
                                      mean.data,
                                      variance.data,
                                      diff_dst.data,
                                      weights.data,
                                      workspace.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get(), diff_weights.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization backward primitive");
    reset(result);
  }

  // Prop_kind == backward_data (+ws or +weights)
  /// @warning This constructor works for backward_data propagation
  ///          - w/ weights but w/o workspace, or
  ///          - w/ workspace but w/o weights
  batch_normalization_backward(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &mean,
                               const primitive::at &variance,
                               const primitive::at &diff_dst,
                               const primitive::at &weights_or_workspace,
                               const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data,
                                      mean.data,
                                      variance.data,
                                      diff_dst.data,
                                      weights_or_workspace.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization backward primitive");
    reset(result);
  }

  // Prop_kind == backward_data
  batch_normalization_backward(const primitive_desc &aprimitive_desc,
                               const primitive::at &src,
                               const primitive::at &mean,
                               const primitive::at &variance,
                               const primitive::at &diff_dst,
                               const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {
        src.data, mean.data, variance.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a batch normalization backward primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_inner_product Inner Product
/// @{

struct inner_product_forward : public primitive {
  struct desc {
    mkldnn_inner_product_desc_t data;
    desc(prop_kind aprop_kind,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_desc) {
      error::wrap_c_api(mkldnn_inner_product_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            &src_desc.data,
                            &weights_desc.data,
                            &bias_desc.data,
                            &dst_desc.data),
                        "could not create a inner product forward descriptor");
    }

    desc(prop_kind aprop_kind,
         const memory::desc &src_desc,
         const memory::desc &weights_desc,
         const memory::desc &dst_desc) {
      error::wrap_c_api(mkldnn_inner_product_forward_desc_init(
                            &data,
                            mkldnn::convert_to_c(aprop_kind),
                            &src_desc.data,
                            &weights_desc.data,
                            nullptr,
                            &dst_desc.data),
                        "could not create a inner product forward descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create a inner product forward primitive descriptor");
      reset(result);
    }

    primitive_desc(const desc &adesc,
                   const primitive_attr &aattr,
                   const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create_v2(
              &result, &adesc.data, aattr.get(), aengine.get(), nullptr),
          "could not create a inner product "
          "forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  inner_product_forward(const primitive_desc &aprimitive_desc,
                        const primitive::at &src,
                        const primitive::at weights,
                        const primitive::at &bias,
                        const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data, bias.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a inner product forward primitive");
    reset(result);
  }

  inner_product_forward(const primitive_desc &aprimitive_desc,
                        const primitive::at &src,
                        const primitive::at weights,
                        const memory &dst) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {dst.get()};
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create a inner product forward primitive");
    reset(result);
  }
};

struct inner_product_backward_data : public primitive {
  struct desc {
    mkldnn_inner_product_desc_t data;
    desc(const memory::desc &diff_src_desc,
         const memory::desc &weights_desc,
         const memory::desc &diff_dst_desc) {
      error::wrap_c_api(
          mkldnn_inner_product_backward_data_desc_init(&data,
                                                       &diff_src_desc.data,
                                                       &weights_desc.data,
                                                       &diff_dst_desc.data),
          "could not create a inner product backward data descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const inner_product_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a inner product backward data primitive "
          "descriptor");
      reset(result);
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff dst primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  inner_product_backward_data(const primitive_desc &aprimitive_desc,
                              const primitive::at &diff_dst,
                              const primitive::at weights,
                              const memory &diff_src) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {diff_dst.data, weights.data};
    const_mkldnn_primitive_t outputs[] = {diff_src.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a inner product backward data primitive");
    reset(result);
  }
};

struct inner_product_backward_weights : public primitive {
  struct desc {
    mkldnn_inner_product_desc_t data;
    desc(const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_bias_desc,
         const memory::desc &diff_dst_desc) {
      error::wrap_c_api(
          mkldnn_inner_product_backward_weights_desc_init(
              &data,
              &src_desc.data,
              &diff_weights_desc.data,
              &diff_bias_desc.data,
              &diff_dst_desc.data),
          "could not create a inner product backward weights descriptor");
    }
    desc(const memory::desc &src_desc,
         const memory::desc &diff_weights_desc,
         const memory::desc &diff_dst_desc) {
      error::wrap_c_api(
          mkldnn_inner_product_backward_weights_desc_init(
              &data,
              &src_desc.data,
              &diff_weights_desc.data,
              nullptr,
              &diff_dst_desc.data),
          "could not create a inner product backward weights descriptor");
    }
  };

  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(
        const desc &adesc,
        const engine &aengine,
        const inner_product_forward::primitive_desc &hint_fwd_primitive_desc) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(&result,
                                       &adesc.data,
                                       aengine.get(),
                                       hint_fwd_primitive_desc.get()),
          "could not create a inner product backward weights primitive "
          "descriptor");
      reset(result);
    }

    memory::primitive_desc diff_dst_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff dst primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a diff bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc src_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  inner_product_backward_weights(const primitive_desc &aprimitive_desc,
                                 const primitive::at &src,
                                 const primitive::at diff_dst,
                                 const memory &diff_weights) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a inner product backward weights primitive");
    reset(result);
  }

  inner_product_backward_weights(const primitive_desc &aprimitive_desc,
                                 const primitive::at &src,
                                 const primitive::at diff_dst,
                                 const memory &diff_weights,
                                 const memory &diff_bias) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = {src.data, diff_dst.data};
    const_mkldnn_primitive_t outputs[] = {diff_weights.get(), diff_bias.get()};
    error::wrap_c_api(
        mkldnn_primitive_create(
            &result, aprimitive_desc.get(), inputs, outputs),
        "could not create a inner product backward weights primitive");
    reset(result);
  }
};

/// @}

/// @addtogroup cpp_api_rnn RNN
/// @{

struct rnn_cell {
  struct desc {
    mkldnn_rnn_cell_desc_t c_rnn_cell_;

    desc(algorithm kind, algorithm activation_f) {
      error::wrap_c_api(
          mkldnn_rnn_cell_desc_init(&c_rnn_cell_,
                                    mkldnn::convert_to_c(kind),
                                    mkldnn::convert_to_c(activation_f),
                                    0U,
                                    0,
                                    0),
          "could not init an rnn cell descriptor");
    }
    desc(algorithm kind) : desc(kind, algorithm::algorithm_undef) {}

    operator const mkldnn_rnn_cell_desc_t *() const { return &c_rnn_cell_; }

    algorithm get_cell_kind() const { return algorithm(c_rnn_cell_.cell_kind); }
    algorithm get_activation() const {
      return algorithm(c_rnn_cell_.activation_kind);
    }

    float get_alpha() const { return c_rnn_cell_.alpha; }
    void set_alpha(float alpha) {
      c_rnn_cell_.flags |= mkldnn_rnn_cell_with_relu;
      c_rnn_cell_.alpha = alpha;
    }

    float get_clipping() const { return c_rnn_cell_.clipping; }
    void set_clipping(float clipping) {
      c_rnn_cell_.flags |= mkldnn_rnn_cell_with_clipping;
      c_rnn_cell_.clipping = clipping;
    }

    int get_gates_count() const {
      return mkldnn_rnn_cell_get_gates_count(&c_rnn_cell_);
    }
    int get_state_count() const {
      return mkldnn_rnn_cell_get_states_count(&c_rnn_cell_);
    }
  };
};

struct rnn_forward : public primitive {
  struct desc {
    mkldnn_rnn_desc_t data;
    desc(prop_kind aprop_kind,
         rnn_cell::desc cell,
         const rnn_direction direction,
         const memory::desc &src_layer_desc,
         const memory::desc &src_iter_desc,
         const memory::desc &weights_layer_desc,
         const memory::desc &weights_iter_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_layer_desc,
         const memory::desc &dst_iter_desc) {
      error::wrap_c_api(
          mkldnn_rnn_forward_desc_init(&data,
                                       mkldnn::convert_to_c(aprop_kind),
                                       cell,
                                       mkldnn::convert_to_c(direction),
                                       &src_layer_desc.data,
                                       &src_iter_desc.data,
                                       &weights_layer_desc.data,
                                       &weights_iter_desc.data,
                                       &bias_desc.data,
                                       &dst_layer_desc.data,
                                       &dst_iter_desc.data),
          "could not create an RNN forward descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                            &result, &adesc.data, aengine.get(), nullptr),
                        "could not create an RNN forward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone an src layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc src_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src iter primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_src_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 2);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t ldesc;
      const_mkldnn_primitive_desc_t const_ldesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(ldesc);
      return adesc;
    }

    memory::primitive_desc dst_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 1);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last iteration primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };

  rnn_forward(const primitive_desc &aprimitive_desc,
              const primitive::at &src_layer,
              const primitive::at &src_iter,
              const primitive::at &weights_layer,
              const primitive::at &weights_iter,
              const primitive::at &bias,
              const memory &dst_layer,
              const memory &dst_iter,
              const memory &workspace) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[5];
    const_mkldnn_primitive_t outputs[3];
    int idx = 0;
    inputs[idx++] = src_layer.data;
    if (!is_null_memory(src_iter.data.primitive)) inputs[idx++] = src_iter.data;
    inputs[idx++] = weights_layer.data;
    inputs[idx++] = weights_iter.data;
    if (!is_null_memory(bias.data.primitive)) inputs[idx++] = bias.data;

    idx = 0;
    outputs[idx++] = dst_layer.get();
    if (!is_null_memory(dst_iter.get())) outputs[idx++] = dst_iter.get();
    if (!is_null_memory(workspace.get())) outputs[idx++] = workspace.get();

    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create an RNN forward primitive");
    reset(result);
  }
};

struct rnn_backward : public primitive {
  struct desc {
    mkldnn_rnn_desc_t data;
    desc(prop_kind aprop_kind,
         rnn_cell::desc cell,
         const rnn_direction direction,
         const memory::desc &src_layer_desc,
         const memory::desc &src_iter_desc,
         const memory::desc &weights_layer_desc,
         const memory::desc &weights_iter_desc,
         const memory::desc &bias_desc,
         const memory::desc &dst_layer_desc,
         const memory::desc &dst_iter_desc,
         const memory::desc &diff_src_layer_desc,
         const memory::desc &diff_src_iter_desc,
         const memory::desc &diff_weights_layer_desc,
         const memory::desc &diff_weights_iter_desc,
         const memory::desc &diff_bias_desc,
         const memory::desc &diff_dst_layer_desc,
         const memory::desc &diff_dst_iter_desc) {
      error::wrap_c_api(
          mkldnn_rnn_backward_desc_init(&data,
                                        mkldnn::convert_to_c(aprop_kind),
                                        cell,
                                        mkldnn::convert_to_c(direction),
                                        &src_layer_desc.data,
                                        &src_iter_desc.data,
                                        &weights_layer_desc.data,
                                        &weights_iter_desc.data,
                                        &bias_desc.data,
                                        &dst_layer_desc.data,
                                        &dst_iter_desc.data,
                                        &diff_src_layer_desc.data,
                                        &diff_src_iter_desc.data,
                                        &diff_weights_layer_desc.data,
                                        &diff_weights_iter_desc.data,
                                        &diff_bias_desc.data,
                                        &diff_dst_layer_desc.data,
                                        &diff_dst_iter_desc.data),
          "could not create an RNN backward descriptor");
    }
  };
  struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const desc &adesc, const engine &aengine) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(
          mkldnn_primitive_desc_create(
              &result, &adesc.data, aengine.get(), nullptr),
          "could not create an RNN backward primitive descriptor");
      reset(result);
    }

    memory::primitive_desc src_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone an src layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc src_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(src_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src iter primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc weights_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(weights_pd), 2);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 0);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc dst_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(dst_pd), 1);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last iteration primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_src_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone an src_layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_src_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_src_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a src iter primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_weights_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_bias_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_weights_pd), 2);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                        "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_layer_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 0);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last layer primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc diff_dst_iter_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(diff_dst_pd), 1);
      error::wrap_c_api(
          mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a dst last iteration primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    memory::primitive_desc workspace_primitive_desc() const {
      memory::primitive_desc adesc;
      mkldnn_primitive_desc_t ldesc;
      const_mkldnn_primitive_desc_t const_ldesc =
          mkldnn_primitive_desc_query_pd(
              get(), mkldnn::convert_to_c(workspace_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                        "could not clone a workspace primitive descriptor");
      adesc.reset(ldesc);
      return adesc;
    }

    engine get_engine() { return engine::query(*this); }
  };
  // With last iteration (with and without input src_iter)
  rnn_backward(const primitive_desc &aprimitive_desc,
               const primitive::at &src_layer,
               const primitive::at &src_iter,
               const primitive::at &weights_layer,
               const primitive::at &weights_iter,
               const primitive::at &bias,
               const primitive::at &dst_layer,
               const primitive::at &dst_iter,
               const memory &diff_src_layer,
               const memory &diff_src_iter,
               const memory &diff_weights_layer,
               const memory &diff_weights_iter,
               const memory &diff_bias,
               const primitive::at &diff_dst_layer,
               const primitive::at &diff_dst_iter,
               const primitive::at &workspace) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[10];
    const_mkldnn_primitive_t outputs[5];
    int idx = 0;
    inputs[idx] = src_layer.data;
    if (!is_null_memory(src_iter.data.primitive)) inputs[idx++] = src_iter.data;
    inputs[idx++] = weights_layer.data;
    inputs[idx++] = weights_iter.data;
    if (!is_null_memory(bias.data.primitive)) inputs[idx++] = bias.data;
    inputs[idx] = dst_layer.data;
    if (!is_null_memory(dst_iter.data.primitive)) inputs[idx++] = dst_iter.data;
    inputs[idx] = diff_dst_layer.data;
    if (!is_null_memory(diff_dst_iter.data.primitive))
      inputs[idx++] = diff_dst_iter.data;
    inputs[idx] = workspace.data;

    idx = 0;
    outputs[idx] = diff_src_layer.get();
    if (!is_null_memory(diff_src_iter.get()))
      outputs[idx++] = diff_src_iter.get();
    outputs[idx] = diff_weights_layer.get();
    outputs[idx] = diff_weights_iter.get();
    if (!is_null_memory(diff_bias.get())) outputs[idx] = diff_bias.get();
    error::wrap_c_api(mkldnn_primitive_create(
                          &result, aprimitive_desc.get(), inputs, outputs),
                      "could not create an RNN backward primitive");
    reset(result);
  }
};

/// @}
/// @} Primitives

/// @addtogroup cpp_api_stream Stream
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct handle_traits<mkldnn_stream_t> {
  static constexpr auto destructor = &mkldnn_stream_destroy;
};
#endif

struct stream : public handle<mkldnn_stream_t> {
  using handle::handle;

  enum kind {
    any = mkldnn_stream_kind_t::mkldnn_any_stream,
    eager = mkldnn_stream_kind_t::mkldnn_eager,
    lazy = mkldnn_stream_kind_t::mkldnn_lazy
  };

  static mkldnn_stream_kind_t convert_to_c(kind akind) {
    return static_cast<mkldnn_stream_kind_t>(akind);
  }
  /// Constructs a stream.
  stream(kind akind) {
    mkldnn_stream_t astream;
    error::wrap_c_api(mkldnn_stream_create(&astream, convert_to_c(akind)),
                      "could not create a stream");
    reset(astream);
  }

  /// Submits a vector of primitives to a stream for computations.
  ///
  /// @param primitives The vector of primitives to submit.
  /// @returns The stream.
  stream &submit(std::vector<primitive> primitives) {
    // TODO: find a proper way to convert vector<primitive> to
    // vector<mkldnn_primitive_t>
    if (primitives.size() == 0) return *this;
    std::vector<mkldnn_primitive_t> c_api_primitives;
    c_api_primitives.reserve(primitives.size());
    auto convert_to_c = [](primitive p) { return p.get(); };
    std::transform(primitives.begin(),
                   primitives.end(),
                   std::back_inserter(c_api_primitives),
                   convert_to_c);

    mkldnn_primitive_t c_api_error_primitive;
    error::wrap_c_api(mkldnn_stream_submit(get(),
                                           c_api_primitives.size(),
                                           &c_api_primitives[0],
                                           &c_api_error_primitive),
                      "could not submit primitives to a stream",
                      &c_api_error_primitive);

    return *this;
  }

  /// Waits for all computations submitted to the stream to complete.
  ///
  /// @param block Specifies whether the operation should wait indefinitely or
  /// return
  ///              immediately.
  /// @returns @c true if all computations completed.
  /// @returns @c false if not all computations completed.
  bool wait(bool block = true) {
    mkldnn_primitive_t c_api_error_primitive;
    mkldnn_status_t status =
        mkldnn_stream_wait(get(), block, &c_api_error_primitive);
    if (status != mkldnn_success && status != mkldnn_try_again)
      error::wrap_c_api(
          status, "could not wait on a stream", &c_api_error_primitive);
    return (status == mkldnn_success);
  }

  stream &rerun() {
    mkldnn_primitive_t c_api_error_primitive;
    error::wrap_c_api(mkldnn_stream_rerun(get(), &c_api_error_primitive),
                      "could not rerun a stream",
                      &c_api_error_primitive);
    return *this;
  }
};

/// @}

/// @} C++ API

}  // namespace mkldnn

#endif
