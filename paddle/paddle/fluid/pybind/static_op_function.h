

#pragma once

#include <Python.h>

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

namespace paddle {

namespace pybind {

PyObject *static_api_abs(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_abs_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_accuracy(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_acos(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_acos_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_acosh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_acosh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_adagrad_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_adam_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_adamax_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_adamw_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_addmm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_addmm_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_affine_grid(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_allclose(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_angle(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_apply_per_channel_scale(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_argmax(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_argmin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_argsort(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_as_complex(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_as_real(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_as_strided(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_asgd_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_asin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_asin_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_asinh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_asinh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_atan(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_atan_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_atan2(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_atanh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_atanh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_auc(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_average_accumulates_(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_bce_loss(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_bce_loss_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_bernoulli(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_bicubic_interp(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_bilinear(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_bilinear_interp(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_bincount(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_binomial(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_bitwise_and(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_bitwise_and_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_bitwise_left_shift(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_bitwise_left_shift_(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_bitwise_not(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_bitwise_not_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_bitwise_or(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_bitwise_or_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_bitwise_right_shift(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_bitwise_right_shift_(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_bitwise_xor(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_bitwise_xor_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_bmm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_box_coder(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_broadcast_tensors(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_ceil(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_ceil_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_celu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_check_finite_and_unscale_(PyObject *self, PyObject *args,
                                               PyObject *kwargs);

PyObject *static_api_check_numerics(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_cholesky(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cholesky_solve(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_class_center_sample(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_clip(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_clip_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_clip_by_norm(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_coalesce_tensor(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_complex(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_concat(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_conj(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_conv2d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_conv3d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_conv3d_transpose(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_copysign(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_copysign_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_cos(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cos_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cosh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cosh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_crop(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cross(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cross_entropy_with_softmax(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_cross_entropy_with_softmax_(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_cummax(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cummin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cumprod(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cumprod_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cumsum(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cumsum_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_data(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_depthwise_conv2d(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_det(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_diag(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_diag_embed(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_diagonal(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_digamma(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_digamma_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_dirichlet(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_dist(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_dot(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_edit_distance(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_eig(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_eigh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_eigvals(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_eigvalsh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_elu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_elu_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_equal_all(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_erf(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_erf_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_erfinv(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_erfinv_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_exp(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_exp_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_expand(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_expand_as(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_expm1(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_expm1_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fill(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fill_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fill_diagonal(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_fill_diagonal_(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_fill_diagonal_tensor(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_fill_diagonal_tensor_(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_flash_attn(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_flash_attn_unpadded(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_flash_attn_with_sparse_mask(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_flatten(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_flatten_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_flip(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_floor(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_floor_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fmax(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fmin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fold(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fractional_max_pool2d(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_fractional_max_pool3d(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_frame(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_full_int_array(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_gammaincc(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_gammaincc_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_gammaln(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_gammaln_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_gather(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_gather_nd(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_gather_tree(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_gaussian_inplace(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_gaussian_inplace_(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_gelu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_generate_proposals(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_graph_khop_sampler(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_graph_sample_neighbors(PyObject *self, PyObject *args,
                                            PyObject *kwargs);

PyObject *static_api_grid_sample(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_group_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_gumbel_softmax(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_hardshrink(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_hardsigmoid(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_hardtanh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_hardtanh_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_heaviside(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_histogram(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_huber_loss(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_i0(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_i0_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_i0e(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_i1(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_i1e(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_identity_loss(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_identity_loss_(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_imag(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_index_add(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_index_add_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_index_put(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_index_put_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_index_sample(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_index_select(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_index_select_strided(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_instance_norm(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_inverse(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_is_empty(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_isclose(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_isfinite(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_isinf(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_isnan(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_kldiv_loss(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_kron(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_kthvalue(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_label_smooth(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_lamb_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_layer_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_leaky_relu(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_leaky_relu_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_lerp(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lerp_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lgamma(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lgamma_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_linear_interp(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_llm_int8_linear(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_log(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log10(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log10_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log1p(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log1p_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log2(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log2_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log_loss(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_log_softmax(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_logcumsumexp(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_logical_and(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_logical_and_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_logical_not(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_logical_not_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_logical_or(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_logical_or_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_logical_xor(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_logical_xor_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_logit(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_logit_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_logsigmoid(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_lstsq(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lu_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_lu_unpack(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_margin_cross_entropy(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_masked_multihead_attention_(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_masked_select(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_matrix_nms(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_matrix_power(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_max_pool2d_with_index(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_max_pool3d_with_index(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_maxout(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_mean_all(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_memory_efficient_attention(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_merge_selected_rows(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_merged_adam_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_merged_momentum_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_meshgrid(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_mode(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_momentum_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_multi_dot(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_multiclass_nms3(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_multinomial(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_multiplex(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_mv(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_nanmedian(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_nearest_interp(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_nextafter(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_nll_loss(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_nms(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_nonzero(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_npu_identity(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_numel(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_overlap_add(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_p_norm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_pad3d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_pixel_shuffle(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_pixel_unshuffle(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_poisson(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_polygamma(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_polygamma_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_pow(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_pow_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_prelu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_prior_box(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_psroi_pool(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_put_along_axis(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_put_along_axis_(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_qr(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_real(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_reciprocal(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_reciprocal_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_reindex_graph(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_relu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_relu_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_relu6(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_renorm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_renorm_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_reverse(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rms_norm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rmsprop_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_roi_align(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_roi_pool(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_roll(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_round(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_round_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rprop_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rsqrt(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rsqrt_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_scale(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_scale_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_scatter(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_scatter_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_scatter_nd_add(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_searchsorted(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_segment_pool(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_selu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_send_u_recv(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_send_ue_recv(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_send_uv(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sgd_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_shape(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_shard_index(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sigmoid_cross_entropy_with_logits(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs);

PyObject *static_api_sigmoid_cross_entropy_with_logits_(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs);

PyObject *static_api_sign(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_silu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sin_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sinh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sinh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_slogdet(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_softplus(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_softshrink(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_softsign(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_solve(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_spectral_norm(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_sqrt(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sqrt_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_square(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_squared_l2_norm(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_squeeze(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_squeeze_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_stack(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_standard_gamma(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_stanh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_svd(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_swiglu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_take_along_axis(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_tan(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tan_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tanh(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tanh_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tanh_shrink(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_temporal_shift(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_tensor_unfold(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_thresholded_relu(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_thresholded_relu_(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_top_p_sampling(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_topk(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_trace(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_triangular_solve(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_trilinear_interp(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_trunc(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_trunc_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_unbind(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_unfold(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_uniform_inplace(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_uniform_inplace_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_unique_consecutive(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_unpool3d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_unsqueeze(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_unsqueeze_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_unstack(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_update_loss_scaling_(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_view_dtype(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_view_shape(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_viterbi_decode(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_warpctc(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_warprnnt(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_weight_dequantize(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_weight_only_linear(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_weight_quantize(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_weighted_sample_neighbors(PyObject *self, PyObject *args,
                                               PyObject *kwargs);

PyObject *static_api_where(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_where_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_yolo_box(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_yolo_loss(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_block_multihead_attention_(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_fc(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fused_bias_act(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_fused_bias_dropout_residual_layer_norm(PyObject *self,
                                                            PyObject *args,
                                                            PyObject *kwargs);

PyObject *static_api_fused_bias_residual_layernorm(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs);

PyObject *static_api_fused_conv2d_add_act(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_fused_dconv_drelu_dbn(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_fused_dot_product_attention(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_fused_dropout_add(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_fused_embedding_eltwise_layernorm(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs);

PyObject *static_api_fused_fc_elementwise_layernorm(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs);

PyObject *static_api_fused_linear_param_grad_add(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_fused_rotary_position_embedding(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs);

PyObject *static_api_fused_scale_bias_add_relu(PyObject *self, PyObject *args,
                                               PyObject *kwargs);

PyObject *static_api_fused_scale_bias_relu_conv_bn(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs);

PyObject *static_api_fusion_gru(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_fusion_repeated_fc_relu(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_fusion_seqconv_eltadd_relu(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_fusion_seqexpand_concat_fc(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_fusion_squared_mat_sub(PyObject *self, PyObject *args,
                                            PyObject *kwargs);

PyObject *static_api_fusion_transpose_flatten_concat(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs);

PyObject *static_api_max_pool2d_v2(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_multihead_matmul(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_self_dp_attention(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_skip_layernorm(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_squeeze_excitation_block(PyObject *self, PyObject *args,
                                              PyObject *kwargs);

PyObject *static_api_variable_length_memory_efficient_attention(
    PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_adadelta_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_add(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_add_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_add_n(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_all(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_all_reduce(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_all_reduce_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_amax(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_amin(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_any(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_assign(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_assign_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_assign_out_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_assign_pos(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_assign_value(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_assign_value_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_batch_fc(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_batch_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_batch_norm_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_c_allgather(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_c_allreduce_avg(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_c_allreduce_avg_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_c_allreduce_max(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_c_allreduce_max_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_c_allreduce_min(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_c_allreduce_min_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_c_allreduce_prod(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_c_allreduce_prod_(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_c_allreduce_sum(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_c_allreduce_sum_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_c_broadcast(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_c_broadcast_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_c_concat(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_c_embedding(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_c_identity(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_c_identity_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_c_reduce_avg(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_c_reduce_avg_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_c_reduce_max(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_c_reduce_max_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_c_reduce_min(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_c_reduce_min_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_c_reduce_prod(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_c_reduce_prod_(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_c_reduce_sum(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_c_reduce_sum_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_c_reducescatter(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_c_scatter(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_c_split(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_c_sync_calc_stream(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_c_sync_calc_stream_(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_c_sync_comm_stream(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_c_sync_comm_stream_(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_cast(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_cast_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_channel_shuffle(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_coalesce_tensor_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_conv2d_transpose(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_conv2d_transpose_bias(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_decayed_adagrad(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_decode_jpeg(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_deformable_conv(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_depthwise_conv2d_transpose(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_dequantize_linear(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_dequantize_linear_(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_dgc_momentum(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_disable_check_model_nan_inf(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_distribute_fpn_proposals(PyObject *self, PyObject *args,
                                              PyObject *kwargs);

PyObject *static_api_distributed_fused_lamb_init(PyObject *self, PyObject *args,
                                                 PyObject *kwargs);

PyObject *static_api_distributed_fused_lamb_init_(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs);

PyObject *static_api_distributed_lookup_table(PyObject *self, PyObject *args,
                                              PyObject *kwargs);

PyObject *static_api_distributed_push_sparse(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_divide(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_divide_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_dropout(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_einsum(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_elementwise_pow(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_embedding(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_empty(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_empty_like(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_enable_check_model_nan_inf(PyObject *self, PyObject *args,
                                                PyObject *kwargs);

PyObject *static_api_equal(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_equal_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_exponential_(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_eye(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fetch(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_floor_divide(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_floor_divide_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_frobenius_norm(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_full_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_full_batch_size_like(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_full_like(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_full_with_tensor(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_fused_adam_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_fused_batch_norm_act(PyObject *self, PyObject *args,
                                          PyObject *kwargs);

PyObject *static_api_fused_batch_norm_act_(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_fused_bn_add_activation(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_fused_bn_add_activation_(PyObject *self, PyObject *args,
                                              PyObject *kwargs);

PyObject *static_api_fused_multi_transformer(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_fused_softmax_mask(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_fused_softmax_mask_upper_triangle(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs);

PyObject *static_api_fused_token_prune(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_gaussian(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_get_tensor_from_selected_rows(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs);

PyObject *static_api_global_scatter(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_greater_equal(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_greater_equal_(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_greater_than(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_greater_than_(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_hardswish(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_hsigmoid_loss(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_increment(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_increment_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_less_equal(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_less_equal_(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_less_than(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_less_than_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_limit_by_capacity(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_linspace(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_logspace(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_logsumexp(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_lrn(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_matmul(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_matmul_with_flatten(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_matrix_rank(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_matrix_rank_tol(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_max(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_maximum(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_mean(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_memcpy(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_memcpy_d2h(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_memcpy_h2d(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_min(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_minimum(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_mish(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_multiply(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_multiply_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_nop(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_nop_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_norm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_not_equal(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_not_equal_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_one_hot(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_pad(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_partial_allgather(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_partial_allgather_(PyObject *self, PyObject *args,
                                        PyObject *kwargs);

PyObject *static_api_partial_concat(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_partial_recv(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_partial_sum(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_pool2d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_pool3d(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_print(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_prod(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_prune_gate_by_capacity(PyObject *self, PyObject *args,
                                            PyObject *kwargs);

PyObject *static_api_push_dense(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_push_sparse_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_push_sparse_v2_(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_quantize_linear(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_quantize_linear_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_randint(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_random_routing(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_randperm(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rank_attention(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_read_file(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_recv_v2(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_remainder(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_remainder_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_repeat_interleave(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_repeat_interleave_with_tensor_index(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs);

PyObject *static_api_reshape(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_reshape_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rnn(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rnn_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_row_conv(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_rrelu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_seed(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_send_v2(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_set_value(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_set_value_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_set_value_with_tensor(PyObject *self, PyObject *args,
                                           PyObject *kwargs);

PyObject *static_api_set_value_with_tensor_(PyObject *self, PyObject *args,
                                            PyObject *kwargs);

PyObject *static_api_shadow_feed(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_shadow_feed_tensors(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_share_data(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_shuffle_batch(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_slice(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_soft_relu(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_softmax(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_softmax_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_split(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_split_with_num(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_strided_slice(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_subtract(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_subtract_(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_sum(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_swish(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sync_batch_norm_(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

PyObject *static_api_tdm_sampler(PyObject *self, PyObject *args,
                                 PyObject *kwargs);

PyObject *static_api_tile(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_trans_layout(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_transpose(PyObject *self, PyObject *args,
                               PyObject *kwargs);

PyObject *static_api_transpose_(PyObject *self, PyObject *args,
                                PyObject *kwargs);

PyObject *static_api_tril(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tril_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_tril_indices(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_triu(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_triu_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_triu_indices(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_truncated_gaussian_random(PyObject *self, PyObject *args,
                                               PyObject *kwargs);

PyObject *static_api_uniform(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_uniform_random_batch_size_like(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs);

PyObject *static_api_unique(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_unpool(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_c_softmax_with_cross_entropy(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs);

PyObject *static_api_dpsgd(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_ftrl(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_fused_attention(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_fused_elemwise_add_activation(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs);

PyObject *static_api_fused_feedforward(PyObject *self, PyObject *args,
                                       PyObject *kwargs);

PyObject *static_api_lars_momentum(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

PyObject *static_api_lars_momentum_(PyObject *self, PyObject *args,
                                    PyObject *kwargs);

PyObject *static_api_match_matrix_tensor(PyObject *self, PyObject *args,
                                         PyObject *kwargs);

PyObject *static_api_moving_average_abs_max_scale(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs);

PyObject *static_api_moving_average_abs_max_scale_(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs);

PyObject *static_api_nce(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_number_count(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_onednn_to_paddle_layout(PyObject *self, PyObject *args,
                                             PyObject *kwargs);

PyObject *static_api_partial_send(PyObject *self, PyObject *args,
                                  PyObject *kwargs);

PyObject *static_api_sparse_momentum(PyObject *self, PyObject *args,
                                     PyObject *kwargs);

PyObject *static_api_arange(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *static_api_sequence_mask(PyObject *self, PyObject *args,
                                   PyObject *kwargs);

}  // namespace pybind

}  // namespace paddle
