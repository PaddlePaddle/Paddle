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

#pragma once
#include <Python.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace py = pybind11;

namespace paddle {
		namespace pybind {
				namespace details {

						template <bool less, size_t I, typename... ARGS>
						struct CastToPyBufferImpl;

						template <size_t I, typename... ARGS>
						struct CastToPyBufferImpl<false, I, ARGS...> {
								pybind11::buffer_info operator()(const framework::Tensor &tensor) {
									PADDLE_THROW("This type of tensor cannot be expose to Python");
									return pybind11::buffer_info();
								}
						};

						template <size_t I, typename... ARGS>
						struct CastToPyBufferImpl<true, I, ARGS...> {
								using CUR_TYPE = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
								pybind11::buffer_info operator()(const framework::Tensor &tensor) {
									if (framework::DataTypeTrait<CUR_TYPE>::DataType == tensor.type()) {
										auto dim_vec = framework::vectorize(tensor.dims());
										std::vector<size_t> dims_outside;
										std::vector<size_t> strides;
										dims_outside.resize(dim_vec.size());
										strides.resize(dim_vec.size());

										size_t prod = 1;
										for (size_t i = dim_vec.size(); i != 0; --i) {
											dims_outside[i - 1] = (size_t)dim_vec[i - 1];
											strides[i - 1] = sizeof(CUR_TYPE) * prod;
											prod *= dims_outside[i - 1];
										}
										framework::Tensor dst_tensor;
										bool is_gpu = paddle::platform::is_gpu_place(tensor.place());
										if (is_gpu) {
#ifdef PADDLE_WITH_CUDA
											auto *src_ptr = static_cast<const void *>(tensor.data<CUR_TYPE>());
        auto *dst_ptr = static_cast<void *>(dst_tensor.mutable_data<CUR_TYPE>(
            tensor.dims(), platform::CPUPlace()));

        paddle::platform::GpuMemcpySync(dst_ptr, src_ptr,
                                        sizeof(CUR_TYPE) * tensor.numel(),
                                        cudaMemcpyDeviceToHost);
#else
											PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
										} else if (paddle::platform::is_cpu_place(tensor.place())) {
											dst_tensor = tensor;
										}

										std::string dtype = std::type_index(typeid(CUR_TYPE)) ==
																				std::type_index(typeid(platform::float16))
																				? std::string("e")  // np.dtype('e') == np.float16
																				: pybind11::format_descriptor<CUR_TYPE>::format();

										if (is_gpu) {
											// manually construct a py_buffer if is_gpu since gpu data is copied
											// into CPU.
											// TODO(yy): Is these following code memleak?
											Py_buffer *py_buffer =
												 reinterpret_cast<Py_buffer *>(malloc(sizeof(Py_buffer)));
											py_buffer->format = strdup(dtype.c_str());
											py_buffer->itemsize = sizeof(CUR_TYPE);
											py_buffer->ndim = framework::arity(dst_tensor.dims());
											py_buffer->len = tensor.numel();
											py_buffer->strides = reinterpret_cast<Py_ssize_t *>(
												 malloc(sizeof(Py_ssize_t) * strides.size()));
											for (size_t i = 0; i < strides.size(); ++i) {
												py_buffer->strides[i] = strides[i];
											}

											py_buffer->shape = reinterpret_cast<Py_ssize_t *>(
												 malloc(sizeof(Py_ssize_t) * tensor.dims().size()));
											for (int i = 0; i < tensor.dims().size(); ++i) {
												py_buffer->shape[i] = tensor.dims()[i];
											}

											py_buffer->readonly = false;
											py_buffer->suboffsets = nullptr;
											py_buffer->obj = nullptr;
											py_buffer->buf =
												 malloc(static_cast<size_t>(py_buffer->len * py_buffer->itemsize));
											memcpy(py_buffer->buf, dst_tensor.data<CUR_TYPE>(),
														 static_cast<size_t>(py_buffer->len * py_buffer->itemsize));
											return pybind11::buffer_info(py_buffer, true);
										} else {
											return pybind11::buffer_info(
												 dst_tensor.data<CUR_TYPE>(), sizeof(CUR_TYPE), dtype,
												 (size_t)framework::arity(dst_tensor.dims()), dims_outside, strides);
										}
									} else {
										constexpr bool less = I + 1 < std::tuple_size<std::tuple<ARGS...>>::value;
										return CastToPyBufferImpl<less, I + 1, ARGS...>()(tensor);
									}
								}
						};
				}  // namespace details

				inline pybind11::buffer_info CastToPyBuffer(const framework::Tensor &tensor) {
					auto buffer_info =
						 details::CastToPyBufferImpl<true, 0, float, int, double, int64_t, bool,
								uint8_t, int8_t, platform::float16>()(tensor);
					return buffer_info;
				}

				template <typename T>
				T TensorGetElement(const framework::Tensor &self, size_t offset) {
					if (platform::is_cpu_place(self.place())) {
						return self.data<T>()[offset];
					} else {
						std::shared_ptr<framework::Tensor> dst(new framework::Tensor);
						framework::TensorCopySync(self, platform::CPUPlace(), dst.get());
						return dst->data<T>()[offset];
					}
				}

// TODO(dzhwinter) : fix the redundant Tensor allocate and free
				template <typename T>
				void TensorSetElement(framework::Tensor *self, size_t offset, T elem) {
					if (platform::is_gpu_place(self->place())) {
						framework::Tensor dst;
						framework::TensorCopySync(*self, platform::CPUPlace(), &dst);
						dst.mutable_data<T>(platform::CPUPlace())[offset] = elem;
						framework::TensorCopySync(dst, self->place(), self);
					} else if (platform::is_cpu_place(self->place())) {
						self->mutable_data<T>(self->place())[offset] = elem;
					}
				}

				template <typename T>
				void PyCPUTensorSetFromArray(
					 framework::Tensor *self,
					 pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
					 array,
					 paddle::platform::CPUPlace place) {
					std::vector<int64_t> dims;
					dims.reserve(array.ndim());
					for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
						dims.push_back(static_cast<int>(array.shape()[i]));
					}

					self->Resize(framework::make_ddim(dims));
					auto *dst = self->mutable_data<T>(place);
					std::memcpy(dst, array.data(), sizeof(T) * array.size());
				}

				template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
				inline void PyCPUTensorSetFromArray(
					 framework::Tensor *self,
					 pybind11::array_t<uint16_t,
							pybind11::array::c_style | pybind11::array::forcecast>
					 array,
					 paddle::platform::CPUPlace place) {
					std::vector<int64_t> dims;
					dims.reserve(array.ndim());
					for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
						dims.push_back(static_cast<int>(array.shape()[i]));
					}

					self->Resize(framework::make_ddim(dims));
					auto *dst = self->mutable_data<platform::float16>(place);
					std::memcpy(dst, array.data(), sizeof(uint16_t) * array.size());
				}

				template <typename T, size_t D>
				void _sliceCompute(const framework::Tensor* in,
													framework::Tensor* out,
													const platform::CPUDeviceContext& ctx,
													std::vector<int>& axes,
													std::vector<int>& starts) {
					auto& eigen_place = *ctx.eigen_device();
					auto place = in->place();
					auto out_dims = out->dims();
					auto in_dims = in->dims();

					auto offsets = Eigen::array<int, D>();
					auto extents = Eigen::array<int, D>();
					for (size_t i = 0; i < D; ++i) {
						offsets[i] = 0;
						extents[i] = out_dims[i];
					}
					int start;
					for (size_t i = 0; i < axes.size(); ++i) {
						start = starts[i];
						if (start < 0) {
							start = (start + in_dims[axes[i]]);
						}
						start = std::max(start, 0);
						offsets[axes[i]] = start;
					}
					auto in_t =
						 framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
								*in);
					auto out_t =
						 framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
								*out);
					out_t.device(eigen_place) = in_t.slice(offsets, extents);
				}

				template <typename T>
				void _concatCompute(const std::vector<paddle::framework::Tensor>& ins,
													 paddle::framework::Tensor* out, const platform::CPUDeviceContext& ctx, int64_t axis) {
					if (axis == 0 && ins.size() < 10) {
						size_t output_offset = 0;
						for (auto& in : ins) {
							auto in_stride = framework::stride_numel(in.dims());
							auto out_stride = framework::stride_numel(out->dims());
							paddle::operators::StridedNumelCopyWithAxis<T>(ctx, axis,
																														 out->data<T>() + output_offset, out_stride,
																														 in.data<T>(), in_stride, in_stride[axis]);
							output_offset += in_stride[axis];
						}
					} else {
						paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, T> concat_functor;
						concat_functor(ctx, ins, static_cast<int>(axis), out);
					}
				}

				void _getSliceinfo(const framework::Tensor &self,
													 py::object obj,
													 const int64_t dim,
													 int64_t& start,
													 int64_t& stop,
													 int64_t& step,
													 int64_t& slicelength) {
					const framework::DDim &srcDDim = self.dims();
					if (dim < 0 || dim >= srcDDim.size()) {
						throw py::index_error();
					}
					if (py::isinstance<py::slice>(obj)) {
						size_t lstart, lstop, lstep, lslicelength;
						py::slice s = static_cast<py::slice>(obj);
						if (!s.compute(srcDDim[dim], &lstart, &lstop, &lstep, &lslicelength)) {
							throw py::index_error();
						}
						start = static_cast<int64_t>(lstart);
						stop = static_cast<int64_t>(lstop);
						step = static_cast<int64_t>(lstep);
						slicelength = static_cast<int64_t>(lslicelength);
					} else if (py::isinstance<py::int_>(obj)) {
						start = static_cast<int64_t>(static_cast<py::int_>(obj));
						if (std::abs(start) >= srcDDim[dim]) {
							throw py::index_error();
						}
						start = (start >= 0) ? start : srcDDim[dim] - start;
						stop = start + 1;
						step = 1;
						slicelength = 1;
					} else {
						throw py::index_error();
					}
				}

				inline framework::Tensor *_getTensor(const framework::Tensor &self, const framework::DDim &ddim) {
					framework::Tensor* output = new framework::Tensor();
					output->Resize(ddim);
					auto place = self.place();
					if (platform::is_cpu_place(place)) {
						output->mutable_data(boost::get<platform::CPUPlace>(place),
																 self.type());
#ifdef PADDLE_WITH_CUDA
						} else {
          if (platform::is_cuda_pinned_place(place)) {
              output->mutable_data(boost::get<platform::CUDAPinnedPlace>(place),
                                   self.type());
            } else if ((platform::is_gpu_place(place))) {
              output->mutable_data(boost::get<platform::CUDAPlace>(place),
                                 self.type());
          }
#endif
					}
					return output;
				}

				template <typename T>
				void _sliceDapper(const framework::Tensor* in,
																							 framework::Tensor* out,
													 const platform::CPUDeviceContext& ctx,
																							 std::vector<int>& axes,
																							 std::vector<int>& starts,
				int size) {
					switch(size) {
						case 1:
							_sliceCompute<T, 1>(in, out, ctx, axes, starts);
							break;
						case 2:
							_sliceCompute<T, 2>(in, out, ctx, axes, starts);
							break;
						case 3:
							_sliceCompute<T, 3>(in, out, ctx, axes, starts);
							break;
						case 4:
							_sliceCompute<T, 4>(in, out, ctx, axes, starts);
							break;
						case 5:
							_sliceCompute<T, 5>(in, out, ctx, axes, starts);
							break;
						case 6:
							_sliceCompute<T, 6>(in, out, ctx, axes, starts);
							break;
						case 7:
							_sliceCompute<T, 7>(in, out, ctx, axes, starts);
							break;
						case 8:
							_sliceCompute<T, 8>(in, out, ctx, axes, starts);
							break;
						case 9:
							_sliceCompute<T, 9>(in, out, ctx, axes, starts);
							break;
						default:
							PADDLE_THROW("dim size not exepected, current is %d", size);
							break;
					}
				}

				template<typename T>
				inline framework::Tensor *_sliceWrapper(const framework::Tensor &self,
																								const platform::CPUDeviceContext& ctx,
																								py::object obj, int dim, int64_t start, int64_t slicelength) {
					framework::DDim dstDDim = self.dims();
					dstDDim[dim] = static_cast<int64_t>(slicelength);
					std::vector<int> axes({dim});
					std::vector<int> starts({static_cast<int>(start)});
					framework::Tensor *output = _getTensor(self, dstDDim);
					_sliceDapper<T>(&self, output, ctx, axes, starts, dstDDim.size());
					return output;
				}

				template<typename T>
				inline framework::Tensor *_sliceAndConcat(const framework::Tensor &self,
																							 py::object obj, int dim) {
					platform::CPUDeviceContext ctx;
					int64_t start, stop, step, slicelength;
					_getSliceinfo(self, obj, dim, start, stop, step, slicelength);
					if (step == 1 || slicelength == 1) {
						return _sliceWrapper<T>(self, ctx, obj, dim, start, slicelength);
					} else {
						std::vector<framework::Tensor> ins;
						for (auto i = 0; i < slicelength; ++i, start += step) {
							ins.emplace_back(*_sliceWrapper<T>(self, ctx, obj, dim, start, 1));
						}

						// do the concat operation
						framework::DDim dstDDim = self.dims();
						dstDDim[dim] = static_cast<int64_t>(slicelength);
						framework::Tensor *output1 = _getTensor(self, dstDDim);
						_concatCompute<T>(ins, output1, ctx, dim);
						return output1;
					}
				}

				inline framework::Tensor *_sliceTensor(const framework::Tensor &self,
																							 py::object obj, int dim) {
					auto src_type = self.type();
					switch (src_type) {
						case framework::proto::VarType::FP16:
							return _sliceAndConcat<paddle::platform::float16>(self, obj, dim);
						case framework::proto::VarType::FP32:
							return _sliceAndConcat<float>(self, obj, dim);
						case framework::proto::VarType::FP64:
							return _sliceAndConcat<double>(self, obj, dim);
						case framework::proto::VarType::INT32:
							return _sliceAndConcat<int>(self, obj, dim);
						case framework::proto::VarType::INT64:
							return _sliceAndConcat<int64_t>(self, obj, dim);
						case framework::proto::VarType::BOOL:
							return _sliceAndConcat<bool>(self, obj, dim);
						case framework::proto::VarType::INT16:
							return _sliceAndConcat<short>(self, obj, dim);
						case framework::proto::VarType::UINT8:
							return _sliceAndConcat<unsigned char>(self, obj, dim);
						default:
							PADDLE_THROW("Not support type %d", src_type);
					}
				}

				inline framework::Tensor *_pySliceTensor(const framework::Tensor &self,
																								py::object obj) {
					if (py::isinstance<py::tuple>(obj)) {
						py::list l = static_cast<py::list>(obj);
						std::unique_ptr<framework::Tensor> target;
						framework::Tensor *src = const_cast<framework::Tensor *>(&self);
						for (auto i = 0; i < static_cast<int>(l.size()); ++i) {
							src = _sliceTensor(*src, l[i], i);
							if (i + 1 == static_cast<int>(l.size())) {
								return src;
							} else {
								target.reset(src);
							}
						}
						return nullptr;
					} else {
						return _sliceTensor(self, obj, 0);
					}
				}

				inline framework::Tensor *PySliceTensor(const framework::Tensor &self,
																								py::object obj) {
					if (platform::is_gpu_place(self.place())) {
						std::unique_ptr<framework::Tensor> holder;
						framework::Tensor src;
						framework::TensorCopySync(self, platform::CPUPlace(), &src);
						framework::Tensor *output = _pySliceTensor(src, obj);
						holder.reset(output);
						framework::Tensor *dst = _getTensor(*output, output->dims());
						framework::TensorCopySync(*output, self.place(), dst);
						return dst;
					} else {
						return _pySliceTensor(self, obj);
					}
				}

#ifdef PADDLE_WITH_CUDA
				template <typename T>
void PyCUDATensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CUDAPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<T>(place);
  paddle::platform::GpuMemcpySync(dst, array.data(), sizeof(T) * array.size(),
                                  cudaMemcpyHostToDevice);
}

template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
inline void PyCUDATensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<uint16_t,
                      pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CUDAPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<platform::float16>(place);
  paddle::platform::GpuMemcpySync(dst, array.data(),
                                  sizeof(uint16_t) * array.size(),
                                  cudaMemcpyHostToDevice);
}

template <typename T>
void PyCUDAPinnedTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        array,
    const paddle::platform::CUDAPinnedPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<T>(place);
  std::memcpy(dst, array.data(), sizeof(T) * array.size());
}

template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
inline void PyCUDAPinnedTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<uint16_t,
                      pybind11::array::c_style | pybind11::array::forcecast>
        array,
    const paddle::platform::CUDAPinnedPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<platform::float16>(place);
  std::memcpy(dst, array.data(), sizeof(uint16_t) * array.size());
}
#endif

		}  // namespace pybind
}  // namespace paddle
