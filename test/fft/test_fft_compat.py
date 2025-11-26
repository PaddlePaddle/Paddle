# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import scipy.fft

import paddle


class TestFFTAliasBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

        self.init_params()
        self.init_data()
        self.paddle_axis_arg = "axes" if self.is_nd_or_shift else "axis"

    def init_params(self):
        self.shape = [2, 4, 6]
        self.test_dim_val = -1
        self.norm = "backward"
        self.dtype = "float32"
        self.is_real_input = False
        self.is_nd_or_shift = False

        self.paddle_api = None
        self.scipy_api = None

    def init_data(self):
        if self.is_real_input:
            self.np_x = np.random.rand(*self.shape).astype(self.dtype)
        else:
            real = np.random.rand(*self.shape).astype(self.dtype)
            imag = np.random.rand(*self.shape).astype(self.dtype)
            self.np_x = real + 1j * imag

    def get_scipy_ref(self):
        kwargs = {"norm": self.norm}

        if 'shift' in self.scipy_api.__name__:
            return self.scipy_api(self.np_x, axes=self.test_dim_val)

        if self.is_nd_or_shift:
            kwargs["axes"] = self.test_dim_val
        else:
            kwargs["axis"] = self.test_dim_val

        return self.scipy_api(self.np_x, **kwargs)

    def test_dygraph_Compatibility(self):
        if self.paddle_api is None:
            return

        paddle.disable_static()
        for place in self.places:
            paddle.set_device(place)

            x = paddle.to_tensor(self.np_x)
            paddle_dygraph_out = []

            # 1. Paddle Usage
            kw_std = {"x": x, self.paddle_axis_arg: self.test_dim_val}
            if 'shift' not in self.paddle_api.__name__:
                kw_std["norm"] = self.norm

            out1 = self.paddle_api(**kw_std)
            paddle_dygraph_out.append(out1)

            # 2. Alias Usage
            kw_alias = {"input": x, "dim": self.test_dim_val}
            if 'shift' not in self.paddle_api.__name__:
                kw_alias["norm"] = self.norm

            out2 = self.paddle_api(**kw_alias)
            paddle_dygraph_out.append(out2)

            # 3. Alias Usage with 'out' parameter (Skip for shift APIs)
            if 'shift' not in self.paddle_api.__name__:
                out_tensor = paddle.empty_like(out1)
                kw_out = {
                    "input": x,
                    "dim": self.test_dim_val,
                    "out": out_tensor,
                    "norm": self.norm,
                }

                self.paddle_api(**kw_out)
                paddle_dygraph_out.append(out_tensor)

            ref_out = self.get_scipy_ref()

            for i, out in enumerate(paddle_dygraph_out):
                np.testing.assert_allclose(
                    out.numpy(),
                    ref_out,
                    rtol=1e-05,
                    atol=1e-08,
                    err_msg=f"Dygraph mismatch case {i} (0=std, 1=alias, 2=out) for {self.paddle_api.__name__} on {place}",
                )
        paddle.enable_static()

    def test_static_Compatibility(self):
        if self.paddle_api is None:
            return

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()

        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape, dtype=self.np_x.dtype
            )

            # 1. Paddle Usage
            kw_std = {"x": x, self.paddle_axis_arg: self.test_dim_val}
            if 'shift' not in self.paddle_api.__name__:
                kw_std["norm"] = self.norm
            out_std = self.paddle_api(**kw_std)

            # 2. Alias Usage
            kw_alias = {"input": x, "dim": self.test_dim_val}
            if 'shift' not in self.paddle_api.__name__:
                kw_alias["norm"] = self.norm
            out_alias = self.paddle_api(**kw_alias)

            ref_out = self.get_scipy_ref()

            for place in self.places:
                exe = paddle.base.Executor(place)
                exe.run(startup)
                fetches = exe.run(
                    main,
                    feed={"x": self.np_x},
                    fetch_list=[out_std, out_alias],
                )

                np.testing.assert_allclose(
                    fetches[0],
                    ref_out,
                    rtol=1e-05,
                    atol=1e-08,
                    err_msg=f"Static graph mismatch (Standard Args) for {self.paddle_api.__name__} on {place}",
                )

                np.testing.assert_allclose(
                    fetches[1],
                    ref_out,
                    rtol=1e-05,
                    atol=1e-08,
                    err_msg=f"Static graph mismatch (Alias Args) for {self.paddle_api.__name__} on {place}",
                )


class TestFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.fft
        self.scipy_api = scipy.fft.fft
        self.is_real_input = False
        self.is_nd_or_shift = False


class TestIFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ifft
        self.scipy_api = scipy.fft.ifft
        self.is_real_input = False
        self.is_nd_or_shift = False


class TestRFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.rfft
        self.scipy_api = scipy.fft.rfft
        self.is_real_input = True
        self.is_nd_or_shift = False


class TestIRFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.irfft
        self.scipy_api = scipy.fft.irfft
        self.is_real_input = False
        self.is_nd_or_shift = False


class TestHFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.hfft
        self.scipy_api = scipy.fft.hfft
        self.is_real_input = False
        self.is_nd_or_shift = False


class TestIHFFT_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ihfft
        self.scipy_api = scipy.fft.ihfft
        self.is_real_input = True
        self.is_nd_or_shift = False


class TestFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.fft2
        self.scipy_api = scipy.fft.fft2
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestIFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ifft2
        self.scipy_api = scipy.fft.ifft2
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestRFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.rfft2
        self.scipy_api = scipy.fft.rfft2
        self.is_real_input = True
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestIRFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.irfft2
        self.scipy_api = scipy.fft.irfft2
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestHFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.hfft2
        self.scipy_api = scipy.fft.hfft2
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestIHFFT2_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ihfft2
        self.scipy_api = scipy.fft.ihfft2
        self.is_real_input = True
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.fftn
        self.scipy_api = scipy.fft.fftn
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestIFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ifftn
        self.scipy_api = scipy.fft.ifftn
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestRFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.rfftn
        self.scipy_api = scipy.fft.rfftn
        self.is_real_input = True
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestIRFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.irfftn
        self.scipy_api = scipy.fft.irfftn
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestHFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.hfftn
        self.scipy_api = scipy.fft.hfftn
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestIHFFTN_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ihfftn
        self.scipy_api = scipy.fft.ihfftn
        self.is_real_input = True
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1, 2)


class TestFFTShift_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.fftshift
        self.scipy_api = scipy.fft.fftshift
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


class TestIFFTShift_Alias(TestFFTAliasBase):
    def init_params(self):
        super().init_params()
        self.paddle_api = paddle.fft.ifftshift
        self.scipy_api = scipy.fft.ifftshift
        self.is_real_input = False
        self.is_nd_or_shift = True
        self.test_dim_val = (0, 1)


if __name__ == "__main__":
    unittest.main()
