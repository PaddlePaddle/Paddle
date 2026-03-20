# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for RestrictedUnpickler security module."""

from __future__ import annotations

import collections
import io
import os
import pickle
import subprocess
import tempfile
import unittest

import numpy as np

from paddle.framework.restricted_unpickler import (
    safe_load_pickle,
    safe_loads_pickle,
)


class TestRestrictedUnpicklerAllowedTypes(unittest.TestCase):
    """Test that legitimate model data types are allowed."""

    def test_numpy_ndarray(self):
        """numpy arrays should be allowed (model parameters)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        np.testing.assert_array_equal(result, data)

    def test_numpy_float64(self):
        """numpy float64 arrays should be allowed."""
        data = np.array([1.0, 2.0], dtype=np.float64)
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        np.testing.assert_array_equal(result, data)

    def test_ordered_dict(self):
        """OrderedDict should be allowed (state_dict structure)."""
        data = collections.OrderedDict(
            [('weight', np.array([1.0])), ('bias', np.array([0.0]))]
        )
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        self.assertIsInstance(result, collections.OrderedDict)
        np.testing.assert_array_equal(result['weight'], data['weight'])

    def test_dict_with_numpy(self):
        """Dict of numpy arrays should be allowed (typical state_dict)."""
        data = {
            'layer1.weight': np.random.randn(3, 3).astype(np.float32),
            'layer1.bias': np.zeros(3, dtype=np.float32),
        }
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        for key in data:
            np.testing.assert_array_equal(result[key], data[key])

    def test_nested_dict(self):
        """Nested dicts should be allowed."""
        data = {'params': {'weight': np.array([1.0, 2.0])}, 'epoch': 10}
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        self.assertEqual(result['epoch'], 10)

    def test_list_of_arrays(self):
        """Lists of numpy arrays should be allowed."""
        data = [np.array([1.0]), np.array([2.0])]
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_load_pickle(buf)
        self.assertEqual(len(result), 2)

    def test_safe_loads_pickle(self):
        """safe_loads_pickle should work with bytes input."""
        data = {'key': np.array([1.0, 2.0, 3.0])}
        pickled = pickle.dumps(data)
        result = safe_loads_pickle(pickled)
        np.testing.assert_array_equal(result['key'], data['key'])


class TestRestrictedUnpicklerBlocked(unittest.TestCase):
    """Test that dangerous payloads are blocked."""

    def test_block_os_system(self):
        """os.system should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (os.system, ('echo pwned',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_subprocess(self):
        """subprocess.call should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (subprocess.call, (['echo', 'pwned'],))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_eval(self):
        """eval should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (eval, ('__import__("os").system("echo pwned")',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_exec(self):
        """exec should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (exec, ('import os; os.system("echo pwned")',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_open(self):
        """builtins.open should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (open, ('/etc/passwd',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_os_popen(self):
        """os.popen should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (os.popen, ('echo pwned',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)

    def test_block_arbitrary_module(self):
        """Importing arbitrary modules should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (__import__, ('shutil',))

        buf = io.BytesIO()
        pickle.dump(Exploit(), buf)
        buf.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load_pickle(buf)


class TestReconstructDenseTensorData(unittest.TestCase):
    """Test the safe DenseTensor reconstruction function."""

    def test_reconstruct_returns_data_unchanged(self):
        """_reconstruct_dense_tensor_data should return data as-is."""
        from paddle.framework.io_utils import _reconstruct_dense_tensor_data

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _reconstruct_dense_tensor_data(data)
        np.testing.assert_array_equal(result, data)
        self.assertIs(result, data)

    def test_reconstruct_with_multidim_array(self):
        """Should work with multi-dimensional arrays."""
        from paddle.framework.io_utils import _reconstruct_dense_tensor_data

        data = np.random.randn(3, 4, 5).astype(np.float32)
        result = _reconstruct_dense_tensor_data(data)
        np.testing.assert_array_equal(result, data)


class TestPickleLoadsMac(unittest.TestCase):
    """Test the Mac-specific pickle loading path."""

    def test_pickle_loads_mac_basic(self):
        """_pickle_loads_mac should load pickled data from a file."""
        from paddle.framework.io_utils import _pickle_loads_mac

        data = {'weight': np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        with tempfile.NamedTemporaryFile(suffix='.pdparams', delete=False) as f:
            pickle.dump(data, f)
            path = f.name
        try:
            with open(path, 'rb') as f:
                result = _pickle_loads_mac(path, f)
            np.testing.assert_array_equal(result['weight'], data['weight'])
        finally:
            os.unlink(path)

    def test_pickle_loads_mac_state_dict(self):
        """_pickle_loads_mac should handle realistic state_dict data."""
        from paddle.framework.io_utils import _pickle_loads_mac

        data = collections.OrderedDict(
            [
                ('layer.weight', np.random.randn(10, 10).astype(np.float32)),
                ('layer.bias', np.zeros(10, dtype=np.float32)),
            ]
        )
        with tempfile.NamedTemporaryFile(suffix='.pdparams', delete=False) as f:
            pickle.dump(data, f)
            path = f.name
        try:
            with open(path, 'rb') as f:
                result = _pickle_loads_mac(path, f)
            self.assertIsInstance(result, collections.OrderedDict)
            for key in data:
                np.testing.assert_array_equal(result[key], data[key])
        finally:
            os.unlink(path)


class TestRestrictedUnpicklerWithFile(unittest.TestCase):
    """Test with actual file I/O (simulating .pdparams loading)."""

    def test_safe_model_file(self):
        """Loading a legitimate model file should work."""
        state_dict = {
            'conv1.weight': np.random.randn(64, 3, 7, 7).astype(np.float32),
            'conv1.bias': np.zeros(64, dtype=np.float32),
            'bn1.running_mean': np.zeros(64, dtype=np.float32),
            'bn1.running_var': np.ones(64, dtype=np.float32),
        }
        with tempfile.NamedTemporaryFile(suffix='.pdparams', delete=False) as f:
            pickle.dump(state_dict, f)
            tmpfile = f.name

        try:
            with open(tmpfile, 'rb') as f:
                loaded = safe_load_pickle(f)
            for key in state_dict:
                np.testing.assert_array_equal(loaded[key], state_dict[key])
        finally:
            os.unlink(tmpfile)

    def test_malicious_model_file(self):
        """Loading a malicious model file should raise an error."""

        class RCEPayload:
            def __reduce__(self):
                return (os.system, ('echo HACKED > /tmp/pwned.txt',))

        with tempfile.NamedTemporaryFile(suffix='.pdparams', delete=False) as f:
            pickle.dump(RCEPayload(), f)
            tmpfile = f.name

        try:
            with (
                open(tmpfile, 'rb') as f,
                self.assertRaises(pickle.UnpicklingError),
            ):
                safe_load_pickle(f)
        finally:
            os.unlink(tmpfile)


if __name__ == '__main__':
    unittest.main()
