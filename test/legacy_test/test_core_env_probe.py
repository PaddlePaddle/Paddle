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

import sys
import unittest
from unittest import mock

from paddle.base import core


class TestRunCommand(unittest.TestCase):
    def test_missing_binary_returns_none(self):
        self.assertIsNone(core._run_command(['paddle-no-such-binary-xyz']))

    def test_never_spawns_a_shell(self):
        # This is the point of _run_command: hardened images may ship no
        # /bin/sh, so the command has to be executed directly.
        with mock.patch('subprocess.Popen') as popen:
            popen.return_value.communicate.return_value = (b'out', b'')
            core._run_command(['echo', 'hello'])
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ['echo', 'hello'])
        self.assertFalse(kwargs.get('shell', False))

    def test_output_on_stderr_returns_none(self):
        with mock.patch('subprocess.Popen') as popen:
            popen.return_value.communicate.return_value = (b'out', b'boom')
            self.assertIsNone(core._run_command(['whatever']))

    def test_merge_stderr_keeps_output(self):
        with mock.patch('subprocess.Popen') as popen:
            popen.return_value.communicate.return_value = (b' out \n', None)
            self.assertEqual(
                core._run_command(['whatever'], merge_stderr=True), 'out'
            )


class TestAvxSupported(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(core.avx_supported(), bool)


class TestGetDsoPath(unittest.TestCase):
    def test_missing_argument_returns_none(self):
        self.assertIsNone(core.get_dso_path(None, 'libgomp'))
        self.assertIsNone(core.get_dso_path('/path/libpaddle.so', None))

    def test_unavailable_ldd_returns_none(self):
        with mock.patch.object(core, '_run_command', return_value=None):
            self.assertIsNone(core.get_dso_path('/no/such.so', 'libgomp'))

    def test_parses_ldd_output(self):
        ldd_out = (
            '\tlinux-vdso.so.1 (0x00007ffd)\n'
            '\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x7f00)\n'
            '\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x7f01)'
        )
        with mock.patch.object(core, '_run_command', return_value=ldd_out):
            self.assertEqual(
                core.get_dso_path('/path/libpaddle.so', 'libgomp'),
                '/lib/x86_64-linux-gnu/libgomp.so.1',
            )

    def test_unmatched_dso_returns_none(self):
        with mock.patch.object(
            core, '_run_command', return_value='\tstatically linked'
        ):
            self.assertIsNone(
                core.get_dso_path('/path/libpaddle.so', 'libgomp')
            )


@unittest.skipUnless(
    sys.platform.startswith('linux'), 'get_libc_ver is only used on Linux'
)
class TestGetLibcVer(unittest.TestCase):
    def test_result_is_usable_by_the_import_path(self):
        libc_type, libc_ver = core.get_libc_ver()
        self.assertIn(libc_type, (None, 'glibc', 'musl'))
        if libc_type is None:
            self.assertIsNone(libc_ver)
        else:
            self.assertTrue(libc_ver)
            # core.py compares the version at import time, it must not raise.
            self.assertIsInstance(core.less_than_ver(libc_ver, '2.23'), bool)

    def test_musl_banner_is_parsed(self):
        banner = 'musl libc (x86_64)\nVersion 1.2.4\nDynamic Program Loader\n'
        with (
            mock.patch.object(core.os, 'confstr', return_value=None),
            mock.patch.object(core, '_run_command', return_value=banner),
        ):
            self.assertEqual(core.get_libc_ver(), ('musl', '1.2.4'))

    def test_no_libc_information_returns_none(self):
        with (
            mock.patch.object(core.os, 'confstr', return_value=None),
            mock.patch.object(core, '_run_command', return_value=None),
        ):
            self.assertEqual(core.get_libc_ver(), (None, None))
            # This used to return ("musl", "") and blow up less_than_ver.
            self.assertFalse(core.less_than_ver(None, '2.23'))


if __name__ == '__main__':
    unittest.main()
