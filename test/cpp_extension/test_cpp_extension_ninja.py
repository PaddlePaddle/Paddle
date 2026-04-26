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

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from setuptools import Distribution

from paddle.utils.cpp_extension.cpp_extension import (
    _as_command_list,
    _get_num_workers,
    _is_ninja_available,
    _join_ninja_shell_list,
    _ninja_escape_path,
    _nt_quote_args,
    _run_ninja_build,
    _write_ninja_file,
)
from paddle.utils.cpp_extension.extension_utils import (
    _write_setup_file,
)


class TestNinjaHelperFunctions(unittest.TestCase):
    def test_is_ninja_available(self):
        result = _is_ninja_available()
        self.assertIsInstance(result, bool)
        if sys.platform.startswith('linux'):
            self.assertTrue(result, "ninja should be available on Linux CI")

    def test_ninja_escape_path(self):
        self.assertEqual(_ninja_escape_path("/path/to/file"), "/path/to/file")
        self.assertEqual(
            _ninja_escape_path("/path with space/file"),
            "/path$ with$ space/file",
        )
        self.assertEqual(
            _ninja_escape_path("C:/path/to/file"), "C$:/path/to/file"
        )
        self.assertEqual(
            _ninja_escape_path("/path/$var/file"), "/path/$$var/file"
        )
        self.assertEqual(
            _ninja_escape_path("C:/path with $var/file"),
            "C$:/path$ with$ $$var/file",
        )

    def test_nt_quote_args(self):
        self.assertEqual(_nt_quote_args(None), [])
        self.assertEqual(_nt_quote_args([]), [])
        self.assertEqual(_nt_quote_args(['-c', '-O2']), ['-c', '-O2'])
        self.assertEqual(
            _nt_quote_args(['/path with space/file', '-c']),
            ['"/path with space/file"', '-c'],
        )
        self.assertEqual(
            _nt_quote_args(['"already quoted"', '-c']),
            ['"already quoted"', '-c'],
        )

    def test_join_ninja_shell_list(self):
        self.assertEqual(
            _join_ninja_shell_list("simple string"), "simple string"
        )
        self.assertEqual(_join_ninja_shell_list([]), "")
        result = _join_ninja_shell_list(['-c', '-O2', '-I/usr/include'])
        self.assertIn('-c', result)
        self.assertIn('-O2', result)
        result = _join_ninja_shell_list(['/path with space', '-c'])
        self.assertIn('/path with space', result)

    def test_as_command_list(self):
        self.assertEqual(_as_command_list("gcc"), ["gcc"])
        self.assertEqual(_as_command_list(["gcc", "-c"]), ["gcc", "-c"])
        self.assertEqual(_as_command_list(("gcc", "-c")), ["gcc", "-c"])

    def test_write_ninja_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ninja_path = os.path.join(tmpdir, "build.ninja")
            _write_ninja_file(ninja_path, "ninja_required_version = 1.5")

            with open(ninja_path, 'r', encoding='utf-8') as f:
                written_content = f.read()
            self.assertTrue(written_content.endswith('\n'))
            self.assertIn("ninja_required_version = 1.5", written_content)

            content_with_newline = "rule compile\n  command = $cxx -c $in\n"
            _write_ninja_file(ninja_path, content_with_newline)
            with open(ninja_path, 'r', encoding='utf-8') as f:
                written_content = f.read()
            self.assertTrue(written_content.endswith('\n'))

            nested_path = os.path.join(tmpdir, "nested", "dir", "build.ninja")
            _write_ninja_file(nested_path, "test content")
            self.assertTrue(os.path.exists(nested_path))

    def test_get_num_workers_with_max_jobs_env(self):
        with mock.patch.dict(os.environ, {'MAX_JOBS': '4'}, clear=False):
            result = _get_num_workers(verbose=False)
            self.assertEqual(result, 4)

    def test_run_ninja_build_windows_vc_env(self):
        if sys.platform != 'win32':
            self.skipTest("Windows-only test")
        with tempfile.TemporaryDirectory() as tmpdir:
            ninja_path = os.path.join(tmpdir, "build.ninja")
            _write_ninja_file(ninja_path, "rule cc\n  command = echo hello\n")

            vc_env_mock = {'PATH': '/vc/bin', 'INCLUDE': '/vc/include'}

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.distutils.util.get_platform',
                    return_value='win-amd64',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_vc_env',
                    return_value=vc_env_mock,
                ),
                mock.patch('subprocess.run') as mock_run,
            ):
                _run_ninja_build(tmpdir, verbose=True, error_prefix="Test")
                mock_run.assert_called_once()

    def test_run_ninja_build_windows_with_vscmd_env(self):
        if sys.platform != 'win32':
            self.skipTest("Windows-only test")
        with tempfile.TemporaryDirectory() as tmpdir:
            ninja_path = os.path.join(tmpdir, "build.ninja")
            _write_ninja_file(ninja_path, "rule cc\n  command = echo hello\n")

            with (
                mock.patch.dict(
                    os.environ, {'VSCMD_ARG_TGT_ARCH': 'x64'}, clear=False
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.IS_WINDOWS', True
                ),
                mock.patch('subprocess.run') as mock_run,
            ):
                _run_ninja_build(tmpdir, verbose=True, error_prefix="Test")
                mock_run.assert_called_once()


class _FakeUnixCompiler:
    compiler_type = 'unix'

    def __init__(self, objects, build_map):
        self.src_extensions = ['.c', '.cc', '.cpp']
        self.compiler_so = ['g++']
        self.compiler_cxx = ['g++']
        self._objects = objects
        self._build_map = build_map

    def compile(self, *args, **kwargs):
        raise AssertionError("original unix compile should be replaced")

    def link_shared_object(self, *args, **kwargs):
        return None

    def object_filenames(self, *args, **kwargs):
        return self._objects

    def _setup_compile(
        self,
        output_dir,
        macros,
        include_dirs,
        sources,
        depends,
        extra_postargs,
    ):
        pp_opts = [f'-I{inc}' for inc in (include_dirs or [])]
        return macros, self._objects, extra_postargs, pp_opts, self._build_map

    def _get_cc_args(self, pp_opts, debug, extra_preargs):
        return list(extra_preargs or []) + list(pp_opts)

    def set_executables(self, **kwargs):
        return None


class _FakeMsvcCompiler:
    compiler_type = 'msvc'

    def __init__(self, objects, build_map):
        self.src_extensions = ['.c', '.cc', '.cpp']
        self._cpp_extensions = ['.c', '.cc', '.cpp']
        self.cc = ['cl.exe']
        self.compile_options = ['/nologo', '/O2', '/W3', '/MD']
        self.compile_options_debug = [
            '/nologo',
            '/Od',
            '/MDd',
            '/Zi',
            '/W3',
            '/D_DEBUG',
        ]
        self.initialized = True
        self.spawn = lambda cmd: None
        self._objects = objects
        self._build_map = build_map

    def compile(self, *args, **kwargs):
        raise AssertionError("original msvc compile should be replaced")

    def initialize(self):
        self.initialized = True

    def object_filenames(self, *args, **kwargs):
        return self._objects

    def _setup_compile(
        self,
        output_dir,
        macros,
        include_dirs,
        sources,
        depends,
        extra_postargs,
    ):
        pp_opts = [f'/I{inc}' for inc in (include_dirs or [])]
        return macros, self._objects, extra_postargs, pp_opts, self._build_map


class TestBuildExtension(unittest.TestCase):
    def _build_extension(self, **kwargs):
        from paddle.utils.cpp_extension.cpp_extension import BuildExtension

        return BuildExtension(dist=Distribution(), **kwargs)

    def _run_build_with_fake_compiler(
        self,
        compiler,
        sources,
        extra_compile_args,
        include_dirs=None,
    ):
        ext = SimpleNamespace(
            name='fake_extension',
            _full_name='fake_extension',
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
        cmd = self._build_extension(use_ninja=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd.extensions = [ext]
            cmd.compiler = compiler
            cmd.build_temp = tmpdir
            cmd.build_lib = tmpdir
            cmd.verbose = False

            captured = {}

            def fake_write_ninja_file(path, content):
                captured['ninja_path'] = path
                captured['ninja_content'] = content

            def fake_build_extensions(_):
                _.compiler.compile(
                    sources,
                    output_dir=_.build_temp,
                    macros=[],
                    include_dirs=include_dirs or [],
                    debug=False,
                    extra_preargs=[],
                    extra_postargs=extra_compile_args,
                    depends=None,
                )

            patches = [
                mock.patch.object(cmd, '_check_abi', return_value=None),
                mock.patch.object(cmd, '_record_op_info', return_value=None),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.clean_object_if_change_cflags',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.define_paddle_extension_name',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._reset_so_rpath',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._write_ninja_file',
                    side_effect=fake_write_ninja_file,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._run_ninja_build',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.build_ext.build_extensions',
                    side_effect=fake_build_extensions,
                ),
            ]

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
            ):
                cmd.build_extensions()

        return captured

    def test_unix_ninja_build_file_contains_multiple_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_sub.cc"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_sub.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cc'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            captured = self._run_build_with_fake_compiler(
                compiler,
                sources,
                extra_compile_args={'cxx': ['-w', '-g'], 'nvcc': []},
                include_dirs=[str(test_dir / "include")],
            )

        content = captured['ninja_content']
        self.assertIn('rule compile', content)
        self.assertIn('deps = gcc', content)
        self.assertIn(_ninja_escape_path(os.path.abspath(objects[0])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(objects[1])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(sources[0])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(sources[1])), content)
        self.assertIn('post_cflags = -w -g', content)
        self.assertIn(
            f'default {_ninja_escape_path(os.path.abspath(objects[0]))} {_ninja_escape_path(os.path.abspath(objects[1]))}',
            content,
        )
        self.assertTrue(captured['ninja_path'].endswith('build.ninja'))

    def test_windows_ninja_build_file_contains_multiple_sources(self):
        if sys.platform != 'win32':
            self.skipTest("Windows-only test")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_sub.cc"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_sub.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cc'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)

            captured = self._run_build_with_fake_compiler(
                compiler,
                sources,
                extra_compile_args={'cxx': ['/wd4244'], 'nvcc': []},
                include_dirs=[str(test_dir / "include dir")],
            )

        content = captured['ninja_content']
        self.assertIn('rule compile', content)
        self.assertIn('deps = msvc', content)
        self.assertIn('command = $cxx /showIncludes', content)
        self.assertIn('cl.exe', content)
        self.assertIn('/wd4244', content)
        self.assertIn(_ninja_escape_path(os.path.abspath(objects[0])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(objects[1])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(sources[0])), content)
        self.assertIn(_ninja_escape_path(os.path.abspath(sources[1])), content)
        self.assertIn('/I', content)
        self.assertTrue(captured['ninja_path'].endswith('build.ninja'))

    def test_use_ninja_attribute_default(self):
        build_ext = self._build_extension()
        self.assertEqual(build_ext.use_ninja, _is_ninja_available())

    def test_use_ninja_attribute_explicit_false(self):
        build_ext = self._build_extension(use_ninja=False)
        self.assertFalse(build_ext.use_ninja)

    def test_use_ninja_attribute_explicit_true(self):
        build_ext = self._build_extension(use_ninja=True)
        self.assertEqual(build_ext.use_ninja, _is_ninja_available())

    def test_unix_compiler_with_use_ninja_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [str(test_dir / "custom_extension.cc")]
            objects = [str(test_dir / "build" / "custom_extension.o")]
            build_map = {objects[0]: (sources[0], '.cc')}
            compiler = _FakeUnixCompiler(objects, build_map)

            ext = SimpleNamespace(
                name='fake_extension',
                _full_name='fake_extension',
                sources=sources,
                extra_compile_args={'cxx': ['-w'], 'nvcc': []},
            )
            cmd = self._build_extension(use_ninja=False)

            cmd.extensions = [ext]
            cmd.compiler = compiler
            cmd.build_temp = tmpdir
            cmd.build_lib = tmpdir
            cmd.verbose = False

            original_compile = compiler.__class__.compile

            patches = [
                mock.patch.object(cmd, '_check_abi', return_value=None),
                mock.patch.object(cmd, '_record_op_info', return_value=None),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.clean_object_if_change_cflags',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.define_paddle_extension_name',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._reset_so_rpath',
                    return_value=None,
                ),
            ]

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):

                def fake_build_extensions(_):
                    self.assertNotEqual(
                        compiler.__class__.compile, original_compile
                    )

                with mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.build_ext.build_extensions',
                    side_effect=fake_build_extensions,
                ):
                    cmd.build_extensions()

    def test_msvc_compiler_with_use_ninja_false(self):
        if sys.platform != 'win32':
            self.skipTest("Windows-only test")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [str(test_dir / "custom_extension.cc")]
            objects = [str(test_dir / "build" / "custom_extension.obj")]
            build_map = {objects[0]: (sources[0], '.cc')}
            compiler = _FakeMsvcCompiler(objects, build_map)

            ext = SimpleNamespace(
                name='fake_extension',
                _full_name='fake_extension',
                sources=sources,
                extra_compile_args={'cxx': ['/wd4244'], 'nvcc': []},
            )
            cmd = self._build_extension(use_ninja=False)

            cmd.extensions = [ext]
            cmd.compiler = compiler
            cmd.build_temp = tmpdir
            cmd.build_lib = tmpdir
            cmd.verbose = False

            original_compile = compiler.compile

            patches = [
                mock.patch.object(cmd, '_check_abi', return_value=None),
                mock.patch.object(cmd, '_record_op_info', return_value=None),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.clean_object_if_change_cflags',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.define_paddle_extension_name',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._reset_so_rpath',
                    return_value=None,
                ),
            ]

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):

                def fake_build_extensions(_):
                    self.assertNotEqual(_.compiler.compile, original_compile)

                with mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.build_ext.build_extensions',
                    side_effect=fake_build_extensions,
                ):
                    cmd.build_extensions()


class TestNinjaGeneratedSetupFile(unittest.TestCase):
    def test_load_setup_file_uses_default_build_extension_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_path = os.path.join(tmpdir, "demo_setup.py")
            build_dir = os.path.join(tmpdir, "build")
            sources = ['custom_extension.cc', 'custom_sub.cc']

            _write_setup_file(
                'demo_extension',
                sources,
                setup_path,
                build_dir,
                ['include_dir'],
                ['library_dir'],
                ['-w', '-g'],
                [],
                [],
            )

            content = Path(setup_path).read_text(encoding='utf-8')

        self.assertIn('BuildExtension.with_options(', content)
        self.assertIn("output_dir=r'", content)
        self.assertIn('no_python_abi_suffix=True', content)
        self.assertIn(
            "sources=['custom_extension.cc', 'custom_sub.cc']", content
        )
        self.assertNotIn('use_ninja=', content)


if __name__ == '__main__':
    unittest.main()
