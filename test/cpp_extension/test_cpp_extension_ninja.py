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
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, SimpleNamespace as ModuleSimpleNamespace
from unittest import mock

from setuptools import Distribution

from paddle.utils.cpp_extension.cpp_extension import (
    _as_command_list,
    _get_num_workers,
    _is_ninja_available,
    _join_ninja_shell_list,
    _ninja_escape_path,
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

    def test_is_ninja_available_handles_probe_failure(self):
        with mock.patch(
            'paddle.utils.cpp_extension.cpp_extension.subprocess.check_output',
            side_effect=OSError("ninja missing"),
        ):
            self.assertFalse(_is_ninja_available())

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

    def test_run_ninja_build_with_work_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = os.path.join(tmpdir, 'build')
            work_dir = os.path.join(tmpdir, 'src')
            os.makedirs(build_dir)
            os.makedirs(work_dir)

            with (
                mock.patch.dict(
                    os.environ, {'VSCMD_ARG_TGT_ARCH': 'x64'}, clear=False
                ),
                mock.patch('subprocess.run') as mock_run,
            ):
                _run_ninja_build(
                    build_dir,
                    verbose=True,
                    error_prefix='Test',
                    work_directory=work_dir,
                )

        command = mock_run.call_args.args[0]
        self.assertIn('-f', command)
        self.assertIn(os.path.join(build_dir, 'build.ninja'), command)
        self.assertEqual(mock_run.call_args.kwargs['cwd'], work_dir)

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
        self._compile_calls = []
        self.verbose = False

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

    def _compile(self, obj, src, ext, cc_args, cflags, pp_opts):
        self._compile_calls.append(
            {
                'obj': obj,
                'src': src,
                'ext': ext,
                'cc_args': list(cc_args),
                'cflags': list(cflags),
                'pp_opts': list(pp_opts),
                'compiler_so': list(self.compiler_so)
                if isinstance(self.compiler_so, list)
                else self.compiler_so,
            }
        )

    def set_executable(self, key, value):
        setattr(self, key, value)

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
        self._objects = objects
        self._build_map = build_map
        self._spawned_cmds = []
        self.verbose = False
        self.spawn = self._spawn

    def compile(self, *args, **kwargs):
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            kwargs.get('output_dir'),
            kwargs.get('macros'),
            kwargs.get('include_dirs'),
            args[0],
            kwargs.get('depends'),
            kwargs.get('extra_postargs'),
        )
        del macros, extra_postargs
        for obj in objects:
            src, ext = build[obj]
            cmd = [
                'cl.exe',
                '/Tp' + src,
                '/Fo' + obj,
                *pp_opts,
            ]
            if ext == '.c':
                cmd[1] = '/Tc' + src
            self.spawn(cmd)
        return objects

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

    def _spawn(self, cmd):
        self._spawned_cmds.append(list(cmd))


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
            cmd.contain_cuda_file = any(
                source.endswith('.cu') for source in sources
            )

            captured = {}

            def fake_write_ninja_file(path, content):
                captured['ninja_path'] = path
                captured['ninja_content'] = content

            def fake_run_ninja_build(
                build_directory, verbose, error_prefix, work_directory=None
            ):
                captured['ninja_build_directory'] = build_directory
                captured['ninja_work_directory'] = work_directory

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
                    side_effect=fake_run_ninja_build,
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

    def _patch_build_extension_common(self, cmd):
        return [
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

    @staticmethod
    def _expected_nvcc_path(cuda_home='/opt/cuda'):
        return os.path.join(cuda_home, 'bin', 'nvcc')

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

    def test_windows_ninja_quotes_flags_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            include_dir = test_dir / "include dir"
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_kernel.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.IS_WINDOWS',
                    True,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_win_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['/DNAME=value with space'],
                        'nvcc': ['--compiler-options=/Wall /WX'],
                    },
                    include_dirs=[str(include_dir)],
                )

        content = captured['ninja_content']
        self.assertIn(subprocess.list2cmdline([f'/I{include_dir}']), content)
        self.assertIn(
            subprocess.list2cmdline(['/DNAME=value with space']), content
        )
        self.assertIn(
            subprocess.list2cmdline(['--compiler-options=/Wall /WX']),
            content,
        )
        self.assertNotIn('\\"/I', content)
        self.assertNotIn('\\"/DNAME', content)

    def test_unix_ninja_build_file_contains_cuda_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_kernel.cu.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_ccache_home',
                    return_value='/usr/bin/ccache',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_unix_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_custom_device',
                    return_value=False,
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['-w'],
                        'nvcc': ['--gpu-flag'],
                    },
                    include_dirs=[str(test_dir / "include")],
                )

        content = captured['ninja_content']
        self.assertIn('cxx = /usr/bin/ccache g++', content)
        self.assertIn(
            f'nvcc = /usr/bin/ccache {self._expected_nvcc_path()}',
            content,
        )
        self.assertIn('rule cuda_compile', content)
        self.assertIn('cuda_post_cflags = --prepared --gpu-flag', content)
        self.assertIn('post_cflags = -w -D_GLIBCXX_USE_CXX11_ABI=1', content)
        self.assertIn('-DPADDLE_WITH_CUDA', content)

    def test_unix_ninja_quotes_flags_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            include_dir = test_dir / "include dir"
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_kernel.cu.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.IS_WINDOWS',
                    False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_ccache_home',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_unix_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_custom_device',
                    return_value=False,
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['-DNAME=value with space'],
                        'nvcc': ['--compiler-options=-Wall -Wextra'],
                    },
                    include_dirs=[str(include_dir)],
                )

        content = captured['ninja_content']
        self.assertIn(shlex.join([f'-I{include_dir}']), content)
        self.assertIn(shlex.join(['-DNAME=value with space']), content)
        self.assertIn(shlex.join(['--compiler-options=-Wall -Wextra']), content)
        self.assertNotIn('"\'"\'"', content)

    def test_unix_ninja_preserves_relative_include_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_kernel.cu.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.os.getcwd',
                    return_value=str(test_dir),
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_ccache_home',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_unix_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_custom_device',
                    return_value=False,
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['-Ithird_party/cxx', '-isystem', 'system/cxx'],
                        'nvcc': ['-Ithird_party/cuda', '-I', 'cuda_split'],
                    },
                    include_dirs=['public/include'],
                )

        content = captured['ninja_content']
        self.assertIn('-Ipublic/include', content)
        self.assertIn('-Ithird_party/cxx', content)
        self.assertIn('-isystem system/cxx', content)
        self.assertIn('-Ithird_party/cuda', content)
        self.assertIn('-I cuda_split', content)
        self.assertEqual(captured['ninja_work_directory'], str(test_dir))

    def test_unix_ninja_uses_hipcc_for_rocm_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_kernel.cu.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.ROCM_HOME',
                    '/opt/rocm',
                    create=True,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_ccache_home',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_unix_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=True,
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['-w'],
                        'hipcc': ['--hip-flag'],
                    },
                    include_dirs=[str(test_dir / "include")],
                )

        content = captured['ninja_content']
        self.assertIn('nvcc = ', content)
        self.assertIn('hipcc', content)
        self.assertIn('cuda_post_cflags = --prepared --hip-flag', content)
        self.assertIn('-D__HIP_PLATFORM_HCC__', content)
        self.assertIn(
            '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP', content
        )
        self.assertIn('-DPADDLE_WITH_HIP', content)

    def test_unix_ninja_reports_missing_corex_compiler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [str(test_dir / "custom_kernel.cu")]
            objects = [str(test_dir / "build" / "custom_kernel.cu.o")]
            build_map = {objects[0]: (sources[0], '.cu')}
            compiler = _FakeUnixCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_custom_device',
                    return_value=True,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.os.path.isfile',
                    return_value=False,
                ),
                self.assertRaisesRegex(ValueError, 'Corex compiler'),
            ):
                self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={'nvcc': ['--gpu-flag']},
                )

    def test_use_ninja_attribute_default(self):
        build_ext = self._build_extension()
        self.assertEqual(build_ext.use_ninja, _is_ninja_available())

    def test_use_ninja_attribute_explicit_false(self):
        build_ext = self._build_extension(use_ninja=False)
        self.assertFalse(build_ext.use_ninja)

    def test_use_ninja_attribute_explicit_true(self):
        build_ext = self._build_extension(use_ninja=True)
        self.assertEqual(build_ext.use_ninja, _is_ninja_available())

    def test_use_ninja_falls_back_when_ninja_missing(self):
        with (
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension._is_ninja_available',
                return_value=False,
            ),
            mock.patch('builtins.print') as mock_print,
        ):
            build_ext = self._build_extension(use_ninja=True)
        self.assertFalse(build_ext.use_ninja)
        mock_print.assert_any_call(
            "Ninja is not available, falling back to the distutils backend."
        )

    def test_unix_compiler_with_use_ninja_false_executes_compile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.o"),
                str(test_dir / "build" / "custom_kernel.o"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeUnixCompiler(objects, build_map)

            ext = SimpleNamespace(
                name='fake_extension',
                _full_name='fake_extension',
                sources=sources,
                extra_compile_args={'cxx': ['-w'], 'nvcc': ['--gpu-flag']},
            )
            cmd = self._build_extension(use_ninja=False)

            cmd.extensions = [ext]
            cmd.compiler = compiler
            cmd.build_temp = tmpdir
            cmd.build_lib = tmpdir
            cmd.verbose = False
            cmd.contain_cuda_file = True

            patches = self._patch_build_extension_common(cmd)

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_unix_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_ccache_home',
                    return_value=None,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension._get_num_workers',
                    return_value=4,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.os.cpu_count',
                    return_value=8,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_rocm',
                    return_value=False,
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.core.is_compiled_with_custom_device',
                    return_value=False,
                ),
                mock.patch('builtins.print') as mock_print,
            ):

                def fake_build_extensions(_):
                    compiled = _.compiler.compile(
                        sources,
                        output_dir=_.build_temp,
                        macros=[],
                        include_dirs=[str(test_dir / "include")],
                        debug=False,
                        extra_preargs=[],
                        extra_postargs=ext.extra_compile_args,
                        depends=None,
                    )
                    self.assertEqual(compiled, objects)

                with mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.build_ext.build_extensions',
                    side_effect=fake_build_extensions,
                ):
                    cmd.build_extensions()

        self.assertEqual(len(compiler._compile_calls), 2)
        cc_call = next(
            call
            for call in compiler._compile_calls
            if call['src'].endswith('.cc')
        )
        cu_call = next(
            call
            for call in compiler._compile_calls
            if call['src'].endswith('.cu')
        )
        self.assertIn('-DPADDLE_WITH_CUDA', cc_call['cflags'])
        self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=1', cc_call['cflags'])
        self.assertIn('--prepared', cu_call['cflags'])
        self.assertEqual(cu_call['compiler_so'], self._expected_nvcc_path())
        mock_print.assert_any_call(
            "Using 2 workers for compilation. HINT: export MAX_JOBS=n to set the number of workers"
        )

    def test_msvc_compiler_with_use_ninja_false_executes_compile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_kernel.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)

            ext = SimpleNamespace(
                name='fake_extension',
                _full_name='fake_extension',
                sources=sources,
                extra_compile_args={
                    'cxx': ['/wd4244'],
                    'nvcc': ['--gpu-flag'],
                },
            )
            cmd = self._build_extension(use_ninja=False)

            cmd.extensions = [ext]
            cmd.compiler = compiler
            cmd.build_temp = tmpdir
            cmd.build_lib = tmpdir
            cmd.verbose = False
            cmd.contain_cuda_file = True

            patches = self._patch_build_extension_common(cmd)

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_win_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
            ):

                def fake_build_extensions(_):
                    _.compiler.compile(
                        sources,
                        output_dir=_.build_temp,
                        macros=[],
                        include_dirs=[str(test_dir / "include dir")],
                        debug=False,
                        extra_preargs=[],
                        extra_postargs=ext.extra_compile_args,
                        depends=None,
                    )

                with mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.build_ext.build_extensions',
                    side_effect=fake_build_extensions,
                ):
                    cmd.build_extensions()

        self.assertEqual(len(compiler._spawned_cmds), 2)
        cc_cmd = next(
            cmd
            for cmd in compiler._spawned_cmds
            if any(arg.endswith('.cc') for arg in cmd)
        )
        cu_cmd = next(
            cmd
            for cmd in compiler._spawned_cmds
            if any(arg.endswith('.cu') for arg in cmd)
        )
        self.assertIn('-DPADDLE_WITH_CUDA', cc_cmd)
        self.assertIn('/wd4244', cc_cmd)
        self.assertEqual(cu_cmd[0], self._expected_nvcc_path())
        self.assertIn('--prepared', cu_cmd)
        self.assertIn('--use-local-env', cu_cmd)

    def test_windows_ninja_build_file_contains_cuda_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_kernel.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_win_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['/wd4244'],
                        'nvcc': ['--gpu-flag'],
                    },
                    include_dirs=[str(test_dir / "include dir")],
                )

        content = captured['ninja_content']
        self.assertIn(f'nvcc = {self._expected_nvcc_path()}', content)
        self.assertIn('rule cuda_compile', content)
        self.assertIn('-Xcompiler /EHsc', content)
        self.assertIn('--prepared', content)
        self.assertIn('/DPADDLE_WITH_CUDA', content)
        self.assertIn('deps = msvc', content)

    def test_windows_ninja_preserves_relative_include_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_kernel.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.os.getcwd',
                    return_value=str(test_dir),
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_win_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args={
                        'cxx': ['/Ithird_party/cxx'],
                        'nvcc': ['/Ithird_party/cuda'],
                    },
                    include_dirs=['public/include'],
                )

        content = captured['ninja_content']
        self.assertIn('/Ipublic/include', content)
        self.assertIn('/Ithird_party/cxx', content)
        self.assertIn('/Ithird_party/cuda', content)
        self.assertEqual(captured['ninja_work_directory'], str(test_dir))

    def test_windows_ninja_ignores_invalid_extra_compile_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            sources = [
                str(test_dir / "custom_extension.cc"),
                str(test_dir / "custom_kernel.cu"),
            ]
            objects = [
                str(test_dir / "build" / "custom_extension.obj"),
                str(test_dir / "build" / "custom_kernel.obj"),
            ]
            build_map = {
                objects[0]: (sources[0], '.cc'),
                objects[1]: (sources[1], '.cu'),
            }
            compiler = _FakeMsvcCompiler(objects, build_map)
            invalid_extra_args = 'not-a-list'

            with (
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.CUDA_HOME',
                    '/opt/cuda',
                ),
                mock.patch(
                    'paddle.utils.cpp_extension.cpp_extension.prepare_win_cudaflags',
                    side_effect=lambda flags: ['--prepared', *flags],
                ),
            ):
                captured = self._run_build_with_fake_compiler(
                    compiler,
                    sources,
                    extra_compile_args=invalid_extra_args,
                    include_dirs=[str(test_dir / "include dir")],
                )

        content = captured['ninja_content']
        self.assertIn('rule cuda_compile', content)
        self.assertIn('cuda_post_cflags = ', content)
        self.assertIn('-Xcompiler /EHsc', content)
        self.assertIn('--prepared --use-local-env', content)
        self.assertIn('/DPADDLE_WITH_CUDA', content)
        self.assertNotIn(invalid_extra_args, content)


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


class TestRunNinjaBuild(unittest.TestCase):
    def test_run_ninja_build_uses_verbose_subprocess_streams(self):
        with (
            mock.patch.dict(
                os.environ, {'VSCMD_ARG_TGT_ARCH': 'x64'}, clear=False
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.subprocess.run'
            ) as mock_run,
        ):
            _run_ninja_build('/tmp/build', verbose=True, error_prefix='Test')

        _, kwargs = mock_run.call_args
        self.assertEqual(mock_run.call_args.args[0], ['ninja', '-v'])
        self.assertEqual(kwargs['cwd'], '/tmp/build')
        self.assertIsNone(kwargs['stdout'])
        self.assertIsNone(kwargs['stderr'])

    def test_run_ninja_build_uses_num_workers_and_quiet_streams(self):
        with (
            mock.patch.dict(
                os.environ,
                {'MAX_JOBS': '3', 'VSCMD_ARG_TGT_ARCH': 'x64'},
                clear=False,
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.subprocess.run'
            ) as mock_run,
        ):
            _run_ninja_build('/tmp/build', verbose=False, error_prefix='Test')

        _, kwargs = mock_run.call_args
        self.assertEqual(mock_run.call_args.args[0], ['ninja', '-v', '-j', '3'])
        self.assertEqual(kwargs['stdout'], subprocess.PIPE)
        self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
        self.assertTrue(kwargs['text'])

    def test_run_ninja_build_windows_merges_vc_env(self):
        vc_env = {'path': 'vc/bin', 'include': 'vc/include'}
        with (
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.IS_WINDOWS', True
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.PLAT_TO_VCVARS',
                {'win-amd64': 'x86_amd64'},
                create=True,
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.distutils',
                ModuleSimpleNamespace(
                    util=ModuleSimpleNamespace(get_platform=lambda: 'win-amd64')
                ),
                create=True,
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension._get_vc_env',
                return_value=vc_env,
                create=True,
            ),
            mock.patch.dict(
                os.environ, {'Path': 'user/bin', 'TMP': 'tmp'}, clear=True
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.subprocess.run'
            ) as mock_run,
        ):
            _run_ninja_build('/tmp/build', verbose=False, error_prefix='Test')

        env = mock_run.call_args.kwargs['env']
        self.assertEqual(env['PATH'], 'vc/bin')
        self.assertEqual(env['INCLUDE'], 'vc/include')
        self.assertEqual(env['TMP'], 'tmp')

    def test_run_ninja_build_raises_runtime_error(self):
        with (
            mock.patch.dict(
                os.environ, {'VSCMD_ARG_TGT_ARCH': 'x64'}, clear=False
            ),
            mock.patch(
                'paddle.utils.cpp_extension.cpp_extension.subprocess.run',
                side_effect=subprocess.CalledProcessError(1, ['ninja', '-v']),
            ),
            self.assertRaisesRegex(RuntimeError, 'Prefix'),
        ):
            _run_ninja_build('/tmp/build', verbose=False, error_prefix='Prefix')

    def test_get_num_workers_verbose_without_env(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch('sys.stderr') as mock_stderr,
            mock.patch('builtins.print') as mock_print,
        ):
            result = _get_num_workers(verbose=True)

        self.assertIsNone(result)
        mock_print.assert_called_with(
            'Allowing ninja to set a default number of workers... '
            '(overridable by setting the environment variable MAX_JOBS=N)',
            file=mock_stderr,
        )


if __name__ == '__main__':
    unittest.main()
