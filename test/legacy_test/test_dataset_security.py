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

import io
import os
import tarfile
import tempfile
import unittest
from unittest import mock

import paddle.vision.datasets.cifar as vision_cifar
from paddle.dataset.cifar import CIFAR10_MD5, CIFAR100_MD5, reader_creator
from paddle.utils.download import _safe_extract_tar
from paddle.vision.datasets import Cifar10, Cifar100


class TestCifarSecurity(unittest.TestCase):
    def _write_tar(self, data_file):
        with tarfile.open(data_file, 'w:gz') as data_tar:
            content = b'not an official CIFAR archive'
            info = tarfile.TarInfo('cifar-10-batches-py/data_batch_1')
            info.size = len(content)
            data_tar.addfile(info, io.BytesIO(content))

    def test_reject_unverified_local_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'cifar-10-python.tar.gz')
            self._write_tar(data_file)

            with self.assertRaisesRegex(
                ValueError,
                'unverified local CIFAR pickle archive|official archive|MD5',
            ):
                Cifar10(data_file=data_file, download=False)

    def test_reject_missing_local_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'missing.tar.gz')

            with self.assertRaisesRegex(
                ValueError, 'Local CIFAR archive does not exist'
            ):
                Cifar10(data_file=data_file, download=False)

    def test_legacy_reader_rechecks_archive_md5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'cifar-10-python.tar.gz')
            self._write_tar(data_file)
            reader = reader_creator(
                data_file,
                'data_batch',
                md5sum=CIFAR10_MD5,
            )

            with self.assertRaisesRegex(
                ValueError, 'unverified CIFAR pickle archive|official MD5'
            ):
                list(reader())

    def _count_local_cifar_md5_calls(
        self,
        data_file,
        dataset_cls=Cifar10,
        md5sum=CIFAR10_MD5,
        mutate=None,
    ):
        md5_calls = []

        def fake_md5file(path):
            md5_calls.append(path)
            return md5sum

        vision_cifar._cached_md5file.cache_clear()
        try:
            with (
                mock.patch.object(vision_cifar, 'md5file', fake_md5file),
                mock.patch.object(
                    dataset_cls,
                    '_load_data',
                    lambda dataset: setattr(dataset, 'data', []),
                ),
            ):
                dataset_cls(data_file=data_file, download=False)
                if mutate is not None:
                    mutate()
                dataset_cls(data_file=data_file, download=False)
        finally:
            vision_cifar._cached_md5file.cache_clear()

        return len(md5_calls)

    def test_local_archive_md5_cache_reuses_unchanged_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'cifar-10-python.tar.gz')
            self._write_tar(data_file)

            self.assertEqual(self._count_local_cifar_md5_calls(data_file), 1)

    def test_cifar100_local_archive_md5_cache_reuses_unchanged_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'cifar-100-python.tar.gz')
            self._write_tar(data_file)

            self.assertEqual(
                self._count_local_cifar_md5_calls(
                    data_file,
                    dataset_cls=Cifar100,
                    md5sum=CIFAR100_MD5,
                ),
                1,
            )

    def test_local_archive_md5_cache_refreshes_changed_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'cifar-10-python.tar.gz')
            self._write_tar(data_file)

            def mutate():
                with open(data_file, 'ab') as f:
                    f.write(b'changed')

            self.assertEqual(
                self._count_local_cifar_md5_calls(data_file, mutate=mutate), 2
            )


class TestFlowersSafeExtractTar(unittest.TestCase):
    def _write_tar_members(self, data_file, members):
        with tarfile.open(data_file, 'w:gz') as data_tar:
            for info, content in members:
                fileobj = io.BytesIO(content) if content is not None else None
                data_tar.addfile(info, fileobj)

    def test_reject_path_traversal_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            outside_file = os.path.join(tmpdir, 'evil.txt')
            os.mkdir(extract_dir)

            with tarfile.open(data_file, 'w:gz') as data_tar:
                content = b'evil'
                info = tarfile.TarInfo('../evil.txt')
                info.size = len(content)
                data_tar.addfile(info, io.BytesIO(content))

            with (
                tarfile.open(data_file) as data_tar,
                self.assertRaises(ValueError),
            ):
                _safe_extract_tar(data_tar, extract_dir)

            self.assertFalse(os.path.exists(outside_file))

    def test_reject_invalid_on_unsafe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            os.mkdir(extract_dir)

            with tarfile.open(data_file, 'w:gz'):
                pass

            with (
                tarfile.open(data_file) as data_tar,
                self.assertRaisesRegex(
                    ValueError, "on_unsafe must be one of 'skip' or 'raise'"
                ),
            ):
                _safe_extract_tar(data_tar, extract_dir, on_unsafe='bad')

    def test_raise_on_symlink_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            os.mkdir(extract_dir)

            symlink_info = tarfile.TarInfo('jpg/link.jpg')
            symlink_info.type = tarfile.SYMTYPE
            symlink_info.linkname = 'image_00001.jpg'
            self._write_tar_members(data_file, [(symlink_info, None)])

            with (
                tarfile.open(data_file) as data_tar,
                self.assertRaisesRegex(ValueError, 'symbolic link'),
            ):
                _safe_extract_tar(data_tar, extract_dir, on_unsafe='raise')

    def test_raise_on_hardlink_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            os.mkdir(extract_dir)

            hardlink_info = tarfile.TarInfo('jpg/hardlink.jpg')
            hardlink_info.type = tarfile.LNKTYPE
            hardlink_info.linkname = 'jpg/image_00001.jpg'
            self._write_tar_members(data_file, [(hardlink_info, None)])

            with (
                tarfile.open(data_file) as data_tar,
                self.assertRaisesRegex(ValueError, 'hard link'),
            ):
                _safe_extract_tar(data_tar, extract_dir, on_unsafe='raise')

    def test_raise_on_special_file_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            os.mkdir(extract_dir)

            special_info = tarfile.TarInfo('jpg/special')
            special_info.type = tarfile.FIFOTYPE
            self._write_tar_members(data_file, [(special_info, None)])

            with (
                tarfile.open(data_file) as data_tar,
                self.assertRaisesRegex(ValueError, 'special file'),
            ):
                _safe_extract_tar(data_tar, extract_dir, on_unsafe='raise')

    def test_skip_unsafe_member_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            os.mkdir(extract_dir)

            symlink_info = tarfile.TarInfo('jpg/link.jpg')
            symlink_info.type = tarfile.SYMTYPE
            symlink_info.linkname = 'image_00001.jpg'
            self._write_tar_members(data_file, [(symlink_info, None)])

            with (
                tarfile.open(data_file) as data_tar,
                self.assertWarnsRegex(UserWarning, 'symbolic link'),
            ):
                _safe_extract_tar(data_tar, extract_dir)

            self.assertFalse(os.path.exists(os.path.join(extract_dir, 'jpg')))

    def test_extract_regular_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'flowers.tgz')
            extract_dir = os.path.join(tmpdir, 'flowers')
            member_name = 'jpg/image_00001.jpg'
            content = b'image content'
            os.mkdir(extract_dir)

            with tarfile.open(data_file, 'w:gz') as data_tar:
                dir_info = tarfile.TarInfo('jpg')
                dir_info.type = tarfile.DIRTYPE
                dir_info.mode = 0o755
                data_tar.addfile(dir_info)

                file_info = tarfile.TarInfo(member_name)
                file_info.mode = 0o644
                file_info.size = len(content)
                data_tar.addfile(file_info, io.BytesIO(content))

            with tarfile.open(data_file) as data_tar:
                _safe_extract_tar(data_tar, extract_dir)

            extracted_file = os.path.join(extract_dir, member_name)
            with open(extracted_file, 'rb') as f:
                self.assertEqual(f.read(), content)
