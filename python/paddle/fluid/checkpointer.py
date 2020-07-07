# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class SerializableBase(object):
    def serialize(self, path):
        raise NotImplementedError

    def deserialize(self, path):
        raise NotImplementedError


class Checkpointer(object):
    def __init__(self, fs):
        self._fs = fs
        self._checkpoint_prefix = "__paddle_fleet_checkpoint__"

    def save_checkpoint(self, path, slists):
        """
        Serialize objects in slists to path
        """
        if not fs.is_exist(path):
            fs.mkdirs(path)
        else:
            assert fs.is_dir(path), "path:%s must be a directory".format(path)

        max_no = self._get_last_checkpoint_no(path, fs=fs)
        if max_no < 0:
            max_no = -1

        real_path = "{}/{}.{}".format(path, self._checkpoint_prefix, max_no + 1)
        tmp_path = "{}.tmp".format(real_path)
        saved_path = tmp_path

        local_fs = LocalFS()

        cache_path = None
        if fs.need_upload_download():
            cache_path = "{}/{}.{}.saved_cache".format(
                local_cache_path, self._checkpoint_prefix, max_no + 1)
            if not local_fs.is_exist(cache_path):
                local_fs.mkdirs(cache_path)
            else:
                assert fs.is_dir(
                    path), "cache path:{} must be a directory".format(
                        cache_path)

            saved_path = cache_path

        for s in slists:
            s.serialize(path)

        if fs.need_upload_download():
            fs.delete(tmp_path)
            fs.upload(cache_path, tmp_path)
        fs.mv(tmp_path, real_path)

        if not remain_all_checkpoint:
            self.clean_redundant_checkpoints(path)

    def load_checkpoint(self, path, slists):
        """
        Deserialize objects in slists from path
        """

        max_no = self._get_last_checkpoint_no(path, fs)

        if not ignore_empty:
            assert max_no >= 0, "Can't find checkpoint"

        if max_no < 0:
            return None

        local_fs = LocalFS()
        if fs.need_upload_download():
            cache_path = "{}/{}.{}.load_cache.{}".format(
                local_cache_path, self._checkpoint_prefix, max_no, trainer_id)
            if not local_fs.is_exist(local_cache_path):
                local_fs.mkdirs(local_cache_path)
            if local_fs.is_exist(cache_path):
                local_fs.delete(cache_path)

        real_path = "{}/{}.{}".format(path, self._checkpoint_prefix, max_no)
        load_path = real_path
        if fs.need_upload_download():
            fs.download(real_path, cache_path)
            load_path = cache_path

        for s in slists:
            s.deserialize(save_path)

    def _get_last_checkpoint_no(self, path):
        pass

    def clean_redundant_check_points(self):
        pass

    def get_path(self, path):
        pass
