# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import re
import shutil

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger
from paddle.distributed.fleet.utils.pp_parallel_adaptor import (
    ParallelConfig,
    PipeLineModelAdaptor,
)


class PipeLineModelLoader:
    def __init__(self, model=None, optimizer=None, hcg=None):
        # distributed model and optimizer
        self._model = model
        self._optimizer = optimizer
        if self._model:
            assert isinstance(
                self._model, fleet.meta_parallel.PipelineParallel
            ), "must be pipeline model"
            if not hcg:
                hcg = fleet.get_hybrid_communicate_group()
        self._hcg = hcg

    def _create_subpath_name(self):
        return "mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
            *self._get_ranks()
        )

    def _get_ranks(self):
        mp_rank = self._hcg.get_model_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        pp_rank = self._hcg.get_stage_id()
        return (mp_rank, sharding_rank, pp_rank)

    def _create_dir(self, dir_path):
        try:
            os.makedirs(dir_path)
        except:
            # dir is already created, do nothing
            pass

    def save(self, save_dir, epoch, step):
        # only dp rank 0 do the work
        if self._hcg.get_data_parallel_rank() == 0:
            self._create_dir(save_dir)
            sub_dir = os.path.join(save_dir, self._create_subpath_name())
            self._create_dir(sub_dir)
            paddle.save(
                self._model.state_dict(),
                os.path.join(sub_dir, "model.pdparams"),
            )
            paddle.save(
                self._optimizer.state_dict(),
                os.path.join(sub_dir, "model_state.pdopt"),
            )
            meta_dict = {
                "epoch": epoch,
                "step": step,
                "cuda_rng_state": paddle.get_cuda_rng_state(),
            }
            paddle.save(meta_dict, os.path.join(sub_dir, "meta_state.pdopt"))

    def _list_subdirs(self, model_dir):
        names = [
            os.path.basename(x)
            for x in os.listdir(model_dir)
            if not os.path.isfile(x)
        ]
        names = [
            e
            for e in names
            if re.match(r'^mp_([0-9]{2})_sharding_([0-9]{2})_pp_([0-9]{2})$', e)
        ]
        return names

    def _parse_grid_dim(self, subdirs):
        regs = [
            re.match(r'^mp_([0-9]{2})_sharding_([0-9]{2})_pp_([0-9]{2})$', e)
            for e in subdirs
        ]
        nodes = [(g.group(1), g.group(2), g.group(3)) for g in regs]
        mp_degree = max([int(e[0]) for e in nodes]) + 1
        sharding_degree = max([int(e[1]) for e in nodes]) + 1
        pp_degree = max([int(e[2]) for e in nodes]) + 1
        assert mp_degree * sharding_degree * pp_degree == len(
            nodes
        ), f"dirs {subdirs} are not valid sub models"
        return (mp_degree, sharding_degree, pp_degree)

    def load(
        self,
        model_dir,
        model_vp_degree=1,
        transformer_layer_num=None,
        segment_method="uniform",
        force_convert=False,
        with_opt=True,
    ):

        subdirs = self._list_subdirs(model_dir)
        assert subdirs, "no subdir found"
        (
            model_mp_degree,
            model_sharding_degree,
            model_pp_degree,
        ) = self._parse_grid_dim(subdirs)
        assert (
            model_mp_degree == self._hcg.get_model_parallel_world_size()
        ), "mp_degree can not change when train from checkpoint"
        assert (
            model_sharding_degree
            == self._hcg.get_sharding_parallel_world_size()
        ), "sharding_degree can not change when train from checkpoint"
        cur_pp_degree = self._hcg.get_pipe_parallel_world_size()
        cur_vp_degree = self._model.get_num_virtual_stages()
        converted_model_path = model_dir
        mp_rank = self._hcg.get_model_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        # convert model as appropriate
        if (
            cur_pp_degree != model_mp_degree
            or model_vp_degree != cur_vp_degree
            or force_convert
        ):
            converted_model_path = (
                "./tmp_converted{:0>2d}_sharding_{:0>2d}".format(
                    mp_rank, sharding_rank
                )
            )
            self._convert_model(
                model_dir,
                converted_model_path,
                model_pp_degree,
                model_vp_degree,
                cur_vp_degree,
                transformer_layer_num,
                segment_method,
            )
        # recover model
        self._load(converted_model_path, with_opt)
        # remove tmp files
        if converted_model_path != model_dir:
            self._remove_converted_model(converted_model_path)

    def _convert_model(
        self,
        src_model_dir,
        dest_model_dir,
        model_pp_degree,
        model_vp_degree,
        cur_vp_degree,
        transformer_layer_num,
        segment_method,
    ):
        # convert model, only dp_rank 0 and pp_rank 0 with do the work
        # barrier cross pp group and pp group
        logger.info(
            f"begin convert model in {src_model_dir} with pp {model_pp_degree} vp {model_vp_degree} to {dest_model_dir}"
        )
        cur_model_degree = self._hcg.get_model_parallel_world_size()
        cur_sharding_degree = self._hcg.get_sharding_parallel_world_size()
        cur_pp_degree = self._hcg.get_pipe_parallel_world_size()

        src_parallel_config = ParallelConfig(
            cur_model_degree,
            model_pp_degree,
            model_vp_degree,
            cur_sharding_degree,
        )

        dst_parallel_config = ParallelConfig(
            cur_model_degree, cur_pp_degree, cur_vp_degree, cur_sharding_degree
        )

        adaptor = PipeLineModelAdaptor(
            src_parallel_config,
            dst_parallel_config,
            transformer_layer_num,
            segment_method,
        )

        mp_rank = self._hcg.get_model_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        pp_rank = self._hcg.get_stage_id()
        dp_rank = self._hcg.get_data_parallel_rank()

        if dp_rank == 0:
            if pp_rank == 0:
                adaptor.apply_for_pp_group(
                    src_model_dir, dest_model_dir, mp_rank, sharding_rank
                )
            # barrier cross pp group
            paddle.distributed.barrier(self._hcg.get_pipe_parallel_group())
        # barrier cross dp group
        paddle.distributed.barrier(self._hcg.get_data_parallel_group())
        logger.info(f"end convert model {src_model_dir} to {dest_model_dir}")

    def _load(self, model_dir, with_opt):
        full_path = os.path.join(model_dir, self._create_subpath_name())
        logger.info(f"begin load model from {full_path}")
        model_path = os.path.join(full_path, "model.pdparams")
        opt_path = os.path.join(full_path, "model_state.pdopt")
        meta_path = os.path.join(full_path, "meta_state.pdopt")
        if os.path.exists(model_path):
            model_dict = paddle.load(model_path)
            if self._model:
                for name, param in self._model.state_dict().items():
                    assert (
                        name in model_dict.keys()
                    ), "No param named `{}` was found in checkpoint file.".format(
                        name
                    )
                    if param.dtype != model_dict[name].dtype:
                        model_dict[name] = model_dict[name].cast(param.dtype)

                self._model.set_state_dict(model_dict)
        else:
            raise ValueError(
                "No model checkpoint file found in %s." % model_path
            )

        if with_opt:
            if os.path.exists(opt_path):
                opt_dict = paddle.load(opt_path)
                if self._optimizer:
                    self._optimizer.set_state_dict(opt_dict)
            else:
                raise ValueError(
                    "No optimizer checkpoint file found in %s." % opt_path
                )
        logger.info(f"load model from {full_path} successfully")

    def _remove_converted_model(self, sub_model_dir):
        paddle.distributed.barrier()

        pp_rank = self._hcg.get_stage_id()
        dp_rank = self._hcg.get_data_parallel_rank()
        if dp_rank == 0:
            if pp_rank == 0:
                shutil.rmtree(sub_model_dir, ignore_errors=True)


if __name__ == "__main__":
    loader = PipeLineModelLoader()
    subdirs = loader._list_subdirs("./output/epoch_0_step_90")
    logger.info(subdirs)
    logger.info(loader._parse_grid_dim(subdirs))
