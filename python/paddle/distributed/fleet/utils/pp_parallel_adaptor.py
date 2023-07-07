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

import argparse
import math
import re
import shutil
from collections import OrderedDict

import paddle


class ParallelConfig:
    def __init__(self, mp: int, pp: int, vpp: int = 1, sharding: int = 1):
        self.mp = mp
        self.pp = pp
        self.vpp = vpp
        self.sharding = sharding

    def pipe_parallel_group(self, i: int, j: int):
        ans = []
        for k in range(self.pp):
            ans.append((i, j, k))
        return ans


class LayerReNamingHelper:
    def __init__(self, template: str):
        self._template = template
        self._i = -1
        self._last_old_layer_name = None

    def get_new_layer_name(self, old_layer_name: str):
        old_layer_name = old_layer_name.split(".")[0]
        if (
            self._last_old_layer_name is None
            or old_layer_name != self._last_old_layer_name
        ):
            self._i = self._i + 1
            self._last_old_layer_name = old_layer_name
        return self._template.format(self._i)


class LayerReNamingManager:
    def __init__(self):
        self._renaming_helpers = OrderedDict()
        self._renaming_helpers["linear"] = LayerReNamingHelper("linear_{}")
        self._renaming_helpers["layer_norm"] = LayerReNamingHelper(
            "layer_norm_{}"
        )
        self._renaming_helpers["embedding"] = LayerReNamingHelper(
            "embedding_{}"
        )

    def get_new_layer_name(self, old_name: str):
        layer_name = ""
        for (k, v) in self._renaming_helpers.items():
            if old_name.startswith(k):
                layer_name = v.get_new_layer_name(old_name)
                break
        return layer_name

    def get_new_param_name(self, old_name: str):
        names = old_name.split(".")
        layer_name = self.get_new_layer_name(names[0])
        assert layer_name, f"can not rename layer {names[0]}"
        names[0] = layer_name
        return ".".join(names)


class PipeLineModelAdaptor:
    def __init__(
        self,
        src_parallel_config: ParallelConfig,
        dst_parallel_config: ParallelConfig,
        transformer_layer_num: int,
        segment_method: str = "layer",
    ):
        self._src_parallel_config = src_parallel_config
        self._dst_parallel_config = dst_parallel_config
        self._transformer_layer_num = transformer_layer_num
        self._segment_method = segment_method

    def apply(self, src_model_path: str, dst_model_path: str):
        for i in range(self._src_parallel_config.mp):
            for j in range(self._src_parallel_config.sharding):
                # TODO(liuzhenhai): use multiple processs
                layers = []

                # 1、extract layers in the same pp group
                group = self._src_parallel_config.pipe_parallel_group(i, j)
                src_dirs = [
                    "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                        src_model_path, *e
                    )
                    for e in group
                ]
                # first rank extract shared layer
                with_shared = True
                for dir in src_dirs:
                    print("extract layer params in dir %s" % dir)
                    layers.extend(self.extract_layers(dir, with_shared))
                    with_shared = False
                # 2、sort and unique layers
                layers = self.sort_layers(layers)

                # 3、resplit layers among pp group according new pp config
                layer_segments = self.segment_layers(
                    layers, self._dst_parallel_config, self._segment_method
                )
                dst_group = self._dst_parallel_config.pipe_parallel_group(i, j)
                dst_dirs = [
                    "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                        dst_model_path, *e
                    )
                    for e in dst_group
                ]

                # 4、merge layers belonging to the same node
                for (layer_segment, dir_) in zip(layer_segments, dst_dirs):
                    print(f"merge {len(layer_segment)} layers to {dir_}")
                    self.merge_layers(layer_segment, dir_)

                # 5、copy meta_state.pdopt
                for (src_dir, dst_dir) in zip(src_dirs, dst_dirs):
                    shutil.copyfile(
                        f"{src_dir}/meta_state.pdopt",
                        f"{dst_dir}/meta_state.pdopt",
                    )

    def peek_model(self, model_dir: str):
        for i in range(self._src_parallel_config.mp):
            for j in range(self._src_parallel_config.sharding):
                group = self._src_parallel_config.pipe_parallel_group(i, j)
                dirs = [
                    "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                        model_dir, *e
                    )
                    for e in group
                ]
                for dir in dirs:
                    print(f"peek partial model in {dir}:")
                    self.peek_partial_model(dir)

    def peek_partial_model(self, sub_dir: str):
        state_dict = paddle.load(f"{sub_dir}/model.pdparams")
        for (k, v) in state_dict.items():
            print(f"\t{k} -> {v.name}")

    def extract_layers(self, dir: str, with_shared: bool):
        opt = paddle.load(dir + "/model_state.pdopt")
        params = paddle.load(dir + "/model.pdparams")
        shared_layer_parsed = False
        # tname -> (layer, param_name)
        tname_to_layer_and_pname = {}
        for (k, v) in params.items():
            layer = self._extract_layer_name(k)
            assert layer
            # special treatment for embedding layer, skip duplicated shared layer
            # shared layer may exist or not, if it exist it share weight with _layers.0
            # _layers.shared_layers.embed.word_embeddings.weight -> embedding_0.w_0
            # _layers.shared_layers.embed.position_embeddings.weight -> embedding_1.w_0
            # _layers.0.word_embeddings.weight -> embedding_0.w_0
            # _layers.0.position_embeddings.weight -> embedding_1.w_0
            shared_layer_parsed = shared_layer_parsed or (
                "_layers.shared_layers" in layer
            )
            if (
                "_layers.shared_layers" not in layer
                and ("word_embeddings" in k or "position_embeddings" in k)
                and shared_layer_parsed
            ):
                continue
            tname_to_layer_and_pname[v.name] = (layer, k)

            # get opt-> param mapping
        tensor_names = list(tname_to_layer_and_pname.keys())
        opt_names = [
            e for e in opt.keys() if e not in ["master_weights", "LR_Scheduler"]
        ]
        opt_to_t = self._opt_name_to_tname(tensor_names, opt_names)
        # gather tensors belonging to one layer togather
        layers = OrderedDict()
        for (k, v) in params.items():
            layer, p = tname_to_layer_and_pname[v.name]
            if layer not in layers:
                layers[layer] = {}
                layers[layer]["opt"] = OrderedDict()
                layers[layer]["params"] = OrderedDict()
                layers[layer]["master_weights"] = OrderedDict()
            layers[layer]["params"][p] = v

        for (k, v) in opt.items():
            if k in ["master_weights", "LR_Scheduler"]:
                continue
            layer, _ = tname_to_layer_and_pname[opt_to_t[v.name]]
            layers[layer]["opt"][k] = v

        if "master_weights" in opt:
            for (k, v) in opt["master_weights"].items():
                layer, _ = tname_to_layer_and_pname[k]
                layers[layer]["master_weights"][k] = v

        if "LR_Scheduler" in opt:
            for layer in layers:
                layers[layer]["LR_Scheduler"] = opt["LR_Scheduler"]

        ans = []

        for (layer_name, layer) in layers.items():
            # special treatment for embedding layer
            if (not with_shared) and "shared_layers" in layer_name:
                continue
            file_name = f"./tmp_layer_files/{layer_name}.tmp"
            paddle.save(layer, file_name)
            ans.append((layer_name, file_name))
            print(f"save layer {layer_name} to {file_name}")
        return ans

    def sort_layers(self, layers: list):
        def priority(elem):
            layer_name = elem[0]
            if "shared_layers" in layer_name:
                return -float(0.5)
            match = re.search(
                r"^_layers((\.\d+)+|(\.shared_layers\.[^\.]+))", layer_name
            )
            assert match, f"{layer_name} not a valid layer name"
            return float(match.group(1).lstrip("."))

        # strictly sort layers
        print("before sort %s" % ("|".join([e[0] for e in layers])))
        layers.sort(key=priority)
        # unique
        unique_layers = []
        for e in layers:
            if unique_layers and e[0] == unique_layers[-1][0]:
                continue
            unique_layers.append(e)
        print("after sort %s " % ("|".join([e[0] for e in unique_layers])))
        return unique_layers

    def segment_layers(
        self,
        layers: list,
        config: ParallelConfig,
        segment_method: str = "layer",
    ):
        layer_num = len(layers)
        stage_num = config.pp * config.vpp

        # segment by weights
        def segment_by_layer():
            # assume model is of the structure below
            # embedding -> n*(transformer layer) -> [optional output layer]
            # segment index
            weights = [0 for _ in range(layer_num)]
            non_zero_layers = range(1, layer_num - 1)
            # input layer is embedding
            if self._transformer_layer_num:
                assert self._transformer_layer_num < layer_num
                non_zero_layers = range(1, 1 + self._transformer_layer_num)
            for i in non_zero_layers:
                weights[i] = 1

            part_size = sum(weights) // stage_num
            result = [0 for _ in range(stage_num + 1)]
            memory_counter = 0
            result_idx = 1
            for idx, weight in enumerate(weights):
                memory_counter += weight
                if memory_counter == part_size:
                    result[result_idx] = idx + 1
                    result_idx += 1
                    memory_counter = 0
            result[stage_num] = layer_num
            return result

        def segment_uniform():
            result = [0 for _ in range(stage_num + 1)]
            part_size = math.floor(layer_num / stage_num)
            extra_layers = layer_num % stage_num
            for i in range(1, stage_num):
                offset = 1 if i > (stage_num - extra_layers) else 0
                result[i] = int(
                    min(result[i - 1] + part_size + offset, layer_num)
                )
            result[stage_num] = layer_num
            return result

        result = (
            segment_uniform()
            if (segment_method == "uniform")
            else segment_by_layer()
        )
        index_segments = [[] for _ in range(config.pp)]
        for i in range(stage_num):
            index_segments[i % config.pp].append((result[i], result[i + 1]))

        # name layers
        segments = [[] for i in range(config.pp)]
        for i in range(config.pp):
            for (start, end) in index_segments[i]:
                for j in range(start, end):
                    if config.vpp > 1:
                        segments[i].append(
                            (
                                [f"_layers.{start}.{j - start}"],
                                layers[j][1],
                            )
                        )
                    else:
                        segments[i].append(([f"_layers.{j}"], layers[j][1]))

        shared_layer_exist = any(
            "_layers.shared_layers" in e[0] for e in layers
        )
        if shared_layer_exist:
            # special treatment for shared layer
            if config.vpp > 1:
                segments[0] = [
                    ([layers[0][0], segments[0][0][0][0]], layers[0][1])
                ] + segments[0][1:]
            else:
                segments[0] = [([layers[0][0]], layers[0][1])] + segments[0][1:]

            for i in range(1, config.pp):
                segments[i] = [([layers[0][0]], layers[0][1])] + segments[i]

        for (pp_rank, segs) in enumerate(segments):
            print(f"segmentment result for pp_rank {pp_rank}:")
            print(50 * "=")
            for seg in segs:
                print(f"{seg[0]} => {seg[1]}")
        return segments

    def merge_layers(self, layers_segment: list, save_dir: str):
        params = OrderedDict()
        opt = OrderedDict()
        master_weights = OrderedDict()
        renaming_manager = LayerReNamingManager()

        def merge(src, dst, map_k=None):
            for (k, v) in src.items():
                k = map_k(k) if map_k is not None else k
                dst[k] = v

        lr_scheduler = None
        for (layer_names, file_path) in layers_segment:
            print("load %s" % file_path)
            layer = paddle.load(file_path)

            def get_param_name_mapper(layer_name):
                # replace layer name
                def map_param_name(param_name):
                    layer_pre = self._extract_layer_name(param_name)
                    return layer_name + param_name[len(layer_pre) :]

                return map_param_name

            (
                layer_params,
                layer_opt,
                layer_master_weight,
            ) = self._map_tensor_names(
                layer["params"],
                layer["opt"],
                layer["master_weights"],
                renaming_manager,
            )
            for layer_name in layer_names:
                merge(layer_params, params, get_param_name_mapper(layer_name))
            merge(layer_opt, opt)
            merge(layer_master_weight, master_weights)
            lr_scheduler = layer["LR_Scheduler"]

        opt = self._pack_opt_state_dict(opt, master_weights, lr_scheduler)
        paddle.save(params, save_dir + "/model.pdparams")
        paddle.save(opt, save_dir + "/model_state.pdopt")

    def _pack_opt_state_dict(self, opt, master_weights, lr_scheduler):
        opt["master_weights"] = master_weights
        opt["LR_Scheduler"] = lr_scheduler
        return opt

    def _extract_layer_name(self, param_name: str):
        match = re.search(
            r"^_layers((\.\d+)+|(\.shared_layers\.[^\.]+))", param_name
        )
        layer_name = ""
        return "" if (not match) else match.group()

    # map opt names to tensor name
    def _opt_name_to_tname(self, tensor_names, opt_names):
        tensor_names = set(tensor_names)
        all_names = []
        all_names.extend(list(tensor_names))
        all_names.extend(opt_names)
        all_names.sort()
        pre_t_name = ""
        opt_to_t = {}
        for n in all_names:
            if n in tensor_names:
                # we get a param
                pre_t_name = n
            else:
                assert pre_t_name
                opt_to_t[n] = pre_t_name
        return opt_to_t

    def _map_tensor_names(self, params, opt, master_weights, renaming_manager):
        opt_renamed = OrderedDict()
        master_weights_renamed = OrderedDict()
        # old name to new name
        t_name_mapping = {}
        # map tensor names
        for (k, v) in params.items():
            t_name_mapping[v.name] = renaming_manager.get_new_param_name(v.name)
            v.name = t_name_mapping[v.name]
        # map opt names
        opt_to_tname = self._opt_name_to_tname(
            t_name_mapping.keys(), opt.keys()
        )
        for (k, v) in opt.items():
            old_t_name = opt_to_tname[k]
            t_name = t_name_mapping[old_t_name]
            opt_name = t_name + k[len(old_t_name) :]
            v.name = opt_name
            opt_renamed[opt_name] = v

        # map master names
        for (k, v) in master_weights.items():
            t_name = t_name_mapping[k]
            v.name = t_name + v.name[len(k) :]
            master_weights_renamed[t_name] = v
        return (params, opt_renamed, master_weights_renamed)


def parse_args():

    parser = argparse.ArgumentParser(
        prog='model converter', description='converter a model'
    )
    parser.add_argument(
        '--src_path',
        type=str,
        default="./output/epoch_0_step_30",
        help='path of the model to convert',
    )

    parser.add_argument(
        '--dst_path',
        type=str,
        default="./test_adapt",
        help='path to saved the converted model',
    )

    parser.add_argument(
        '--src_mp',
        type=int,
        default=2,
        help='mp degree of the origin triaing task that dumpped this model',
    )

    parser.add_argument(
        '--src_pp',
        type=int,
        default=2,
        help='pp degree of the origin triaing task that dumpped this model',
    )

    parser.add_argument(
        '--src_vp',
        type=int,
        default=2,
        help='vp degree of the origin triaing task that dumpped this model',
    )

    parser.add_argument(
        '--dst_mp',
        type=int,
        default=None,
        help='mp degree of the origin triaing task that dumpped this model',
    )

    parser.add_argument(
        '--dst_pp',
        type=int,
        default=None,
        help='pp degree of the expected triaing task that would recover this model',
    )

    parser.add_argument(
        '--dst_vp',
        type=int,
        default=2,
        help='vp degree of the expected triaing task that would recover this model',
    )

    parser.add_argument(
        '--sharding',
        type=int,
        default=1,
        help=" sharding degree of both the origin triaing task that dumpped this model and the expected triaing task that would recover this model",
    )

    parser.add_argument(
        '--method',
        type=str,
        default="adapt_model",
        help='vp degree of the expected triaing task that would recover this model',
    )

    parser.add_argument(
        '--segment_method',
        type=str,
        default="layer",
        help='method to segment layers to pp or vp stages',
    )

    parser.add_argument(
        '--transformer_layer_num',
        type=int,
        default=0,
        help='transformer_layer_num of the model',
    )
    # assume model is of the structure below
    # embedding -> n*[transformer layer] -> optional output layer

    args = parser.parse_args()

    if args.dst_mp is None:
        args.dst_mp = args.src_mp
    if args.dst_pp is None:
        args.dst_pp = args.src_pp

    assert args.src_mp == args.dst_mp, "src mp {} dst mp {}".format(
        args.src_mp, args.dst_mp
    )

    assert args.method in [
        'peek_model',
        'adapt_model',
    ], "method should be in ['peek_model', 'adapt_model']"
    assert args.segment_method in [
        "uniform",
        "layer",
    ], "segment_method should be 'uniform' or 'layer"

    print(
        "adapt model dumped by task with pp degree:{}, vp degree:{}, mp degree:{} to task with pp degree:{}, vp degree:{}, mp degree:{}".format(
            args.src_pp,
            args.src_vp,
            args.src_mp,
            args.dst_pp,
            args.dst_vp,
            args.dst_mp,
        )
    )

    return args


def adaptor_from_args(args):
    src_parallel_config = ParallelConfig(
        args.src_mp, args.src_pp, args.src_vp, args.sharding
    )

    dst_parallel_config = ParallelConfig(
        args.dst_mp, args.dst_pp, args.dst_vp, args.sharding
    )

    adaptor = PipeLineModelAdaptor(
        src_parallel_config,
        dst_parallel_config,
        args.transformer_layer_num,
        args.segment_method,
    )
    return adaptor


def main():

    args = parse_args()
    adaptor = adaptor_from_args(args)
    if args.method == "peek_model":
        adaptor.peek_model(args.dst_path)
    elif args.method == "adapt_model":
        adaptor.apply(args.src_path, args.dst_path)


if __name__ == "__main__":
    """
    Usage:
        python pp_parallel_adaptor.py --src_mp xxx --src_path xxx --method \
          adapt_model/peek_model --dst_path xxx --sharding xxx --segment_method xxx --transformer_layer_num xxx

    for the meaning of a specific arg, please use:
        python  pp_parallel_adaptor.py  -h
    """
    main()
