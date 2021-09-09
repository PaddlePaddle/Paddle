#!/usr/bin/env python
# coding=utf-8
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
import yaml
import copy
import six
import sys
import os
import logging
import traceback

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def is_distributed_env():
    node_role = os.getenv("TRAINING_ROLE")
    logger.info("-- Role: {} --".format(node_role))
    if node_role is None:
        return False
    else:
        return True


def lazy_instance_by_fliename(abs, class_name):
    try:
        dirname = os.path.dirname(abs)
        sys.path.append(dirname)
        package = os.path.splitext(os.path.basename(abs))[0]
        model_package = __import__(package,
                                   globals(), locals(), package.split("."))
        instance = getattr(model_package, class_name)
        return instance
    except Exception as err:
        traceback.print_exc()
        print('Catch Exception:%s' % str(err))
        return None


def get_utils_file_path():
    return os.path.dirname(os.path.abspath(__file__))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class YamlHelper(object):
    def load_yaml(self, yaml_file, other_part=None):
        part_list = [
            "running_config", "workspace", "static_benchmark",
            "hyper_parameters"
        ]
        if other_part:
            part_list += other_part
        running_config = self.get_all_inters_from_yaml(yaml_file, part_list)
        running_config = self.workspace_adapter(running_config)
        return running_config

    def print_yaml(self, config):
        print(self.pretty_print_envs(config))

    def parse_yaml(self, config):
        vs = [int(i) for i in yaml.__version__.split(".")]
        if vs[0] < 5:
            use_full_loader = False
        elif vs[0] > 5:
            use_full_loader = True
        else:
            if vs[1] >= 1:
                use_full_loader = True
            else:
                use_full_loader = False

        if os.path.isfile(config):
            if six.PY2:
                with open(config, 'r') as rb:
                    if use_full_loader:
                        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                    else:
                        _config = yaml.load(rb.read())
                    return _config
            else:
                with open(config, 'r', encoding="utf-8") as rb:
                    if use_full_loader:
                        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                    else:
                        _config = yaml.load(rb.read())
                    return _config
        else:
            raise ValueError("config {} can not be supported".format(config))

    def get_all_inters_from_yaml(self, file, filters):
        _envs = self.parse_yaml(file)
        return _envs

        #all_flattens = {}

        def fatten_env_namespace(namespace_nests, local_envs):
            for k, v in local_envs.items():
                if isinstance(v, dict):
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    fatten_env_namespace(nests, v)
                else:
                    global_k = ".".join(namespace_nests + [k])
                    all_flattens[global_k] = v

        #fatten_env_namespace([], _envs)
        #ret = {}
        #for k, v in all_flattens.items():
        #    for f in filters:
        #        if k.startswith(f):
        #            ret[k] = v
        #return ret

    def workspace_adapter(self, config):
        workspace = config.get("workspace")
        for k, v in config.items():
            if isinstance(v, str) and "{workspace}" in v:
                config[k] = v.replace("{workspace}", workspace)
        return config

    def pretty_print_envs(self, envs, header=None):
        spacing = 2
        max_k = 40
        max_v = 45

        for k, v in envs.items():
            max_k = max(max_k, len(k))

        h_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(max_k, " " *
                                                              spacing, max_v)
        l_format = "    " + "|{{:>{}s}}{{}}{{:^{}s}}|\n".format(max_k, max_v)
        length = max_k + max_v + spacing

        border = "    +" + "".join(["="] * length) + "+"
        line = "    +" + "".join(["-"] * length) + "+"

        draws = ""
        draws += border + "\n"

        if header:
            draws += h_format.format(header[0], header[1])
        else:
            draws += h_format.format("PaddleRec Benchmark Envs", "Value")

        draws += line + "\n"

        for k, v in sorted(envs.items()):
            if isinstance(v, str) and len(v) >= max_v:
                str_v = "... " + v[-41:]
            else:
                str_v = v

            draws += l_format.format(k, " " * spacing, str(str_v))

        draws += border

        _str = "\n{}\n".format(draws)
        return _str
