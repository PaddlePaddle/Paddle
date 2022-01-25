# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = []


def log(ctx):
    ctx.logger.debug("int args {}".format(ctx.args).replace(", ", "\n"))
    ctx.logger.debug("int envs {}".format(ctx.envs))


def fill_job(ctx):
    if ctx.args.host:
        ctx.node.ip = ctx.args.host
        ctx.envs["POD_IP"] = ctx.args.host


def process_args(ctx):
    # reset device by args
    argdev = ctx.args.gpus or ctx.args.xpus or ctx.args.npus
    if argdev:
        ctx.node.device.labels = argdev.split(',')
        ctx.node.device.count = len(ctx.node.device.labels)
        ctx.logger.debug('device reset by args {}'.format(argdev))


enabled_plugins = [log, fill_job, process_args]
