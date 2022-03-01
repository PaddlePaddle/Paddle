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

import six

__all__ = []


def log(ctx):
    ctx.logger.debug("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(ctx.args))):
        ctx.logger.debug("%s: %s" % (arg, value))
    ctx.logger.debug("------------------------------------------------")


def process_args(ctx):
    # reset device by args
    #argdev = ctx.args.gpus or ctx.args.xpus or ctx.args.npus
    argdev = ctx.args.devices
    if argdev:
        ctx.node.device.labels = argdev.split(',')
        ctx.node.device.count = len(ctx.node.device.labels)
        ctx.logger.debug('device reset by args {}'.format(argdev))


def collective_compatible(ctx):
    if 'PADDLE_TRAINER_ENDPOINTS' in ctx.envs:
        ctx.master = ctx.envs['PADDLE_TRAINER_ENDPOINTS'].split(',')[0]


enabled_plugins = [process_args, log]
