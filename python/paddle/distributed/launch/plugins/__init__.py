# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os

__all__ = []


# print configuration after args are well filled in controller init
def log(ctx):
    ctx.logger.info("-----------  Configuration  ----------------------")
    for arg, value in sorted(six.iteritems(vars(ctx.args))):
        ctx.logger.info("%s: %s" % (arg, value))
    ctx.logger.info("--------------------------------------------------")


def process_args(ctx):
    # reset device by args
    #argdev = ctx.args.gpus or ctx.args.xpus or ctx.args.npus
    argdev = ctx.args.devices
    if argdev:
        for d in argdev.split(','):
            if d not in ctx.node.device.labels:
                ctx.logger.error(
                    f'Device not found {d} from {argdev} for setting {ctx.node.device.labels}'
                )

    if ctx.args.ips:
        ips = ctx.args.ips.split(',')
        if '127.0.0.1' in ips and len(ips) != 1:
            raise "127.0.0.1 in ips is not allowed in multi-nodes."


def collective_compatible(ctx):
    if 'PADDLE_TRAINER_ENDPOINTS' in ctx.envs:
        eps = ctx.envs['PADDLE_TRAINER_ENDPOINTS'].split(',')
        hosts = set([h.split(':')[0] for h in eps])
        ctx.args.master = eps[0] if ':' in eps[0] else '{}:6768'.format(eps[0])
        ctx.args.nnodes = len(hosts)
        ctx.logger.info(
            'args reset by env PADDLE_TRAINER_ENDPOINTS\n{}'.format(eps))

    if 'DISTRIBUTED_TRAINER_ENDPOINTS' in ctx.envs:
        eps = ctx.envs['DISTRIBUTED_TRAINER_ENDPOINTS'].split(',')
        hosts = set([h.split(':')[0] for h in eps])
        ctx.args.master = eps[0]
        ctx.args.nnodes = len(hosts)
        ctx.logger.info(
            'args reset by env DISTRIBUTED_TRAINER_ENDPOINTS\n{}'.format(eps))


def rewrite_host_ip(ctx):
    if ctx.args.host is not None and "." in ctx.args.host:
        ctx.logger.warning('Host ip reset to {}'.format(ctx.args.host))
        ctx.node.ip = ctx.args.host


def test_mode(ctx):
    if ctx.args.training_script == 'run_check':
        ctx.logger.info('Paddle Distributed Test begin...')
        if int(ctx.args.nnodes) < 2:
            ctx.args.nnodes = 2
        ctx.args.training_script = '{}/test.py'.format(
            os.path.dirname(__file__))


enabled_plugins = [
    test_mode, collective_compatible, rewrite_host_ip, process_args
]
