# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer.config_parser import parse_config
from paddle.trainer.config_parser import logger
from py_paddle import swig_paddle
import util


def main():
    trainer_config = parse_config("./testTrainConfig.py", "")
    model = swig_paddle.GradientMachine.createFromConfigProto(
        trainer_config.model_config)
    trainer = swig_paddle.Trainer.create(trainer_config, model)
    trainer.startTrain()
    for train_pass in xrange(2):
        trainer.startTrainPass()
        num = 0
        cost = 0
        while True:  # Train one batch
            batch_size = 1000
            data, atEnd = util.loadMNISTTrainData(batch_size)
            if atEnd:
                break
            trainer.trainOneDataBatch(batch_size, data)
            outs = trainer.getForwardOutput()
            cost += sum(outs[0]['value'])
            num += batch_size
        trainer.finishTrainPass()
        logger.info('train cost=%f' % (cost / num))

        trainer.startTestPeriod()
        num = 0
        cost = 0
        while True:  # Test one batch
            batch_size = 1000
            data, atEnd = util.loadMNISTTrainData(batch_size)
            if atEnd:
                break
            trainer.testOneDataBatch(batch_size, data)
            outs = trainer.getForwardOutput()
            cost += sum(outs[0]['value'])
            num += batch_size
        trainer.finishTestPeriod()
        logger.info('test cost=%f' % (cost / num))

    trainer.finishTrain()


if __name__ == '__main__':
    swig_paddle.initPaddle("--use_gpu=0", "--trainer_count=1")
    main()
