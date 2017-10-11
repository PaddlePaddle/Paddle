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

from py_paddle import swig_paddle
import paddle.trainer.config_parser
import numpy
import util


def init_params(params):
    def init_param(p):
        assert isinstance(p, swig_paddle.Parameter)
        val = p.getBuf(swig_paddle.PARAMETER_VALUE)
        assert isinstance(val, swig_paddle.Vector)
        arr = val.toNumpyArrayInplace()
        for i in xrange(len(arr)):
            arr[i] = numpy.random.uniform(-1.0, 1.0)

    for p in params:
        init_param(p)


def init_optimizers(opt_conf, params):
    opts = {}
    for param in params:
        param_conf = param.getConfig().toProto()
        opts[param.getID()] = swig_paddle.ParameterOptimizer.create(opt_conf)
        opts[param.getID()].init(param_conf.dims[1], param.getConfig())
    retv_opts = [None for _ in xrange(len(opts))]
    for k in opts:
        assert k < len(retv_opts)
        retv_opts[k] = opts[k]
    return retv_opts


def main():
    trainer_config = paddle.trainer.config_parser.parse_config(
        "./testTrainConfig.py", "")
    opt_config = trainer_config.opt_config
    print "========Optimization Config ======="
    print opt_config
    print "==================================="
    opt_config = swig_paddle.OptimizationConfig.createFromProto(opt_config)
    _temp_optimizer_ = swig_paddle.ParameterOptimizer.create(opt_config)
    enable_types = _temp_optimizer_.getParameterTypes()
    m = swig_paddle.GradientMachine.createFromConfigProto(
        trainer_config.model_config, swig_paddle.CREATE_MODE_NORMAL,
        enable_types)
    assert m is not None
    assert isinstance(m, swig_paddle.GradientMachine)
    init_params(m.getParameters())

    optimizers = init_optimizers(opt_config, m.getParameters())

    # Train One Pass.
    for optimizer in optimizers:
        optimizer.startPass()
    batch_id = 0
    while True:  # Train one batch
        batch_size = 1000
        inArgs, atEnd = util.loadMNISTTrainData(batch_size)
        if atEnd:
            break
        outArgs = swig_paddle.Arguments.createArguments(0)

        for optimizer in optimizers:
            optimizer.startBatch(batch_size)

        def update_callback(param):
            try:
                bufs = list(param.getBufs())
                opt = optimizers[param.getID()]
                opt.update(bufs, param.getConfig())
                callback = opt.needSpecialTraversal(param.getConfig())
                if callback is not None:
                    callback(bufs, param.getConfig(), swig_paddle.NO_SPARSE_ID)

            except Exception as e:
                print e

        ev = m.makeEvaluator()
        ev.start()
        m.forwardBackward(inArgs, outArgs, swig_paddle.PASS_TRAIN,
                          update_callback)
        m.eval(ev)
        ev.finish()
        for name in ev.getNames():
            print name, ev.getValue(name)
        for optimizer in optimizers:
            optimizer.finishBatch()

        cost_vec = outArgs.getSlotValue(0)
        assert isinstance(cost_vec, swig_paddle.Matrix)
        cost_vec = cost_vec.copyToNumpyMat()
        print 'Finish Batch', batch_id, 'with cost ', cost_vec.sum(
        ) / batch_size
        batch_id += 1

    for optimizer in optimizers:
        optimizer.finishPass()


if __name__ == '__main__':
    swig_paddle.initPaddle("--use_gpu=0", "--trainer_count=1")
    main()
