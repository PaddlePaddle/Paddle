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

from ..fluid.optimizer import ExponentialMovingAverage
from ..fluid.wrapped_decorator import signature_safe_contextmanager
from ..fluid.framework import Variable
import six


class EMA(ExponentialMovingAverage):
    r"""
	:api_attr: Static Graph

    Compute the moving average of parameters with exponential decay.
    Given a parameter :math:`\\theta`, its exponential moving average (EMA)
    will be

    ..  math::

        \\text{EMA}_0 & = 0

	\\text{EMA}_t & = \\text{decay} * \\text{EMA}_{t-1} + (1 - \\text{decay}) * \\theta_t

    The average results calculated by **update()** method will be saved in 
    temporary variables which are created and maintained by the object, and can 
    be applied to parameters of current model by calling **apply()** method. And 
    the **restore()** method is used to restore the parameters.

    **Bias correction**. All EMAs are initialized to :math:`0` and hence they will be 
    zero biased, which can be corrected by divided by a factor 
    :math:`(1 - \\text{decay}^t)` , i.e., the actual EMAs applied to parameters 
    when calling **apply()** method would be 

    ..  math::
    
        \\widehat{\\text{EMA}}_t = \\frac{\\text{EMA}_t}{1 - \\text{decay}^t}

    **Decay rate scheduling**. A large decay rate very close to 1 would result 
    in that the averages move very slowly. And a better strategy is to set a 
    relative smaller decay rate in the very beginning. The argument **thres_steps**
    allows users to pass a Variable to schedule the decay rate, in this case, 
    the actual decay rate becomes
     
    ..  math::
    
        \\min(\\text{decay}, \\frac{1 + \\text{thres_steps}}{10 + \\text{thres_steps}})

    Usually **thres_steps** can be the global training steps.


    Args:
	decay (float, optional): The exponential decay rate, usually close to 1, such as 
            0.999, 0.9999, ... . Default 0.999.
        thres_steps (Variable|None): If not `None`, schedule the decay rate. 
            Default None.
        name (str|None): For detailed information, please refer to 
            :ref:`api_guide_Name`. Usually name is no need to set and None by 
            default.


    Examples:

	.. code-block:: python

	    import numpy
	    import paddle
	    import paddle.fluid as fluid

	    data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
	    hidden = fluid.layers.fc(input=data, size=10)
	    cost = fluid.layers.mean(hidden)

	    test_program = fluid.default_main_program().clone(for_test=True)

	    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	    optimizer.minimize(cost)

	    global_steps = fluid.layers.autoincreased_step_counter()
	    ema = fluid.optimizer.ExponentialMovingAverage(0.999, thres_steps=global_steps)
	    ema.update()

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    for pass_id in range(3):
		for batch_id in range(6):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=fluid.default_main_program(),
			feed={'x': data}, 
			fetch_list=[cost.name])

		# usage 1
		with ema.apply(exe):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
			    

		 # usage 2
		with ema.apply(exe, need_restore=False):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
		ema.restore(exe)
    """

    def update(self):
        """ 
        Update Exponential Moving Average. Should only call this method in 
        train program.
        """
        ExponentialMovingAverage.update(self)

    @signature_safe_contextmanager
    def apply(self, executor, feed=None, fetch_list=None, need_restore=True):
        """
        Apply moving average to parameters for evaluation.
        
        Args:
            executor (Executor or ParallelExecutor): The Executor to execute applying.
            fetch_list (list or tuple, optional): The Tensors that need to be returned after the 
                model runs, default None. However when executor is :code:`ParallelExecutor`, 
                :code:`fetch_list` should not be None.
                
            need_restore (bool, optional): Whether to restore parameters after 
                applying. Default True.
        """
        exe_class = executor.__class__.__name__
        if exe_class == "Executor":
            executor.run(self.apply_program)
            try:
                yield
            finally:
                if need_restore:
                    self.restore(executor)
        elif exe_class == "ParallelExecutor":
            if feed is None or fetch_list is None:
                raise ValueError(
                    "While applying moving average to parameters "
                    "for {}, feed and fetch_list should not be None.".format(
                        exe_class))
            assert isinstance(fetch_list, tuple) or isinstance(fetch_list, list), \
                "Currently , The fetch_list type only should be list or tuple, \n"\
                "but the input type is {}. For more information please refer to \n"\
                "the executor.run(...).".format(type(fetch_list))
            executor.run(feed=feed, fetch_list=fetch_list)
            try:
                yield
            finally:
                if need_restore:
                    self.restore(executor, feed, fetch_list)

    def restore(self, executor, feed=None, fetch_list=None):
        """Restore parameters.
        
        Args:
            executor (Executor or ParallelExecutor): The Executor to execute restoring.
        """
        if feed is None and fetch_list is None:
            executor.run(self.restore_program)
        else:
            executor.run(feed=feed, fetch_list=fetch_list)
