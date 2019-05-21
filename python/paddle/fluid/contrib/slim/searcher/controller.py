#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""The controller used to search hyperparameters or neural architecture"""


class Controller(object):
    """Controller for Neural Architecture Search.
    """

    def __init__(self, *args, **kwargs):
        pass

    def check(self, reward_new, reward, iteration):
        """Check if the var should be updated.
        """
        raise NotImplementedError('Abstract method.')

    def generate_init_var(self):
        """Generate init var.
        """
        raise NotImplementedError('Abstract method.')

    def generate_new_var(self, var):
        """Generate new var.
        """
        raise NotImplementedError('Abstract method.')


class SaController(Controller):
    """Simulated annealing controller."""

    def __init__(self,
                 range_table,
                 reduce_rate=0.85,
                 init_temperature=1024,
                 constrain_func=None,
                 max_threshold=None,
                 min_threshold=None,
                 max_iter_number=300):
        """Initialize.
        Args:
            range_table(list): variable range table.
            reduce_rate(float): reduce rate.
            init_temperature(float): init temperature.
            constrain(bool): whether constrain.
            constrain_command(str): constrain command.
            max_threshold(float): max threshold.
            min_threshold(float): min threshold.
            max_iter_number(int): max iteration number.
        """
        super(SaController, self).__init__()
        self._range_table = range_table
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._constrain_func = constrain_func
        self._max_threshold = max_threshold
        self._min_threshold = min_threshold
        self._max_iter_number = max_iter_number

    def reset(self, range_table):
        self._range_table = range_table
        return self._generate_init_var()

    def check(self, reward_new, reward, iteration):
        """Check if the var should be updated using general policy.
        Args:
            reward_new(float): new reward.
            reward(float): reward.
            iteration(int): iteration.
        Returns:
            bool: a list of new variables.
        """
        if reward_new > reward:
            return True
        temperature = self._init_temperature * self._reduce_rate**iteration
        return np.random.random() <= math.exp(
            (reward_new - reward) / temperature)

    def _generate_init_var(self):
        """Generate init var.
        Returns:
            list: a list of variables.
        """
        if self._constrain is None:
            return [np.random.randint(_) for _ in self._range_table]
        ratio_start = 0
        ratio_end = self._range_table[0]
        net_arc = [(ratio_start + ratio_end) // 2
                   for _ in range(len(self._range_table))]
        init_flops = self.get_flops(net_arc)
        while ratio_start != ratio_end and self._constrain_func(net_arc):
            if init_flops > self._max_threshold * self._constrain:
                ratio_start = (ratio_start + ratio_end) // 2
            elif init_flops < self._min_threshold * self._constrain:
                ratio_end = (ratio_start + ratio_end) // 2
            net_arc = [(ratio_start + ratio_end) // 2
                       for _ in range(len(self._range_table))]
            init_flops = self.get_flops(net_arc)
        return net_arc

    def next_tokens(self, tokens):
        """Generate new tokens.
        Args:
            tokens(list): a list of token.
        Returns:
            list: a list of new token.
        """
        var_new = var[:]
        index = int(len(self._range_table) * np.random.random())
        var_new[index] = int(self._range_table[index] * np.random.random())
        if self._constrain is not None:
            for _ in range(self._max_iter_number):
                init_flops = self.get_flops(var_new)
                if init_flops > self._max_threshold * self._constrain or \
                    init_flops < self._min_threshold * self._constrain:
                    index = int(len(self._range_table) * np.random.random())
                    var_new[index] = int(self._range_table[index] *
                                         np.random.random())
                else:
                    break
        return var_new

    def get_flops(self, net_arc):
        """Get flops.
        Args:
            net_arc(list): network architecture.
        Returns:
            float: flops.
        """
        temp_command = copy.deepcopy(self._constrain_command)
        temp_command.append(str(net_arc).replace(' ', ''))
        child_proc = subprocess.Popen(
            temp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            if child_proc.poll() is None:
                time.sleep(0.1)
                continue
            else:
                break
        out = child_proc.communicate()[0].strip('\n')
        try:
            flops, params = map(float, out.split())
        except Exception as err:
            flops, params = 1e9, 1e9
            print('[ERROR] {}'.format(err))
        print('net_arc: {}, flops: {}, params: {}'.format(net_arc, flops,
                                                          params))
        return flops
