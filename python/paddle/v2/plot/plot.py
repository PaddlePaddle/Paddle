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

import os


class PlotData(object):
    def __init__(self):
        self.step = []
        self.value = []

    def append(self, step, value):
        self.step.append(step)
        self.value.append(value)

    def reset(self):
        self.step = []
        self.value = []


class Ploter(object):
    def __init__(self, *args):
        self.__args__ = args
        self.__plot_data__ = {}
        for title in args:
            self.__plot_data__[title] = PlotData()
        # demo in notebooks will use Ploter to plot figure, but when we convert
        # the ipydb to py file for testing, the import of matplotlib will make the
        # script crash. So we can use `export DISABLE_PLOT=True` to disable import
        # these libs
        self.__disable_plot__ = os.environ.get("DISABLE_PLOT")
        if not self.__plot_is_disabled__():
            import matplotlib.pyplot as plt
            from IPython import display
            self.plt = plt
            self.display = display

    def __plot_is_disabled__(self):
        return self.__disable_plot__ == "True"

    def append(self, title, step, value):
        assert isinstance(title, basestring)
        assert self.__plot_data__.has_key(title)
        data = self.__plot_data__[title]
        assert isinstance(data, PlotData)
        data.append(step, value)

    def plot(self, path=None):
        if self.__plot_is_disabled__():
            return

        titles = []
        for title in self.__args__:
            data = self.__plot_data__[title]
            assert isinstance(data, PlotData)
            if len(data.step) > 0:
                titles.append(title)
                self.plt.plot(data.step, data.value)
        self.plt.legend(titles, loc='upper left')
        if path is None:
            self.display.clear_output(wait=True)
            self.display.display(self.plt.gcf())
        else:
            self.plt.savefig(path)
        self.plt.gcf().clear()

    def reset(self):
        for key in self.__plot_data__:
            data = self.__plot_data__[key]
            assert isinstance(data, PlotData)
            data.reset()
