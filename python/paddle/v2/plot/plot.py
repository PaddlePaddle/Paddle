from IPython import display
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


class Plot(object):
    def __init__(self, *args):
        self.args = args
        self.__plot_data__ = {}
        for title in args:
            self.__plot_data__[title] = PlotData()
        self.__disable_plot__ = os.environ.get("DISABLE_PLOT")
        if not self.__plot_is_disabled__():
            import matplotlib.pyplot as plt
            self.plt = plt

    def __plot_is_disabled__(self):
        return self.__disable_plot__ == "True"

    def append(self, title, step, value):
        assert isinstance(title, basestring)
        assert self.__plot_data__.has_key(title)
        data = self.__plot_data__[title]
        assert isinstance(data, PlotData)
        data.append(step, value)

    def plot(self):
        if self.__plot_is_disabled__():
            return

        titles = []
        for title in self.args:
            data = self.__plot_data__[title]
            assert isinstance(data, PlotData)
            if len(data.step) > 0:
                titles.append(title)
                self.plt.plot(data.step, data.value)
        self.plt.legend(titles, loc='upper left')
        display.clear_output(wait=True)
        display.display(self.plt.gcf())
        self.plt.gcf().clear()

    def reset(self):
        for key in self.__plot_data__:
            data = self.__plot_data__[key]
            assert isinstance(data, PlotData)
            data.reset()

if __name__ == '__main__':
    title = "cost"
    plot_test = Plot(title)
    plot_test.append(title, 1, 1)
    plot_test.append(title, 2, 2)
    for k, v in plot_test.__plot_data__.iteritems():
        print k, v.step, v.value
    plot_test.reset()
    for k, v in plot_test.__plot_data__.iteritems():
        print k, v.step, v.value
gg
