import matplotlib.pyplot as plt
from IPython import display


class PlotCost(object):
    """
    append train and test cost in event_handle and then call plot.
    """

    def __init__(self):
        self.train_costs = ([], [])
        self.test_costs = ([], [])

    def plot(self):
        plt.plot(*self.train_costs)
        plt.plot(*self.test_costs)
        title = []
        if len(self.train_costs[0]) > 0:
            title.append('Train Cost')
        if len(self.test_costs[0]) > 0:
            title.append('Test Cost')
        plt.legend(title, loc='upper left')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.gcf().clear()

    def append_train_cost(self, step, cost):
        self.train_costs[0].append(step)
        self.train_costs[1].append(cost)

    def append_test_cost(self, step, cost):
        self.test_costs[0].append(step)
        self.test_costs[1].append(cost)

    def reset(self):
        self.train_costs = ([], [])
        self.test_costs = ([], [])
