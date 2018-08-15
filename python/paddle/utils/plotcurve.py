#!/usr/bin/python
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
"""Plot training and testing curve from paddle log.

It takes input from a file or stdin, and output to a file or stdout.

Note: must have numpy and matplotlib installed in order to use this tool.

usage: Plot training and testing curves from paddle log file.
       [-h] [-i INPUT] [-o OUTPUT] [--format FORMAT] [key [key ...]]

positional arguments:
  key                   keys of scores to plot, the default will be AvgCost

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input filename of paddle log, default will be standard
                        input
  -o OUTPUT, --output OUTPUT
                        output filename of figure, default will be standard
                        output
  --format FORMAT       figure format(png|pdf|ps|eps|svg)


The keys must be in the order of paddle output(!!!).

For example, paddle.INFO contrains the following log
   I0406 21:26:21.325584  3832 Trainer.cpp:601]  Pass=0 Batch=7771 AvgCost=0.624935 Eval: error=0.260972

To use this script to generate plot for AvgCost, error:
   python plotcurve.py -i paddle.INFO -o figure.png AvgCost error
"""

import sys
import matplotlib
# the following line is added immediately after import matplotlib
# and before import pylot. The purpose is to ensure the plotting
# works even under remote login (i.e. headless display)
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as pyplot
import numpy
import argparse
import re
import os


def plot_paddle_curve(keys, inputfile, outputfile, format='png',
                      show_fig=False):
    """Plot curves from paddle log and save to outputfile.

    :param keys: a list of strings to be plotted, e.g. AvgCost
    :param inputfile: a file object for input
    :param outputfile: a file object for output
    :return: None
    """
    pass_pattern = r"Pass=([0-9]*)"
    test_pattern = r"Test samples=([0-9]*)"
    if not keys:
        keys = ['AvgCost']
    for k in keys:
        pass_pattern += r".*?%s=([0-9e\-\.]*)" % k
        test_pattern += r".*?%s=([0-9e\-\.]*)" % k
    data = []
    test_data = []
    compiled_pattern = re.compile(pass_pattern)
    compiled_test_pattern = re.compile(test_pattern)
    for line in inputfile:
        found = compiled_pattern.search(line)
        found_test = compiled_test_pattern.search(line)
        if found:
            data.append([float(x) for x in found.groups()])
        if found_test:
            test_data.append([float(x) for x in found_test.groups()])
    x = numpy.array(data)
    x_test = numpy.array(test_data)
    if x.shape[0] <= 0:
        sys.stderr.write("No data to plot. Exiting!\n")
        return
    m = len(keys) + 1
    for i in xrange(1, m):
        pyplot.plot(
            x[:, 0],
            x[:, i],
            color=cm.jet(1.0 * (i - 1) / (2 * m)),
            label=keys[i - 1])
        if (x_test.shape[0] > 0):
            pyplot.plot(
                x[:, 0],
                x_test[:, i],
                color=cm.jet(1.0 - 1.0 * (i - 1) / (2 * m)),
                label="Test " + keys[i - 1])
    pyplot.xlabel('number of epoch')
    pyplot.legend(loc='best')
    if show_fig:
        pyplot.show()
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()


def main(argv):
    """
    main method of plotting curves.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot training and testing curves from paddle log file.")
    cmdparser.add_argument(
        'key', nargs='*', help='keys of scores to plot, the default is AvgCost')
    cmdparser.add_argument(
        '-i',
        '--input',
        help='input filename of paddle log, '
        'default will be standard input')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output filename of figure, '
        'default will be standard output')
    cmdparser.add_argument('--format', help='figure format(png|pdf|ps|eps|svg)')
    args = cmdparser.parse_args(argv)
    keys = args.key
    if args.input:
        inputfile = open(args.input)
    else:
        inputfile = sys.stdin
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout
    plot_paddle_curve(keys, inputfile, outputfile, format)
    inputfile.close()
    outputfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
