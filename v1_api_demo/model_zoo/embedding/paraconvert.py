#!/bin/env python
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
"""
Example:
    python paraconvert.py --b2t -i INPUT -o OUTPUT -d DIM
    python paraconvert.py --t2b -i INPUT -o OUTPUT

Options:
    -h, --help  show this help message and exit
    --b2t       convert parameter file of embedding model from binary to text
    --t2b       convert parameter file of embedding model from text to binary
    -i INPUT    input parameter file name
    -o OUTPUT   output parameter file name
    -d DIM      dimension of parameter
"""
from optparse import OptionParser
import struct


def binary2text(input, output, paraDim):
    """
    Convert a binary parameter file of embedding model to be a text file.  
    input: the name of input binary parameter file, the format is:
           1) the first 16 bytes is filehead:
                version(4 bytes): version of paddle, default = 0
                floatSize(4 bytes): sizeof(float) = 4
                paraCount(8 bytes): total number of parameter
           2) the next (paraCount * 4) bytes is parameters, each has 4 bytes 
    output: the name of output text parameter file, for example:
           0,4,32156096
           -0.7845433,1.1937413,-0.1704215,...
           0.0000909,0.0009465,-0.0008813,...
           ...
           the format is:
           1) the first line is filehead: 
              version=0, floatSize=4, paraCount=32156096
           2) other lines print the paramters
              a) each line prints paraDim paramters splitted by ','
              b) there is paraCount/paraDim lines (embedding words)
    paraDim: dimension of parameters 
    """
    fi = open(input, "rb")
    fo = open(output, "w")
    """
    """
    version, floatSize, paraCount = struct.unpack("iil", fi.read(16))
    newHead = ','.join([str(version), str(floatSize), str(paraCount)])
    print >> fo, newHead

    bytes = 4 * int(paraDim)
    format = "%df" % int(paraDim)
    context = fi.read(bytes)
    line = 0

    while context:
        numbers = struct.unpack(format, context)
        lst = []
        for i in numbers:
            lst.append('%8.7f' % i)
        print >> fo, ','.join(lst)
        context = fi.read(bytes)
        line += 1
    fi.close()
    fo.close()
    print "binary2text finish, total", line, "lines"


def get_para_count(input):
    """
    Compute the total number of embedding parameters in input text file. 
    input: the name of input text file
    """
    numRows = 1
    paraDim = 0
    with open(input) as f:
        line = f.readline()
        paraDim = len(line.split(","))
        for line in f:
            numRows += 1
    return numRows * paraDim


def text2binary(input, output, paddle_head=True):
    """
    Convert a text parameter file of embedding model to be a binary file.
    input: the name of input text parameter file, for example:
           -0.7845433,1.1937413,-0.1704215,...
           0.0000909,0.0009465,-0.0008813,... 
           ...
           the format is:
           1) it doesn't have filehead
           2) each line stores the same dimension of parameters, 
              the separator is commas ','
    output: the name of output binary parameter file, the format is:
           1) the first 16 bytes is filehead: 
             version(4 bytes), floatSize(4 bytes), paraCount(8 bytes)
           2) the next (paraCount * 4) bytes is parameters, each has 4 bytes
    """
    fi = open(input, "r")
    fo = open(output, "wb")

    newHead = struct.pack("iil", 0, 4, get_para_count(input))
    fo.write(newHead)

    count = 0
    for line in fi:
        line = line.strip().split(",")
        for i in range(0, len(line)):
            binary_data = struct.pack("f", float(line[i]))
            fo.write(binary_data)
        count += 1
    fi.close()
    fo.close()
    print "text2binary finish, total", count, "lines"


def main():
    """
    Main entry for running paraconvert.py 
    """
    usage = "usage: \n" \
            "python %prog --b2t -i INPUT -o OUTPUT -d DIM \n" \
            "python %prog --t2b -i INPUT -o OUTPUT"
    parser = OptionParser(usage)
    parser.add_option(
        "--b2t",
        action="store_true",
        help="convert parameter file of embedding model from binary to text")
    parser.add_option(
        "--t2b",
        action="store_true",
        help="convert parameter file of embedding model from text to binary")
    parser.add_option(
        "-i", action="store", dest="input", help="input parameter file name")
    parser.add_option(
        "-o", action="store", dest="output", help="output parameter file name")
    parser.add_option(
        "-d", action="store", dest="dim", help="dimension of parameter")
    (options, args) = parser.parse_args()
    if options.b2t:
        binary2text(options.input, options.output, options.dim)
    if options.t2b:
        text2binary(options.input, options.output)


if __name__ == '__main__':
    main()
