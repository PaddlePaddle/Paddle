# -*- coding: utf-8 -*-

import sys
import os
import json
import random
from StringIO import StringIO
import gzip
import urllib2
from optparse import OptionParser

defaultFile = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'

def parse(inputFile, topN, outputDir):
    """
    Parse Amazon Reviews as a Gzip-ed JSON file and generate train.txt and test.txt.
    """
    train = open(os.path.join(outputDir, 'train.txt'), 'w')
    test = open(os.path.join(outputDir, 'test.txt'), 'w')

    if inputFile.startswith("http"):
        request = urllib2.Request(inputFile)
        response = urllib2.urlopen(request)
        buf = StringIO(response.read())
        g = gzip.GzipFile(fileobj=buf)
    else:
        g = gzip.open(inputFile, 'r')

    lines = 0
    for l in g:
        lines = lines + 1
        if lines > topN:
            break

        o = train
        if random.random() < 0.1:
            o = test

        j = json.loads(l)
        text = " ".join(j["reviewText"].lower().split())
        rate = j["overall"]
        if text:
            if rate == 5.0:
                o.write('1\t%s\n' % text)
            elif rate < 3.0:
                o.write('0\t%s\n' % text)
    g.close()
    train.close()
    test.close()

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option("-n", "--firstn",
                      type="int",
                      action="store",
                      dest="firstN",
                      default=100,
                      help="Use the first N number of instances",)
    (options, args) = parser.parse_args()

    inputFile = args[0] if len(args) > 0 else defaultFile
    print "Downloading and processing the first %d records from %s ..." % (options.firstN, inputFile)

    random.seed(1)
    parse(inputFile, options.firstN, os.path.dirname(sys.argv[0]) or './')
