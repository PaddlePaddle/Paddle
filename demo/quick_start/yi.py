# -*- coding: utf-8 -*-

import sys
import os
import operator
import json
import random
from StringIO import StringIO
import gzip
import urllib2
from optparse import OptionParser

defaultFile = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'

def parse(inputFile, firstNRecords, outputDir, topNWords):
    """
    Parse Amazon Reviews data and generate {train,test,pred,dict}.{txt,list}.
    """
    train = open(os.path.join(outputDir, 'train.txt'), 'w')
    test = open(os.path.join(outputDir, 'test.txt'), 'w')
    pred = open(os.path.join(outputDir, 'pred.txt'), 'w')

    if inputFile.startswith("http"):
        request = urllib2.Request(inputFile)
        response = urllib2.urlopen(request)
        buf = StringIO(response.read())
        g = gzip.GzipFile(fileobj=buf)
    else:
        g = gzip.open(inputFile, 'r')

    written = 0
    wc = {}

    for l in g:
        if written >= firstNRecords:
            break

        o = train
        u = random.random()
        if u < 0.01:
            o = pred
        elif u < 0.1:
            o = test

        j = json.loads(l)
        words = j["reviewText"].lower().split()
        for w in words:
            if w not in wc:
                wc[w] = 1
            else:
                wc[w] += 1

        text = " ".join(words)
        rate = j["overall"]
        if text:
            if rate == 5.0:
                o.write('1\t%s\n' % text)
                written = written + 1
            elif rate < 3.0:
                o.write('0\t%s\n' % text)
                written = written + 1

    train.close()
    test.close()
    g.close()

    with open(os.path.join(outputDir, 'dict.txt'), 'w') as dict:
        dict.write('%s\t%s\n' % ('unk', '-1'))
        c = 0
        for k, v in sorted(wc.items(), key=operator.itemgetter(1), reverse=True):
            dict.write('%s\t%s\n' % (k, v))
            c = c + 1
            if c > topNWords:
                break

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option("-n", "--firstn", type="int", dest="firstN", default=100,
                      help="Generate train & test sets with N instances in total.")
    parser.add_option("-t", "--topn", type="int", dest="topN", default=30001,
                      help="Save the top N frequent words into dict.txt",)
    (options, args) = parser.parse_args()

    inputFile = args[0] if len(args) > 0 else defaultFile
    outputDir = os.path.dirname(sys.argv[0]) or './'
    print "Processing the first %d records in %s. Writing to %s." % (options.firstN, inputFile, outputDir)

    random.seed(1)
    parse(inputFile, options.firstN, outputDir, max(1, options.topN))

    with open(os.path.join(outputDir, 'train.list'), 'w') as l:
        l.write('%s\n' % os.path.join(outputDir, 'train.txt'))

    with open(os.path.join(outputDir, 'test.list'), 'w') as l:
        l.write('%s\n' % os.path.join(outputDir, 'test.txt'))

    with open(os.path.join(outputDir, 'pred.list'), 'w') as l:
        l.write('%s\n' % os.path.join(outputDir, 'pred.txt'))

    with open(os.path.join(outputDir, 'labels.list'), 'w') as l:
        l.write('neg\t0\npos\t1\n')
