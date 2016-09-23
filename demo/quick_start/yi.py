# -*- coding: utf-8 -*-
import json
import random

def parse(jsonFile, topN, trainFile, testFile):
    """
    Parse Amazon Reviews JSON file and generate train.txt and test.txt.
    """
    train = open(trainFile, 'w')
    test = open(testFile, 'w')

    g = open(jsonFile, 'r')
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
    random.seed(1)
    parse('data/reviews.json', 1000, '/tmp/train.txt', '/tmp/test.txt')
