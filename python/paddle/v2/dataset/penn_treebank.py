# The module paddle.v2.datasets.penn_treebank.py provides convenient
# access to the Penn TreeBank dataset
# (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) via
# the following reader creators:
#
#   train() -- reads line-by-line the ./data/ptb.train.txt
#   valid() -- reads ./data/ptb.valid.txt
#   test() -- reads ./.data/ptb.test.txt

import tarfile
import hashlib
import os.path
import requests

TARBALL_FILENAME = 'simple-examples.tgz'
TARBALL_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/' + TARBALL_FILENAME
CACHE_DIR = '/usr/local/paddle/dataset/penn_treebank/'
CACHE_FILENAME = CACHE_DIR + TARBALL_FILENAME
MD5_HASH = '30177ea32e27c525793142b6bf2c8e2d'


# Shamelessly copied from http://stackoverflow.com/a/3431838/724872
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def tarball_exists():
    return os.path.exists(CACHE_FILENAME) and md5(CACHE_FILENAME) == MD5_HASH


def fetch_tarball(progress_handler):
    with open(CACHE_FILENAME, 'wb') as f:
        response = requests.get(TARBALL_URL, stream=True)

        if not response.ok:
            raise ValueError("Failed to download " + TARBALL_URL)

        for block in response.iter_content(1024):
            f.write(block)
            progress_handler()


def reader_creator(filename, download_progress_handler):
    if not tarball_exists():
        fetch_tarball(download_progress_handler)

    def reader():
        tarf = tarfile.open(CACHE_FILENAME)
        f = tarf.extractfile(filename)
        for l in f:
            print l.split()
        tarf.close()

    return reader


def train(download_progress_handler):
    return reader_creator('./simple-examples/data/ptb.train.txt',
                          download_progress_handler)


def valid(download_progress_handler):
    return reader_creator('./simple-examples/data/ptb.valid.txt',
                          download_progress_handler)


def test(download_progress_handler):
    return reader_creator('./simple-examples/data/ptb.test.txt',
                          download_progress_handler)


def download_progress_handler():
    print "Block written ..."


if __name__ == "__main__":
    r = train(download_progress_handler)
    for l in r():
        print l
