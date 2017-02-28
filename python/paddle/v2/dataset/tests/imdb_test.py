import paddle.v2.dataset.common
import tarfile

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

tarf = tarfile.open(paddle.v2.dataset.common.download(URL, 'imdb', MD5))

tf = tarf.next()
while tf != None:
    print tf.name
    tf = tarf.next()
