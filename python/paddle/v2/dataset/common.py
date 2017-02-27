import hashlib
import os
import shutil
import urllib2

__all__ = ['DATA_HOME', 'download', 'md5file']

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')

if not os.path.exists(DATA_HOME):
    os.makedirs(DATA_HOME)


def download(url, package_name, md5):
    filename = os.path.split(url)[-1]
    assert DATA_HOME is not None
    filepath = os.path.join(DATA_HOME, md5)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    __full_file__ = os.path.join(filepath, filename)

    def __file_ok__():
        if not os.path.exists(__full_file__):
            return False
        md5_hash = hashlib.md5()
        with open(__full_file__, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest() == md5

    while not __file_ok__():
        response = urllib2.urlopen(url)
        with open(__full_file__, mode='wb') as of:
            shutil.copyfileobj(fsrc=response, fdst=of)
    return __full_file__


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()
