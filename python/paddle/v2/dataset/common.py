import requests
import hashlib
import os
import shutil

__all__ = ['DATA_HOME', 'download', 'md5file']

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')

if not os.path.exists(DATA_HOME):
    os.makedirs(DATA_HOME)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, url.split('/')[-1])
    if not (os.path.exists(filename) and md5file(filename) == md5sum):
        r = requests.get(url, stream=True)
        with open(filename, 'w') as f:
            shutil.copyfileobj(r.raw, f)

    return filename


def dict_add(a_dict, ele):
    if ele in a_dict:
        a_dict[ele] += 1
    else:
        a_dict[ele] = 1
