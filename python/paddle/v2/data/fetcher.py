"""
Fetcher is a utility package for paddle.data package. It used to download
dataset and decompress dataset.
"""

import gzip
import hashlib
import os
import tarfile

import requests

__all__ = ['Fetcher']


def __get_md5__(f, chunk_size):
    content_hash = hashlib.md5()
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        content_hash.update(data)
    return content_hash.hexdigest()


def __is_file_matched__(filename, md5, chunk_size):
    if md5 is not None:
        with open(filename, 'rb') as f:
            if md5 != __get_md5__(f, chunk_size):
                return False
    return True


class IFileOpener(object):
    """
    File Opener is a interface to open a file inside dataset. Because the
    dataset file could be a tar.gz package or gzipped file, IFileOpener could
    get every file-like object in dataset by open method.
    """

    def __init__(self, working_dir, filename):
        self.__working_dir__ = working_dir
        self.__filename__ = filename

    def open(self, path=None):
        """
        open a file inside the dataset. If the dataset is only one file, the
        path argument will be ignored.

        :param path: File path inside dataset package, like tgz file.
        :return: file-like object
        :rtype: file
        """
        raise NotImplementedError()

    def walk(self, *args, **kwargs):
        raise ValueError("Not support")


class GZipFileOpener(IFileOpener):
    """
    Open a gzipped file. The filename must end with '.gz'

    Like XXX.json.gz XXX.txt.gz
    """

    extension = ".gz"

    def open(self, path=None):
        __filename__ = os.path.join(self.__working_dir__, self.__filename__)
        return gzip.open(__filename__)


class TarGzFileOpener(IFileOpener):
    """
    Opener for a tar gz package. The filename must end with '.tar.gz'
    """

    extension = ".tar.gz"

    def open(self, path=None):
        if not os.path.exists(os.path.join(self.__working_dir__, 'done')):
            tar = tarfile.open(
                os.path.join(self.__working_dir__, self.__filename__))
            tar.extractall(self.__working_dir__)
            tar.close()
            with open(os.path.join(self.__working_dir__, 'done'), 'w') as f:
                f.write('ok\n')

        if path is None:
            raise ValueError("Path must be set")
        try:
            f = open(os.path.join(self.__working_dir__, path))
            return f
        except IOError as e:
            e.filename = path
            raise

    def walk(self, top='.', *args, **kwargs):
        top = os.path.join(self.__working_dir__, top)
        return os.walk(top, *args, **kwargs)


class TGZFileOpener(TarGzFileOpener):
    """
    Opener for a tar.gz package. The filename must end with '.tgz'
    """
    extension = ".tgz"


class NormalFileOpener(IFileOpener):
    """
    Naive Opener for any file, just return a file object.
    """

    def open(self, path=None):
        return open(os.path.join(self.__working_dir__, self.__filename__))


extension_readers = [TarGzFileOpener, TGZFileOpener, GZipFileOpener]


class Fetcher(object):
    """
    Fetch a url with a filename. If the data is fetched before, then do not
    fetch again.

    :param url: The remote url string. Only support http/https protocol now.
    :type url: str
    :param filename: The downloaded filename.
    :type filename: str
    :param md5: The md5 hash sum of downloaded file. It is OK not set the md5
                checksum, but it is not recommended.
    :type md5: str
    :param local_cached_dir: The local directory used for download file.
    :type local_cached_dir: str
    :param download_progress: The download progress callback.
    :type download_progress: callable (downloaded_bytes, total_bytes) => None
    :param chunk_size: The chunk size during downloading.
    :type chunk_size: int
    """

    def __init__(self,
                 url,
                 filename,
                 md5=None,
                 local_cached_dir=None,
                 download_progress=None,
                 chunk_size=1024):
        self.__filename__ = filename
        if local_cached_dir is None:
            local_cached_dir = "~/.cache/paddle.fetcher/"
        url_hash = hashlib.md5()
        url_hash.update(url)
        self.__cached_dir__ = os.path.join(
            os.path.expanduser(local_cached_dir), url_hash.hexdigest())

        self.__reader__ = None
        for each_extension in extension_readers:
            if self.__filename__.endswith(each_extension.extension):
                self.__reader__ = each_extension(self.__cached_dir__,
                                                 self.__filename__)
                break
        if self.__reader__ is None:
            self.__reader__ = NormalFileOpener(self.__cached_dir__,
                                               self.__filename__)

        out_fn = os.path.join(self.__cached_dir__, self.__filename__)
        if os.path.exists(out_fn):
            if __is_file_matched__(out_fn, md5, chunk_size):
                return
            else:
                os.remove(out_fn)
        try:  # removed previous directory.
            os.rmdir(self.__cached_dir__)
        except OSError:
            pass
        os.makedirs(self.__cached_dir__)

        req = requests.get(url=url, stream=True)
        length = req.headers['content-length']
        content_hash = None
        if md5 is not None:
            content_hash = hashlib.md5()
        downloaded_length = 0
        with open(out_fn, 'wb') as f:
            for i, chunk in enumerate(req.iter_content(chunk_size=chunk_size)):
                if chunk:
                    downloaded_length += len(chunk)
                    f.write(chunk)
                    if download_progress is not None:
                        download_progress(downloaded_length, length)
                    if content_hash is not None:
                        content_hash.update(chunk)
        req.close()

        if md5 is not None and md5 != content_hash.hexdigest():
            raise ValueError("Hash sum mismatch")

    def open(self, path=None):
        """
        Open a file inside dataset.

        :param path: The path inside dataset.
        :type path: str
        :return: The file-like object.
        :rtype: file
        """
        return self.__reader__.open(path=path)

    def walk(self, *args, **kwargs):
        return self.__reader__.walk(*args, **kwargs)


def unittest():
    test_set = (
        ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
         'train-images-idx3-ubyte.gz', (
             (None, '6bbc9ace898e44ae57da46a324031adb'), )),
        ('http://snap.stanford.edu/data/amazon/productGraph/'
         'categoryFiles/reviews_Electronics_5.json.gz',
         'reviews_Electronics_5.json.gz', (
             (None, '89fd3bb8a4a701ecf5be75c840bb6979'), )),
        ('http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz',
         'aclImdb_v1.tar.gz', (
             ('./aclImdb/imdb.vocab', '23c86a0533c0151b6f12fa52b106dcc2'),
             ('./aclImdb/train/neg/10000_4.txt',
              'd122bc3b46ac372847d610f6382318c7'))),
        ('http://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz',
         'conll05st-tests.tar.gz', (
             ('./conll05st-release/test.wsj/words/test.wsj.words.gz',
              'd923123c71b0b19026ecaf8c2e9e4e8a'), )), )

    for url, filename, files in test_set:
        fetcher = Fetcher(url=url, filename=filename)
        for fn, md5 in files:
            with fetcher.open(fn) as f:
                fmd5 = __get_md5__(f, 1024)
                assert fmd5 == md5


if __name__ == '__main__':
    unittest()
