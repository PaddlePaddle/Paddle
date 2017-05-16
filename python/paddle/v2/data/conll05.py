import fetcher
import word_dict
import os
import gzip
import itertools

__all__ = ['Conll05']

test_file_url = 'http://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz'
md5sum = '387719152ae52d60422c016e92a742fc'


class Conll05(object):
    """
    Conll 2005 dataset.  Paddle semantic role labeling Book and demo use this
    dataset as an example. Because Conll 2005 is not free in public, the default
    downloaded URL is test set of Conll 2005 (which is public). Users can change
    URL and MD5 to their Conll dataset.

    :param url: The HTTP URL to download Conll05 dataset.
    :type url: basestring
    :param md5: The MD5 checksum of downloaded file.
    :type md5: basestring
    """

    def __init__(self, url=test_file_url, md5=md5sum):
        self.__fetcher__ = fetcher.Fetcher(
            url=url, md5=md5, filename=os.path.split(url)[-1])
        _, dirnames, _ = next(self.__fetcher__.walk(top="."))
        self.__base_dir__ = dirnames[0]

    def corpus_names(self):
        """
        corpus_names return the name of each corpus in the Conll05 dataset.
        :return: names of each corpus
        :rtype: list
        """
        _, dirnames, _ = next(self.__fetcher__.walk(top=self.__base_dir__))
        return dirnames

    def get_word_dictionary(self, limit=None):
        """
        get_word_dictionary returns the word dictionary, which key is a word and
        the value is word index. The words in this dictionary are all words in
        corpora. The word indices are sorted by word frequency in corpora in
        descending order.


        :param limit: The max length of the dictionary.
        :return: word dictionary
        :rtype: dict
        """
        wd = word_dict.WordDict()
        for corpus_name in self.corpus_names():
            with self.__open__(corpus_name, 'words') as f:
                for line in f:
                    line = line.strip().lower()
                    if len(line) != 0:
                        wd.append(line)
        return wd.to_dict(limit=limit)

    def get_label_dictionary(self):
        """
        get_label_dictionary returns the label dictionary. It contains all
        semantic role labels as key, including 'Other' label as 'O'.

        :return: the label dictionary
        :rtype: dict
        """
        srl_label = set()
        for corpus_name in self.corpus_names():
            with self.__open__(corpus_name, "props") as f:
                for line in f:
                    line = line.strip().split()
                    for w in line[1:]:
                        assert isinstance(w, basestring)
                        w = w.strip('*()')
                        if len(w) != 0:
                            srl_label.add(w)
        srl_label = list(srl_label)
        srl_label.sort()

        ret = dict()
        for i, label in enumerate(
                itertools.chain(
                    ['O'],  # O means other.
                    *(('B-' + l, 'I-' + l) for l in srl_label))):
            ret[label] = i
        return ret

    def read_corpus(self, name, word_dict=None, label_dict=None, unk_id=None):
        """
        Read one corpus by corpus name. It returns an iterator. Each element of
        this iterator is a tuple including sentence and labels. The sentence is
        consist of a list of word IDs. The labels include a list of label IDs.

        :param name: corpus name.
        :type name: basestring
        :param word_dict: word dictionary. If not set, then use
                          `get_word_dictionary` to generate one.
        :type word_dict: None|dict
        :param label_dict: label dictionary. If not set, then use
                           `get_label_dictionary` to generate one.
        :type label_dict: None|dict
        :param unk_id: Unknown Key ID. Used for the word not in word dictionary.
                       None if unk_id is the length of word_dict.
        :type unk_id: None|int
        :return: a iterator of data.
        :rtype: iterator
        """

        if word_dict is None:
            word_dict = self.get_word_dictionary()
        if label_dict is None:
            label_dict = self.get_label_dictionary()

        if unk_id is None:
            unk_id = len(word_dict)

        if name not in self.corpus_names():
            raise ValueError("No such corpus name %s" % name)

        with self.__open__(
                name=name, filename='props') as props_file, self.__open__(
                    name=name, filename='words') as words_file:
            sentences = []
            labels = None
            status = None
            for word, label in itertools.izip(words_file, props_file):
                word = word.strip()
                label = label.strip().split()
                if len(label) == 0:  # end of sentence
                    sentences = [
                        word_dict.get(word, unk_id) for word in sentences
                    ]
                    for lbl in labels:
                        if len(lbl) != 0:
                            assert len(sentences) == len(lbl)
                            yield sentences, [label_dict[l] for l in lbl]

                    sentences = []
                    labels = None
                    status = None
                else:
                    if labels is None:
                        labels = [[] for _ in xrange(len(label))]
                        status = [None] * (len(label) - 1)

                    sentences.append(word)
                    for i, l in enumerate(label[1:]):
                        if l == '*':
                            if status[i] is None:
                                labels[i].append('O')
                            else:
                                labels[i].append('I-' + status[i])
                        elif l[0] == '(':
                            l = l.strip('*()')
                            status[i] = l
                            labels[i].append('B-' + l)
                        elif l[-1] == ')':
                            labels[i].append('I-' + status[i])
                            status[i] = None

    def __open__(self, name, filename):
        gzipped_file = self.__fetcher__.open(
            os.path.join(self.__base_dir__, name, filename, name + "." +
                         filename + ".gz"))
        return gzip.GzipFile(fileobj=gzipped_file)

    def __call__(self, names=None, *args, **kwargs):
        """
        Reader Creator interface. It is a wrapper of read_corpus method.
        """

        def reader():
            if names is None:
                nms = self.corpus_names()
            else:
                nms = names
            return itertools.chain(*(self.read_corpus(nm, *args, **kwargs)
                                     for nm in nms))

        return reader


def main():
    conll05 = Conll05()
    reader = conll05()
    #  conll05 is also a reader creator
    #  assert isinstance(conll05, reader_creator)
    for sentence, lbl in reader():
        print(sentence, lbl)


if __name__ == '__main__':
    main()
