import collections

__all__ = ['WordDict']


class WordDict(object):
    def __init__(self):
        self.__word_counter__ = collections.defaultdict(lambda: 0)

    def append(self, word):
        self.__word_counter__[word] += 1

    def to_dict(self, limit=None):
        word_list = []
        for word in self.__word_counter__.keys():
            word_list.append((word, self.__word_counter__[word]))
        word_list.sort(key=lambda x: x[1], reverse=True)
        if limit is not None and len(word_list) > limit:
            word_list = word_list[:limit]

        ret_dict = dict()
        for i, word in enumerate((w for w, _ in word_list)):
            ret_dict[word] = i
        return ret_dict
