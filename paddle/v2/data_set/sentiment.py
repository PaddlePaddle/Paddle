import random
import nltk
import numpy as np
from nltk.corpus import movie_reviews
from config import DATA_HOME

__all__ = ['train', 'test', 'get_label_dict', 'get_word_dict']
SPLIT_NUM = 800
TOTAL_DATASET_NUM = 1000


def get_label_dict():
    label_dict = {'neg': 0, 'pos': 1}
    return label_dict


def is_download_data():
    try:
        nltk.data.path.append(DATA_HOME)
        movie_reviews.categories()
    except LookupError:
        print "dd"
        nltk.download('movie_reviews', download_dir=DATA_HOME)
        nltk.data.path.append(DATA_HOME)


def get_word_dict():
    words_freq_sorted = list()
    is_download_data()
    words_freq = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    words_sort_list = words_freq.items()
    words_sort_list.sort(cmp=lambda a, b: b[1] - a[1])
    print words_sort_list
    for index, word in enumerate(words_sort_list):
        words_freq_sorted.append(word[0])
    return words_freq_sorted


def load_sentiment_data():
    label_dict = get_label_dict()
    is_download_data()
    words_freq = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    data_set = [([words_freq[word]
                for word in movie_reviews.words(fileid)], label_dict[category])
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
    random.shuffle(data_set)
    return data_set


data_set = load_sentiment_data()


def reader_creator(data_type):
    if data_type == 'train':
        for each in data_set[0:SPLIT_NUM]:
            train_sentences = np.array(each[0], dtype=np.int32)
            train_label = np.array(each[1], dtype=np.int8)
            yield train_sentences, train_label
    else:
        for each in data_set[SPLIT_NUM:]:
            test_sentences = np.array(each[0], dtype=np.int32)
            test_label = np.array(each[1], dtype=np.int8)
            yield test_sentences, test_label


def train():
    return reader_creator('train')


def test():
    return reader_creator('test')


if __name__ == '__main__':
    for train in train():
        print "train"
        print train
    for test in test():
        print "test"
        print test
