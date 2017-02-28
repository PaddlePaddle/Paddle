import paddle.v2 as paddle


def main():
    movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
    title_word_count = len(movie_title_dict)

    paddle.layer.mixed


if __name__ == '__main__':
    main()
