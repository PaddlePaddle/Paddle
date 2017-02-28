import paddle.v2 as paddle


def main():
    movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_user_id() + 1))
    usr_emb = paddle.layer.embedding(input=uid, size=32)

    usr_gender_id = paddle.layer.data(
        name='gender_id', type=paddle.data_type.integer_value(2))
    usr_gender_emb = paddle.layer.embedding(input=usr_gender_id, size=16)

    usr_age_id = paddle.layer.data(
        name='age_id',
        type=paddle.data_type.integer_value(
            len(paddle.dataset.movielens.age_table)))
    usr_age_emb = paddle.embedding(input=usr_age_id, size=16)

    usr_combined_features = paddle.fc(
        input=[usr_emb, usr_gender_emb, usr_age_emb],
        size=200,
        act=paddle.activation.Tanh())

    mov_id = paddle.layer.data(
        name='movie_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_movie_id() + 1))
    mov_emb = paddle.layer.embedding(input=mov_id, size=32)

    mov_title_id = paddle.layer.data(
        name='movie_title',
        type=paddle.data_type.integer_value(len(movie_title_dict)))
    mov_title_emb = paddle.embedding(input=mov_title_id, size=32)
    with paddle.layer.mixed() as mixed:
        pass


if __name__ == '__main__':
    main()
