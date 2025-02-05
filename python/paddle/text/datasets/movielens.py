#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import re
import zipfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from paddle.dataset.common import _check_exists_and_download
from paddle.io import Dataset

if TYPE_CHECKING:
    import numpy.typing as npt

    _MovieLensDataSetMode = Literal["train", "test"]
__all__ = []

age_table = [1, 18, 25, 35, 45, 50, 56]

URL = 'https://dataset.bj.bcebos.com/movielens%2Fml-1m.zip'
MD5 = 'c4d9eecfca2ab87c1945afe126590906'


class MovieInfo:
    """
    Movie id, title and categories information are stored in MovieInfo.
    """

    index: int
    categories: list[str]
    title: str

    def __init__(self, index: str, categories: list[str], title: str) -> None:
        self.index = int(index)
        self.categories = categories
        self.title = title

    def value(self, categories_dict, movie_title_dict):
        """
        Get information from a movie.
        """
        return [
            [self.index],
            [categories_dict[c] for c in self.categories],
            [movie_title_dict[w.lower()] for w in self.title.split()],
        ]

    def __str__(self) -> str:
        return f"<MovieInfo id({self.index}), title({self.title}), categories({self.categories})>"

    def __repr__(self) -> str:
        return self.__str__()


class UserInfo:
    """
    User id, gender, age, and job information are stored in UserInfo.
    """

    index: int
    is_male: bool
    age: int
    job_id: int

    def __init__(self, index: str, gender: str, age: str, job_id: str) -> None:
        self.index = int(index)
        self.is_male = gender == 'M'
        self.age = age_table.index(int(age))
        self.job_id = int(job_id)

    def value(self):
        """
        Get information from a user.
        """
        return [
            [self.index],
            [0 if self.is_male else 1],
            [self.age],
            [self.job_id],
        ]

    def __str__(self) -> str:
        gender = "M" if self.is_male else "F"
        return f"<UserInfo id({self.index}), gender({gender}), age({age_table[self.age]}), job({self.job_id})>"

    def __repr__(self) -> str:
        return str(self)


class Movielens(Dataset):
    """
    Implementation of `Movielens 1-M <https://grouplens.org/datasets/movielens/1m/>`_ dataset.

    Args:
        data_file(str|None): path to data tar file, can be set None if
            :attr:`download` is True. Default None.
        mode(str): 'train' or 'test' mode. Default 'train'.
        test_ratio(float): split ratio for test sample. Default 0.1.
        rand_seed(int): random seed. Default 0.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True.

    Returns:
        Dataset: instance of Movielens 1-M dataset.

    Examples:

        .. code-block:: python

            >>> # doctest: +TIMEOUT(75)
            >>> import paddle
            >>> from paddle.text.datasets import Movielens

            >>> class SimpleNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            ...     def forward(self, category, title, rating):
            ...         return paddle.sum(category), paddle.sum(title), paddle.sum(rating)


            >>> movielens = Movielens(mode='train')

            >>> for i in range(10):
            ...     category, title, rating = movielens[i][-3:]
            ...     category = paddle.to_tensor(category)
            ...     title = paddle.to_tensor(title)
            ...     rating = paddle.to_tensor(rating)
            ...
            ...     model = SimpleNet()
            ...     category, title, rating = model(category, title, rating)
            ...     print(category.shape, title.shape, rating.shape)
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []
            [] [] []

    """

    mode: _MovieLensDataSetMode
    data_file: str | None
    test_ratio: float
    rand_seed: int
    movie_info: dict[int, MovieInfo]
    movie_title_dict: dict[str, int]
    categories_dict: dict[str, int]
    user_info: dict[int, UserInfo]
    data: list[list[float]]

    def __init__(
        self,
        data_file: str | None = None,
        mode: _MovieLensDataSetMode = 'train',
        test_ratio: float = 0.1,
        rand_seed: int = 0,
        download: bool = True,
    ) -> None:
        assert mode.lower() in [
            'train',
            'test',
        ], f"mode should be 'train', 'test', but got {mode}"
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert (
                download
            ), "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, URL, MD5, 'sentiment', download
            )

        self.test_ratio = test_ratio
        self.rand_seed = rand_seed

        np.random.seed(rand_seed)
        self._load_meta_info()
        self._load_data()

    def _load_meta_info(self) -> None:
        pattern = re.compile(r'^(.*)\((\d+)\)$')
        self.movie_info = {}
        self.movie_title_dict = {}
        self.categories_dict = {}
        self.user_info = {}
        with zipfile.ZipFile(self.data_file) as package:
            for info in package.infolist():
                assert isinstance(info, zipfile.ZipInfo)
                title_word_set = set()
                categories_set = set()
                with package.open('ml-1m/movies.dat') as movie_file:
                    for i, line in enumerate(movie_file):
                        line = line.decode(encoding='latin')
                        movie_id, title, categories = line.strip().split('::')
                        categories = categories.split('|')
                        for c in categories:
                            categories_set.add(c)
                        title = pattern.match(title).group(1)
                        self.movie_info[int(movie_id)] = MovieInfo(
                            index=movie_id, categories=categories, title=title
                        )
                        for w in title.split():
                            title_word_set.add(w.lower())

                for i, w in enumerate(title_word_set):
                    self.movie_title_dict[w] = i

                for i, c in enumerate(categories_set):
                    self.categories_dict[c] = i

                with package.open('ml-1m/users.dat') as user_file:
                    for line in user_file:
                        line = line.decode(encoding='latin')
                        uid, gender, age, job, _ = line.strip().split("::")
                        self.user_info[int(uid)] = UserInfo(
                            index=uid, gender=gender, age=age, job_id=job
                        )

    def _load_data(self) -> None:
        self.data = []
        is_test = self.mode == 'test'
        with zipfile.ZipFile(self.data_file) as package:
            with package.open('ml-1m/ratings.dat') as rating:
                for line in rating:
                    line = line.decode(encoding='latin')
                    if (np.random.random() < self.test_ratio) == is_test:
                        uid, mov_id, rating, _ = line.strip().split("::")
                        uid = int(uid)
                        mov_id = int(mov_id)
                        rating = float(rating) * 2 - 5.0

                        mov = self.movie_info[mov_id]
                        usr = self.user_info[uid]
                        self.data.append(
                            usr.value()
                            + mov.value(
                                self.categories_dict, self.movie_title_dict
                            )
                            + [[rating]]
                        )

    def __getitem__(self, idx: int) -> tuple[npt.NDArray[Any], ...]:
        data = self.data[idx]
        return tuple([np.array(d) for d in data])

    def __len__(self) -> int:
        return len(self.data)
