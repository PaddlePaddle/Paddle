# /usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
A utility for fetching, reading amazon product review data set.

http://jmcauley.ucsd.edu/data/amazon/
"""

import os
from http_download import download
from logger import logger
import hashlib

BASE_URL = 'http://snap.stanford.edu/data/' \
           'amazon/productGraph/categoryFiles/reviews_%s_5.json.gz'


class Categories(object):
    Books = "Books"
    Electronics = "Electronics"
    MoviesAndTV = "Movies_and_TV"
    CDsAndVinyl = "CDs_and_Vinyl"
    ClothingShoesAndJewelry = "Clothing_Shoes_and_Jewelry"
    HomeAndKitchen = "Home_and_Kitchen"
    KindleStore = "Kindle_Store"
    SportsAndOutdoors = "Sports_and_Outdoors"
    CellPhonesAndAccessories = "Cell_Phones_and_Accessories"
    HealthAndPersonalCare = "Health_and_Personal_Care"
    ToysAndGames = "Toys_and_Games"
    VideoGames = "Video_Games"
    ToolsAndHomeImprovement = "Tools_and_Home_Improvement"
    Beauty = "Beauty"
    AppsForAndroid = "Apps_for_Android"
    OfficeProducts = "Office_Products"
    PetSupplies = "Pet_Supplies"
    Automotive = "Automotive"
    GroceryAndGourmetFood = "Grocery_and_Gourmet"
    PatioLawnAndGarden = "Patio_Lawn_and_Garden"
    Baby = "Baby"
    DigitalMusic = "Digital_Music"
    MusicalInstruments = "Musical_Instruments"
    AmazonInstantVideo = "Amazon_Instant_Video"

    __md5__ = dict()

    __md5__[AmazonInstantVideo] = '10812e43e99c345f63333d8ee10aef6a'
    __md5__[AppsForAndroid] = 'a7d1ae198b862eea6910fe45c842b0c6'
    __md5__[Automotive] = '757fdb1ab2c5e2fc0934047721082011'
    __md5__[Baby] = '7698a4179a1d8385e946ed9083490d22'
    __md5__[Beauty] = '5d2ccdcd86641efcfbae344317c10829'


__all__ = ['fetch', 'Categories']


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.

    Args:

    Returns:
        path for the data untar.
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    if not os.path.exists(directory):
        os.makedirs(directory)
    logger.info("Downloading amazon review dataset for %s category" % category)
    return download(BASE_URL % category,
                    os.path.join(directory, '%s.json.gz' % category))


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    categories = filter(
        lambda c: getattr(Categories, c) not in Categories.__md5__.keys(),
        filter(lambda c: c[0] != '_', dir(Categories)))

    for each in categories:
        try:
            filename = fetch(category=getattr(Categories, each))
        except Exception as e:
            print type(e)
            continue
        print each, calculate_md5(filename)
        os.remove(filename)


if __name__ == '__main__':
    main()
