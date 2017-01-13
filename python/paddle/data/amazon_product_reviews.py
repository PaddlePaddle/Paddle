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
    __md5__[Books] = 'bc1e2aa650fe51f978e9d3a7a4834bc6'
    __md5__[CDsAndVinyl] = '82bffdc956e76c32fa655b98eca9576b'
    __md5__[CellPhonesAndAccessories] = '903a19524d874970a2f0ae32a175a48f'
    __md5__[ClothingShoesAndJewelry] = 'b333fba48651ea2309288aeb51f8c6e4'
    __md5__[DigitalMusic] = '35e62f7a7475b53714f9b177d9dae3e7'
    __md5__[Electronics] = 'e4524af6c644cd044b1969bac7b62b2a'
    __md5__[GroceryAndGourmetFood] = 'd8720f98ea82c71fa5c1223f39b6e3d9'
    __md5__[HealthAndPersonalCare] = '352ea1f780a8629783220c7c9a9f7575'
    __md5__[HomeAndKitchen] = '90221797ccc4982f57e6a5652bea10fc'
    __md5__[KindleStore] = 'b608740c754287090925a1a186505353'
    __md5__[MoviesAndTV] = 'd3bb01cfcda2602c07bcdbf1c4222997'
    __md5__[MusicalInstruments] = '8035b6e3f9194844785b3f4cee296577'
    __md5__[OfficeProducts] = '1b7e64c707ecbdcdeca1efa09b716499'
    __md5__[PatioLawnAndGarden] = '4d2669abc5319d0f073ec3c3a85f18af'
    __md5__[PetSupplies] = '40568b32ca1536a4292e8410c5b9de12'
    __md5__[SportsAndOutdoors] = '1df6269552761c82aaec9667bf9a0b1d'
    __md5__[ToolsAndHomeImprovement] = '80bca79b84621d4848a88dcf37a1c34b'
    __md5__[ToysAndGames] = 'dbd07c142c47473c6ee22b535caee81f'
    __md5__[VideoGames] = '730612da2d6a93ed19f39a808b63993e'


__all__ = ['fetch', 'Categories']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


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

    fn = os.path.join(directory, '%s.json.gz' % category)

    if os.path.exists(fn) and \
                    calculate_md5(category) == Categories.__md5__[category]:
        # already download.
        return fn

    logger.info("Downloading amazon review dataset for %s category" % category)
    return download(BASE_URL % category, fn)
