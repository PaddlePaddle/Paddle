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
MQ2007 dataset

MQ2007 is a query set from Million Query track of TREC 2007. There are about 1700 queries in it with labeled documents. In MQ2007, the 5-fold cross
validation strategy is adopted and the 5-fold partitions are included in the package. In each fold, there are three subsets for learning: training set,
validation set and testing set. 

MQ2007 dataset from 
http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar and parse training set and test set into paddle reader creators

"""


import os
import random
import functools
import rarfile
from common import download
import numpy as np


# URL = "http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar"
URL = "http://www.bigdatalab.ac.cn/benchmark/upload/download_source/7b6dbbe2-842c-11e4-a536-bcaec51b9163_MQ2007.rar"
MD5 = "7be1640ae95c6408dab0ae7207bdc706"


def __initialize_meta_info__():
  """
  download and extract the MQ2007 dataset
  """
  fn = fetch()
  rar = rarfile.RarFile(fn)
  dirpath = os.path.dirname(fn)
  rar.extractall(path=dirpath)
  return dirpath


class Query(object):
  """
  queries used for learning to rank algorithms. It is created from relevance scores,  query-document feature vectors

  Parameters:
  ----------
  query_id : int
    query_id in dataset, mapping from query to relevance documents
  relevance_score : int 
    relevance score of query and document pair
  feature_vector : array, dense feature
    feature in vector format
  description : string
    comment section in query doc pair data
  """
  def __init__(self, query_id=-1, relevance_score=-1,
               feature_vector=None, description=""):
    self.query_id = query_id
    self.relevance_score = relevance_score
    if feature_vector is None:
      self.feature_vector = []
    else:
      self.feature_vector = feature_vector
    self.description = description

  def __str__(self):
    string = "%s %s %s" %(str(self.relevance_score), str(self.query_id), " ".join(str(f) for f in self.feature_vector))
    return string

  # @classmethod
  def _parse_(self, text):
    """
    parse line into Query
    """
    comment_position = text.find('#')
    line = text[:comment_position].strip()
    self.description = text[comment_position+1:].strip()
    parts = line.split()
    assert(len(parts) == 48), "expect 48 space split parts, get %d" %(len(parts))
    # format : 0 qid:10 1:0.000272 2:0.000000 .... 
    self.relevance_score = int(parts[0])
    self.query_id = int(parts[1].split(':')[1])
    for p in parts[2:]: 
      pair = p.split(':')
      self.feature_vector.append(float(pair[1]))
    return self

class QueryList(object):
  """
  group query into list, every item in list is a Query
  """
  def __init__(self, querylist=None):
    self.query_id = -1
    if querylist is None:
      self.querylist = []
    else:
      self.querylist = querylist
      for query in self.querylist:
        if self.query_id == -1:
          self.query_id = query.query_id
        else:
          if self.query_id != query.query_id:
            raise ValueError("query in list must be same query_id")

  def __iter__(self):
    for query in self.querylist:
      yield query

  def __len__(self):
    return len(self.querylist)

  def _correct_ranking_(self):
    if self.querylist is None:
      return 
    self.querylist.sort(key=lambda x:x.relevance_score, reverse=True)

  def _add_query(self, query):
      if self.query_id == -1:
        self.query_id = query.query_id
      else:
        if self.query_id != query.query_id:
          raise ValueError("query in list must be same query_id")
      self.querylist.append(query)



def gen_pair(querylist, partial_order="full"):
  """
  gen pair for pair-wise learning to rank algorithm
  Paramters:
  --------
  querylist : querylist, one query match many docment pairs in list, see QueryList
  pairtial_order : "full" or "neighbour"
  gen pairs for neighbour items or the full partial order pairs

  return :
  ------
  label : np.array, shape=(1)
  query_left : np.array, shape=(1, feature_dimension)
  query_right : same as left
  """
  if not isinstance(querylist, QueryList):
    querylist = QueryList(querylist)
  querylist._correct_ranking_()
  # C(n,2)
  if partial_order == "full":
    for i, query_left in enumerate(querylist):
      for j, query_right in enumerate(querylist):
        if query_left.relevance_score > query_right.relevance_score:
          yield 1, np.array(query_left.feature_vector), np.array(query_right.feature_vector)
        else:
          yield 1, np.array(query_left.feature_vector), np.array(query_right.feature_vector)

  elif partial_order == "neighbour":
    # C(n)
    k = 0 
    while k < len(querylist)-1:
      query_left = querylist[k]
      query_right = querylist[k+1]
      if query_left.relevance_score > query_right.relevance_score:
        yield 1, np.array(query_left.feature_vector), np.array(query_right.feature_vector)
      else:
        yield 1, np.array(query_left.feature_vector), np.array(query_right.feature_vector)
      k += 1
  else:
    raise ValueError("unsupport parameter of partial_order, Only can be neighbour or full")

  
def gen_list(querylist):
  """
  gen item in list for list-wise learning to rank algorithm
  Paramters:
  --------
  querylist : querylist, one query match many docment pairs in list, see QueryList

  return :
  ------
  label : np.array, shape=(samples_num, )
  querylist : np.array, shape=(samples_num, feature_dimension)
  """
  if not isinstance(querylist, QueryList):
    querylist = QueryList(querylist)
  querylist._correct_ranking_()
  relevance_score_list = [query.relevance_score for query in querylist]
  feature_vector_list = [query.feature_vector for query in querylist]
  # yield np.array(relevance_score_list).T, np.array(feature_vector_list)
  for i in range(len(querylist)):
    yield relevance_score_list[i], np.array(feature_vector_list[i])


def load_from_text(filepath, shuffle=True, fill_missing=-1):
  """
  parse data file into querys
  """
  prev_query_id = -1;
  querylists = []
  querylist = None
  fn = __initialize_meta_info__()
  with open(os.path.join(fn, filepath)) as f:
    for line in f:
      query = Query()
      query = query._parse_(line)
      if query.query_id != prev_query_id:
        if querylist is not None:
          querylists.append(querylist)
        querylist = QueryList()
        prev_query_id = query.query_id
      querylist._add_query(query)
  if shuffle == True:
    random.shuffle(querylists)
  return querylists


def __reader__(filepath, format="pairwise", shuffle=True, fill_missing=-1):
  """
  Parameters
  --------
  filename : string
  shuffle : shuffle query-doc pair under the same query
  fill_missing : fill the missing value. default in MQ2007 is -1
  
  Returns
  ------
  yield
    label query_left, query_right  # format = "pairwise"
    label querylist # format = "listwise"
  """
  querylists = load_from_text(filepath, shuffle=shuffle, fill_missing=fill_missing)
  for querylist in querylists:
    if format == "pairwise":
      for pair in gen_pair(querylist):
        yield pair
    elif format == "listwise":
      # yield next(gen_list(querylist))
      for instance in gen_list(querylist):
        yield instance

train = functools.partial(__reader__,filepath="MQ2007/MQ2007/Fold1/train.txt")
test = functools.partial(__reader__, filepath="MQ2007/MQ2007/Fold1/test.txt")


def fetch():
  return download(URL, "MQ2007", MD5)

if __name__ == "__main__":
  fetch()

