## CTR READER

An multi-thread cpp reader that has the same interface with py_reader. It
uses cpp multi-thread to read file and is much more faster then the Python read
thread in py_reader.

Currently, it support two types of file:
 - gzip
 - plain text file

and two types of data format:
 - cvs data format is :
   * label dense_fea,dense_fea sparse_fea,sparse_fea
 - the svm data format is :
   * label slot1:fea_sign slot2:fea_sign slot1:fea_sign
