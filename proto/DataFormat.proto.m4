/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
ifdef(`proto3', `syntax = "proto2";')

package paddle;

/*
 If values is not empty and ids is empty, this is a dense vector.
 If values is not empty and ids is not empty, this is a sparse vector. The position of each value
 is specified by ids.
 If values is empty and ids is not empty, this is a sparse vector whose non-zero values are 1.
 The position of each 1 is specified by ids.
*/
message VectorSlot {
  repeated float values = 1 [packed = true];
  repeated uint32 ids = 2 [packed = true];
  /* For multidimensional data, for example "image width height depth" */
  repeated uint32 dims = 3 [packed = true];
  repeated string strs = 4; 
};

/*
 SubseqSlot use to record whether VectorSlot or any other slot in future has subseq.
 If not all VectorSlot have subseq, we only store the one who has subseq, and use *slot_id* to record it.
 One vector_slots has one sequence, and it may have N subseq, thus the number of *lens* will be N too. 
*/
message SubseqSlot {
  required uint32 slot_id = 1; //the id of slot who has subseq
  repeated uint32 lens = 2; // lengths of sub-sequence in the slot
};

message SlotDef {
  enum SlotType {
    VECTOR_DENSE = 0;
    VECTOR_SPARSE_NON_VALUE = 1;
    VECTOR_SPARSE_VALUE = 2;
    INDEX = 3;  // This can be used as label, or word id, etc.
    VAR_MDIM_DENSE = 4;
    VAR_MDIM_INDEX = 5;
    STRING = 6;
  }
  required SlotType type = 1;
  required uint32 dim = 2;  // For INDEX slots, this means the maximal index plus 1.
};

message DataHeader {
  // INDEX slot should be always after VECTOR slots.
  repeated SlotDef slot_defs = 1;
};

message DataSample {
  optional bool is_beginning = 1 [default = true]; // is the beginning of a sequence
  repeated VectorSlot vector_slots = 2;
  repeated uint32 id_slots = 3 [packed = true];
  /* use ids of VectorSlot */
  repeated VectorSlot var_id_slots = 4;
  repeated SubseqSlot subseq_slots = 5;
};

/*
 Usually, training data consists of several types of data, where each type of data is defined as a slot.
 For example, input vector and corresponding label are needed for common supervised learning tasks,
 which are regarded as two kinds of input slots.
 Data type and sequence type are orthogonal attribute of a slot. PaddlePaddle has four data types and
 three sequence types. You must assign a data type and a sequence type of each slot.
 DataHeader2 describes the data type and sequence type of input data which contains one or more slots.
 slot_defs and seq_type are defined as repeated field. The size of slot_defs and seq_type must be the
 same as the slots number.
*/
message DataHeader2 {
  enum SeqType {
    NON_SEQ = 0;
    SEQ = 1;
    SUB_SEQ = 2;
  }
  repeated SlotDef slot_defs = 1;
  repeated SeqType seq_type = 2;
}

/*
 SlotSample describes the content of each slot. Each slot has an unique slot id, which is identified
 by slot_id. slot_id starts from 0 to (slots number - 1).
 A repeated field vector_slots is defined for storing data units of each slot data, which are able to
 represent both NON_SEQ data and SEQ data. SUB_SEQ data needs to set subseq_start_positions additionally.
 Following shows how data is organized in SlotSample.
 +-------------------------+---------------------+-----------------------------------+------------------------------------------------+
 |                         | NO_SEQ              | SEQ                               |  SUB_SEQ                                       |
 +=========================+=====================+===================================+================================================+
 | VECTOR_DENSE            | [f, f, ...]         | [[f, ...], [f, ...], ...]         | [[[f, ...], ...], [[f, ...], ...],...]         |
 +-------------------------+---------------------+-----------------------------------+------------------------------------------------+
 | VECTOR_SPARSE_NON_VALUE | [i, i, ...]         | [[i, ...], [i, ...], ...]         | [[[i, ...], ...], [[i, ...], ...],...]         |
 +-------------------------+---------------------+-----------------------------------+------------------------------------------------+
 | VECTOR_SPARSE_VALUE     | [(i,f), (i,f), ...] | [[(i,f), ...], [(i,f), ...], ...] | [[[(i,f), ...], ...], [[(i,f), ...], ...],...] |
 +-------------------------+---------------------+-----------------------------------+------------------------------------------------+
 | INDEX                   |  i                  | [i, i, ...]                       | [[i, ...], [i, ...], ...]                      |
 +-------------------------+---------------------+-----------------------------------+------------------------------------------------+
 Note: In VectorSlot, i is stored in ids field and f is stored in values field.
*/
message SlotSample {
  required uint32 slot_id = 1;
  repeated VectorSlot vector_slots = 2;
  repeated uint32 subseq_start_positions = 3;
}
/*
 DataSample2 represents a sample of input data containing one or more slots defined in DataHeader2.
 The size of slots_data must be the same as the slots number defined in DataHeader2. The order of
 SlotSample in DataSample2 must also be the same as the slot_defs and seq_type order in DataHeader2.
 The slot_id of the i'th SlotSample is i(start from 0). Missing slots in a DataSample2 is not allowed.
*/
message DataSample2 {
  repeated SlotSample slots_data = 1;
}

