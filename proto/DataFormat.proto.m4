/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
 DataHeader2 describes the format and type of input data. Input data consists of a couple of slots.
 For each slot, a SlotDef and a SeqType type must be appointed. For example, if input data contains
 3 kinds of slot, the size of slot_defs and seq_type will both be 3.
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
 SlotSample describes the content of each slot. The unique slot id of input data is identified by slot_id.
 A repeated field vector_slots is defined for the content of slot data. SeqType and SlotDef is orthogonal.
 NON_SEQ : size of vector_slots is 1.
 SEQ : size of vector_slots is the length of sequence.
 SUB_SEQ : subseq_start_positions is defined to record start positions of the subsequence in a sequence.
 INDEX : size of ids in VectorSlot is 1.
 VECTOR_DENSE : a dense vector.
 VECTOR_SPARSE_NON_VALUE : size of ids in VectorSlot >= 1.
 VECTOR_SPARSE_VALUE : a sparse vector.
*/
message SlotSample {
  required uint32 slot_id = 1;
  repeated VectorSlot vector_slots = 2;
  repeated uint32 subseq_start_positions = 3;
}
/*
 A DataSample2 represents a sample of input data, which contains a repeated field SlotSample.
 For example, if input data contains 3 kinds of slot, the size of slots_data is 3.
*/
message DataSample2 {
  repeated SlotSample slots_data = 1;
}

