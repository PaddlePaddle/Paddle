/* Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved.

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


message Attribute {
  required string name = 1; 
  enum AttributeType {
  	REAL = 0;
  	BOOL = 1;
  	INT32 = 2;
  	STRING = 3;
  }

  required AttributeType type = 2;

  optional real r_val = 3;
  optional bool b_val = 4;
  optional int32 i_val = 5;
  optional string s_val = 6;
}
