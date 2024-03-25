// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <nlohmann/json.hpp>
#include "paddle/pir/include/core/serialize_deserialize/json_utils.h"

namespace pir{

class ProgramWriter{
    public:
        explicit ProgramWriter(const uint64_t version):version_(version){};

        ProgramWriter(ProgramWriter&&) = delete;
        ProgramWriter(const ProgramWriter& ProgramWriter) = delete;
        ProgramWriter& operator=(const ProgramWriter&) = delete;
        ProgramWriter& operator=(ProgramWriter&&);
        
        //static void staticInit()
        
        Json GetProgramJson(const pir::Program* program);
        ~ProgramWriter() = default;

    private:
        uint64_t version_;
        Json program_json;
        std::map<pir::Value, int64_t> value_id_map;
        int64_t region_id_ = 0;
        int64_t block_id_ = 0;
        int64_t value_id_ = 1;
        int64_t blockarg_id_ = -1;

        Json WriteProgram(const pir::Program* program);
        Json WriteRegion(const pir::Region* region, const std::string& region_name);
        Json WriteBlock(const pir::Block* block, const std::string& block_name);
        Json WriteOp(const pir::Operation& op);
        Json WriteBlockArg(const pir::Value& value);
        Json WriteValue(const pir::Value& value);
        Json WriteOpOperand(const pir::OpOperand& op_operand);
        Json WriteAttributesMap_0(const AttributeMap& attr_map);
        Json WriteAttributesMap_1(const AttributeMap& attr_map);
        Json WriteAttribute(const std::string& op_attr_name, const pir::Attribute& attr);
        Json WriteType(const pir::Type& type);
};

void WriteModule(const pir::Program &program, const std::string& file_path, const uint64_t &pir_version, bool overwrite);
}//namepsace pir