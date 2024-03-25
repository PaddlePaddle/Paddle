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
#include "paddle/pir/include/core/serialize_deserialize/json_utils.h"

using Json = nlohmann::json;
namespace pir{

    Json writeType(const pir::Type& type){

        if (type.isa<pir::BoolType>()){
        return pir::SerializeTypeToJson<pir::BoolType>(type.dyn_cast<pir::BoolType>());
        }else if(type.isa<pir::BFloat16Type>()){
            return pir::SerializeTypeToJson<pir::BFloat16Type>(type.dyn_cast<pir::BFloat16Type>());
        }else if(type.isa<pir::Float16Type>()){
            return pir::SerializeTypeToJson<pir::Float16Type>(type.dyn_cast<pir::Float16Type>());
        }else if(type.isa<pir::Float32Type>()){
            return pir::SerializeTypeToJson<pir::Float32Type>(type.dyn_cast<pir::Float32Type>());
        }else if(type.isa<pir::Float64Type>()){
            return pir::SerializeTypeToJson<pir::Float64Type>(type.dyn_cast<pir::Float64Type>());
        }else if(type.isa<pir::Int8Type>()){
            return pir::SerializeTypeToJson<pir::Int8Type>(type.dyn_cast<pir::Int8Type>());
        }else if(type.isa<pir::UInt8Type>()){
            return pir::SerializeTypeToJson<pir::UInt8Type>(type.dyn_cast<pir::UInt8Type>());
        }else if(type.isa<pir::Int16Type>()){
            return pir::SerializeTypeToJson<pir::Int16Type>(type.dyn_cast<pir::Int16Type>());
        }else if(type.isa<pir::Int32Type>()){
            return pir::SerializeTypeToJson<pir::Int32Type>(type.dyn_cast<pir::Int32Type>());
        }else if(type.isa<pir::Int64Type>()){
            return pir::SerializeTypeToJson<pir::Int64Type>(type.dyn_cast<pir::Int64Type>());
        }else if(type.isa<pir::IndexType>()){
            return pir::SerializeTypeToJson<pir::IndexType>(type.dyn_cast<pir::IndexType>());
        }else if(type.isa<pir::Complex64Type>()){
            return pir::SerializeTypeToJson<pir::Complex64Type>(type.dyn_cast<pir::Complex64Type>());
        }else if(type.isa<pir::Complex128Type>()){
            return pir::SerializeTypeToJson<pir::Complex64Type>(type.dyn_cast<pir::Complex64Type>());
        }else if(type.isa<pir::VectorType>()){
            return pir::SerializeTypeToJson<pir::VectorType>(type.dyn_cast<pir::VectorType>());
        }else if(type.isa<pir::DenseTensorType>()){
            return pir::SerializeTypeToJson<pir::DenseTensorType>(type.dyn_cast<pir::DenseTensorType>());
        }
        VLOG(0) << "Finish write Type ... ";
        Json type_json = Json::object();
    return type_json;
}

    template<typename T>
    Json SerializeTypeToJson(const T type){
        Json j;
        j["id"] = type.name();
        return j;
    }
    
    template<>
    Json SerializeTypeToJson<pir::VectorType>(const pir::VectorType type){
        Json j;
        j["id"] = type.name();
        Json content = Json::array();
        for (auto type_ : type.data()){
            content.push_back(writeType(type_));
        }
        j["data"] = content;
        return j;
    }
    
    template<>
    Json SerializeTypeToJson<pir::DenseTensorType>(const pir::DenseTensorType type){
        Json j;
        j["id"] = type.name();
        Json content = Json::array();
        content.push_back(writeType(type.dtype()));

        std::vector<int64_t> dims_;
        for (auto i = 0;i < type.dims().size(); i++){
            dims_.push_back(type.dims().at(i));
        }
        content.push_back(dims_);

        content.push_back(DataLayoutToString(type.data_layout()));
        
        content.push_back(type.lod());
        
        content.push_back(type.offset());
        j["data"] = content;
        return j;
    }
    

    ///////////////////////////////////////////////////////////
    template<typename T>
    Json SerializeAttrToJson(const T attr){
        Json j;
        j["id"] = attr.name();
        j["data"] = attr.data();
        return j;
    }
    
    template<>
    Json SerializeAttrToJson<pir::Complex64Attribute>(const pir::Complex64Attribute attr){
        Json j;
        j["id"] = attr.name();
        j["data"] = {attr.data().real, attr.data().imag};
        return j;
    }
    
    template<>
    Json SerializeAttrToJson<pir::Complex128Attribute>(const pir::Complex128Attribute attr){
        Json j;
        j["id"] = attr.name();
        j["data"] = {attr.data().real, attr.data().imag};
        return j;
    }

    template<>
    Json SerializeAttrToJson<pir::StrAttribute>(const pir::StrAttribute attr){
        Json j;
        j["id"] = attr.name();
        j["data"] = attr.AsString();
        return j;
    }
    
    template<>
    Json SerializeAttrToJson<pir::TypeAttribute>(const pir::TypeAttribute attr){
        Json j;
        j["id"] = attr.name();
        j["data"] = SerializeTypeToJson(attr.data());
        return j;
    }

    template<>
    Json SerializeAttrToJson<pir::ArrayAttribute>(const pir::ArrayAttribute attr){
        std::vector<int64_t> val;
        for (size_t i = 0; i < attr.size(); i++) {
            val.push_back(attr.at(i).dyn_cast<pir::Int64Attribute>().data());
        }
        Json j;
        j["id"] = attr.name();
        j["data"] = val;
        return j;
    }
    
    Json writeAttr(const pir::Attribute& attr){

        if (attr.isa<pir::BoolAttribute>()){
            return pir::SerializeAttrToJson<pir::BoolAttribute>(attr.dyn_cast<pir::BoolAttribute>());
        }else if(attr.isa<pir::FloatAttribute>()){
            return pir::SerializeAttrToJson<pir::FloatAttribute>(attr.dyn_cast<pir::FloatAttribute>());
        }else if(attr.isa<pir::DoubleAttribute>()){
            return pir::SerializeAttrToJson<pir::DoubleAttribute>(attr.dyn_cast<pir::DoubleAttribute>());
        }else if(attr.isa<pir::Int32Attribute>()){
            return pir::SerializeAttrToJson<pir::Int32Attribute>(attr.dyn_cast<pir::Int32Attribute>());
        }else if(attr.isa<pir::Int64Attribute>()){
            return pir::SerializeAttrToJson<pir::Int64Attribute>(attr.dyn_cast<pir::Int64Attribute>());
        }else if(attr.isa<pir::IndexAttribute>()){
            return pir::SerializeAttrToJson<pir::IndexAttribute>(attr.dyn_cast<pir::IndexAttribute>());
        }else if(attr.isa<pir::ArrayAttribute>()){
            return pir::SerializeAttrToJson<pir::ArrayAttribute>(attr.dyn_cast<pir::ArrayAttribute>());
        }else if(attr.isa<pir::TypeAttribute>()){
            return pir::SerializeAttrToJson<pir::TypeAttribute>(attr.dyn_cast<pir::TypeAttribute>());
        }else if(attr.isa<pir::TensorNameAttribute>()){
            return pir::SerializeAttrToJson<pir::TensorNameAttribute>(attr.dyn_cast<pir::TensorNameAttribute>());
        }else if(attr.isa<pir::Complex64Attribute>()){
            return pir::SerializeAttrToJson<pir::Complex64Attribute>(attr.dyn_cast<pir::Complex64Attribute>());
        }else if(attr.isa<pir::Complex128Attribute>()){
            return pir::SerializeAttrToJson<pir::Complex128Attribute>(attr.dyn_cast<pir::Complex128Attribute>());
        }else if(attr.isa<pir::StrAttribute>()){
            return pir::SerializeAttrToJson<pir::StrAttribute>(attr.dyn_cast<pir::StrAttribute>());
        }

    }
}//namepsace pir