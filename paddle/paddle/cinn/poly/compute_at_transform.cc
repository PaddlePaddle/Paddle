// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/poly/compute_at_transform.h"

namespace cinn {
namespace poly {

void ComputeAtTransform::AdjustPdomain() {
  isl::map ct_with_params = ctransform_with_params();
  isl::set ct_domain = ct_with_params.domain();

  isl::set cdomain1 = isl::manage(AddParamsTo(cdomain_.copy()));

  VLOG(3) << "ct_domain: " << ct_domain.space();
  VLOG(3) << "cdomain1: " << cdomain1.space();

  ct_domain = ct_domain.intersect(cdomain1);
  VLOG(3) << "ct_domain: " << ct_domain;

  // get producer domain from access
  isl::map access_with_params = isl::manage(AddParamsTo(access_.copy()));

  isl::set pdomain = ct_domain.apply(access_with_params);

  // intect with the original producer domain
  auto pdomain_params = isl::manage(AddParamsTo(pdomain_.copy()));
  VLOG(4) << "pdomain: " << pdomain;
  VLOG(4) << "pdomain_params: " << pdomain_params;
  adjusted_pdomain_ = isl::manage(
      isl_set_intersect(pdomain.release(), pdomain_params.release()));
  adjusted_pdomain_ = isl::manage(isl_simplify(adjusted_pdomain_.release()));
  VLOG(4) << "adjusted pdomain: " << adjusted_pdomain_;
}

void ComputeAtTransform::AdjustPtransform() {
  // insert level+1 dims from ctransform's range into ptransform's range

  {
    // insert empty dims to ptransform's range
    adjusted_ptransform_ = ptransform_;
    adjusted_ptransform_ = isl::manage(isl_map_insert_dims(
        adjusted_ptransform_.release(), isl_dim_out, 0, level_ + 1));

    // update the tuple name
    adjusted_ptransform_ = isl::manage(isl_map_set_tuple_name(
        adjusted_ptransform_.release(), isl_dim_in, ptuple()));
    adjusted_ptransform_ = isl::manage(isl_map_set_tuple_name(
        adjusted_ptransform_.release(), isl_dim_out, ptuple()));
  }

  {
    // make ctransform range the same space with ptransform's range so that we
    // can copy the dims
    isl::set ct_range = cdomain_.apply(ctransform_);
    isl::set ct_range1 = isl::manage(isl_set_project_out(
        ct_range.release(),
        isl_dim_set,
        level_ + 1,
        isl_set_dim(ct_range.get(), isl_dim_set) - level_ - 1));
    ct_range1 = isl::manage(isl_set_add_dims(
        ct_range1.release(),
        isl_dim_set,
        isl_map_dim(adjusted_ptransform_.get(), isl_dim_out) - level_ - 1));
    // set as the producer's tuple to make a same space
    ct_range1 =
        isl::manage(isl_set_set_tuple_name(ct_range1.release(), ptuple()));

    adjusted_ptransform_ = adjusted_ptransform_.intersect_range(ct_range1);
    VLOG(4) << "adjusted_ptransform: " << adjusted_ptransform_;
  }

  {  // add params
    adjusted_ptransform_ =
        isl::manage(AddParamsTo(adjusted_ptransform_.release()));
  }
}

isl::set ComputeAtTransform::cdomain_with_params() {
  // add level+1 param to consumer transform
  isl::set cd_with_params =
      isl::manage(isl_set_add_dims(cdomain_.copy(), isl_dim_param, level_ + 1));
  return cd_with_params;
}

isl::map ComputeAtTransform::ctransform_with_params() {
  // add level+1 param to consumer transform
  int num_existing_param = isl_map_dim(ctransform_.get(), isl_dim_param);
  isl::map ct_with_params = isl::manage(AddParamsTo(ctransform_.copy()));
  {
    isl_local_space* local_space =
        isl_local_space_from_space(ct_with_params.space().release());
    for (int i = 0; i < level_ + 1; i++) {
      isl_constraint* cst =
          isl_constraint_alloc_equality(isl_local_space_copy(local_space));
      cst = isl_constraint_set_coefficient_val(
          cst,
          isl_dim_param,
          num_existing_param + i,
          isl_val_int_from_si(ctransform_.ctx().get(), -1));
      cst = isl_constraint_set_coefficient_val(
          cst, isl_dim_out, i, isl_val_int_from_si(ctransform_.ctx().get(), 1));
      ct_with_params =
          isl::manage(isl_map_add_constraint(ct_with_params.release(), cst));
    }
    isl_local_space_free(local_space);
  }
  return ct_with_params;
}

void ComputeAtTransform::DisplayC(isl_map* pschedule, isl_map* cschedule) {
  VLOG(3) << "adjusted cdomain: " << adjusted_cdomain_;
  VLOG(3) << "adjusted ctransform: " << adjusted_ctransform_;

  auto adjusted_ctransform = adjusted_ctransform_;
  auto adjusted_ptransform = adjusted_ptransform_;

  if (cschedule) {
    adjusted_ctransform = isl::manage(
        isl_map_apply_range(adjusted_ctransform.release(), cschedule));
  }
  if (pschedule) {
    adjusted_ptransform = isl::manage(
        isl_map_apply_range(adjusted_ptransform.release(), pschedule));
  }

  auto whole_domain =
      isl::manage(isl_union_set_from_set(adjusted_pdomain_.copy()));
  whole_domain = isl::manage(
      isl_union_set_add_set(whole_domain.release(), adjusted_cdomain_.copy()));
  VLOG(3) << "whole domain: " << whole_domain;

  auto whole_schedule =
      isl::manage(isl_union_map_from_map(adjusted_ptransform.copy()));
  whole_schedule = isl::manage(isl_union_map_add_map(
      whole_schedule.release(), adjusted_ctransform.copy()));
  VLOG(3) << "whole_schedule: " << whole_schedule;

  isl::set context(whole_domain.ctx(), "{:}");

  auto intersect_schedule = whole_schedule.intersect_domain(whole_domain);

  auto* build = isl_ast_build_from_context(context.release());
  auto* node =
      isl_ast_build_node_from_schedule_map(build, intersect_schedule.release());

  VLOG(3) << "code:\n\n" << isl_ast_node_to_C_str(node);

  isl_ast_node_free(node);
}

isl_set* ComputeAtTransform::AddParamsTo(isl_set* set) {
  int existing_params = isl_set_dim(set, isl_dim_param);
  set = isl_set_add_dims(set, isl_dim_param, level_ + 1);

  // set name
  for (int i = 0; i < level_ + 1; i++) {
    std::string pname = GenConsumerParamName(ctuple(), i);
    set = isl_set_set_dim_name(
        set, isl_dim_param, existing_params + i, pname.c_str());
  }
  return set;
}

isl_map* ComputeAtTransform::AddParamsTo(isl_map* map) {
  int existing_params = isl_map_dim(map, isl_dim_param);
  map = isl_map_add_dims(map, isl_dim_param, level_ + 1);

  // set name
  for (int i = 0; i < level_ + 1; i++) {
    std::string pname = GenConsumerParamName(ctuple(), i);
    map = isl_map_set_dim_name(
        map, isl_dim_param, existing_params + i, pname.c_str());
  }
  return map;
}

ComputeAtTransform::ComputeAtTransform(isl::set pdomain,
                                       isl::set cdomain,
                                       isl::map access,
                                       isl::map ptransform,
                                       isl::map ctransform,
                                       int level)
    : pdomain_(pdomain),
      cdomain_(cdomain),
      access_(access),
      ptransform_(ptransform),
      ctransform_(ctransform),
      level_(level) {
  VLOG(2) << "pdomain: " << pdomain;
  VLOG(2) << "ptransform: " << ptransform;
  VLOG(2) << "cdomain: " << cdomain;
  VLOG(2) << "ctransform: " << ctransform;
  VLOG(2) << "access: " << access;

  adjusted_ctransform_ = isl::manage(AddParamsTo(ctransform_.copy()));
  adjusted_cdomain_ = isl::manage(AddParamsTo(cdomain_.copy()));
}

std::string GenConsumerParamName(const char* tuple, int id) {
  return utils::StringFormat("%s%s_%d", kConsumerParamPrefix, tuple, id);
}

std::vector<int> ComputeAtTransform::GetProducerAdjustedShape() const {
  VLOG(3) << "domain: " << adjusted_pdomain();
  isl::set param_limit =
      isl::manage(isl_set_universe(adjusted_pdomain().space().release()));
  // set all the params to 0
  isl_local_space* local_space =
      isl_local_space_from_space(param_limit.space().release());
  for (int i = 0; i < isl_set_dim(param_limit.get(), isl_dim_param); i++) {
    isl_constraint* cst =
        isl_constraint_alloc_equality(isl_local_space_copy(local_space));
    cst = isl_constraint_set_coefficient_val(
        cst, isl_dim_param, i, isl_val_int_from_si(ctransform_.ctx().get(), 1));
    param_limit =
        isl::manage(isl_set_add_constraint(param_limit.release(), cst));
  }

  VLOG(3) << "param_limit: " << param_limit;
  isl::set domain = adjusted_pdomain().intersect(param_limit);

  std::vector<int> shape;
  // collect the min and max and get the num elements for each axis.
  for (int i = 0; i < isl_set_dim(domain.get(), isl_dim_set); i++) {
    auto _minv_maxv_ = isl_set_get_axis_range(domain.get(), i);
    auto& minv = std::get<0>(_minv_maxv_);
    auto& maxv = std::get<1>(_minv_maxv_);
    int num_elements = maxv.num_si() - minv.num_si() + 1;
    shape.push_back(num_elements);
  }
  return shape;
}

std::vector<int>
ComputeAtTransform::GetAccessesPrecedingIndicesMinAssumingParamsZero() {
  std::vector<int> res;

  isl::set cdomain_with_param = isl::manage(AddParamsTo(cdomain_.copy()));
  VLOG(4) << "cdomain_with_param: " << cdomain_with_param;
  isl::map access_with_param = isl::manage(AddParamsTo(access_.copy()));

  VLOG(4) << "applied: " << cdomain_with_param.apply(access_with_param);
  isl::set param_limited_cdomain = ctransform_with_params().domain();
  VLOG(4) << "ctransform.domain: " << param_limited_cdomain;
  isl::set access_domain = param_limited_cdomain.apply(access_with_param);

  // set all the params to 0
  isl_local_space* local_space =
      isl_local_space_from_space(access_domain.space().release());
  for (int i = 0; i < isl_set_dim(access_domain.get(), isl_dim_param); i++) {
    isl_constraint* cst =
        isl_constraint_alloc_equality(isl_local_space_copy(local_space));
    cst = isl_constraint_set_coefficient_val(
        cst, isl_dim_param, i, isl_val_int_from_si(ctransform_.ctx().get(), 1));
    access_domain =
        isl::manage(isl_set_add_constraint(access_domain.release(), cst));
  }
  isl_local_space_free(local_space);

  access_domain = access_domain.intersect(adjusted_pdomain());

  VLOG(3) << "access_with_param: " << access_domain;

  for (int i = 0; i < level_ + 1; i++) {
    auto _minv_maxv_ = isl_set_get_axis_range(access_domain.get(), i);
    auto& minv = std::get<0>(_minv_maxv_);
    auto& maxv = std::get<1>(_minv_maxv_);
    res.push_back(minv.get_num_si());
  }

  return res;
}

}  // namespace poly
}  // namespace cinn
