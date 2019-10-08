#!/bin/bash
if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
API_FILES=("CMakeLists.txt"
           "paddle/fluid/API.spec"
           "paddle/fluid/op_use_default_grad_op_maker.spec"
           "paddle/fluid/framework/operator.h"
           "paddle/fluid/framework/tensor.h"
           "paddle/fluid/framework/details/op_registry.h"
           "paddle/fluid/framework/grad_op_desc_maker.h"
           "paddle/fluid/framework/lod_tensor.h"
           "paddle/fluid/framework/selected_rows.h"
           "paddle/fluid/framework/op_desc.h"
           "paddle/fluid/framework/block_desc.h"
           "paddle/fluid/framework/var_desc.h"
           "paddle/fluid/framework/scope.h"
           "paddle/fluid/framework/ir/node.h"
           "paddle/fluid/framework/ir/graph.h"
           "paddle/fluid/framework/framework.proto"
           "python/requirements.txt"
           "python/paddle/fluid/__init__.py"
           "python/paddle/fluid/compiler.py"
           "python/paddle/fluid/parallel_executor.py"
           "python/paddle/fluid/framework.py"
           "python/paddle/fluid/backward.py"
           "paddle/fluid/operators/distributed/send_recv.proto.in")

approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=0
echo_list=()
if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
  APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 38231817`
  if [ "${APPROVALS}" == "FALSE" ]; then
    failed_num=`expr $failed_num + 1`
    echo_line="You must have Dianhai approval for change 20+ files or add than 1000+ lines of content\n"
    echo_list=(${echo_list[@]}$failed_num "." $echo_line)
  fi
fi    

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" | grep -v "/CMakeLists.txt" || true`
  echo "checking ${API_FILE} change, PR: ${GIT_PR_ID}, changes: ${API_CHANGE}"
  if [ "${API_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
      # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
      # approval_user_list: XiaoguangHu01 46782768,chengduoZH 30176695,Xreki 12538138,luotao1 6836917,sneaxiy 32832641,tensor-tang 21351065,xsrobin 50069408,qingqing01 7845005,guoshengCS 14105589,heavengate 12605721,kuke 3064195,Superjomn 328693,lanxianghit 47554610,cyj1986 39645414,hutuxian 11195205,frankwhzhang 20274488,nepeplwu 45024560,Dianhai 38231817,JiabinYang 22361972,chenwhql 22561442. 
      if [ "${API_FILE}" == "paddle/fluid/API.spec" ];then
        APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 2 46782768 7534971 14105589 12605721 3064195 328693 47554610 39645414 11195205 20274488 45024560 `
      elif [ "${API_FILE}" == "paddle/fluid/op_use_default_grad_op_maker.spec" ];then
        APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 32832641 6836917`
      elif [ "${API_FILE}" == "CMakeLists.txt" ];then
        APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 6836917 46782768 30176695`
      elif [ "${API_FILE}" == "python/paddle/fluid/__init__.py" ];then
         APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 6836917 47554610`
      elif [ "${API_FILE}" == "python/requirements.txt" ];then
         APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 6836917 22361972`
      else
        APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 21351065 3048612 46782768 30176695 12538138 6836917 32832641`
      fi
      echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
      if [ "${APPROVALS}" == "FALSE" ]; then
        if [ "${API_FILE}" == "paddle/fluid/API.spec" ];then
          failed_num=`expr $failed_num + 1`
          echo_line="You must have two RD (XiaoguangHu01 or wanghaoshuang or guoshengCS or heavengate or kuke or Superjomn or lanxianghit or cyj1986 or hutuxian or frankwhzhang or nepeplwu) approval for the api change! ${API_FILE} for the management reason of API interface and API document.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        elif [ "${API_FILE}" == "paddle/fluid/op_use_default_grad_op_maker.spec" ];then
          failed_num=`expr $failed_num + 1` 
          echo_line="You must have one RD (sneaxiy (Recommend) or luotao1) approval for op_use_default_grad_op_maker.spec, which manages the grad_op memory optimization.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        elif [ "${API_FILE}" == "CMakeLists.txt" ];then
          failed_num=`expr $failed_num + 1`
          echo_line="You must have one RD (luotao1 or chengduoZH or XiaoguangHu01) approval for CMakeLists.txt, which manages the compilation parameter.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        elif [ "${API_FILE}" == "python/requirements.txt" ];then
          failed_num=`expr $failed_num + 1`
          echo_line="You must have one RD (JiabinYang (Recommend) or luotao1) approval for python/requirements.txt, which manages the third-party python package.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        elif [ "${API_FILE}" == "python/paddle/fluid/__init__.py" ];then
          failed_num=`expr $failed_num + 1`
          echo_line="You must have one RD (lanxianghit (Recommend) or luotao1) approval for the python/paddle/fluid/init.py, which manages the environment variables.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        else
          failed_num=`expr $failed_num + 1`
          echo_line="You must have one RD (XiaoguangHu01,chengduoZH,Xreki,luotao1,sneaxiy,tensor-tang) approval for ${API_FILE}, which manages the underlying code for fluid.\n"
          echo_list=(${echo_list[@]}$failed_num "." $echo_line)
        fi
      fi
  fi
done

HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "const_cast" || true`
if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
    APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
    python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 21351065 3048612 46782768 30176695 12538138 6836917 32832641`
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        failed_num=`expr $failed_num + 1`
        echo_line="You must have one RD (XiaoguangHu01,chengduoZH,Xreki,luotao1,sneaxiy,tensor-tang) approval for the usage (either add or delete) of const_cast.\n"
        echo_list=(${echo_list[@]}$failed_num "." $echo_line)
    fi
fi

HAS_DEFINE_FLAG=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "DEFINE_int32" |grep -o -m 1 "DEFINE_bool" | grep -o -m 1 "DEFINE_string" || true`
if [ ${HAS_DEFINE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
    python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 47554610` 
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        failed_num=`expr $failed_num + 1`
        echo_line="You must have one RD lanxianghit approval for the usage (either add or delete) of DEFINE_int32/DEFINE_bool/DEFINE_string flag.\n"
        echo_list=(${echo_list[@]}$failed_num "." $echo_line)
    fi
fi

HAS_PADDLE_ENFORCE_FLAG=`git diff -U0 upstream/$BRANCH |grep "+" |grep -v "PADDLE_ENFORCE_" |grep -o -m 1 "PADDLE_ENFORCE" || true`
if [ ${HAS_PADDLE_ENFORCE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
    python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 6836917 47554610 22561442`
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        failed_num=`expr $failed_num + 1`
        echo_line="PADDLE_ENFORCE is not recommended. Please use PADDLE_ENFORCE_EQ/NE/GT/GE/LT/LE or PADDLE_ENFORCE_NOT_NULL or PADDLE_ENFORCE_CUDA_SUCCESS instead.\nYou must have one RD (chenwhql (Recommend) , luotao1 (Recommend) or lanxianghit) approval for the usage (either add or delete) of PADDLE_ENFORCE.\n"
        echo_list=(${echo_list[@]}$failed_num "." $echo_line)
    fi
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e ${echo_list[@]}
  git diff -U0 upstream/$BRANCH |grep "+" |grep -v "PADDLE_ENFORCE_" |grep "PADDLE_ENFORCE"
  echo "There are ${failed_num} approved errors."
  echo "****************"
  exit 1
fi
