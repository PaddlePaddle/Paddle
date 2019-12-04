#!/bin/bash
if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
API_FILES=("CMakeLists.txt"
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
           "paddle/fluid/operators/distributed/send_recv.proto.in"
           "paddle/fluid/framework/unused_var_check.cc")

approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=0
echo_list=()


function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py $1 $person_num`
    if [ "${APPROVALS}" == "FALSE" ]; then
        add_failed $failed_num $echo_line
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    add_line=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    echo_list=(${echo_list[@]}$1 "." $add_line)
} 


if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
    echo_line="You must have Dianhai approval for change 20+ files or add than 1000+ lines of content.\n"
    check_approval 1 38231817
fi    

api_spec_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.api  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.api` 
if [ "$api_spec_diff" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01 or lanxianghit or saxon-zh)approval for the api change for the management reason of API interface.\n"
    check_approval 1 46782768 47554610 2870059 
fi

api_doc_spec_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.doc  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.doc` 
if [ "$api_doc_spec_diff" != "" ]; then
    echo_line="You must have one TPM (saxon-zh or Boyan-Liu) approval for the api change for the management reason of API document.\n"
    check_approval 1 31623103 2870059
fi


for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" | grep -v "/CMakeLists.txt" || true`
  echo "checking ${API_FILE} change, PR: ${GIT_PR_ID}, changes: ${API_CHANGE}"
  if [ "${API_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
      # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
      # approval_user_list: XiaoguangHu01 46782768,Xreki 12538138,luotao1 6836917,sneaxiy 32832641,qingqing01 7845005,guoshengCS 14105589,heavengate 12605721,kuke 3064195,Superjomn 328693,lanxianghit 47554610,cyj1986 39645414,hutuxian 11195205,frankwhzhang 20274488,nepeplwu 45024560,Dianhai 38231817,JiabinYang 22361972,chenwhql 22561442,zhiqiu 6888866,seiriosPlus 5442383,gongweibao 10721757,saxon-zh 2870059,Boyan-Liu 2870059. 
      if [ "${API_FILE}" == "paddle/fluid/op_use_default_grad_op_maker.spec" ];then
          echo_line="You must have one RD (sneaxiy (Recommend) or luotao1) approval for op_use_default_grad_op_maker.spec, which manages the grad_op memory optimization.\n"
          check_approval 1 32832641 6836917
      elif [ "${API_FILE}" == "CMakeLists.txt" ];then
          echo_line="You must have one RD (luotao1 or XiaoguangHu01) approval for CMakeLists.txt, which manages the compilation parameter.\n"
          check_approval 1 6836917 46782768
      elif [ "${API_FILE}" == "python/paddle/fluid/__init__.py" ];then
          echo_line="You must have one RD (lanxianghit (Recommend) or luotao1) approval for the python/paddle/fluid/init.py, which manages the environment variables.\n"
          check_approval 1 6836917 47554610
      elif [ "${API_FILE}" == "python/requirements.txt" ];then
          echo_line="You must have one RD (JiabinYang (Recommend) or luotao1) approval for python/requirements.txt, which manages the third-party python package.\n"
          check_approval 1 6836917 22361972
      elif [ "${API_FILE}" == "paddle/fluid/operators/distributed/send_recv.proto.in" ];then
          echo_line="You must have one RD (gongweibao or seiriosPlus) approval for the paddle/fluid/operators/distributed/send_recv.proto.in, which manages the environment variables.\n"
          check_approval 1 10721757 5442383
      elif [ "${API_FILE}" == "paddle/fluid/framework/unused_var_check.cc" ];then
          echo_line="You must have one RD (zhiqiu (Recommend) , sneaxiy or luotao1) approval for the paddle/fluid/framework/unused_var_check.cc, which manages the white list of operators that have unused input variables. Before change the white list, please read the spicification [https://github.com/PaddlePaddle/Paddle/wiki/OP-Should-Not-Have-Unused-Input] and try to refine code first. \n"
          check_approval 1 6888866 32832641 6836917
      else
          echo_line="You must have one RD (XiaoguangHu01,Xreki,luotao1,sneaxiy) approval for ${API_FILE}, which manages the underlying code for fluid.\n"
          check_approval 1 3048612 46782768 12538138 6836917 32832641
      fi
  fi
done

HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "const_cast" || true`
if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01,Xreki,luotao1,sneaxiy) approval for the usage (either add or delete) of const_cast.\n"
    check_approval 1 3048612 46782768 12538138 6836917 32832641
fi

HAS_DEFINE_FLAG=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "DEFINE_int32" |grep -o -m 1 "DEFINE_bool" | grep -o -m 1 "DEFINE_string" || true`
if [ ${HAS_DEFINE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD lanxianghit approval for the usage (either add or delete) of DEFINE_int32/DEFINE_bool/DEFINE_string flag.\n"
    check_approval 1 47554610 
fi

HAS_UNITTEST_SKIP=`git diff -U0 upstream/$BRANCH | grep "^+[[:space:]]\{0,\}@unittest.skip" || true`
if [ "${HAS_UNITTEST_SKIP}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Unittest is not allowed to be disabled.\nYou must have one RD (XiaoguangHu01, phlrain, luotao1 or lanxianghit) approval for the usage of @unittest.skip or @unittest.skipIf.\n${HAS_UNITTEST_SKIP}\n"
    check_approval 1 46782768 6836917 47554610 43953930
fi

ALL_PADDLE_ENFORCE=`git diff -U0 upstream/$BRANCH |grep "+" |grep -zoE "PADDLE_ENFORCE\(.[^,\);]+.[^;]*\);\s" || true`
if [ "${ALL_PADDLE_ENFORCE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="PADDLE_ENFORCE is not recommended. Please use PADDLE_ENFORCE_EQ/NE/GT/GE/LT/LE or PADDLE_ENFORCE_NOT_NULL or PADDLE_ENFORCE_CUDA_SUCCESS instead, see [ https://github.com/PaddlePaddle/Paddle/wiki/PADDLE_ENFORCE-Rewriting-Specification ] for details.\nYou must have one RD (chenwhql (Recommend) , luotao1 (Recommend) or lanxianghit) approval for the usage (either add or delete) of PADDLE_ENFORCE.\n${ALL_PADDLE_ENFORCE}\n"
    check_approval 1 6836917 47554610 22561442
fi

ALL_PADDLE_CHECK=`git diff -U0 upstream/$BRANCH |grep "+" |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true`
VALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' || true`
INVALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true`
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The error message you wrote in PADDLE_ENFORCE{_**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please specify chenwhql (Recommend), luotao1 or lanxianghit review and approve.\nThe PADDDLE_ENFORCE or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
    check_approval 1 6836917 47554610 22561442
fi

NEW_OP_ADDED=`git diff --name-only --diff-filter=A upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${NEW_OP_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    GET_KERNEL_TYPE_FUNC_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -czoE "GetExpectedKernelType[(][^(){}]+[)][^{]+[{][^}]+[}]" || true`
    INDICATE_VAR_DTYPE_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -co "IndicateVarDataType" || true`
    if [ ${GET_KERNEL_TYPE_FUNC_CNT} -gt ${INDICATE_VAR_DTYPE_CNT} ]; then
        echo_line="If you override GetExpectedKernelType method of OperatorWithKernel, please use OperatorWithKernel::IndicateVarDataType() method to get specific input variable's dtype, which checked whether the input variable is initialized (The details in https://github.com/PaddlePaddle/FluidDoc/pull/1527). If you don't use this method to check, you must have one RD (chenwhql (Recommend) , luotao1 or lanxianghit) approval for the usage of other methods.\n"
        check_approval 1 6836917 47554610 22561442
    fi
fi

OP_FILE_CHANGE=`git diff --name-only --diff-filter=AM upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${OP_FILE_CHANGE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    CHECK_SHAREDATAWITH_CNT=`git diff -U0 --diff-filter=AM upstrem/$BRANCH |grep "+" |grep -coE "ShareDataWith[(]" || true`
    CHECK_SHAREBUFFERWITH_CNT=`git diff -U0 --diff-filter=AM upstream/$BRANCH |grep "+" |grep -co "ShareBufferWith[(]" || true`
    if [ ${CHECK_SHAREDATAWITH_CNT} != ""] || [ ${CHECK_SHAREBUFFERWITH_CNT} != ""]; then
        echo_line="If you use the ShareDataWith or ShareBufferWith method, please change the method to others. Because this method cloud casus some bug of memory. If you don't want to change the method, you must have one RD (lanxianghit or luotao1 or sneaxiy) approval for the usage of other method.\n"
        check_approval 1 6836917 32832641 47554610
    fi
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "****************"
fi

python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec

if [ -n "${echo_list}" ]; then
  exit 1
fi
