#!/bin/bash

check_failed_cnt() {
  max_failed=$1
  failed_count=$(python /root/k8s_tools.py count_pods_by_phase tf-job-trainer=${JOB_NAME} Failed)
  if [ $failed_count -gt $max_failed ]; then
    stdbuf -oL echo "Failed trainer count beyond the threadhold: "$max_failed
    echo "Failed trainer count beyond the threshold: " $max_failed > /dev/termination-log
    exit 0
  fi
}

check_trainer_ret() {
  ret=$1
  stdbuf -oL echo "job returned $ret...setting pod return message..."
  stdbuf -oL echo "==============================="

  if [ $ret -eq 136 ] ; then
    echo "Error Arithmetic Operation(Floating Point Exception)" > /dev/termination-log
  elif [ $ret -eq 139 ] ; then
    echo "Segmentation Fault" > /dev/termination-log
  elif [ $ret -eq 1 ] ; then
    echo "General Error" > /dev/termination-log
  elif [ $ret -eq 134 ] ; then
    echo "Program Abort" > /dev/termination-log
  fi
  stdbuf -oL echo "termination log wroted..."
  exit $ret
}

start_tf() {
  pserver_label="tf-job-pserver=${JOB_NAME}"
  trainer_label="tf-job-trainer=${JOB_NAME}"

  stdbuf -oL python /root/k8s_tools.py wait_pods_running ${pserver_label} ${PSERVERS}
  if [ "${TRAINING_ROLE}" == "TRAINER" ]; then
    check_failed_cnt ${TRAINERS}
    sleep 5
    stdbuf -oL python /root/k8s_tools.py wait_pods_running ${trainer_label} ${TRAINERS}
    export INIT_TRAINER_ID=$(python /root/k8s_tools.py fetch_trainer_id ${trainer_label})
  fi
  export INIT_PSERVERS=$(python /root/k8s_tools.py fetch_pserver_ips ${pserver_label} ${INIT_PORT})
  cmd="${ENTRY} --ps_hosts=$"
  stdbuf -oL sh -c "${ENTRY}"
  check_trainer_ret $?
}

usage() {
    echo "usage: tf_k8s [<args>]:"
    echo "  start_tf         Start tensorflow jobs"
}

case "$1" in
    start_tf)
        start_tf
        ;;
    --help)
        usage
        ;;
    *)
        usage
        ;;
esac
