#!/bin/bash


function rocm() {
  # # ROCM 3.3 - not work as rocthrust build fail without AMD GPU
  # sed 's#<rocm_repo_version>#3.3#g'  Dockerfile.rocm >test/rocm33.dockerfile
  # sed -ri 's#<rocprim_version>#3.3.0#g' test/rocm33.dockerfile
  # sed -ri 's#<rocthrust_version>#3.3.0#g' test/rocm33.dockerfile
  # sed -ri 's#<hipcub_version>#3.3.0#g' test/rocm33.dockerfile

  # ROCM 3.5
  sed 's#<rocm_repo_version>#3.5.1#g'  Dockerfile.rocm >test/rocm35.dockerfile
  sed -ri 's#<rocprim_version>#3.5.1#g' test/rocm35.dockerfile
  sed -ri 's#<rocthrust_version>#3.5.0#g' test/rocm35.dockerfile
  sed -ri 's#<hipcub_version>#3.5.0#g' test/rocm35.dockerfile

  # ROCM 3.9
  sed 's#<rocm_repo_version>#3.9.1#g'  Dockerfile.rocm >test/rocm39.dockerfile
  sed -ri 's#<rocprim_version>#3.9.0#g' test/rocm39.dockerfile
  sed -ri 's#<rocthrust_version>#3.9.0#g' test/rocm39.dockerfile
  sed -ri 's#<hipcub_version>#3.9.0#g' test/rocm39.dockerfile
}

function main() {
  if [ ! -d "test" ];then
    mkdir test
  fi
  rocm
}

main
