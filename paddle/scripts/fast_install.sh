#!/bin/bash


function linux(){
  path='http://paddlepaddle.org/download?url='
  release_version='1.2.0'
  AVX=`cat /proc/cpuinfo |grep avx|tail -1|grep avx`
  GPU=`nvidia-smi|grep GPU`
  while true
    do
     echo "please enter the CUDA version.txt directory, if you are not sure about it please press enter to continue."
     read -p "" cuda_version
	 if [ "$cuda_version" == "" ];then	
		CUDA=`echo ${CUDA_VERSION}|awk -F "[ .]" '{print $1}'`
		if [ "$CUDA" == "" ];then
			version_file=`sudo find /usr/local -name version.txt`
			if [ "${version_file}" == "" ];then
				version_file=`sudo find /usr/local/cuda -name version.txt`
				if [ "${version_file}" == "" ];then
					echo "No CUDA FOUND"
					break
				fi
			fi
			availability=`cat $version_file | grep 'CUDA Version'`
			if [  "${availability}" == "" ];then
				echo "No CUDA FOUND"
				break	
			fi
			CUDA=`cat ${version_file}|awk -F "[ .]" '{print $3}'`
		fi
	  else
		CUDA=`cat ${cuda_version}|awk -F "[ .]" '{print $3}'`
	  fi
	  
	  if [ "$CUDA" == "" ];then
	  	echo "No CUDA FOUND"
	  else
	  	echo "CUDA $CUDA FOUND" 
	  	break
	  fi
	 done
   
  while true
  	do
  	 echo "please enter the cudnnn.h directory, if you are not sure about it please press enter to continue." 
  	 read -p "" cudnn_h
  	 if [ "$cudnn_h" == "" ];then
  	 	cudnn_h=`find /usr -name cudnn.h`
  	 	CUDNN=`cat ${cudnn_h} | grep CUDNN_MAJOR -A 2|awk 'NR==1{print $NF}'`
  	 else
  	 	CUDNN=`cat ${cudnn_h} | grep CUDNN_MAJOR -A 2|awk 'NR==1{print $NF}'`
  	 fi
  	 
  	 if [ "$cudnn_h" == "" ];then
  	 	echo "No cudnn_h FOUND"
  	 else
  	 	echo "cuDNN $CUDNN FOUND" 
  	 	break
  	 fi
    done
  
  while true
    do
      read -p "Please input which math lib would you like to use? openblas or mkl?：
          openblas
          mkl
          Please select：" math
        if [ "$math" == "openblas" ]||[ $math == "mkl" ];then
          break
        fi
        echo "wrong input please input again"
    done 
 

  while true
    do
      read -p "Please select the Paddle Version：
          develop
          release
          Please Select：" paddle_version
        if [ "$paddle_version" == "develop" ]||[ $paddle_version == "release" ];then
          break
        fi
        echo "wrong input please input again"
    done

   echo "Please input the directory of the pip you would like to use："
   read -p "" pip_path
   
   python_version=`$pip_path --version|awk -F "[ |)]" '{print $6}'|sed 's#\.##g'`
   if [[ "$python_version" == "27" ]];then
     uncode=`python -c "import pip._internal;print(pip._internal.pep425tags.get_supported())"|grep "cp27"` 
     if [[ "$uncode" == "" ]];then
        uncode=mu
     else 
        uncode=m
     fi
   fi

   if [[ "$python_version" == "" ]];then
     echo "Can't find available pip" 
     exit
   fi


  if [[ "$AVX" != "" ]];then
    AVX=avx
  else
    AVX=noavx
  fi


  if [[ "$GPU" != "" ]];then
    GPU=gpu
  else
    GPU=cpu
  fi


  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-${noavx}-${math}/paddlepaddle-1.2.0-cp${python_version}-cp${python_version}${uncode}-linux_x86_64.whl"
  wheel_gpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0.post${CUDA}${CUDNN}-cp${python_version}-cp${python_version}${uncode}-linux_x86_64.whl"
  wheel_gpu_release_noavx="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0-cp${python_version}-cp${python_version}${uncode}-linux_x86_64.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-${AVX}-${math}/paddlepaddle-latest-cp${python_version}-cp${python_version}${uncode}-linux_x86_64.whl"
  wheel_gpu_develop="http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-latest-cp${python_version}-cp${python_version}${uncode}-linux_x86_64.whl"


  if [[ "$paddle_version" == "release" ]];then
    if [[ "$GPU" == "gpu" ]];then
        if [[ ${AVX} == "avx" ]];then
          $pip_path install ${path}$wheel_gpu_release
        else
          $pip_path install ${path}$wheel_gpu_release_noavx 
        fi
    else
        $pip_path install ${path}$wheel_cpu_release
    fi
  else 
    if [[ "$GPU" == "gpu" ]];then
        $pip_path install ${path}$wheel_gpu_develop
    else
        $pip_path install ${path}$wheel_cpu_develop
    fi
  fi
}


function macos() {
  path='http://paddlepaddle.org/download?url='
  release_version="1.2.0"
  AVX=`sysctl -a | grep cpu | grep AVX1.0 | tail -1 | grep AVX`

  while true
    do
      read -p "Please select the Paddle Version：
          develop
          release
          Please Select：" paddle_version
        if [ $paddle_version == "develop" ]||[ $paddle_version == "release" ];then
          break
        fi
        echo "wrong input please input again"
    done
	
   echo "Please input the directory of the pip you would like to use："
   read -p "" pip_path
   
   python_version=`$pip_path --version|awk -F "[ |)]" '{print $6}'|sed 's#\.##g'`
   if [[ $python_version == "27" ]];then
     uncode=`python -c "import pip._internal;print(pip._internal.pep425tags.get_supported())"|grep "cp27"` 
     if [[ $uncode == "" ]];then
        uncode=mu
     else 
        uncode=m
     fi
   fi

   if [[ $python_version == "" ]];then
     echo "Can't find available pip" 
     exit
   fi


  if [[ $AVX != "" ]];then
    AVX=avx
  else
    AVX=noavx
  fi


  if [[ $GPU != "" ]];then
    GPU=gpu
  else
    GPU=cpu
  fi


  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-mac/paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  whl_cpu_release="paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-mac/paddlepaddle-latest-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  whl_cpu_develop="paddlepaddle-latest-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"

  if [[ $paddle_version == "release" ]];then
        wget ${path}$wheel_cpu_release -O $whl_cpu_release
        $pip_path install $whl_cpu_release
  else 
        wget ${path}$wheel_cpu_develop -O $whl_cpu_develop
        $pip_path install $whl_cpu_develop
  fi
}

function main() {
  SYSTEM=`uname -s`
  if [ "$SYSTEM" == "Darwin" ];then
  	echo "Your are using MACOS"
  	macos
  else
 	echo "Your are using Linux"
	  OS=`cat /etc/issue|awk 'NR==1 {print $1}'`
	  if [ $OS == "CentOS" ]||[ $OS == "Ubuntu" ];then
	    linux
	  else 
	    echo 系统不支持
	  fi
  fi
}
main
