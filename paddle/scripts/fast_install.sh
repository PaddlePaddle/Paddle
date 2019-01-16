#!/bin/bash

gpu_list=("GeForce 410M"
"GeForce 610M"
"GeForce 705M"
"GeForce 710M"
"GeForce 800M"
"GeForce 820M"
"GeForce 830M"
"GeForce 840M"
"GeForce 910M"
"GeForce 920M"
"GeForce 930M"
"GeForce 940M"
"GeForce GT 415M"
"GeForce GT 420M"
"GeForce GT 430"
"GeForce GT 435M"
"GeForce GT 440"
"GeForce GT 445M"
"GeForce GT 520"
"GeForce GT 520M"
"GeForce GT 520MX"
"GeForce GT 525M"
"GeForce GT 540M"
"GeForce GT 550M"
"GeForce GT 555M"
"GeForce GT 610"
"GeForce GT 620"
"GeForce GT 620M"
"GeForce GT 625M"
"GeForce GT 630"
"GeForce GT 630M"
"GeForce GT 635M"
"GeForce GT 640"
"GeForce GT 640 (GDDR5)"
"GeForce GT 640M"
"GeForce GT 640M LE"
"GeForce GT 645M"
"GeForce GT 650M"
"GeForce GT 705"
"GeForce GT 720"
"GeForce GT 720M"
"GeForce GT 730"
"GeForce GT 730M"
"GeForce GT 735M"
"GeForce GT 740"
"GeForce GT 740M"
"GeForce GT 745M"
"GeForce GT 750M"
"GeForce GTS 450"
"GeForce GTX 1050"
"GeForce GTX 1060"
"GeForce GTX 1070"
"GeForce GTX 1080"
"GeForce GTX 1080 Ti"
"GeForce GTX 460"
"GeForce GTX 460M"
"GeForce GTX 465"
"GeForce GTX 470"
"GeForce GTX 470M"
"GeForce GTX 480"
"GeForce GTX 480M"
"GeForce GTX 485M"
"GeForce GTX 550 Ti"
"GeForce GTX 560M"
"GeForce GTX 560 Ti"
"GeForce GTX 570"
"GeForce GTX 570M"
"GeForce GTX 580"
"GeForce GTX 580M"
"GeForce GTX 590"
"GeForce GTX 650"
"GeForce GTX 650 Ti"
"GeForce GTX 650 Ti BOOST"
"GeForce GTX 660"
"GeForce GTX 660M"
"GeForce GTX 660 Ti"
"GeForce GTX 670"
"GeForce GTX 670M"
"GeForce GTX 670MX"
"GeForce GTX 675M"
"GeForce GTX 675MX"
"GeForce GTX 680"
"GeForce GTX 680M"
"GeForce GTX 680MX"
"GeForce GTX 690"
"GeForce GTX 750"
"GeForce GTX 750 Ti"
"GeForce GTX 760"
"GeForce GTX 760M"
"GeForce GTX 765M"
"GeForce GTX 770"
"GeForce GTX 770M"
"GeForce GTX 780"
"GeForce GTX 780M"
"GeForce GTX 780 Ti"
"GeForce GTX 850M"
"GeForce GTX 860M"
"GeForce GTX 870M"
"GeForce GTX 880M"
"GeForce GTX 950"
"GeForce GTX 950M"
"GeForce GTX 960"
"GeForce GTX 960M"
"GeForce GTX 965M"
"GeForce GTX 970"
"GeForce GTX 970M"
"GeForce GTX 980"
"GeForce GTX 980M"
"GeForce GTX 980 Ti"
"GeForce GTX TITAN"
"GeForce GTX TITAN Black"
"GeForce GTX TITAN X"
"GeForce GTX TITAN Z"
"Jetson TK1"
"Jetson TX1"
"Jetson TX2"
"Mobile Products"
"NVIDIA NVS 310"
"NVIDIA NVS 315"
"NVIDIA NVS 510"
"NVIDIA NVS 810"
"NVIDIA TITAN V"
"NVIDIA TITAN X"
"NVIDIA TITAN Xp"
"NVS 4200M"
"NVS 5200M"
"NVS 5400M"
"Quadro 410"
"Quadro GP100"
"Quadro K1100M"
"Quadro K1200"
"Quadro K2000"
"Quadro K2000D"
"Quadro K2100M"
"Quadro K2200"
"Quadro K2200M"
"Quadro K3100M"
"Quadro K4000"
"Quadro K4100M"
"Quadro K420"
"Quadro K4200"
"Quadro K4200M"
"Quadro K5000"
"Quadro K500M"
"Quadro K5100M"
"Quadro K510M"
"Quadro K5200"
"Quadro K5200M"
"Quadro K600"
"Quadro K6000"
"Quadro K6000M"
"Quadro K610M"
"Quadro K620"
"Quadro K620M"
"Quadro M1000M"
"Quadro M1200"
"Quadro M2000"
"Quadro M2000M"
"Quadro M2200"
"Quadro M3000M"
"Quadro M4000"
"Quadro M4000M"
"Quadro M5000"
"Quadro M5000M"
"Quadro M500M"
"Quadro M520"
"Quadro M5500M"
"Quadro M6000"
"Quadro M6000 24GB"
"Quadro M600M"
"Quadro M620"
"Quadro Mobile Products"
"Quadro P1000"
"Quadro P2000"
"Quadro P3000"
"Quadro P400"
"Quadro P4000"
"Quadro P5000"
"Quadro P600"
"Quadro P6000"
"Quadro Plex 7000"
"Tegra K1"
"Tegra X1"
"Tesla C2050/C2070"
"Tesla C2075"
"Tesla Data Center Products"
"Tesla K10"
"Tesla K20"
"Tesla K40"
"Tesla K80"
"Tesla M40"
"Tesla M60"
"Tesla P100"
"Tesla P4"
"Tesla P40"
"Tesla V100")

function linux(){
  path='http://paddlepaddle.org/download?url='
  release_version='1.2.0'
  AVX=`cat /proc/cpuinfo |grep avx|tail -1|grep avx`
  which_gpu=`lspci |grep -i nvidia`
  if [ "$which_gpu" == "" ];then
    GPU='cpu'
    echo "您使用的是CPU机器"
  else
    GPU='gpu'
    echo "您使用的是GPU机器"
  fi
  if [ "$GPU" == 'gpu' ];then
    while true
      do
       gpu_model=`nvidia-smi |awk 'NR==8{print $3,$4}'|sed 's#m$##g'`
       Flag=False
       for i in "${gpu_list[@]}"
         do
           if [ "$gpu_model" == "$i" ];then
             Flag=True
           fi
       done 
  
       if [ "$Flag" != "True" ];then
         echo "GPU型号不符合"
         exit
       fi

       CUDA=`echo ${CUDA_VERSION}|awk -F "[ .]" '{print $1}'`
       
       if [ "$CUDA" == "" ];then
         if [ -f "/usr/local/cuda/version.txt" ];then
           CUDA=`cat /usr/local/cuda/version.txt | grep 'CUDA Version'|awk -F '[ .]' '{print $3}'`
           tmp_cuda=$CUDA
         fi
         if [ -f "/usr/local/cuda8/version.txt" ];then
           CUDA=`cat /usr/local/cuda8/version.txt | grep 'CUDA Version'|awk -F '[ .]' '{print $3}'`
           tmp_cuda8=$CUDA
         fi
         if [ -f "/usr/local/cuda9/version.txt" ];then
           CUDA=`cat /usr/local/cuda9/version.txt | grep 'CUDA Version'|awk -F '[ .]' '{print $3}'`
           tmp_cuda9=$CUDA
         fi
       fi

       if [ "$tmp_cuda" != "" ];then
         echo "找到$tmp_cuda"
       fi
       if [ "$tmp_cudai8" != "" ];then
         echo "找到$tmp_cuda8"
       fi
       if [ "$tmp_cuda9" != "" ];then
         echo "找到$tmp_cuda9"
       fi

       if [ "$CUDA" == "" ];then
           echo "没有找到cuda/version.txt文件"
           read -p "请提供cuda version.txt的路径:" cuda_version
           if [ "$cuda_version" == "" ];then
             exit
           fi
         CUDA=`cat $cuda_version | grep 'CUDA Version'|awk -F '[ .]' '{print $3}'`
         if [ "$CUDA" == "" ];then
           echo "没有找到CUDA"
           exit
         fi
       fi

       if [ "$CUDA" == "8" ] || [ "$CUDA" == "9" ];then
         echo "为您安装CUDA$CUDA"
         break
       else
         echo "你的CUDA${CUDA}版本不支持,目前支持CUDA8/9"
         exit
       fi
    done
     
    while true
    	do
         version_file='/usr/local/cuda/include/cudnn.h'
         if [ -f "$version_file" ];then
            CUDNN=`cat $version_file | grep CUDNN_MAJOR |awk 'NR==1{print $NF}'`
         fi
         if [ "$CUDNN" == "" ];then 
             version_file=`sudo find /usr -name "cudnn.h"|head -1`
             if [ "$version_file" != "" ];then
               CUDNN=`cat ${cudnn_h} | grep CUDNN_MAJOR -A 2|awk 'NR==1{print $NF}'`
             else
                echo "没有找到cuda/include/cudnn.h文件"
                read -p "请提供cudnn.h的路径:" cudnn_version
                if [ "$cudnn_version" == "" ];then
                   exit
                else
                  CUDNN=`cat $cudnn_version | grep CUDNN_MAJOR |awk 'NR==1{print $NF}'`
                fi
             fi
         fi
         if [ "$CUDA" == "9" -a "$CUDNN" != "7" ];then
           echo CUDA9目前只支持CUDNN7
           exit
         fi
         if [ "$CUDNN" == 5 ] || [ "$CUDNN" == 7 ];then
           echo "您的CUDNN版本是CUDNN$CUDNN"
           break
         else 
           echo "你的CUDNN${CUDNN}版本不支持,目前支持CUDNN5/7"
           exit
         fi
      done
  fi

  while true
    do
      if [ "$GPU" == "gpu" ];then
        math='mkl'
        break
      else
        read -p "Please input which math lib would you like to use? openblas or mkl?：
            openblas
            mkl
            Please select：" math
          if [ "$math" == "openblas" ]||[ $math == "mkl" ];then
            break
          fi
          echo "wrong input please input again"
      fi
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
     uncode=`python -c "import pip._internal;print(pip._internal.pep425tags.get_supported())"|grep "cp27mu"` 
     if [[ "$uncode" == "" ]];then
        uncode=
     else
        uncode=u
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


  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-${AVX}-${math}/paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0.post${CUDA}${CUDNN}-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release_noavx="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-${AVX}-${math}/paddlepaddle-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_develop="http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"


  if [[ "$paddle_version" == "release" ]];then
    if [[ "$GPU" == "gpu" ]];then
        if [[ ${AVX} == "avx" ]];then
          rm -rf $wheel_cpu_release
          wget $wheel_cpu_develop
          $pip_path install $wheel_gpu_release
          rm -rf $wheel_cpu_release
        else
          rm -rf $wheel_cpu_develop_noavx
          wget $wheel_cpu_release_novax
          $pip_path install $wheel_gpu_release_noavx 
          rm -rf $wheel_cpu_release_noavx
        fi
    else
        rm -rf $wheel_cpu_release
        wget $wheel_cpu_develop
        $pip_path install $wheel_cpu_release
        rm -rf $wheel_cpu_release
    fi
  else 
    if [[ "$GPU" == "gpu" ]];then
        rm -rf $wheel_cpu_develop
        wget $wheel_cpu_develop
        $pip_path install $wheel_gpu_develop
        rm -rf $wheel_cpu_develop
    else
        rm -rf $wheel_cpu_develop
        wget $wheel_cpu_develop
        $pip_path install $wheel_cpu_develop
        rm -rf $wheel_cpu_develop
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
        rm -rf $wheel_cpu_develop
        wget ${path}$wheel_cpu_release -O $whl_cpu_release
        $pip_path install $whl_cpu_release
        rm -rf $wheel_cpu_release
  else 
        rm -rf $wheel_cpu_develop
        wget ${path}$wheel_cpu_develop -O $whl_cpu_develop
        $pip_path install $whl_cpu_develop
        rm -rf $wheel_cpu_develop
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
	  if [ $OS == "\S" ] || [ "$OS" == "CentOS" ] || [ $OS == "Ubuntu" ];then
	    linux
	  else 
	    echo 系统不支持
	  fi
  fi
}
main
