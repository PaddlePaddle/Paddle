#!/bin/bash

path='http://paddlepaddle.org/download?url='
release_version='1.2.0'

function use_cpu(){
   while true
    do
     echo "是否安装CPU版本的PaddlePaddle？(yes/no)， 或使用ctrl + c退出"
     typeset -l cpu_option
     read -p "" cpu_option
     # TODO user wrong input process
     if [ "$cpu_option" == "" || "$cpu_option" == "no" ];then
        echo "退出安装中...."
        exit
     else
        GPU='cpu'
        break
     fi
    done
}


function linux(){
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

  AVX=`cat /proc/cpuinfo |grep avx|tail -1|grep avx`
  which_gpu=`lspci |grep -i nvidia`
  if [ "$which_gpu" == "" ];then
    GPU='cpu'
    echo "您使用的是不包含支持的GPU的机器"
  else
    GPU='gpu'
    echo "您使用的是包含我们支持的GPU机器"
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
             echo "目前我们还不支持您使用的GPU型号"
             use_cpu
             if [ "$GPU" == "cpu" ];then
                break
             fi
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
                while true
                do
                    read -p "请提供cuda version.txt的路径:" cuda_version
                    if [ "$cuda_version" == "" || ! -f "$cuda_version" ];then
                        read -p "未找到CUDA,只能安装cpu版本的PaddlePaddle，是否安装（yes/no）,  或使用ctrl + c退出" cpu_option
                        if [ "$cpu_option" == "yes" ];then
                            GPU='cpu'
                            break
                        else
                            echo "重新输入..."
                        fi
                    else
                        CUDA=`cat $cuda_version | grep 'CUDA Version'|awk -F '[ .]' '{print $3}'`
                        if [ "$CUDA" == "" ];then
                            echo "未找到CUDA，重新输入..."
                        else
                            break
                        fi
                    fi
                done
                if [ "$GPU" == "cpu" ];then
                    break
                fi
           fi

           if [ "$CUDA" == "8" ] || [ "$CUDA" == "9" ];then
              echo "您的CUDA版本是${CUDA}"
           else
              echo "你的CUDA${CUDA}版本不支持,目前支持CUDA8/9"
              use_cpu
           fi

           if [ "$GPU" == "cpu" ];then
              break
           fi

           version_file='/usr/local/cuda/include/cudnn.h'
           if [ -f "$version_file" ];then
              CUDNN=`cat $version_file | grep CUDNN_MAJOR |awk 'NR==1{print $NF}'`
           fi
           if [ "$CUDNN" == "" ];then
               version_file=`sudo find /usr -name "cudnn.h"|head -1`
               if [ "$version_file" != "" ];then
                  CUDNN=`cat ${version_file} | grep CUDNN_MAJOR -A 2|awk 'NR==1{print $NF}'`
               else
                  echo "未找到cuda/include/cudnn.h文件"
                  while true
                    do
                      read -p "请提供cudnn.h的路径:" cudnn_version
                      if [ "$cudnn_version" == "" ] || [ ! -f "$cudnn_version" ];then
                            read -p "未找到cuDNN,只能安装cpu版本的PaddlePaddle，是否安装（yes/no）, 或使用ctrl + c退出:" cpu_option
                            if [ "$cpu_option" == "yes" ];then
                                GPU='cpu'
                                break
                            else
                                echo "重新输入..."
                            fi
                      else
                         CUDNN=`cat $cudnn_version | grep CUDNN_MAJOR |awk 'NR==1{print $NF}'`
                         echo "您的CUDNN版本是${CUDNN}"
                         break
                      fi
                     done
                 if [ "$GPU" == "cpu" ];then
                    break
                 fi
               fi
           fi
           if [ "$CUDA" == "9" -a "$CUDNN" != "7" ];then
              echo CUDA9目前只支持CUDNN7
              use_cpu()
              if [ "$GPU"=="cpu" ];then
                 break
              fi
           fi
           if [ "$CUDNN" == 5 ] || [ "$CUDNN" == 7 ];then
              echo "您的CUDNN版本是CUDNN$CUDNN"
              break
           else
              echo "你的CUDNN${CUDNN}版本不支持,目前支持CUDNN5/7"
              use_cpu
              if [ "$GPU"=="cpu" ];then
                 break
              fi
           fi
       done
  fi

  while true
    do
      if [ "$AVX" ==  "" ];then
        math='mkl'
        break
      elif [ "$GPU" == "gpu" ];then
        math='mkl'
        break
      else
        read -p "请输入您想使用哪个数学库？OpenBlas或MKL？：
            openblas
            mkl
            请选择：" math
          if [ "$math" == "openblas" ]||[ $math == "mkl" ];then
            break
          fi
          echo "输入错误，请再次输入"
      fi
    done 
 

  while true
    do
      read -p "请选择Paddle版本：
          develop
          release
          请选择：" paddle_version
        if [ "$paddle_version" == "develop" ]||[ $paddle_version == "release" ];then
          break
        fi
        echo "输入错误，请再次输入"
    done
  while true
    do
       echo "请输入您要使用的pip目录（您可以使用which pip来查看）："
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
       echo $python_version
       if [ "$python_version" == "27" -o "$python_version" == "35" -o "$python_version" == "36" -o "$python_version" == "37" ];then
         echo "找到python${python_version}版本"
         break
       else
         echo "找不到可用的 pip, 我们只支持Python27/35/36/37及其对应的pip, 请重新输入， 或使用ctrl + c退出 "
       fi
    done

  if [[ "$AVX" != "" ]];then
    AVX=avx
  else
    if [ "$CUDA" == "8" -a "$CUDNN" == "7" ] || [ "$GPU" == "cpu" ];then
        AVX=noavx
    else
        echo "我们仅支持纯CPU或GPU with CUDA 8 cuDNN 7 下noavx版本的安装，请使用cat /proc/cpuinfo | grep avx检查您计算机的avx指令集支持情况"
        exit
    fi
  fi


  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-${AVX}-${math}/paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0.post${CUDA}${CUDNN}-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release_noavx="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-1.2.0-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-${AVX}-${math}/paddlepaddle-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_develop="http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"


  if [[ "$paddle_version" == "release" ]];then
    if [[ "$GPU" == "gpu" ]];then
        if [[ ${AVX} == "avx" ]];then
          rm -rf $wheel_gpu_release
          wget $wheel_cpu_develop
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release
          rm -rf $wheel_gpu_release
        else
          rm -rf $wheel_gpu_develop_noavx
          wget $wheel_cpu_release_novax
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release_noavx
          rm -rf $wheel_gpu_release_noavx
        fi
    else
        rm -rf $wheel_cpu_release
        wget $wheel_cpu_develop
        $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_release
        rm -rf $wheel_cpu_release
    fi
  else 
    if [[ "$GPU" == "gpu" ]];then
        rm -rf $wheel_gpu_develop
        wget $wheel_gpu_develop
        $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_develop
        rm -rf $wheel_gpu_develop
    else
        rm -rf $wheel_cpu_develop
        wget $wheel_cpu_develop
        $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_develop
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
      read -p "请选择Paddle版本：
          develop
          release
          Please Select：" paddle_version
        if [ $paddle_version == "develop" ]||[ $paddle_version == "release" ];then
          break
        fi
        echo "输入错误，请再次输入"
    done
	
   echo "请输入您要使用的pip目录（您可以使用which pip来查看）："
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
     echo "找不到可用的pip"
     exit
   fi


  if [[ $AVX != "" ]];then
    AVX=avx
  else
    echo "您的Mac不支持AVX指令集，目前不能安装PaddlePaddle"
  fi


  if [[ $GPU != "" ]];then
    echo "MacOS上暂不支持GPU版本的PaddlePaddle"
  else
    echo "MacOS上暂不支持GPU版本的PaddlePaddle"
    GPU=cpu
  fi


  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-mac/paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  whl_cpu_release="paddlepaddle-1.2.0-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-mac/paddlepaddle-latest-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"
  whl_cpu_develop="paddlepaddle-latest-cp${python_version}-cp${python_version}m-macosx_10_6_intel.whl"

  if [[ $paddle_version == "release" ]];then
        rm -rf $wheel_cpu_develop
        wget ${path}$wheel_cpu_release -O $whl_cpu_release
        $pip_path --user install $whl_cpu_release
        rm -rf $wheel_cpu_release
  else 
        rm -rf $wheel_cpu_develop
        wget ${path}$wheel_cpu_develop -O $whl_cpu_develop
        $pip_path --user install $whl_cpu_develop
        rm -rf $wheel_cpu_develop
  fi
}

function main() {
  echo "一键安装脚本将会基于您的系统和硬件情况为您安装适合的PaddlePaddle"
  SYSTEM=`uname -s`
  if [ "$SYSTEM" == "Darwin" ];then
  	echo "您正在使用MAC OSX"
  	macos
  else
 	echo "您正在使用Linux"
	  OS=`cat /etc/issue|awk 'NR==1 {print $1}'`
	  if [ $OS == "\S" ] || [ "$OS" == "CentOS" ] || [ $OS == "Ubuntu" ];then
	    linux
	  else 
	    echo 系统不支持
	  fi
  fi
}
main
