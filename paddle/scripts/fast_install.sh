#!/bin/bash

path='http://paddlepaddle.org/download?url='
#release_version=`curl -s https://pypi.org/project/paddlepaddle/|grep -E "/project/paddlepaddle/"|grep "release"|awk -F '/' '{print $(NF-1)}'|head -1`
release_version=1.2.0
python_list=(
"27"
"35"
"36"
"37"
)


function use_cpu(){
   while true
    do
     read -p "是否安装CPU版本的PaddlePaddle？(y/n)， 或使用ctrl + c退出: " cpu_option
     cpu_option=`echo $cpu_option | tr 'A-Z' 'a-z'`
     if [[ "$cpu_option" == "" || "$cpu_option" == "n" ]];then
        echo "退出安装中...."
        exit
     else
        GPU='cpu'
        echo "为您安装CPU版本"
        break
     fi
    done
}

function checkMacPython2(){
    while true
       do
          read -p "未发现除MacOS自带的python外的可用python，
                   请安装brew或从pypi.org下载的python2.7.15或更高版本，
                   或 输入您安装的python路径（可以使用ctrl + c后退出后使用which python查询）,
                   或 使用ctrl + c退出: " python_root
          python_version=`$python_root --version 2>&1 1>&1`
          if [ $? == "0" ];then
            :
          else
            python_version=""
          fi
          check_python=`echo $python_version | grep "Python 2"`
          echo $check_python
          if [ "$python_version" == "" ] || [ "$python_root" == "/usr/bin/python" -a "$python_version" == "Python 2.7.10" ] ;then
               python_version=""
          elif [ -n "$check_python" ];then
              while true
                do
                  read -p "找到：$python_version, 是否使用：(y/n)，输入n来输入自定义使用的python路径，或者按ctrl + c退出： " use_python
                  use_python=`echo $use_python | tr 'A-Z' 'a-z'`
                  if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                       use_python="y"
                       break
                  elif [ "$use_python" == "n" ];then
                       python_root=""
                       break
                  else
                      echo "输入错误，请重新输入"
                  fi
                done
              if [ "$use_python" == "y" ];then
                break
              fi
          else
               echo "您输入Python的不是Python2"
               python_version=""
          fi
       done
}

function checkMacPython3(){
    while true
       do
          read -p "未发现可用的python3，
                   请安装brew或从pypi.org下载的python3或更高版本，
                   或输入您安装的python3路径（可使用which python3查询），
                   或使用ctrl + c退出: " python_root
          python_version=`$python_root --version  2>&1 1>&1`
          if [ $? == "0" ];then
              :
          else
              python_version=""
          fi
          check_python=`echo $python_version | grep "Python 3"`
          if [ "$python_version" == "" ] || [ "$python_root" == "/usr/bin/python" -a "$python_version" == "Python 2.7.10" ] ;then
               python_version=""
          elif [ -n "$check_python" ] ;then
              while true
                do
                  read -p "找到：$python_version, 是否使用：(y/n)，输入n来输入自定义使用的python路径，或者按ctrl + c退出： " use_python
                  use_python=`echo $use_python | tr 'A-Z' 'a-z'`
                  if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                       use_python="y"
                       break
                  elif [ "$use_python" == "n" ];then
                        python_root=""
                        break
                  else
                      echo "输入错误，请重新输入"
                  fi
                done
              if [ "$use_python" == "y" ];then
                    break
              fi
          else
              echo "您输入Python的不是Python2"
              python_version=""
          fi
       done
}

function checkLinuxCUDNN(){
   while true
   do
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
                        read -p "未找到cuDNN,只能安装cpu版本的PaddlePaddle，是否安装（y/n）, 或使用ctrl + c退出:" cpu_option
                        cpu_option=`echo $cpu_option | tr 'A-Z' 'a-z'`
                        if [ "$cpu_option" == "y" -o "$cpu_option" == "" ];then
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
}

function checkLinuxCUDA(){
   while true
   do
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
         echo "找到CUDA $tmp_cuda"
       fi
       if [ "$tmp_cudai8" != "" ];then
         echo "找到CUDA $tmp_cuda8"
       fi
       if [ "$tmp_cuda9" != "" ];then
         echo "找到CUDA $tmp_cuda9"
       fi

       if [ "$CUDA" == "" ];then
            echo "没有找到cuda/version.txt文件"
            while true
            do
                read -p "请提供cuda version.txt的路径:" cuda_version
                if [ "$cuda_version" == "" || ! -f "$cuda_version" ];then
                    read -p "未找到CUDA,只能安装cpu版本的PaddlePaddle，是否安装（y/n）,  或使用ctrl + c退出" cpu_option
                    cpu_option=`echo $cpu_option | tr 'A-Z' 'a-z'`
                    if [ "$cpu_option" == "y" || "$cpu_option" == "" ];then
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
          break
       else
          echo "你的CUDA${CUDA}版本不支持,目前支持CUDA8/9"
          use_cpu
       fi

       if [ "$GPU" == "cpu" ];then
          break
       fi
   done
}

function checkLinuxMathLibrary(){
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
            输入1：openblas
            输入2：mkl
            请选择：" math
          if [ "$math" == "" ];then
            math="mkl"
            echo "为您安装mkl"
            break
          fi
          if [ "$math" == "1" ];then
            math=openblas
            echo "为您安装openblas"
            break
          elif [ "$math" == "2" ];then
            math=mkl
            echo "为您安装mkl"
            break
          fi
          echo "输入错误，请再次输入"
      fi
    done
}

function checkLinuxPaddleVersion(){
  while true
    do
      read -p "请选择Paddle版本：
          输入1：develop
          输入2：release-${release_version}
          请选择：" paddle_version
        if [ "$paddle_version" == "" ];then
          paddle_version="release-${release_version}"
          echo "为您安装release-${release_version}"
          break
        fi
        if [ "$paddle_version" == "1" ];then
          echo "为您安装develop"
          break
        elif [ "$paddle_version" == "2" ];then
          echo "为您安装release-${release_version}"
          break
        fi
        echo "输入错误，请再次输入"
    done
}

function checkLinuxPip(){
  while true
    do
       echo "请输入您要使用的pip目录（您可以使用which pip来查看）："
       read -p "" pip_path
       if [ "$pip_path" == "" -o ! -f "$pip_path" ];then
         echo "pip不存在,请重新输入"
         continue
       fi
       python_version=`$pip_path --version|awk -F "[ |)]" '{print $6}'|sed 's#\.##g'`
       if [ "$python_version" == "27" ];then
         uncode=`python -c "import pip._internal;print(pip._internal.pep425tags.get_supported())"|grep "cp27mu"`
         if [[ "$uncode" == "" ]];then
            uncode=
         else
            uncode=u
         fi
       fi
       if [ "$python_version" == "" ];then
         echo "pip不存在,请重新输入" 
       else
         version_list=`echo "${python_list[@]}" | grep "$python_version" `
         if [ "$version_list" != "" ];then
           echo "找到python${python_version}版本"
           break
         else
           echo "找不到可用的 pip, 我们只支持Python27/35/36/37及其对应的pip, 请重新输入， 或使用ctrl + c退出 "
         fi
       fi
    done
}

function checkLinuxAVX(){
  while true
  do
    if [[ "$AVX" != "" ]];then
      AVX="avx"
      break
    else
      if [ "$CUDA" == "8" -a "$CUDNN" == "7" ] || [ "$GPU" == "cpu" ];then
        AVX="noavx"
        break
      else
        echo "我们仅支持纯CPU或GPU with CUDA 8 cuDNN 7 下noavx版本的安装，请使用cat /proc/cpuinfo | grep avx检查您计算机的avx指令集支持情况"
        break
      fi
    fi
  done
}

function PipLinuxInstall(){
  wheel_cpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-${GPU}-${AVX}-${math}/paddlepaddle-${release_version}-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-${release_version}.post${CUDA}${CUDNN}-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_release_noavx="http://paddle-wheel.bj.bcebos.com/${release_version}-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-${release_version}-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_cpu_develop="http://paddle-wheel.bj.bcebos.com/latest-cpu-${AVX}-${math}/paddlepaddle-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"
  wheel_gpu_develop="http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda${CUDA}-cudnn${CUDNN}-${AVX}-${math}/paddlepaddle_gpu-latest-cp${python_version}-cp${python_version}m${uncode}-linux_x86_64.whl"


  if [[ "$paddle_version" == "2" ]];then
    if [[ "$GPU" == "gpu" ]];then
        if [[ ${AVX} == "avx" ]];then
          rm -rf `echo $wheel_gpu_release|awk -F '/' '{print $NF}'`
          wget -q $wheel_gpu_release
          if [ "$?" != "0" ];then
            $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release
          else
            echo paddlepaddle whl包下载失败
            exit 1
          fi
        else
          rm -rf `echo $wheel_gpu_release_novax|awk -F '/' '{print $NF}'`
          wget -q $wheel_gpu_release_novax
          if [ "$?" != "0" ];then
            $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release_noavx
          else
            echo paddlepaddle whl包下载失败
            exit 1
          fi
        fi
    else
        rm -rf `echo $wheel_cpu_release|awk -F '/' '{print $NF}'`
        wget -q $wheel_cpu_release
        if [ "$?" != "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_release
        else
          echo paddlepaddle whl包下载失败
          exit 1
        fi
    fi
  else
    if [[ "$GPU" == "gpu" ]];then
        rm -rf `echo $wheel_gpu_develop|awk -F '/' '{print $NF}'`
        wget -q $wheel_gpu_develop
        if [ "$?" != "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_develop
        else
          echo paddlepaddle whl包下载失败
          exit 1
        fi
    else
        rm -rf `echo $wheel_cpu_develop|awk -F '/' '{print $NF}'`
        wget -q $wheel_cpu_develop
        if [ "$?" != "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_develop
        else
          echo paddlepaddle whl包下载失败
          exit 1
        fi
    fi
  fi
}


function checkLinuxGPU(){
  AVX=`cat /proc/cpuinfo |grep avx|tail -1|grep avx`
  which nvidia-smi >/dev/null 2>&1
  if [ "$?" != "0" ];then
    GPU='cpu'
    echo "您使用的是不包含支持的GPU的机器"
  else
    GPU='gpu'
    echo "您使用的是包含我们支持的GPU机器"
  fi
  if [ "$GPU" == 'gpu' ];then
    checkLinuxCUDA
    checkLinuxCUDNN
  fi
}

function linux(){
gpu_list=(
"GeForce 410M"
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
  checkLinuxGPU
  checkLinuxMathLibrary
  checkLinuxPaddleVersion
  checkLinuxPip
  checkLinuxAVX
  PipLinuxInstall
}

function checkMacPaddleVersion(){
  while true
    do
      read -p "请选择Paddle版本(默认是release)：
               输入 1 来使用develop版本
               输入 2 来使用release ${release_version}
               请输入，或者按ctrl + c退出： " paddle_version
      if [ "$paddle_version" == "1" ]||[ "$paddle_version" == "2" ];then
          break
      else
          paddle_version="2"
          echo "将会下载release版本PaddlePaddle"
          break
      fi
    done
}

function checkMacPythonVersion(){
  while true
    do
       read -p "请您选择希望使用的python版本
                输入 2 使用python2.x
                输入 3 使用python3.x
                请选择(默认为2)，或者按ctrl + c退出：" python_V
       if [ "$python_V" == "" ];then
            python_V="2"
       fi
       if [ "$python_V" == "2" ];then
           python_root=`which python2.7`
           if [ "$python_root" == "" ];then
                python_root=`which python`
           fi
           python_version=`$python_root --version 2>&1 1>&1`
           if [ $? == "0" ];then
               :
           else
               python_version=""
           fi
           if [ "$python_root" == "" ]||[ "$python_root" == "/usr/bin/python" -a "$python_version" == "Python 2.7.10" ]||[ "$python_root" == "/usr/bin/python2.7" -a "$python_version" == "Python 2.7.10" ];then
               checkMacPython2
           fi
           while true
             do
               read -p "找到：$python_version, 是否使用：(y/n)，输入n来输入自定义使用的python路径，或者按ctrl + c退出：  " use_python
               use_python=`echo $use_python | tr 'A-Z' 'a-z'`
               if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                    break
               elif [ "$use_python" == "n" ];then
                    python_root=""
                    checkMacPython2
                    break
               else
                    echo "输入错误，请重新输入"
               fi
            done

       elif [ "$python_V" == "3" ];then
           python_root=`which python3`
           python_version=`$python_root --version 2>&1 1>&1`
           if [ $? == "0" ];then
               :
           else
               python_version=""
           fi
           if [ "$python_root" == "" ]||[ "$python_root" == "/usr/bin/python" -a "$python_version" == "Python 2.7.10" ];then
               checkMacPython3
           fi
           while true
             do
               read -p "找到：$python_version, 是否使用：(y/n), 输入n来输入自定义使用的python路径，或者按ctrl + c退出：" use_python
               use_python=`echo $use_python | tr 'A-Z' 'a-z'`
               if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                   break
               elif [ "$use_python" == "n" ];then
                    checkMacPython3
                    break
               else
                    echo "输入错误，请重新输入"
               fi
           done
       else
           :
       fi


       if [ "$python_V" == "2" ]||[ "$python_V" == "3" ];then
           python_brief_version=`$python_root -m pip -V |awk -F "[ |)]" '{print $6}'|sed 's#\.##g'`
           if [[ $python_brief_version == "27" ]];then
              uncode=`python -c "import pip._internal;print(pip._internal.pep425tags.get_supported())"|grep "cp27"`
              if [[ $uncode == "" ]];then
                 uncode="mu"
              else
                 uncode="m"
              fi
           fi
           echo ${python_list[@]}
           version_list=`echo "${python_list[@]}" | grep "$python_brief_version" `
           if [ "$version_list" != "" ];then
              break
            else
              echo "未发现可用的pip或pip3/pip3.x, 我们只支持Python2.7/3.5/3.6/3.7及其对应的pip, 请重新输入， 或使用ctrl + c退出"
           fi
        else
            echo "输入错误，请重新输入"
        fi
  done
}

function checkMacAVX(){
    if [[ $AVX != "" ]];then
        AVX="avx"
    else
        echo "您的Mac不支持AVX指令集，目前不能安装PaddlePaddle"
    fi
}

function checkMacGPU(){
    if [[ $GPU != "" ]];then
        echo "MacOS上暂不支持GPU版本的PaddlePaddle, 将为您安装CPU版本的PaddlePaddle"
    else
        echo "MacOS上暂不支持GPU版本的PaddlePaddle, 将为您安装CPU版本的PaddlePaddle"
        GPU=cpu
    fi
}

function macos() {
  path='http://paddlepaddle.org/download?url='
  AVX=`sysctl -a | grep cpu | grep AVX1.0 | tail -1 | grep AVX`

  while true
      do
        checkMacPaddleVersion
        checkMacPythonVersion
        checkMacAVX
        checkMacGPU


        if [[ $paddle_version == "2" ]];then
            $python_root -m pip install paddlepaddle
            if [ $? == "0" ];then
               echo "安装成功，可以使用: ${python_root} 来启动安装了PaddlePaddle的Python解释器"
               break
            else
               rm  $whl_cpu_release
               echo "未能正常安装PaddlePaddle，请尝试更换您输入的python路径，或者ctrl + c退出后请检查您使用的python对应的pip或pip源是否可用"
               echo""
               echo "=========================================================================================="
               echo""
               exit 1
            fi
        else
            if [ -f $whl_cpu_develop ];then
                $python_root -m pip install $whl_cpu_develop
                if [ $? == "0" ];then
                   rm -rf $whl_cpu_develop
                   echo "安装成功，可以使用: ${python_root} 来启动安装了PaddlePaddle的Python解释器"
                   break
                else
                   echo "未能正常安装PaddlePaddle，请尝试更换您输入的python路径，或者ctrl + c退出后请检查您使用的python对应的pip或pip源是否可用"
                   echo""
                   echo "=========================================================================================="
                   echo""
                   exit 1
                fi
            else
                wget ${path}$whl_cpu_develop -O $whl_cpu_develop
                if [ $? == "0" ];then
                    $python_root -m pip install $whl_cpu_develop
                    if [ $? == "0" ];then
                       rm  $wheel_cpu_develop
                       echo "安装成功，可以使用: ${python_root} 来启动安装了PaddlePaddle的Python解释器"
                       break
                    else
                       rm  $whl_cpu_release
                       echo "未能正常安装PaddlePaddle，请尝试更换您输入的python路径，或者ctrl + c退出后请检查您使用的python对应的pip或pip源是否可用"
                       echo""
                       echo "=========================================================================================="
                       echo""
                       exit 1
                    fi
                else
                      rm  $whl_cpu_develop
                      echo "未能正常安装PaddlePaddle，请检查您的网络 或者确认您是否安装有 wget，或者ctrl + c退出后反馈至https://github.com/PaddlePaddle/Paddle/issues"
                      echo""
                      echo "=========================================================================================="
                      echo""
                      exit 1
                fi
            fi
        fi
  done
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
