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
     read -p "是否安装CPU版本的PaddlePaddle？(y/n)" cpu_option
     cpu_option=`echo $cpu_option | tr 'A-Z' 'a-z'`
     if [[ "$cpu_option" == "" || "$cpu_option" == "n" ]];then
        echo "退出安装中..."
        exit
     else
        GPU='cpu'
        echo "将为您安装CPU版本的PaddlePaddle"
        break
     fi
    done
}

function checkLinuxCUDNN(){
   echo
   read -n1 -p "请按回车键进行下一步..."
   echo
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
               echo "检测结果：未在常规路径下找到cuda/include/cudnn.h文件"
               while true
               do
                  read -p "请核实cudnn.h位置，并在此输入路径（请注意，路径需要输入到“cudnn.h”这一级）:" cudnn_version
                  echo
                  if [ "$cudnn_version" == "" ] || [ ! -f "$cudnn_version" ];then
                        read -p "仍未找到cuDNN，输入y将安装CPU版本的PaddlePaddle，输入n可重新录入cuDNN路径，请输入（y/n）" cpu_option
                        echo
                        cpu_option=`echo $cpu_option | tr 'A-Z' 'a-z'`
                        if [ "$cpu_option" == "y" -o "$cpu_option" == "" ];then
                            GPU='cpu'
                            break
                        else
                            echo "请重新输入"
                            echo
                        fi
                  else
                     CUDNN=`cat $cudnn_version | grep CUDNN_MAJOR |awk 'NR==1{print $NF}'`
                     echo "检测结果：找到cudnn.h"
                     break
                  fi
                 done
             if [ "$GPU" == "cpu" ];then
                break
             fi
           fi
       fi
       if [ "$CUDA" == "9" -a "$CUDNN" != "7" ];then
           echo
           echo "目前CUDA9下仅支持cuDNN7，暂不支持您机器上的CUDNN${CUDNN}。您可以访问NVIDIA官网下载适合版本的CUDNN，请ctrl+c退出安装进程。按回车键将为您安装CPU版本的PaddlePaddle"
           echo
          use_cpu()
          if [ "$GPU"=="cpu" ];then
             break
          fi
       fi

       if [ "$CUDNN" == 5 ] || [ "$CUDNN" == 7 ];then
          echo
          echo "您的CUDNN版本是: CUDNN$CUDNN"
          break
       else
          echo
          read -n1 -p "目前支持的CUDNN版本为5和7,暂不支持您机器上的CUDNN${CUDNN}，将为您安装CPU版本的PaddlePaddle,请按回车键开始安装"
          echo
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
         echo "检测结果：找到CUDA $tmp_cuda"
       fi
       if [ "$tmp_cudai8" != "" ];then
         echo "检测结果：找到CUDA $tmp_cuda8"
       fi
       if [ "$tmp_cuda9" != "" ];then
         echo "检测结果：找到CUDA $tmp_cuda9"
       fi

       if [ "$CUDA" == "" ];then
            echo "检测结果：没有在常规路径下找到cuda/version.txt文件"
            while true
            do
                read -p "请输入cuda/version.txt的路径:" cuda_version
                if [ "$cuda_version" == "" || ! -f "$cuda_version" ];then
                    read -p "仍未找到CUDA，输入y将安装CPU版本的PaddlePaddle，输入n可重新录入CUDA路径，请输入（y/n）" cpu_option
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
                        echo "未能在version.txt中找到CUDA相关信息"
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
          echo "目前支持CUDA8/9，暂不支持您的CUDA${CUDA}，将为您安装CPU版本的PaddlePaddle"
          echo
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
        echo "正在检测您环境中是否存在AVX指令集..."
        echo
        echo "检测结果：您电脑上没有AVX指令集，目前针对无AVX指令集的环境，我们仅提供支持mkl数学库的PaddlePaddle，将为您安装此版本的PaddlePaddle"
        math='mkl'
        break
      elif [ "$GPU" == "gpu" ];then
        math='mkl'
        echo "检测到您的机器上配备GPU，推荐您使用mkl数学库"
        break
      else
        read -p "请输入您希望使用的数学库：
            1：openblas 一个高性能多核 BLAS 库
            2：mkl（推荐） 英特尔数学核心函数库
            => 请输入数字1或2。如输入其他字符或直接回车，将会默认选择【 2. mkl 】 。请在这里输入并回车：" math
          if [ "$math" == "" ];then
            math="mkl"
            echo "您选择了数字【2】"
            break
          fi
          if [ "$math" == "1" ];then
            math=openblas
            echo "您选择了数字【1】"
            break
          elif [ "$math" == "2" ];then
            math=mkl
            echo "您选择了数字【2】"
            break
          fi
          echo "输入错误，请再次输入"
      fi
    done
}

function checkLinuxPaddleVersion(){
  read -n1 -p "请按回车键继续..."
  while true
    do
      read -p "
               1. 开发版：对应Github上develop分支，如您需要开发、或希望使用PaddlePaddle最新功能，请选用此版本
               2. 稳定版（推荐）：如您无特殊开发需求，建议使用此版本，目前最新的版本号为 ${release_version}
                => 请输入数字1或2。如输入其他字符或直接回车，将会默认选择【 2. 稳定版 】 。请在这里输入并回车：" paddle_version
        if [ "$paddle_version" == "" ];then
          paddle_version="release-${release_version}"
          echo "您选择了数字【2】，为您安装release-${release_version}"
          break
        fi
        if [ "$paddle_version" == "1" ];then
          echo "您选择了数字【1】，将为您安装开发版"
          break
        elif [ "$paddle_version" == "2" ];then
          echo "您选择了数字【2】，为您安装release-${release_version}"
          break
        fi
        echo "输入错误，请再次输入"
    done
}

function checkLinuxPip(){
  while true
    do
       echo "请输入您要使用的pip目录（您可以另起终端，并使用which pip来查看）："
       read -p "" pip_path
       if [ "$pip_path" == "" -o ! -f "$pip_path" ];then
         echo "检测结果：pip不存在,请重新输入"
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
         echo "检测结果：pip不存在,请重新输入"
       else
         version_list=`echo "${python_list[@]}" | grep "$python_version" `
         if [ "$version_list" != "" ];then
           echo "检测结果：找到python${python_version}版本"
           break
         else
           echo "检测结果：找不到可用的 pip, 我们只支持Python27/35/36/37及其对应的pip, 请重新输入， 或使用ctrl + c退出 "
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
        echo "Step 6. 检测是否有avx"
        echo
        echo "检测结果：未能找到avx，我们仅提供CPU版本或配置为CUDA8 cuDNN7的GPU版本的安装包"
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
          if [ "$?" == "0" ];then
            $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release
          else
            echo "paddlepaddle whl包下载失败"
            exit 1
          fi
        else
          rm -rf `echo $wheel_gpu_release_novax|awk -F '/' '{print $NF}'`
          wget -q $wheel_gpu_release_novax
          if [ "$?" == "0" ];then
            $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_release_noavx
          else
            echo "paddlepaddle whl包下载失败"
            exit 1
          fi
        fi
    else
        rm -rf `echo $wheel_cpu_release|awk -F '/' '{print $NF}'`
        wget -q $wheel_cpu_release
        if [ "$?" == "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_release
        else
          echo "paddlepaddle whl包下载失败"
          exit 1
        fi
    fi
  else
    if [[ "$GPU" == "gpu" ]];then
        rm -rf `echo $wheel_gpu_develop|awk -F '/' '{print $NF}'`
        wget -q $wheel_gpu_develop
        if [ "$?" == "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_gpu_develop
        else
          echo "paddlepaddle whl包下载失败"
          exit 1
        fi
    else
        rm -rf `echo $wheel_cpu_develop|awk -F '/' '{print $NF}'`
        wget -q $wheel_cpu_develop
        if [ "$?" == "0" ];then
          $pip_path install --user -i https://mirrors.aliyun.com/pypi/simple --trusted-host=mirrors.aliyun.com $wheel_cpu_develop
        else
          echo "paddlepaddle whl包下载失败"
          exit 1
        fi
    fi
  fi
}


function checkLinuxGPU(){
  read -n1 -p "即将检测您的机器是否含GPU，请按回车键继续..."
  echo
  AVX=`cat /proc/cpuinfo |grep avx|tail -1|grep avx`
  which nvidia-smi >/dev/null 2>&1
  if [ "$?" != "0" ];then
    GPU='cpu'
    echo "未在机器上找到GPU，或PaddlePaddle暂不支持此型号的GPU"
  else
    GPU='gpu'
    echo "已在您的机器上找到GPU，即将确认CUDA和CUDNN版本..."
    echo
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

  echo "Step 2. 检测GPU型号和CUDA/cuDNN版本"
  echo
  checkLinuxGPU
  echo
  echo "Step 3. 检测数学库"
  echo
  checkLinuxMathLibrary
  echo
  echo "Step 4. 选择要安装的PaddlePaddle版本"
  echo
  checkLinuxPaddleVersion
  echo
  echo "Step 5. 检测pip版本"
  echo
  checkLinuxPip
  echo
  checkLinuxAVX
  echo "*********************2. 开始安装*****************************"
  PipLinuxInstall
}

function checkMacPython2(){
    while true
       do
          read -p "
                => 未能在常规路径下找到Python2，请使用ctrl+c命令退出安装程序，并使用brew或pypi.org下载安装Python2（注意Python版本不能低于2.7.15）
                如希望自定义Python路径，请输入路径：" python_root
          echo
          python_version=`$python_root --version 2>&1 1>&1`
          if [ $? == "0" ];then
            :
          else
            python_version=""
          fi
          check_python=`echo $python_version | grep "Python 2"`
          if [ "$python_version" == "" ] || [ "$python_root" == "/usr/bin/python" -a "$python_version" == "Python 2.7.10" ]  ;then
               python_version=""
          elif [ -n "$check_python" ];then
              while true
                do
                  read -p "
                => 在您的环境中找到 $python_version, 确认使用此版本请输入y；如您希望自定义Python路径请输入n。请在这里输入（y/n）并回车: " use_python
                  echo
                  use_python=`echo $use_python | tr 'A-Z' 'a-z'`
                  if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                       use_python="y"
                       break
                  elif [ "$use_python" == "n" ];then
                       python_root=""
                       break
                  else
                      echo "输入错误，请重新输入(y/n)"
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
          read -p "
                => 未能在常规路径下找到Python3，请使用ctrl+c命令退出安装程序，并使用brew或pypi.org下载Python3
                如希望自定义Python路径，请输入路径：" python_root
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
                  read -p "
                => 在您的环境中找到 $python_version, 确认使用此版本请输入y；如您希望自定义Python路径请输入n。请在这里输入（y/n）并回车: " use_python
                  echo
                  use_python=`echo $use_python | tr 'A-Z' 'a-z'`
                  if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                       use_python="y"
                       break
                  elif [ "$use_python" == "n" ];then
                        python_root=""
                        break
                  else
                      echo "输入错误，请重新输入(y/n)"
                  fi
                done
              if [ "$use_python" == "y" ];then
                    break
              fi
            else
              echo "您输入Python的不是Python3"
              python_version=""
            fi
       done
}

function checkMacPaddleVersion(){
  while true
    do
      read -n1 -p "Step 2. 选择PaddlePaddle的版本，请按回车键继续..."
      echo
      read -p "
               1. 开发版：对应Github上develop分支，如您需要开发、或希望使用PaddlePaddle最新功能，请选用此版本
               2. 稳定版（推荐）：如您无特殊开发需求，建议使用此版本，目前最新的版本号为 ${release_version}

               => 请输入数字1或2。如输入其他字符或直接回车，将会默认选择【 2. 稳定版 】 。请在这里输入并回车：" paddle_version
      if [ "$paddle_version" == "1" ]||[ "$paddle_version" == "2" ];then
          echo
          echo "您选择了数字【"$paddle_version" 】"
          echo
          break
      else
          paddle_version="2"
          echo
          echo "您选择了数字【2】"
          echo
          break
      fi
    done
}

function checkMacPythonVersion(){
  while true
    do
       read -n1 -p "Step 3. 选择Python版本，请按回车键继续..."
       read -p "
               2. 使用python 2.x
               3. 使用python 3.x

                => 请输入数字2或3。如输入其他字符或直接回车，将会默认使用【Python 2 】。请在这里输入并回车：" python_V
                echo
       if [ "$python_V" == "" ];then
            python_V="2"
       fi
       echo "您选择了数字【"$python_V"】，正在寻找符合您要求的Python版本，请按回车键继续..."
       echo
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
               read -p "
                => 在您的环境中找到 $python_version, 确认使用此版本请输入y；如您希望自定义Python路径请输入n。请在这里输入（y/n）并回车：" use_python
               echo
               use_python=`echo $use_python | tr 'A-Z' 'a-z'`
               if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                    break
               elif [ "$use_python" == "n" ];then
                    python_root=""
                    checkMacPython2
                    break
               else
                    echo "输入错误，请重新输入(y/n)"
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
               read -p "
                => 在您的环境中找到 $python_version, 确认使用此版本请输入y；如您希望自定义Python路径请输入n。请在这里输入（y/n）并回车：" use_python
               echo
               use_python=`echo $use_python | tr 'A-Z' 'a-z'`
               if [ "$use_python" == "y" ]||[ "$use_python" == "" ];then
                   break
               elif [ "$use_python" == "n" ];then
                    checkMacPython3
                    break
               else
                    echo "输入错误，请重新输入(y/n)"
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
           version_list=`echo "${python_list[@]}" | grep "$python_brief_version" `
           if [ "$version_list" != "" ];then
              break
            else
              echo "未找到可用的pip或pip3。PaddlePaddle目前支持：Python2.7/3.5/3.6/3.7及其对应的pip, 请重新输入，或使用ctrl + c退出"
           fi
        else
            echo "输入错误，请重新输入"
        fi
  done
}

function checkMacAVX(){
    read -n1 -p "Step 4. 检测您的Mac是否支持AVX指令集，请按回车键继续..."
    echo
    if [[ $AVX != "" ]];then
        AVX="avx"
        echo "检测结果：支持"
    else
        read -n1 -p "检测结果：不支持。非常抱歉，PaddlePaddle在Mac系统暂不提供no_avx类型的安装包，您可以选择在Linux系统中安装no_avx版的PaddlePaddle, 请按回车键退出..."
        exit
    fi
    echo
}

function checkMacGPU(){
    read -n1 -p "Step 5. 选择CPU/GPU版本，请按回车键继续..."
    echo
    if [[ $GPU != "" ]];then
        echo "MacOS环境下，暂未提供GPU版本的PaddlePaddle安装包，将为您安装CPU版本的PaddlePaddle"
    else
        echo "MacOS环境下，暂未提供GPU版本的PaddlePaddle安装包，将为您安装CPU版本的PaddlePaddle"
        GPU=cpu
    fi
    echo
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


        echo "*********************2. 开始安装*****************************"
        echo
        read -n1 -p "即将为您下载并安装PaddlePaddle，请按回车键继续..."
        echo
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
                   echo "安装成功！小提示：可以使用: ${python_root} 来启动安装了PaddlePaddle的Python解释器"
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
  echo "*********************************"
  echo "欢迎使用PaddlePaddle快速安装脚本"
  echo "*********************************"
  echo
  echo "如果您在安装过程中遇到任何问题，请在https://github.com/PaddlePaddle/Paddle/issues反馈，我们的工作人员将会帮您答疑解惑"
  echo
  echo "本安装包将帮助您在Linux或Mac系统下安装PaddlePaddle，包括 1）安装前的准备和 2）开始安装 两部分"
  echo
  read -n1 -p "请按回车键进行下一步..."
  echo
  echo
  echo "*********************1. 安装前的准备*****************************"
  echo
  echo "Step 1. 正在检测您的操作系统信息..."
  echo
  SYSTEM=`uname -s`
  if [ "$SYSTEM" == "Darwin" ];then
  	echo "您的系统为：MAC OSX"
    echo
  	macos
  else
 	echo "您的系统为：Linux"
  echo
	  OS=`cat /etc/issue|awk 'NR==1 {print $1}'`
	  if [ $OS == "\S" ] || [ "$OS" == "CentOS" ] || [ $OS == "Ubuntu" ];then
	    linux
	  else
	    echo "您的系统不在本安装包的支持范围，如您需要在windows环境下安装PaddlePaddle，请您参考PaddlePaddle官网的windows安装文档"
	  fi
  fi
}
main
