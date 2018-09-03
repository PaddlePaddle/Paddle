#!/bin/bash

#################################################
#
# Usage: anakin_docker_build_and_run.sh -p -o -m 
#
#################################################

ANAKIN_DOCKER_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"

# help_anakin_docker_run() to print help msg.
help_anakin_docker_run() {
	echo "Usage: $0 -p -o -m"
    echo ""
	echo "Options:"
    echo ""
	echo " -p Hardware Place where docker will running [ NVIDIA-GPU / AMD-GPU / X86-ONLY / ARM ] "
	echo " -o Operating system docker will reside on [ Centos / Ubuntu ] "
	echo " -m Script exe mode [ Build / Run ] default mode is build and run"
	exit 1
}

# install nvidia-docker 2 
install_nvidia_docker_v2() {
	echo "Setting env nvidia-docker2 in background ..."
	distribution=$(source /etc/os-release;echo $ID$VERSION_ID)
	if [ $ID == 'centos'];then
		docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
		sudo yum remove nvidia-docker
		curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
			  sudo tee /etc/yum.repos.d/nvidia-docker.repo
		sudo yum install -y nvidia-docker2 --skip-broken
		sudo pkill -SIGHUP dockerd
	else
		# default ubuntu
		# remove nv-doker v1
		docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
		sudo apt-get purge nvidia-docker
		curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
		curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
					sudo tee /etc/apt/sources.list.d/nvidia-docker.list
		sudo apt-get update
		sudo apt-get install -y nvidia-docker2
		sudo pkill -SIGHUP dockerd
	fi
}

# building and running docker for nvidia gpu
building_and_run_nvidia_gpu_docker() {
	if [ ! $# -eq 2 ]; then
		exit 1
	fi
	DockerfilePath=$1
	MODE=$2
	tag="$(echo $DockerfilePath | awk -F/ '{print tolower($(NF-3) "_" $(NF-1))}')"
	if [ ! $MODE = "Run" ]; then
		echo "Building nvidia docker ... [ docker_image_name: anakin image_tag: $tag ]"	
		docker build --network=host -t anakin:$tag"-base" . -f $DockerfilePath
        docker run --network=host -it anakin:$tag"-base"  Anakin/tools/gpu_build.sh
        container_id=$(docker ps -l | sed -n 2p | awk '{print $1}')
        docker commit $container_id anakin:$tag
	else
		echo "Running nvidia docker ... [ docker_image_name: anakin image_tag: $tag ]" 
		docker run --network=host --runtime=nvidia --rm -it anakin:$tag  /bin/bash
	fi
}

# buiding and running docker for amd gpu
building_and_run_amd_gpu_docker() {
	if [ ! $# -eq 2 ]; then
		exit 1
	fi
	DockerfilePath=$1
	MODE=$2
	tag="$(echo $DockerfilePath | awk -F/ '{print tolower($(NF-3) "_" $(NF-1))}')"
	if [ ! $MODE = "Run" ]; then
		echo "Building amd docker ... [ docker_image_name: anakin image_tag: $tag ]"	
		docker build --network=host -t anakin:$tag . -f $DockerfilePath
	else
		echo "Running amd docker ... [ docker_image_name: anakin image_tag: $tag ]" 
		docker run -it --device=/dev/kfd --device=/dev/dri --group-add video anakin:$tag /bin/bash
	fi
}

# building and running docker for x86
building_and_run_x86_docker() { 
	if [ ! $# -eq 2 ]; then
		exit 1
	fi
	DockerfilePath=$1
	MODE=$2
	tag="$(echo $DockerfilePath | awk -F/ '{print tolower($(NF-3) "_" $(NF-1))}')"
	if [ ! $MODE = "Run" ]; then
		echo "Building X86 docker ... [ docker_image_name: anakin image_tag: $tag ]"	
		docker build --network=host -t anakin:$tag . -f $DockerfilePath
	else
		echo "Running X86 docker ... [ docker_image_name: anakin image_tag: $tag ]" 
		docker run -it anakin:$tag /bin/bash
	fi
}

# building docker for arm
building_and_arm_docker() { 
	if [ ! $# -eq 2 ]; then
		exit 1
	fi
	DockerfilePath=$1
	MODE=$2
	tag="$(echo $DockerfilePath | awk -F/ '{print tolower($(NF-3) "_" $(NF-1))}')"
	if [ ! $MODE = "Run" ]; then
		echo "Building ARM docker ... [ docker_image_name: anakin image_tag: $tag ]"	
		docker build --network=host -t anakin:$tag . -f $DockerfilePath
	else
		echo "Running ARM docker ... [ docker_image_name: anakin image_tag: $tag ]" 
		docker run -it anakin:$tag /bin/bash
	fi
}

# dispatch user args to target docker path
dispatch_docker_path() {
	# declare associative map from place to relative path
	declare -A PLACE2PATH
	PLACE2PATH["NVIDIA-GPU"]=NVIDIA
	PLACE2PATH["AMD-GPU"]=AMD
	PLACE2PATH["X86-ONLY"]=X86
	PLACE2PATH["ARM"]=ARM
	# declare associative map from os to relative path
	declare -A OS2PATH
	OS2PATH["Centos"]=centos
	OS2PATH["Ubuntu"]=ubuntu

	if [ $# -eq 2 ]; then
		place=$1
		os=$2
		if [ ${PLACE2PATH[$place]+_} ]; then
			echo "+ Found ${PLACE2PATH[$place]} path..."
		else
			echo "+ Error: -p place: $place is not support yet !"
			exit 1
		fi
		if [ ${OS2PATH[$os]+_} ]; then
			echo "+ Found ${OS2PATH[$os]} path..."
		else
			echo "+ Error: -o os: $os is not support yet !"
			exit 1
		fi
		PlaceRelativePath=${PLACE2PATH[$place]}
		OSRelativePath=${OS2PATH[$os]}
	else
		exit 1
	fi
	tag_info="$( ls $ANAKIN_DOCKER_ROOT/$PlaceRelativePath/$OSRelativePath/ )"
	SupportDockerFilePath=$ANAKIN_DOCKER_ROOT/$PlaceRelativePath/$OSRelativePath/$tag_info/Dockerfile
	if [ ! -f $SupportDockerFilePath ];then
		echo "Error: can't find Dockerfile in path: $ANAKIN_DOCKER_ROOT/$PlaceRelativePath/$OSRelativePath/$tag_info"
		exit 1
	fi
}

# get args
if [ $# -lt 2 ]; then
	help_anakin_docker_run
	exit 1
fi

place=0
os=0
mode=Build
while getopts p:o:m:hold opt
do
	case $opt in
		p) place=$OPTARG;;
		o) os=$OPTARG;;
		m) mode=${OPTARG};;
		*) help_anakin_docker_run;;
	esac
done

echo "User select place:             $place"
echo "User select operating system:  $os"
echo "User select mode:              $mode"

dispatch_docker_path $place $os
#echo $SupportDockerFilePath

if [ $place = "NVIDIA-GPU" ]; then
	building_and_run_nvidia_gpu_docker $SupportDockerFilePath $mode
elif [ $place = "AMD-GPU" ]; then
	building_and_run_amd_gpu_docker $SupportDockerFilePath $mode
elif [ $place = "X86-ONLY" ]; then
	building_and_run_x86_docker $SupportDockerFilePath $mode
elif [ $place = "ARM" ]; then
	building_and_arm_docker $SupportDockerFilePath $mode
else
	echo "Error: target place is unknown! " 
fi
