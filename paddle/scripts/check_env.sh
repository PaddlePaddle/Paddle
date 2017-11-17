#!/bin/bash

if [ "`uname -s`" != "Linux" ]; then
  echo "Current scenario only support in Linux yet!"
  exit 0
fi

echo "========================= Hardware Information ========================="
sockets=`grep 'physical id' /proc/cpuinfo | sort -u | wc -l`
cores_per_socket=`grep 'core id' /proc/cpuinfo | sort -u | wc -l`
ht=`lscpu |grep "per core" |awk -F':' '{print $2}'|xargs`
physical_cores=$((sockets * cores_per_socket))
virtual_cores=`grep 'processor' /proc/cpuinfo | sort -u | wc -l`
numa_nodes=`lscpu |grep "NUMA node(s)"|awk -F':' '{print $2}'|xargs`
echo "CPU Name               : `lscpu |grep \"name\" |awk -F':' '{print $2}'|xargs`"
echo "CPU Family             : `lscpu |grep \"CPU family\" |awk -F':' '{print $2}'|xargs`"
echo "Socket Number          : $sockets"
echo "Cores Per Socket       : $cores_per_socket"
echo "Total Physical Cores   : $physical_cores"
echo "Total Virtual Cores    : $virtual_cores"
if [ $ht -eq 1 ]; then
  echo "Hyper Threading        : OFF"
  if [ $physical_cores -ne $virtual_cores ]; then
    echo "Error: HT logical error"
  fi
else
  echo "Hyper Threading        : ON"
  if [ $physical_cores -ge $virtual_cores ]; then
    echo "Error: HT logical error"
  fi
fi
echo "NUMA Nodes             : $numa_nodes"
if [ $numa_nodes -lt $sockets ]; then
  echo "Warning: NUMA node is not enough for the best performance,\
 at least $sockets"
fi

echo "-------------------------- Memory Information --------------------------"
echo "DIMMs max slots        : `dmidecode | grep "Bank Locator" | wc -l`"
# dmidecode support start from 2.11
num_dimms_installed=`dmidecode | grep "Memory Device Mapped" | wc -l`
num_clock_configed=`dmidecode | grep -i "Configured Clock Speed" |grep -i "Hz" |wc -l`
echo "Installed DIMM number  : $num_dimms_installed"
if [ $num_dimms_installed -ne $num_clock_configed ]; then
  echo "Error: installed DIMMs do ont match configured clocks: $num_clock_configed"
fi
echo "Memory Size            : `free -h |grep -i mem |awk -F' ' '{print $2}'|xargs`"
echo "Swap Memory Size       : `free -h |grep -i swap |awk -F' ' '{print $2}'|xargs`"
echo "Total Memory Size      : `free -th |grep -i total |tail -n 1| awk -F' ' '{print $2}'|xargs`"
echo "Max Memory Capacity    : `dmidecode |grep -i \"maximum capacity\"|sort -u|awk -F':' '{print $2}'|xargs`"
# DIMMs fequency
clock_speeds=`dmidecode | grep -i "Configured Clock Speed" | grep -i "Hz" |sort -u | awk -F':' '{print $2}'|xargs`
echo "Configed Clock Speed   : $clock_speeds"
num_clock_type=`dmidecode | grep -i "Configured Clock Speed" | grep -i "Hz" |sort -u | wc -l`
if [ $num_clock_type -ne 1 ]; then
  echo "Warning: Have more than 1 speed type, all DIMMs should have same fequency: $clock_speeds"
fi

echo "-------------------------- Turbo Information  --------------------------"
scaling_drive=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver`
echo "Scaling Driver         : $scaling_drive"
if [ $scaling_drive == "intel_pstate" ] && [ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  turbo=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`
  if [ $turbo -eq 1 ]; then
    echo "Turbo Status           : OFF"
  else
    echo "Turbo Status           : ON"
  fi
else
  echo "Warning: Scaling driver is not intel_pstarte, maybe should enable it in BIOS"
  echo "Turbo Status           : Unknown"
fi
# cpu frequency
num_max_freq=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq| sort -u |wc -l`
num_min_freq=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq| sort -u |wc -l`
if [ $num_max_freq -ne 1 ]; then
  echo "Error: the max_frequency of all CPU should be equal"
fi
if [ $num_min_freq -ne 1 ]; then
  echo "Error: the min_frequency of all CPU should be equal"
fi
max_freq=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq| uniq|xargs` # kHz
max_freq=`awk 'BEGIN{printf "%.2f",('$max_freq' / 1000000)}'` # GHz
min_freq=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq| uniq|xargs` # kHz
min_freq=`awk 'BEGIN{printf "%.2f",('$min_freq' / 1000000)}'` # GHz
echo "CPU Max Frequency      : $max_freq GHz"
echo "CPU Min Frequency      : $min_freq GHz"
# cpu governor
num_governor=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor| sort -u |wc -l`
if [ $num_governor -ne 1 ]; then
  echo "Error: the governor of all CPU should be the same"
fi
governor=`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor| sort -u |uniq`
echo "CPU Freq Governor      : $governor"


echo "========================= Software Information ========================="
echo "BIOS Release Date      : `dmidecode | grep "Release Date"|awk -F ':' '{print $2}'|xargs`"
echo "OS Version             : `cat /etc/redhat-release`"
echo "Kernel Release Version : `uname -r`"
echo "Kernel Patch Version   : `uname -v`"
echo "GCC Version            :`gcc --version | head -n 1|awk -F '\\\(GCC\\\)' '{print $2}'`"
echo "CMake Version          :`cmake --version | head -n 1 | awk -F 'version' '{print $2}'`"
echo "------------------ Environment Variables Information -------------------"
kmp_affinity=`env | grep KMP_AFFINITY`
omp_dynamic=`env | grep OMP_DYNAMIC`
omp_nested=`env | grep OMP_NESTED`
omp_num_threads=`env | grep OMP_NUM_THREADS`
mkl_num_threads=`env | grep MKL_NUM_THREADS`
mkl_dynamic=`env | grep MKL_DYNAMIC`
if [ ! $kmp_affinity ]; then kmp_affinity="unset"; fi
if [ ! $omp_dynamic ]; then omp_dynamic="unset"; fi
if [ ! $omp_nested ]; then omp_nested="unset"; fi
if [ ! $omp_num_threads ]; then omp_num_threads="unset"; fi
if [ ! $mkl_num_threads ]; then mkl_num_threads="unset"; fi
if [ ! $mkl_dynamic ]; then mkl_dynamic="unset"; fi
echo "KMP_AFFINITY           : $kmp_affinity"
echo "OMP_DYNAMIC            : $omp_dynamic"
echo "OMP_NESTED             : $omp_nested"
echo "OMP_NUM_THREADS        : $omp_num_threads"
echo "MKL_NUM_THREADS        : $mkl_num_threads"
echo "MKL_DYNAMIC            : $mkl_dynamic"
# Check if any MKL related libraries have been installed in LD_LIBRARY_PATH
for path in `echo $LD_LIBRARY_PATH | awk -F ':' '{for(i=1;i<=NF;++i)print $i}'`; do
  mkldnn_found=`find $path -name "libmkldnn.so"`
  if [ "$mkldnn_found" ]; then
    echo "Found MKL-DNN          : $mkldnn_found"
  fi
  mklml_found=`find $path -name "libmklml_intel.so"`
  if [ "$mklml_found" ]; then
    echo "Found MKLML            : $mklml_found"
  fi
  iomp_found=`find $path -name "libiomp5.so"`
  if [ "$iomp_found" ]; then
    echo "Found IOMP             : $iomp_found"
  fi
done

# dump all details for fully check
lscpu > lscpu.dump
dmidecode > dmidecode.dump
