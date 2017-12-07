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
echo "CPU Name               : `cat /proc/cpuinfo |grep -i "model name" |uniq |awk -F ':' '{print $2}'|xargs`"
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
# dmidecode support start from 2.11
dmi_ver=`dmidecode --version|awk -F '.' '{print $1}'|xargs`
if [ $dmi_ver -lt 2 ]; then
  echo "Error: dmidecode unknown or version is too old"
  exit 0
fi
if [ `dmidecode | grep -ic "Permission denied"` -ne 0 ]; then
  echo "Error: need root to run dmidecode"
  exit 0
fi
max_dimms=0
num_dimms_installed=0
for dimm_id in `dmidecode |grep Locator|sort -u | awk -F ':' '{print $2}'`; do
  num_refered=`dmidecode |grep -wc "$dimm_id"`
  # the actual dimm id should be refered only once
  if [ $num_refered -eq 1 ]; then
    num_unknown=`dmidecode | awk '/'$dimm_id'/ {s=1; f=0};
      /Unknown/ {f=1};
      /Manufacturer/ {if (s==1) {print f; exit 0;}};'`
    if [ $num_unknown -eq 0 ]; then
      dimms_installed="$dimms_installed \n $dimm_id"
      ((num_dimms_installed++))
    else
      dimms_uninstalled="$dimms_uninstalled \n $dimm_id"
    fi
    ((max_dimms++))
  fi
done
echo "Installed DIMM number  : $num_dimms_installed"
num_dimms_mapped=`dmidecode | grep "Memory Device Mapped" | wc -l`
if [ $num_dimms_installed -ne $num_dimms_mapped ]; then
  echo "Error: The installed DIMMs number does ont match the mapped memory device: $num_dimms_mapped"
fi
num_clock_configed=`dmidecode | grep -i "Configured Clock Speed" |grep -ic "Hz"`
if [ $num_dimms_installed -ne $num_clock_configed ]; then
  echo "Error: The installed DIMMs number does ont match configured clocks: $num_clock_configed"
fi
echo -e "Installed DIMMs Locator: $dimms_installed"
echo -e "Not installed DIMMs    : $dimms_uninstalled"
max_dimm_slots=`dmidecode | grep -c "Bank Locator"`
echo "DIMMs max slots        : $max_dimm_slots"
if [ $max_dimms -ne $max_dimm_slots ]; then
  echo "Error: The max dimm slots do not match the max dimms: $max_dimms"
fi
free_ver_main=`free -V|awk -F ' ' '{print $NF}'|awk -F '.' '{print $1}'`
free_ver_sub=`free -V|awk -F ' ' '{print $NF}'|awk -F '.' '{print $2}'`
if [ $free_ver_main -lt 3 ] || [ $free_ver_sub -lt 3 ]; then
  mem_sz=`free |grep -i mem |awk -F' ' '{print $2}'|xargs`
  swap_sz=`free |grep -i swap |awk -F' ' '{print $2}'|xargs`
  total_sz=`free -t |grep -i total |tail -n 1| awk -F' ' '{print $2}'|xargs`
  mem_sz="`awk 'BEGIN{printf "%.1f\n",('$mem_sz'/1024/1024)}'` GB" 
  swap_sz="`awk 'BEGIN{printf "%.1f\n",('$swap_sz'/1024/1024)}'` GB"
  total_sz="`awk 'BEGIN{printf "%.1f\n",('$total_sz'/1024/1024)}'` GB"
else
  mem_sz=`free -h |grep -i mem |awk -F' ' '{print $2}'|xargs`
  swap_sz=`free -h |grep -i swap |awk -F' ' '{print $2}'|xargs`
  total_sz=`free -th |grep -i total |tail -n 1| awk -F' ' '{print $2}'|xargs`
fi
echo "Memory Size            : $mem_sz"
echo "Swap Memory Size       : $swap_sz"
echo "Total Memory Size      : $total_sz"
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
if command -v cmake >/dev/null 2>&1; then 
  cmake_ver=`cmake --version | head -n 1 | awk -F 'version' '{print $2}'`
else
  cmake_ver=" Not installed"
fi
echo "CMake Version          :$cmake_ver"
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

# The expected result would be like:
# ========================= Hardware Information =========================
# CPU Name               : Intel(R) Xeon(R) Gold 6148M CPU @ 2.40GHz
# CPU Family             : 6
# Socket Number          : 2
# Cores Per Socket       : 20
# Total Physical Cores   : 40
# Total Virtual Cores    : 40
# Hyper Threading        : OFF
# NUMA Nodes             : 2
# -------------------------- Memory Information --------------------------
# Installed DIMM number  : 12
# Installed DIMMs Locator:
#  CPU1_DIMM_A1
#  CPU1_DIMM_B1
#  CPU1_DIMM_C1
#  CPU1_DIMM_D1
#  CPU1_DIMM_E1
#  CPU1_DIMM_F1
#  CPU2_DIMM_A1
#  CPU2_DIMM_B1
#  CPU2_DIMM_C1
#  CPU2_DIMM_D1
#  CPU2_DIMM_E1
#  CPU2_DIMM_F1
# Not installed DIMMs    :
#  CPU1_DIMM_A2
#  CPU1_DIMM_B2
#  CPU1_DIMM_C2
#  CPU1_DIMM_D2
#  CPU1_DIMM_E2
#  CPU1_DIMM_F2
#  CPU2_DIMM_A2
#  CPU2_DIMM_B2
#  CPU2_DIMM_C2
#  CPU2_DIMM_D2
#  CPU2_DIMM_E2
#  CPU2_DIMM_F2
# DIMMs max slots        : 24
# Memory Size            : 376G
# Swap Memory Size       : 4.0G
# Total Memory Size      : 380G
# Max Memory Capacity    : 2304 GB
# Configed Clock Speed   : 2666 MHz
# -------------------------- Turbo Information  --------------------------
# Scaling Driver         : intel_pstate
# Turbo Status           : ON
# CPU Max Frequency      : 3.70 GHz
# CPU Min Frequency      : 1.00 GHz
# CPU Freq Governor      : performance
# ========================= Software Information =========================
# BIOS Release Date      : 03/10/2017
# OS Version             : CentOS Linux release 7.3.1611 (Core)
# Kernel Release Version : 3.10.0-514.el7.x86_64
# Kernel Patch Version   : #1 SMP Tue Nov 22 16:42:41 UTC 2016
# GCC Version            : 4.8.5 20150623 (Red Hat 4.8.5-11)
# CMake Version          : 3.5.2
# ------------------ Environment Variables Information -------------------
# KMP_AFFINITY           : unset
# OMP_DYNAMIC            : unset
# OMP_NESTED             : unset
# OMP_NUM_THREADS        : unset
# MKL_NUM_THREADS        : unset
# MKL_DYNAMIC            : unset
