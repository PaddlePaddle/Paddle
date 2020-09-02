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

