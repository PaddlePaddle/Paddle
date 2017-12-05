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
# dmidecode support start from 2.11
max_dimms=0
num_dimms_installed=0
for dimm_id in `dmidecode |grep Locator|sort -u | awk -F ':' '{print $2}'`; do
  num_refered=`dmidecode |grep -c "$dimm_id"`
  # the acutal dimm id should be refered only once
  if [ $num_refered -eq 1 ]; then
    num_unknown=`dmidecode | awk '/'$dimm_id'/ {s=1}; {if (s==1) {a[NR]=$0}};
      /Manufacturer/ {s=0; for (i in a) print a[i]; delete a}' |grep -ic unknown`
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
