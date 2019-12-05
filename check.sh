abc=`git diff -U0 upstream/develop |grep "+" |grep -E "inplace_atol[[:space:]]*=.*" || true`
echo "lalalalalallalalalallalala${abc}"
#echo $@|awk '{for (i=2;i<=NF;i++)print $i}'
