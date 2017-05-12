# Name  
cp - copy files and directories

# Synopsis
` cp [OPTION]...
<LocalPath> <PFSPath> or <PFSPath> <LocalPath> or <PFSPath> <PFSPath>`

# Description

```		
--preserve--links
   Reserve links when copy links
   
-R, -r, --recursive
   Copy directories recursively
```

# Examples
- The following command copies a single file to pfs

```
paddle pfs cp ./text1.txt /pfs/mydir/text1.txt
```

Output

```
upload ./text1.txt to /pfs/mydir/text1.txt
```

- The following command copies pfs file to a local file

```
paddle pfs cp /pfs/mydir/text1.txt ./text1.txt
```

Output

```
download /pfs/mydir/text1.txt to ./text1.txt
```
