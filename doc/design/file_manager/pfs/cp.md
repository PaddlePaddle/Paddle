# Name  
cp - copy files

# Synopsis
```
cp [-r] [-f | -n] [-v] [--preserve--links] <LocalPath> <PFSPath>
cp [-r] [-f | -n] [-v] [--preserve--links] <LocalPath> ... <PFSPath>
cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> <LocalPath> 
cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> ... <LocalPath>
cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> <PFSPath> 
cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> ... <PFSPath>
```

# Description
```	   
The following options are available:

-r
   Copy directories recursively
   
-f      
	Do not prompt for confirmation before overwriting the destination path.  (The -f option overrides previous -n options.)

-n      
	Do not overwrite an existing file.  (The -n option overrides previous -f options.)

-v      
	Cause cp to be verbose, showing files after they are copied.

--preserve--links
   Reserve links when copy links
```

# Examples
- The following command copies a single file to pfs

```
paddle pfs cp ./text1.txt /pfs/$DATACENTER/home/$USER/text1.txt
```

- The following command copies pfs file to a local file

```
paddle pfs cp /pfs/$DATACENTER/home/$USER/text1.txt ./text1.txt
```
