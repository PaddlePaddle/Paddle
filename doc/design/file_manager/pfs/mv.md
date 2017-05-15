# Name  
mv - move (rename) files


# Synopsis
```
mv [-f | -n] [-v] <LocalPath> <PFSPath>
mv [-f | -n] [-v] <LocalPath> ... <PFSPath>
mv [-f | -n] [-v] <PFSPath> <LocalPath> 
mv [-f | -n] [-v] <PFSPath> ... <LocalPath> 
mv [-f | -n] [-v] <PFSPath> <PFSPath> 
mv [-f | -n] [-v] <PFSPath> ... <PFSPath> 
```

# Description
```
The following options are available:

-f      
	Do not prompt for confirmation before overwriting the destination path.  (The -f option overrides previous -n options.)

-n      
	Do not overwrite an existing file.  (The -n option overrides previous -f options.)

-v      
	Cause mv to be verbose, showing files after they are moved.
```

# Examples
- The following command moves a single file to pfs

```
paddle pfs mv ./text1.txt /pfs/$DATACENTER/home/$USER/text1.txt
```
