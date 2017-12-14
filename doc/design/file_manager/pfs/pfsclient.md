# PFSClient

## Description
The `pfs` command is a Command Line Interface to manage your files on PaddlePaddle Cloud

## Synopsis
```
paddle [options] pfs <subcommand> [parameters]
```

## Options
```
--profile (string)
	Use a specific profile from your credential file.

--help (string)
	Display more information about command

--version
	Output version information and exit

--debug
	Show detailed debugging log	
	
--only-show-errors (boolean) 
	Only errors and warnings are displayed. All other output is suppressed.
```

## Path Arguments
When using a command, we need to specify path arguments. There are two path argument type: `localpath` and `pfspath`.  

A `pfspath` begin with `/pfs`, eg: `/pfs/$DATACENTER/home/$USER/folder`.

[Here](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/cluster_train/data_dispatch.md#上传训练文件) is how to config datacenters.

## order of Path Arguments
Commonly, if there are two path arguments, the first is the source, and the second is the destination.

## Subcommonds
- rm - remove files or directories

```
Synopsis:
	rm [-r] [-v] <PFSPath> ...

Options:
	-r 
		Remove directories and their contents recursively 
	-v      
		Cause rm to be verbose, showing files after they are removed.
	
Examples:
	paddle pfs rm /pfs/$DATACENTER/home/$USER/file
	paddle pfs rm -r /pfs/$DATACENTER/home/$USER/folder
```
- mv - move (rename) files

```
Synopsis:
	mv [-f | -n] [-v] <LocalPath> <PFSPath>
	mv [-f | -n] [-v] <LocalPath> ... <PFSPath>
	mv [-f | -n] [-v] <PFSPath> <LocalPath> 
	mv [-f | -n] [-v] <PFSPath> ... <LocalPath> 
	mv [-f | -n] [-v] <PFSPath> <PFSPath> 
	mv [-f | -n] [-v] <PFSPath> ... <PFSPath> 
	
Options:
	-f      
		Do not prompt for confirmation before overwriting the destination path.  (The -f option overrides previous -n options.)
	-n      
		Do not overwrite an existing file.  (The -n option overrides previous -f options.)
	-v      
		Cause mv to be verbose, showing files after they are moved.
		
Examples:
	paddle pfs mv ./text1.txt /pfs/$DATACENTER/home/$USER/text1.txt
```
- cp - copy files or directories

```
Synopsis:
	cp [-r] [-f | -n] [-v] [--preserve--links] <LocalPath> <PFSPath>
	cp [-r] [-f | -n] [-v] [--preserve--links] <LocalPath> ... <PFSPath>
	cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> <LocalPath> 
	cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> ... <LocalPath>
	cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> <PFSPath> 
	cp [-r] [-f | -n] [-v] [--preserve--links] <PFSPath> ... <PFSPath>

Options:
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
	   
Examples:
	paddle pfs cp ./file /pfs/$DATACENTER/home/$USER/file
	paddle pfs cp /pfs/$DATACENTER/home/$USER/file ./file
```
- ls- list files

```
Synopsis:
	ls [-r] <PFSPath> ...
	
Options:
	-R
   		List directory(ies) recursively

Examples:
	paddle pfs ls  /pfs/$DATACENTER/home/$USER/file
	paddle pfs ls  /pfs/$DATACENTER/home/$USER/folder
```

- mkdir - mkdir directory(ies)
Create intermediate directory(ies) as required.

```
Synopsis:
	mkdir <PFSPath> ...

Examples:
	paddle pfs mkdir  /pfs/$DATACENTER/home/$USER/folder
```
