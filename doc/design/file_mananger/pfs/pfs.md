# PFS Client

## Description
The `pfs` command is a Command Line Interface to manage your files on Paddle Cloud

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
A `pfspath` begin with `/pfs`, eg: `/pfs/mydir/text1.txt`.

## order of Path Arguments
Commonly, if there are two path arguments. The first is the source, and the second is the destination.

## Subcommonds
- [rm](rm.md)
- [mv](mv.md)
- [cp](cp.md)
- [ls](ls.md)
- [mkdir](mkdir.md)
- [sync](sync.md)
