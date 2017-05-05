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
--help pfs 
	Display more information about pfs
--version
	Output version information and exit
--debug
	Show detailed debugging log	
```

## Path Arguments
When using a commnd, we need to specify path arguments. There are two path argument type: `localpath` and `pfspath`.
A pfspath begin with `pfs://`, eg: `pfs://mydir/text1.txt`.

## Path Arguments‘s order
Commonly, there maybe two path arguments. The first is source, and the second is destination.

## Subcommonds
- [rm](rm.md)
- [mv](mv.md)
- [cp](cp.md)
- [ls](ls.md)
- [mkdir](mkdir.md)
- [sync](sync.md)
