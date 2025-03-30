# NTU Physics MPL Course b11 Monday Class Group 1 Codebase

Source code used to perform data analysis on MPL experiments.  

## Introduction  

Top level directories are experiment labels. 
Under each experiment the structure is introduced as the following table:  

|Directory|Description|
|:-:|:-:|
|`analysis/`|Python source code and redirected stdout text files|
|`pics/`|Stores figure outputs|
|`data/`|Data used to perform analysis|

## Prerequisites

Here's a list of potentially used (nix-)packages/python modules used in this codebase.  

|Package/Python library|Description|
|:-:|:-:|
|[Python](https://www.python.org) 3.12.8|The Python programming language|
|[PyQt5 (PyQt6)](https://search.nixos.org/packages?channel=24.11&query=pyqt)|Python bindings for the Qt graphics framework|
|[numpy](https://numpy.org)|Python package for scientific computing|
|[matplotlib](https://matplotlib.org)|A graph plotting Python library|
|[scipy](https://scipy.org)|Fundamental algorithms for scientific computing in Python|
|[pandas](https://pandas.pydata.org)|A Python data analysis library|

All packages are installed via nix on NixOS from the [Nixpkgs](https://github.com/NixOS/nixpkgs) repository. 
This repository is also a nix flake, so nix could be used to easily reproduce the results. 
Except for the matplotlib QtAgg backend part, all results should be entirely reproducible with a proper python environment setup. 
For more information about the packages used (versions), checkout the [`shell.nix`](shell.nix) file for packages used, and [`flake.nix`](flake.nix) and [`flake.lock`](flake.lock) file for versions used.  

## Usage

Please setup a python environment matching the [prerequisites](#prerequisites). 
If using nix with flakes support, cloning and running `nix develop` will do the job smoothly. 
Manually setup the python environment otherwise.  

To reproduce figures and text outputs (redirecting stdout), under each `analysis/` folder, run (on Linux/Unix-like):  

```
$ python <python_file_name>.py > <python_file_name>-output.txt
```
