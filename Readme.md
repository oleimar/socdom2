
# socdom2: C++ code for evolutionary simulation of dominance hierarchy formation


## Overview

This repository contains C++ code and example data.
The executable program `EvoDom`, built from this code, will run evolutionary simulations of a population of several groups of individuals that, in each generation, have social dominance interactions.
The program was used to produce the evolutionary simulation results for the paper "Reproductive skew, fitness costs, and winner-loser effects in social-dominance evolution" by Olof Leimar and Redouan Bshary.


## System requirements

This program has been compiled and run on a Linux server with Ubuntu 20.04 LTS.
The C++ compiler was g++ version 9.3.0, provided by Ubuntu, with compiler flags for c++17, and `cmake` (<https://cmake.org/>) was used to build the program.
It can be run multithreaded using OpenMP, which speeds up execution times.
Most likely the instructions below will work for many Linux distributions.
For single-threaded use, the program has also been compiled and run on macOS, using the Apple supplied Clang version of g++.

The program reads input parameters from TOML files (<https://github.com/toml-lang/toml>), using the open source `cpptoml.h` header file (<https://github.com/skystrife/cpptoml>), which is included in this repository.

The program stores evolving populations in HDF5 files (<https://www.hdfgroup.org/>), which is an open source binary file format.
The program uses the open source HighFive library (<https://github.com/BlueBrain/HighFive>) to read and write to such files.
These pieces of software need to be installed in order for `cmake` to successfully build the program.


## Installation guide

Install the repository from Github to a local computer.
There is a single directory `socdom2` for source code and executable, a subdirectory `Data` where input data and data files containing simulated populations are kept, and a subdirectory `build` used by `cmake` for files generated during building, including the executable `EvoDom`.


## Building the program

The CMake build system is used.
If it does not exist, create a build subdirectory in the project folder (`mkdir build`) and make it the current directory (`cd build`).
If desired, for a build from scratch, delete any previous content (`rm -rf *`).
Run CMake from the build directory. For a release build:
```
cmake -D CMAKE_BUILD_TYPE=Release ../
```
and for a debug build replace Release with Debug.
If this succeeds, i.e. if the `CMakeLists.txt` file in the project folder is processed without problems, build the program:
```
make
```
This should produce an executable in the `build` directory.


## Running

Make the Data directory current.
Assuming that the executable is called `EvoDom` and with an input file called `Run_n8v00s1nb.toml`, corresponding to case 1 in Table 2, run the program as
```
../build/EvoDom Run_n8v00s1nb.toml
```
Alternatively, using an R script file `Run_n8v00s1nb_run.R`, run the script as
```
Rscript Run_n8v00s1nb_run.R
```
where `Rscript` is the app for running R scripts.
You need to have `R` installed for this to work.


## Description of the evolutionary simulations

There is an input file, for instance `Run_n8v00s1nb.toml`, for each case, which typically simulates 5,000 generations, inputting the populations from, e.g., the HDF5 file `Run_n8v00s1nb.h5` and outputting to the same file.
Without an existing `Run_n8v00s1nb.h5` data file, the program can start by constructing individuals with genotypes from the allelic values given by `all0`in the input file.
To make this happen, use `read_from_file = false` in the input file.
There is an R script, e.g. `Run_n8v00s1nb_run.R`, which repeats such runs a number of times, for instance 25, and for each run computes statistics on the evolving leaning traits and adds a row to a TSV data file, e.g. `Run_n8v00s1nb_data.tsv`.

Each case is first run for many generations. When a seeming evolutionary equilibrium has been reached, 100 runs are kept in the summary data file, for instance `Run_n8v00s1nb_data.tsv`. These then represent evolution over `100*5,000 = 500,000` generations. The mean of all these means, for each learning trait, is then used for simulations of dominance hierarchy formation for groups of individuals that are adapted to a particular situation, e.g. using the input file `Run_n8v00s1nb0.toml` (these simulations are used for figures, e.g. for Fig. 3, in the paper).

### The cases in Table 2 of the paper

As a convention, an input file like `Run_n8v00s1nb.toml` corresponds to 8 interacting individuals per group, V0 = 0, shape of V1 given by V11 (see Fig. 1 in the paper), and no bystanders.
The basic input files for the cases in Table 2 are as follows:
1. `Run_n8v00s1nb.toml`
2. `Run_n8v01s1nb.toml`
3. `Run_n8v04s1nb.toml`
4. `Run_n8v09s1nb.toml`
5. `Run_n8v00s2nb.toml`
6. `Run_n8v01s2nb.toml`
7. `Run_n8v04s2nb.toml`
8. `Run_n8v09s2nb.toml`
9. `Run_n8v00s3nb.toml`
10. `Run_n8v01s3nb.toml`
11. `Run_n8v04s3nb.toml`
12. `Run_n8v09s3nb.toml`
13. `Run_n8v00s4nb.toml`
14. `Run_n8v01s4nb.toml`
15. `Run_n8v04s4nb.toml`
16. `Run_n8v09s4nb.toml`


## License

The `EvoDom` program runs evolutionary simulations of social hierarchy formation.

Copyright (C) 2021  Olof Leimar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

