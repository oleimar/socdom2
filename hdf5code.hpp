#ifndef HDF5CODE_HPP
#define HDF5CODE_HPP

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <string>

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

//************************* Class hdf5R **********************************

// This class is used to read a population of individuals from a HDF5 file. The
// file is assumed to have number of datasets in its root group, with names and
// content (data spaces) corresponding to the data fields in the individuals.
// The first dimension of each dataset is assumed to be equal to the number of
// individuals in the population.

class h5R {
public:
    using flt = float;
    using v_type = std::vector<flt>;
    using vv_type = std::vector<v_type>;
    using ui_type = std::vector<unsigned>;
    using vui_type = std::vector<ui_type>;
    using i_type = std::vector<int>;
    // constructor opens file for reading
    h5R(std::string in_name) : file(in_name, HighFive::File::ReadOnly) {}
    void read_flt(std::string ds_name, v_type& dat);
    void read_flt_arr(std::string ds_name, vv_type& dat);
    void read_uint(std::string ds_name, ui_type& dat);
    void read_uint_arr(std::string ds_name, vui_type& dat);
    void read_int(std::string ds_name, i_type& dat);
private:
    HighFive::File file;
};

//*************************** Class h5W **********************************

// This class is used to write a population of individuals to a HDF5 file,
// overwriting any content if the file already exists. The data is written as a
// number of datasets in the files root group, with names and content
// (data spaces) corresponding to the data fields in the individuals. The first
// dimension of each dataset is equal to the number of individuals in the
// population.

class h5W {
public:
    using flt = float;
    using v_type = std::vector<flt>;
    using vv_type = std::vector<v_type>;
    using ui_type = std::vector<unsigned>;
    using vui_type = std::vector<ui_type>;
    using i_type = std::vector<int>;
    // constructor opens file for writing, truncating any previous file/content
    h5W(std::string out_name) :
        file(out_name,
             HighFive::File::ReadWrite |
             HighFive::File::Create |
             HighFive::File::Truncate) {}
    void write_flt(std::string ds_name, const v_type& dat);
    void write_flt_arr(std::string ds_name, const vv_type& dat);
    void write_uint(std::string ds_name, const ui_type& dat);
    void write_uint_arr(std::string ds_name, const vui_type& dat);
    void write_int(std::string ds_name, const i_type& dat);
private:
    HighFive::File file;
};

#endif // HDF5CODE_HPP
