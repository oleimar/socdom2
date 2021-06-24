#include "hdf5code.hpp"
#include <iostream>
#include <string>

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

//*************************** Class h5R **********************************

void h5R::read_flt(std::string ds_name, v_type& dat)
{
    HighFive::DataSet ds = file.getDataSet(ds_name);
    ds.read(dat);
}

void h5R::read_flt_arr(std::string ds_name, vv_type& dat)
{
    HighFive::DataSet ds = file.getDataSet(ds_name);
    ds.read(dat);
}

void h5R::read_uint(std::string ds_name, ui_type& dat)
{
    HighFive::DataSet ds = file.getDataSet(ds_name);
    ds.read(dat);
}

void h5R::read_uint_arr(std::string ds_name, vui_type& dat)
{
    HighFive::DataSet ds = file.getDataSet(ds_name);
    ds.read(dat);
}

void h5R::read_int(std::string ds_name, i_type& dat)
{
    HighFive::DataSet ds = file.getDataSet(ds_name);
    ds.read(dat);
}


//*************************** Class h5W **********************************

void h5W::write_flt(std::string ds_name, const v_type& dat)
{
    HighFive::DataSet ds =
        file.createDataSet<flt>(ds_name, HighFive::DataSpace::From(dat));
    ds.write(dat);
}

void h5W::write_flt_arr(std::string ds_name, const vv_type& dat)
{
    HighFive::DataSet ds =
        file.createDataSet<flt>(ds_name, HighFive::DataSpace::From(dat));
    ds.write(dat);
}

void h5W::write_uint(std::string ds_name, const ui_type& dat)
{
    HighFive::DataSet ds =
        file.createDataSet<unsigned>(ds_name, HighFive::DataSpace::From(dat));
    ds.write(dat);
}

void h5W::write_uint_arr(std::string ds_name, const vui_type& dat)
{
    HighFive::DataSet ds =
        file.createDataSet<unsigned>(ds_name, HighFive::DataSpace::From(dat));
    ds.write(dat);
}

void h5W::write_int(std::string ds_name, const i_type& dat)
{
    HighFive::DataSet ds =
        file.createDataSet<int>(ds_name, HighFive::DataSpace::From(dat));
    ds.write(dat);
}
