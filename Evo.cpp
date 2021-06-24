#include "EvoCode.hpp"
#include <iostream>

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

int main(int argc, char* argv[])
{
    // Open input file and read indata
    EvoInpData eid(argv[1]);
    if (!eid.OK) {
        std::cout << "Input failed!" << "\n";
        return -1;
    }
    // Run the iteration
    Evo evo(eid);
    evo.Run();
    return 0;
}
