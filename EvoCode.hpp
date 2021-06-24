#ifndef EVOCODE_HPP
#define EVOCODE_HPP

// NOTE: for easier debugging one can comment out the following
#ifdef _OPENMP
#define PARA_RUN
#endif

#include "Genotype.hpp"
#include "Phenotype.hpp"
#include "Individual.hpp"
#include "CmpGrp.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

// An individual has 11 genetically determined traits, w0, th0, g0, ga0, alphw,
// alphth, beta, v, gf, mf, mfcgr (see Phenotype.hpp), and there is one locus
// for each trait

//************************* Class EvoInpData ***************************

// This class is used to 'package' input data in a single place; the
// constructor extracts data from an input file

class EvoInpData {
public:
    using flt = double;         // the TOML class requires double
    using int_type = int64_t;   // the TOML class perhaps requires int64_t
    using v_type = std::vector<flt>;
    using vint_type = std::vector<int_type>;
    std::size_t max_num_thrds;  // Max number of threads to use
    unsigned num_loci;          // Number of loci in individual's genotype
    unsigned G;                 // Number of local groups (subpopulations)
    unsigned M;                 // Number of cohorts (age classes)
    unsigned n1;                // Size of age 1 cohort of males
    unsigned np1;               // Size of age 1 cohort of females
    unsigned CG0;               // Number of cmp-groups in pre-mating season
    unsigned CG1;               // Number of cmp-groups in mating season
    unsigned numI0;             // Number of interactions per pair per CG0
    unsigned numI1;             // Number of interactions per pair per CG1
    unsigned irndx;             // Maximum number of rounds in an interaction
    unsigned minrnds;           // Minimum number of rounds until dominance
    unsigned domrnds;           // AS (or SA) rounds until dominance is clear
    unsigned drawrnds;          // SS rounds until draw
    unsigned numV1;             // Number of values for mating groups
    unsigned numg0;             // Number of pre-mating group sizes
    unsigned numg1;             // Number of mating group sizes
    unsigned numyrs;            // Number of generation to simulate
    bool bystl;                 // Whether there is bystander learning
    flt V0;                     // Baseline fitness payoff in mating group
    flt cdmg;                   // Parameter for cost from damage
    flt c0;                     // Parameter for pre to season mortality
    flt c1;                     // Parameter for mating-season mortality
    flt c2;                     // Parameter for between-year mortality
    flt mu0;                    // Parameter for pre to season mortality
    flt mu1;                    // Parameter for mating-season mortality
    flt mu2;                    // Parameter for between-year mortality
    flt eta0;                   // Parameter for pre to season damage healing
    flt eta1;                   // Parameter for mating-season damage healing
    flt eta2;                   // Parameter for end of season damage healing
    flt a1;                     // Parameter for 'own perceived quality'
    flt b1;                     // Parameter for 'displayed quality'
    flt sq;                     // Between-individual SD of quality variation
    flt sqm;                    // Within-cohort SD of quality variation
    flt sigma;                  // Parameter for 'relative quality' obs
    flt sigp;                   // Parameter for 'penalty of AA interaction'
    flt pmax;                   // Maximum value for probability to use A
    flt betelo;                 // parameter for Elo score updates
    v_type V1;                  // Fitness payoffs in mating groups
    v_type Qm;                  // Age class qualities (fighting abilities)
    vint_type g0;               // Pre-mating group sizes
    v_type pg0;                 // Probability of pre-mating group size
    vint_type g1;               // Mating group sizes
    v_type pg1;                 // Probability of mating group size
    v_type mut_rate;            // Probability of mutation at each locus
    v_type SD;                  // SD of mutational increments at each locus
    v_type max_val;             // Maximal allelic value at each locus
    v_type min_val;             // Minimal allelic value at each locus
    v_type rho;                 // Recombination rates
    v_type all0;                // Starting allelic values (if not from file)
    bool hist;                  // Whether to compute and save history
    bool read_from_file;        // Whether to read population from file
    std::string h5InName;       // File name for input of population
    std::string h5OutName;      // File name for output of population
    std::string h5HistName;     // File name for output of history

    std::string InpName;  // Name of input data file
    bool OK;              // Whether input data has been successfully read

    EvoInpData(const char* filename);
};


//***************************** Class Evo ******************************

class Evo {
public:
    // types needed to define individual
    using mut_rec_type = MutRec<MutIncrNorm<>>;
    using gam_type = Gamete<mut_rec_type>;
    using gen_type = Diplotype<gam_type>;
    using phen_type = Phenotype<gen_type>;
    using lp_type = typename phen_type::lp_type;
    using ind_type = Individual<gen_type, phen_type>;
    using stat_type = CmpStat<phen_type>;
    // use std::vector containers for male and female populations
    using vind_type = std::vector<ind_type>;
    using vgen_type = std::vector<gen_type>;
    using vph_type = std::vector<phen_type>;
    using cg_type = CmpGrp<phen_type>;
    using flt = float;
    using v_type = std::vector<flt>;
    using vs_type = std::vector<stat_type>;
    using ui_type = std::vector<unsigned>;
    using rand_eng = std::mt19937;
    using vre_type = std::vector<rand_eng>;
    using vmr_type = std::vector<mut_rec_type>;
    using rand_ui = std::uniform_int_distribution<unsigned>;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_norm = std::normal_distribution<flt>;
    using rand_discr = std::discrete_distribution<int>;
    Evo(const EvoInpData& eid);
    void Run();
    void h5_read_pop(const std::string& infilename);
    void h5_write_pop(const std::string& outfilename) const;
    void h5_write_hist(const std::string& histfilename) const;
private:
    void SelectReproduce(vind_type& lmp, vgen_type& lfp,
                         vgen_type& lmo, vgen_type& lfo,
                         mut_rec_type& mr);
    void ForgetCgrp(phen_type& ph);
    void Forget(phen_type& ph);
    void UpdateAgeStr(unsigned yr1, rand_eng& eng);
    void Set_stat_tm();

    EvoInpData id;
    unsigned num_loci;
    unsigned G;
    unsigned M;
    unsigned n1;
    unsigned np1;
    unsigned Nl;
    unsigned Npl;
    unsigned N1;
    unsigned Np1;
    unsigned N;
    unsigned Np;
    unsigned CG0;
    unsigned CG1;
    unsigned numI0;
    unsigned numI1;
    unsigned irndx;
    unsigned minrnds;
    unsigned domrnds;
    unsigned drawrnds;
    unsigned numV1;
    unsigned numg0;
    unsigned numg1;
    unsigned numyrs;
    bool bystl;
    flt V0;
    flt cdmg;
    flt c0;
    flt c1;
    flt c2;
    flt mu0;
    flt mu1;
    flt mu2;
    flt eta0;
    flt eta1;
    flt eta2;
    flt a1;
    flt b1;
    flt sq;
    flt sqm;
    flt sigma;
    flt sigp;
    flt pmax;
    flt betelo;
    v_type V1;
    v_type Qm;
    ui_type g0;
    v_type pg0;
    ui_type g1;
    v_type pg1;
    bool hist;
    std::size_t num_thrds;
    ui_type sds;
    vre_type vre;
    vmr_type vmr;
    vind_type m_pop;
    vgen_type f_pop;
    vgen_type m_offspr;
    vgen_type f_offspr;
    vs_type stat;
};

#endif // EVOCODE_HPP
