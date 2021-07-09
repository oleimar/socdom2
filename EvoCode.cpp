#include "cpptoml.h"   // to read input parameters from TOML file
#include "EvoCode.hpp"
#include "hdf5code.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

#ifdef PARA_RUN
#include <omp.h>
#endif

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

//************************** Read and ReadArr ****************************

// convenience functions to read from TOML input file

// this template function can be used for any type of single value
template<typename T>
void Get(std::shared_ptr<cpptoml::table> infile,
         T& value, const std::string& name)
{
    auto val = infile->get_as<T>(name);
    if (val) {
        value = *val;
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}

// this template function can be used for a vector or array (but there is no
// checking how many elements are read)
template<typename It>
void GetArr(std::shared_ptr<cpptoml::table> infile,
            It beg, const std::string& name)
{
    using valtype = typename std::iterator_traits<It>::value_type;
    auto vp = infile->get_array_of<valtype>(name);
    if (vp) {
        std::copy(vp->begin(), vp->end(), beg);
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}


//************************** class EvoInpData ****************************

EvoInpData::EvoInpData(const char* filename) :
      OK(false)
{
    auto idat = cpptoml::parse_file(filename);
    Get(idat, max_num_thrds, "max_num_thrds");
    Get(idat, num_loci, "num_loci");
    Get(idat, G, "G");
    Get(idat, M, "M");
    Get(idat, n1, "n1");
    Get(idat, np1, "np1");
    Get(idat, CG0, "CG0");
    Get(idat, CG1, "CG1");
    Get(idat, numI0, "numI0");
    Get(idat, numI1, "numI1");
    Get(idat, irndx, "irndx");
    Get(idat, minrnds, "minrnds");
    Get(idat, domrnds, "domrnds");
    Get(idat, drawrnds, "drawrnds");
    Get(idat, numV1, "numV1");
    Get(idat, numg0, "numg0");
    Get(idat, numg1, "numg1");
    Get(idat, numyrs, "numyrs");
    Get(idat, bystl, "bystl");
    Get(idat, V0, "V0");
    Get(idat, cdmg, "cdmg");
    Get(idat, c0, "c0");
    Get(idat, c1, "c1");
    Get(idat, c2, "c2");
    Get(idat, mu0, "mu0");
    Get(idat, mu1, "mu1");
    Get(idat, mu2, "mu2");
    Get(idat, eta0, "eta0");
    Get(idat, eta1, "eta1");
    Get(idat, eta2, "eta2");
    Get(idat, a1, "a1");
    Get(idat, b1, "b1");
    Get(idat, sq, "sq");
    Get(idat, sqm, "sqm");
    Get(idat, sigma, "sigma");
    Get(idat, sigp, "sigp");
    Get(idat, pmax, "pmax");
    Get(idat, betelo, "betelo");
    V1.resize(numV1);
    GetArr(idat, V1.begin(), "V1");
    Qm.resize(M);
    GetArr(idat, Qm.begin(), "Qm");
    g0.resize(numg0);
    GetArr(idat, g0.begin(), "g0");
    pg0.resize(numg0);
    GetArr(idat, pg0.begin(), "pg0");
    g1.resize(numg1);
    GetArr(idat, g1.begin(), "g1");
    pg1.resize(numg1);
    GetArr(idat, pg1.begin(), "pg1");
    mut_rate.resize(num_loci);
    GetArr(idat, mut_rate.begin(), "mut_rate");
    SD.resize(num_loci);
    GetArr(idat, SD.begin(), "SD");
    max_val.resize(num_loci);
    GetArr(idat, max_val.begin(), "max_val");
    min_val.resize(num_loci);
    GetArr(idat, min_val.begin(), "min_val");
    rho.resize(num_loci);
    GetArr(idat, rho.begin(), "rho");
    Get(idat, hist, "hist");
    Get(idat, read_from_file, "read_from_file");
    if (read_from_file) {
        Get(idat, h5InName, "h5InName");
    } else {
        all0.resize(num_loci);
        GetArr(idat, all0.begin(), "all0");
    }
    Get(idat, h5OutName, "h5OutName");
    if (hist) {
        Get(idat, h5HistName, "h5HistName");
    }
    InpName = std::string(filename);
    OK = true;
}


//****************************** Class Evo *****************************

Evo::Evo(const EvoInpData& eid) :
    id{eid},
    num_loci{id.num_loci},
    G{id.G},
    M{id.M},
    n1{id.n1},
    np1{id.np1},
    Nl{n1*M},
    Npl{np1*M},
    N1{n1*G},
    Np1{np1*G},
    N{Nl*G},
    Np{Npl*G},
    CG0{id.CG0},
    CG1{id.CG1},
    numI0{id.numI0},
    numI1{id.numI1},
    irndx{id.irndx},
    minrnds{id.minrnds},
    domrnds{id.domrnds},
    drawrnds{id.drawrnds},
    numV1{id.numV1},
    numg0{id.numg0},
    numg1{id.numg1},
    numyrs{id.numyrs},
    bystl{id.bystl},
    V0{static_cast<flt>(id.V0)},
    cdmg{static_cast<flt>(id.cdmg)},
    c0{static_cast<flt>(id.c0)},
    c1{static_cast<flt>(id.c1)},
    c2{static_cast<flt>(id.c2)},
    mu0{static_cast<flt>(id.mu0)},
    mu1{static_cast<flt>(id.mu1)},
    mu2{static_cast<flt>(id.mu2)},
    eta0{static_cast<flt>(id.eta0)},
    eta1{static_cast<flt>(id.eta1)},
    eta2{static_cast<flt>(id.eta2)},
    a1{static_cast<flt>(id.a1)},
    b1{static_cast<flt>(id.b1)},
    sq{static_cast<flt>(id.sq)},
    sqm{static_cast<flt>(id.sqm)},
    sigma{static_cast<flt>(id.sigma)},
    sigp{static_cast<flt>(id.sigp)},
    pmax{static_cast<flt>(id.pmax)},
    betelo{static_cast<flt>(id.betelo)},
    V1{id.V1.begin(), id.V1.end()},
    Qm{id.Qm.begin(), id.Qm.end()},
    g0{id.g0.begin(), id.g0.end()},
    pg0{id.pg0.begin(), id.pg0.end()},
    g1{id.g1.begin(), id.g1.end()},
    pg1{id.pg1.begin(), id.pg1.end()},
    hist{id.hist},
    num_thrds{1}
{
    // decide on number of threads for parallel processing
#ifdef PARA_RUN
    num_thrds = omp_get_max_threads();
    if (num_thrds > id.max_num_thrds) num_thrds = id.max_num_thrds;
    std::cout << "Number of threads: "
              << num_thrds << '\n';
#endif
    // generate one seed for each thread
    sds.resize(num_thrds);
    std::random_device rd;
    for (unsigned i = 0; i < num_thrds; ++i) {
        // set up thread-local to be random number engines
        sds[i] = rd();
        vre.push_back(rand_eng(sds[i]));
    }
    for (unsigned i = 0; i < num_thrds; ++i) {
        // set up thread-local to be mutation records, with thread-local engine
        // and parameters controlling mutation, segregation and recombination
        rand_eng& eng = vre[i];
        mut_rec_type mr(eng, num_loci);
        for (unsigned l = 0; l < num_loci; ++l) {
            mr.mut_rate[l] = static_cast<flt>(id.mut_rate[l]);
            mr.SD[l] = static_cast<flt>(id.SD[l]);
            mr.max_val[l] = static_cast<flt>(id.max_val[l]);
            mr.min_val[l] = static_cast<flt>(id.min_val[l]);
            mr.rho[l] = static_cast<flt>(id.rho[l]);
        }
        vmr.push_back(mr);
    }

    int threadn = 0;

    // Note concerning thread safety: in order to avoid possible problems with
    // multiple threads, the std::vector containers m_pop, f_pop, and stat are
    // allocated once and for all here, and thread-local data are then copied
    // into position in these (thus avoiding potentially unsafe push_back and
    // insert).

    // Create N male "placeholder individuals" in male population (based on the
    // constructor these are not alive). The convention is that individuals in
    // local group k (k = 0, ..., G-1) are found as alive phenotypes at
    // m_pop[j] with j = k*Nl, k*Nl+1, ..., k*Nl+Nl-1, i.e. j = k*Nl+i with i =
    // 0, ..., Nl-1. These values of i (i.e., i = 0, ..., Nl-1) correspond to
    // the original values of inum in each group gnum = k, assigned to
    // individuals when read_from_file is false. Further, when an individual
    // with age beyond the maximum is replaced by a yearling (in UpdateAgeStr),
    // the new individual 'inherits' the old inum.
    gam_type gam(num_loci);
    ind_type m_indi(Nl, gam);
    m_pop.resize(N, m_indi);
    // create Np female "placeholder genotypes" as female population
    gen_type gen(gam);
    f_pop.resize(Np, gen);
    // also create male and female offspring placeholders
    m_offspr.resize(N1, gen);
    f_offspr.resize(Np1, gen);
    // history stats
    if (hist) {
        stat.reserve(irndx);
    }

    // check if population data should be read from file
    if (id.read_from_file) {
        // Read_pop(id.InName);
        h5_read_pop(id.h5InName);
        UpdateAgeStr(0, vre[threadn]);
    } else {
        rand_eng& eng = vre[threadn];
        rand_norm nrq(0, sq);
        rand_norm nrqm(0, sqm);
        // construct all individuals with the same genotypes
        gam_type gam(num_loci); // starting gamete
        for (unsigned l = 0; l < num_loci; ++l) {
            gam.gamdat[l] = static_cast<flt>(id.all0[l]);
        }
        unsigned j = 0;
        unsigned jp = 0;
        for (unsigned k = 0; k < G; ++k) { // local groups
            for (unsigned m = 0; m < M; ++m) { // cohorts in groups
                // construct n1 males (genotype and phenotype) in each cohort
                for (unsigned i = 0; i < n1; ++i) { // males in cohort
                    ind_type ind(Nl, gam);
                    phen_type& ph = ind.phenotype;
                    ph.dq = nrq(eng);
                    ph.q = Qm[m] + ph.dq + nrqm(eng);
                    ph.y = a1*ph.q;
                    ph.z = b1*ph.q;
                    ph.gnum = k;         // set local group number
                    ph.age = m;          // set age
                    ph.dage = M;         // death age if survival over all ages
                    ph.inum = m*n1 + i;  // set individual number
                    ph.female = false;
                    ph.alive = true;
                    m_pop[j++] = ind;
                }
                // construct np1 female genotypes in each cohort
                for (unsigned ip = 0; ip < np1; ++ip) { // females in cohort
                    gen_type gen(gam);
                    f_pop[jp++] = gen;
                }
            }
        }
    }
}

void Evo::Run()
{
    Timer timer(std::cout);
    timer.Start();
    ProgressBar PrBar(std::cout, numyrs);
    // run through years
    // Time sequence within a year: (i)
    for (unsigned yr = 0; yr < numyrs; ++yr) {
        // use parallel for processing over the male-male interactions in local
        // groups, within a year
#pragma omp parallel for num_threads(num_thrds)
        for (unsigned k = 0; k < G; ++k) {
#ifdef PARA_RUN
            int threadn = omp_get_thread_num();
#else
            int threadn = 0;
#endif
            // thread-local random number engine
            rand_eng& eng = vre[threadn];
            // thread-local mutation record
            mut_rec_type& mr = vmr[threadn];
            // distribution needed
            rand_uni uni(0, 1);
            // thread-local container for local group males
            vind_type lmpop;
            lmpop.reserve(Nl);
            for (unsigned i = 0; i < Nl; ++i) {
                unsigned j = k*Nl + i;
                lmpop.push_back(m_pop[j]);
            }
            // thread-local container for local group male phenotypes
            vph_type lgph;
            lgph.reserve(Nl);
            // vector of indices in lmpop of males in container lgph
            ui_type lgi;
            lgi.reserve(Nl);
            for (unsigned i = 0; i < Nl; ++i) {
                phen_type& ph = lmpop[i].phenotype;
                if (ph.alive) { // only include alive individuals
                    lgph.push_back(ph);
                    lgi.push_back(i);
                }
            }
            // get history only for single threaded and final M years
            bool lhist = false;
            if (num_thrds == 1 && yr >= numyrs - M && hist) {
                lhist = true;
            }
            // run through competition groups in pre-mating season
            for (unsigned cgrp = 0; cgrp < CG0; ++cgrp) {
                unsigned g = g0[0]; // pre-mating group size
                if (numg0 > 0) {
                    // randomly select pre-mating group size
                    rand_discr dscr(pg0.begin(), pg0.end());
                    g = g0[dscr(eng)];
                }
                ui_type i0;
                for (unsigned i = 0; i < lgph.size(); ++i) {
                    if (lgph[i].alive) {
                        i0.push_back(i);
                    }
                }
                if (i0.size() > 0) {
                    if (i0.size() < g) {
                        // if there are too few males, adjust g
                        g = i0.size();
                    }
                    // randomly select alive males from lgph to be in
                    // pre-mating group; the indices of these males are in i0;
                    // pmgph contains the phenotypes of the males in the
                    // pre-mating group and pmgi the indices in local group
                    // lgph of these males (note that std::sample is 'stable',
                    // i.e. preserves the relative ordering)
                    ui_type pmgi;
                    pmgi.reserve(g);
                    std::sample(i0.begin(), i0.end(),
                        std::back_inserter(pmgi), g, eng);
                    vph_type pmgph;
                    pmgph.reserve(g);
                    for (unsigned i = 0; i < g; ++i) {
                        pmgph.push_back(lgph[pmgi[i]]);
                    }
                    // construct pre-mating group
                    unsigned cper = 0;
                    cg_type cg(g, numI0,
                        irndx, minrnds, domrnds, drawrnds, yr,
                        cper, cgrp, V0, cdmg, sigma, sigp, pmax,
                        betelo, V1, pmgph, bystl, lhist);
                    cg.Interact(eng);
                    const vph_type& memb = cg.Get_memb();
                    // put back these males from the pre-mating group into
                    // lgph
                    for (unsigned i = 0; i < g; ++i) {
                        lgph[pmgi[i]] = memb[i];
                    }
                    if (lhist) {
                        // append histories
                        const vs_type& st = cg.Get_stat();
                        stat.insert(stat.end(), st.begin(), st.end());
                    }
                }
            }
            // damage healing and survival from pre-mating to mating season;
            // run through lgph
            for (unsigned i = 0; i < lgph.size(); ++i) {
                phen_type& ph = lgph[i];
                if (ph.alive) {
                    // first do healing
                    ph.dmg *= 1 - eta0;
                    // then do survival
                    if (uni(eng) >= std::exp(-mu0 - c0*ph.dmg)) {
                        ph.alive = false;
                        ph.dage = ph.age;
                    }
                }
            }
            // run through competition groups in mating season
            for (unsigned cgrp = 0; cgrp < CG1; ++cgrp) {
                unsigned g = g1[0]; // mating group size
                if (numg1 > 0) {
                    // randomly select mating group size
                    rand_discr dscr(pg1.begin(), pg1.end());
                    g = g1[dscr(eng)];
                }
                ui_type i1;
                for (unsigned i = 0; i < lgph.size(); ++i) {
                    if (lgph[i].alive) {
                        i1.push_back(i);
                    }
                }
                if (i1.size() > 0) {
                    if (i1.size() < g) {
                        // if there are too few males, adjust g
                        g = i1.size();
                    }
                    // randomly select alive males from lgph to be in mating
                    // group (std::sample is 'stable')
                    ui_type mgi;
                    mgi.reserve(g);
                    std::sample(i1.begin(), i1.end(),
                        std::back_inserter(mgi), g, eng);
                    // mgph: phenotypes of males in mating group; mgi: indices
                    // in local group lgph of these males
                    vph_type mgph;
                    mgph.reserve(g);
                    for (unsigned i = 0; i < g; ++i) {
                        mgph.push_back(lgph[mgi[i]]);
                    }
                    // construct mating group
                    unsigned cper = 1;
                    cg_type cg(g, numI1,
                        irndx, minrnds, domrnds, drawrnds, yr,
                        cper, cgrp, V0, cdmg, sigma, sigp, pmax,
                        betelo, V1, mgph, bystl, lhist);
                    cg.Interact(eng);
                    const vph_type& memb = cg.Get_memb();
                    // Put males in the mating group back into lgph and apply
                    // healing, survival, and forgetting
                    for (unsigned i = 0; i < g; ++i) {
                        lgph[mgi[i]] = memb[i];
                        phen_type& ph = lgph[mgi[i]];
                        if (ph.alive) {
                            // first do healing
                            ph.dmg *= 1 - eta1;
                            // then do survival
                            if (uni(eng) >= std::exp(-mu1 - c1*ph.dmg)) {
                                ph.alive = false;
                                ph.dage = ph.age;
                            }
                        }
                        if (ph.alive) {
                            ForgetCgrp(ph);
                        }
                    }
                    if (lhist) {
                        // append histories
                        const vs_type& st = cg.Get_stat();
                        stat.insert(stat.end(), st.begin(), st.end());
                    }
                } else {
                    // NOTE: all males are dead, and to avoid failure of
                    // reproduction the current year, allocate expected
                    // reproduction to all lowest age males in lgph
                    for (unsigned i = 0; i < lgph.size(); ++i) {
                        if (lgph[i].age == 0) {
                            lgph[i].erepr += 1;
                        }
                    }
                }
            }
            // damage healing and survival from mating season, over
            // reproduction and to the pre-mating season next year; run through
            // lgph; check survival
            for (unsigned i = 0; i < lgph.size(); ++i) {
                phen_type& ph = lgph[i];
                if (ph.alive) {
                    // first do healing
                    ph.dmg *= 1 - eta2;
                    // then do survival
                    if (uni(eng) >= std::exp(-mu2 - c2*ph.dmg)) {
                        ph.alive = false;
                        ph.dage = ph.age;
                    }
                }
            }
            // put male phenotypes from lgph into the correct local group
            // individuals
            for (unsigned i = 0; i < lgph.size(); ++i) {
                lmpop[lgi[i]].phenotype = lgph[i];
            }
            // thread-local container for local group females
            vgen_type lfpop;
            lfpop.reserve(Nl);
            for (unsigned i = 0; i < Npl; ++i) {
                unsigned j = k*Npl + i;
                lfpop.push_back(f_pop[j]);
            }
            // reproduction in local group: produce n1 males and np1 females
            vgen_type lmo;
            vgen_type lfo;
            SelectReproduce(lmpop, lfpop, lmo, lfo, mr);
            // copy local offspring to global containers
            if (lmo.size() == n1 && lfo.size() == np1) {
                for (unsigned i = 0; i < n1; ++i) {
                    unsigned j = k*n1 + i;
                    m_offspr[j] = lmo[i];
                }
                for (unsigned i = 0; i < np1; ++i) {
                    unsigned j = k*np1 + i;
                    f_offspr[j] = lfo[i];
                }
            }
            // copy males from lmpop back to m_pop
            for (unsigned i = 0; i < Nl; ++i) {
                unsigned j = k*Nl + i;
                m_pop[j] = lmpop[i];
            }
        } // end of parallel for (over local groups)
        // if not final year, update age structure
        if (yr < numyrs - 1) {
            UpdateAgeStr(yr + 1, vre[0]);
        }
        // all set to start next year
        ++PrBar;
    }

    PrBar.Final();
    timer.Stop();
    timer.Display();
    h5_write_pop(id.h5OutName);
    if (hist) {
        Set_stat_tm();
        h5_write_hist(id.h5HistName);
    }
}

void Evo::SelectReproduce(vind_type& lmp, vgen_type& lfp,
                          vgen_type& lmo, vgen_type& lfo,
                          mut_rec_type& mr)
{
    unsigned nm = lmp.size();
    unsigned nf = lfp.size();
    if (nm > 0 && nf > 0) {
        // get discrete distribution with male expected reproduction as weights
        v_type wei(nm);
        flt tot_wei = 0;
        for (unsigned i = 0; i < nm; ++i) {
            const phen_type& ph = lmp[i].phenotype;
            wei[i] = ph.erepr;
            tot_wei += wei[i];
        }
        if (tot_wei > 0) {
            rand_discr dsm(wei.begin(), wei.end());
            // get uniform distribution for female genotypes
            rand_ui dsf(0, nf - 1);
            // get n1 male offspring
            lmo.reserve(n1);
            for (unsigned j = 0; j < n1; ++j) {
                // find mother genotype for individual to be constructed
                unsigned imat = dsf(mr.eng);
                const gen_type& matind = lfp[imat];
                // find father for individual to be constructed
                unsigned ipat = dsm(mr.eng);
                ind_type& patind = lmp[ipat];
                // construct offspring genotype
                gen_type offspr(matind.GetGamete(mr), patind.GetGamete(mr));
                // append new male genotype to lmo
                lmo.push_back(offspr);
                // update father's record of reproductive success
                patind.phenotype.nOffspr += 1;
            }
            // get np1 female offspring
            lfo.reserve(np1);
            for (unsigned j = 0; j < np1; ++j) {
                // find mother genotype for individual to be constructed
                unsigned imat = dsf(mr.eng);
                const gen_type& matind = lfp[imat];
                // find father for individual to be constructed
                unsigned ipat = dsm(mr.eng);
                ind_type& patind = lmp[ipat];
                // construct offspring genotype
                gen_type offspr(matind.GetGamete(mr), patind.GetGamete(mr));
                // append new male genotype to lfo
                lfo.push_back(offspr);
                // update father's record of reproductive success
                patind.phenotype.nOffspr += 1;
            }
        }
    }
}

void Evo::ForgetCgrp(phen_type& ph)
{
    // move learning parameters of the phenotype towards 'naive' state
    lp_type& lp = ph.lp;
    // deviations of w and th from initial value are
    // multiplied by memory factor
    if (ph.mfcgr < 1) {
        for (unsigned j = 0; j < lp.w.size(); ++j) {
            lp.w[j] = ph.w0 + ph.mfcgr*(lp.w[j] - ph.w0);
            lp.th[j] = ph.th0 + ph.mfcgr*(lp.th[j] - ph.th0);
        }
    }
}

void Evo::Forget(phen_type& ph)
{
    // move learning parameters of the phenotype towards 'naive' state
    lp_type& lp = ph.lp;
    // deviations of w and th from initial value are
    // multiplied by memory factor
    for (unsigned j = 0; j < lp.w.size(); ++j) {
        // lp.w[j] = ph.w0 + ph.mf*(lp.w[j] - ph.w0);
        // lp.th[j] = ph.th0 + ph.mf*(lp.th[j] - ph.th0);
        lp.w[j] *= ph.mf;
        lp.th[j] *= ph.mf;
    }
}

void Evo::UpdateAgeStr(unsigned yr1, rand_eng& eng)
{
    rand_uni uni(0, 1);
    rand_norm nrq(0, sq);
    rand_norm nrqm(0, sqm);
    // first increase the age of each individual in m_pop, and put those
    // individuals who reach beyond maximum age as dead;  for those below
    // maximum age, adjust the fighting ability, and put erepr for all to zero
    for (unsigned i = 0; i < N; ++i) {
        phen_type& ph = m_pop[i].phenotype;
        ph.age += 1;
        if (ph.age >= M) {
            ph.alive = false;
        } else {
            // update fighting ability and apply forgetting for alive
            // individuals
            if (ph.alive) {
                ph.q = Qm[ph.age] + ph.dq + nrqm(eng);
                ph.y = a1*ph.q;
                ph.z = b1*ph.q;
                Forget(ph);
            }
        }
        ph.erepr = 0;
    }
    // then copy male offspring genotypes from a reshuffled m_offspr into the
    // positions of individuals who reach beyond maximum age and adjust
    // phenotypes
    ui_type moi;
    moi.reserve(N1);
    for (unsigned i = 0; i < N1; ++i) {
        moi.push_back(i);
    }
    // randomly shuffle indices
    std::shuffle(moi.begin(), moi.end(), eng);
    // replace males with ages beyond maximum with male offspring, and update
    // the learning parameters of other local group members
    unsigned j = 0;
    for (unsigned i = 0; i < N; ++i) {
        phen_type& ph = m_pop[i].phenotype;
        if (ph.age >= M) {
            gen_type& gen = m_offspr[moi[j]];
            m_pop[i].genotype = gen;
            // keep individual and local group numbers
            unsigned inum = ph.inum;
            unsigned gnum = ph.gnum;
            ph.Assign(gen);
            // count number of replacements
            ++j;
            // set aspects of phenotype not set correctly by Assign
            ph.dq = nrq(eng);
            ph.q = Qm[ph.age] + ph.dq + nrqm(eng); // ph.age is zero
            ph.y = a1*ph.q;
            ph.z = b1*ph.q;
            ph.dage = M;
            ph.inum = inum;
            ph.yr1 = yr1;
            ph.gnum = gnum;
            ph.alive = true;
            // update relevant learning parameters of older local group males
            for (unsigned k = gnum*Nl; k < (gnum + 1)*Nl; ++k) {
                phen_type& phl = m_pop[k].phenotype;
                if (phl.age > 0 && phl.age < M && phl.inum != inum) {
                    lp_type& lp = phl.lp;
                    lp.w[inum] = phl.w0;
                    lp.th[inum] = phl.th0;
                    lp.wins[inum] = 0;
                    lp.losses[inum] = 0;
                    lp.draws[inum] = 0;
                }
            }
        }
    }
    // then copy (move) females in f_pop to correspond to increased age
    for (unsigned k = 0; k < G; ++ k) {
        // in each group (of size Npl = np1*M), the first np1 positions in
        // f_pop of the group are the youngest cohort, etc.
        for (unsigned i = Npl - 1; i > np1 - 1; --i) {
            unsigned j = k*Npl + i;
            f_pop[j] = f_pop[j - np1];
        }
    }
    // then copy female offspring in a reshuffled f_offspr into the positions
    // corresponding to the youngest age
    ui_type foi;
    foi.reserve(Np1);
    for (unsigned i = 0; i < Np1; ++i) {
        foi.push_back(i);
    }
    // randomly shuffle indices
    std::shuffle(foi.begin(), foi.end(), vre[0]);
    for (unsigned k = 0; k < G; ++ k) {
        for (unsigned i = 0; i < np1; ++i) {
            f_pop[k*Npl + i] = f_offspr[foi[k*np1 + i]];
        }
    }
}

void Evo::Set_stat_tm()
{
    // compute and assign tminyr and tm fields of history in stat
    unsigned hlen = stat.size();
    if (hlen > 0) {
        unsigned kh = 0; // counter for stat elements
        unsigned ncgrp = CG0 + CG1; // number of competition groups per year
        flt dtcgrp = 1.0/ncgrp; // time increment per competition group
        // run through history
        while (kh < hlen) {
            stat_type& st = stat[kh];
            unsigned yr = st.yr;
            unsigned cper = st.cper;
            unsigned cgrp = st.cgrp;
            unsigned cnum = st.cnum;
            unsigned g = st.g;
            // number of interactions per pair
            unsigned nI = (cper == 0) ? numI0 : numI1;
            // number of interactions in current competition group
            unsigned nInt = nI*g*(g - 1)/2;
            flt dInt = 1.0/nInt;
            flt tminyr = (cper*CG0 + cgrp)*dtcgrp + cnum*dInt*dtcgrp;
            st.tminyr = tminyr;
            st.tm = st.yr + tminyr;
            ++kh;
        }
    }
}

void Evo::h5_read_pop(const std::string& infilename)
{
    // read data and put in pop
    h5R h5(infilename);
    // male gametes
    std::vector<v_type> mgams(N, v_type(num_loci));
    // read maternal gametes of males
    h5.read_flt_arr("MlMatGam", mgams);
    for (unsigned i = 0; i < N; ++i) {
        gam_type& gam = m_pop[i].genotype.mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = mgams[i][l];
        }
    }
    // read paternal gametes of males
    h5.read_flt_arr("MlPatGam", mgams);
    for (unsigned i = 0; i < N; ++i) {
        gam_type& gam = m_pop[i].genotype.pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = mgams[i][l];
        }
    }
    // female gametes
    std::vector<v_type> fgams(Np, v_type(num_loci));
    // read maternal gametes of females
    h5.read_flt_arr("FlMatGam", fgams);
    for (unsigned i = 0; i < Np; ++i) {
        gam_type& gam = f_pop[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = fgams[i][l];
        }
    }
    // read paternal gametes of females
    h5.read_flt_arr("FlPatGam", fgams);
    for (unsigned i = 0; i < Np; ++i) {
        gam_type& gam = f_pop[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = fgams[i][l];
        }
    }
    // male offspring gametes
    std::vector<v_type> mogams(N1, v_type(num_loci));
    // read maternal gametes of male offspring
    h5.read_flt_arr("MloMatGam", mogams);
    for (unsigned i = 0; i < N1; ++i) {
        gam_type& gam = m_offspr[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = mogams[i][l];
        }
    }
    // read paternal gametes of male offspring
    h5.read_flt_arr("MloPatGam", mogams);
    for (unsigned i = 0; i < N1; ++i) {
        gam_type& gam = m_offspr[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = mogams[i][l];
        }
    }
    // female offspring gametes
    std::vector<v_type> fogams(Np1, v_type(num_loci));
    // read maternal gametes of female offspring
    h5.read_flt_arr("FloMatGam", fogams);
    for (unsigned i = 0; i < Np1; ++i) {
        gam_type& gam = f_offspr[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = fogams[i][l];
        }
    }
    // read paternal gametes of female offspring
    h5.read_flt_arr("FloPatGam", fogams);
    for (unsigned i = 0; i < Np1; ++i) {
        gam_type& gam = f_offspr[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = fogams[i][l];
        }
    }
    v_type fval(N);
    // w0
    h5.read_flt("w0", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.w0 = fval[i];
    }
    // th0
    h5.read_flt("th0", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.th0 = fval[i];
    }
    // g0
    h5.read_flt("g0", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.g0 = fval[i];
    }
    // ga0
    h5.read_flt("ga0", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.ga0 = fval[i];
    }
    // alphw
    h5.read_flt("alphw", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.alphw = fval[i];
    }
    // alphth
    h5.read_flt("alphth", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.alphth = fval[i];
    }
    // beta
    h5.read_flt("beta", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.beta = fval[i];
    }
    // v
    h5.read_flt("v", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.v = fval[i];
    }
    // gf
    h5.read_flt("gf", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.gf = fval[i];
    }
    // mf
    h5.read_flt("mf", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.mf = fval[i];
    }
    // mfcgr
    h5.read_flt("mfcgr", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.mfcgr = fval[i];
    }
    // dq
    h5.read_flt("dq", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.dq = fval[i];
    }
    // q
    h5.read_flt("q", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.q = fval[i];
    }
    // y
    h5.read_flt("y", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.y = fval[i];
    }
    // z
    h5.read_flt("z", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.z = fval[i];
    }
    // dmg
    h5.read_flt("dmg", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.dmg = fval[i];
    }
    // erepr
    h5.read_flt("erepr", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.erepr = fval[i];
    }
    // elor
    h5.read_flt("elor", fval);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.elor = fval[i];
    }
    // std::vector to hold unsigned int member
    ui_type uival(N);
    // age
    h5.read_uint("age", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.age = uival[i];
    }
    // dage
    h5.read_uint("dage", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.dage = uival[i];
    }
    // nInts
    h5.read_uint("nInts", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.nInts = uival[i];
    }
    // nRnds
    h5.read_uint("nRnds", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.nRnds = uival[i];
    }
    // nAA
    h5.read_uint("nAA", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.nAA = uival[i];
    }
    // nOffspr
    h5.read_uint("nOffspr", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.nOffspr = uival[i];
    }
    // inum
    h5.read_uint("inum", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.inum = uival[i];
    }
    // yr1
    h5.read_uint("yr1", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.yr1 = uival[i];
    }
    // gnum
    h5.read_uint("gnum", uival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.gnum = uival[i];
    }
    // std::vector to hold int (actually bool) member
    std::vector<int> ival(N);
    // female
    h5.read_int("female", ival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.female = ival[i];
    }
    // alive
    h5.read_int("alive", ival);
    for (unsigned i = 0; i < N; ++i) {
        m_pop[i].phenotype.alive = ival[i];
    }
    // read learning parameters w and th
    std::vector<v_type> pars(N, v_type(Nl));
    // read w parameters
    h5.read_flt_arr("w", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = m_pop[i].phenotype.lp.w;
        for (unsigned j = 0; j < Nl; ++j) {
            par[j] = pars[i][j];
        }
    }
    // read th parameters
    h5.read_flt_arr("th", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = m_pop[i].phenotype.lp.th;
        for (unsigned j = 0; j < Nl; ++j) {
            par[j] = pars[i][j];
        }
    }
    // read expected reproduction per competition group
    // read erepcg parameters
    h5.read_flt_arr("erepcg", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = m_pop[i].phenotype.lp.erepcg;
        for (unsigned j = 0; j < Nl; ++j) {
            par[j] = pars[i][j];
        }
    }
    // read parameters ncg, wins, losses, and draws
    std::vector<ui_type> uipars(N, ui_type(Nl));
    // read ncg parameters
    h5.read_uint_arr("ncg", uipars);
    for (unsigned i = 0; i < N; ++i) {
        ui_type& uipar = m_pop[i].phenotype.lp.ncg;
        for (unsigned j = 0; j < Nl; ++j) {
            uipar[j] = uipars[i][j];
        }
    }
    // read wins parameters
    h5.read_uint_arr("wins", uipars);
    for (unsigned i = 0; i < N; ++i) {
        ui_type& uipar = m_pop[i].phenotype.lp.wins;
        for (unsigned j = 0; j < Nl; ++j) {
            uipar[j] = uipars[i][j];
        }
    }
    // read losses parameters
    h5.read_uint_arr("losses", uipars);
    for (unsigned i = 0; i < N; ++i) {
        ui_type& uipar = m_pop[i].phenotype.lp.losses;
        for (unsigned j = 0; j < Nl; ++j) {
            uipar[j] = uipars[i][j];
        }
    }
    // read draws parameters
    h5.read_uint_arr("draws", uipars);
    for (unsigned i = 0; i < N; ++i) {
        ui_type& uipar = m_pop[i].phenotype.lp.draws;
        for (unsigned j = 0; j < Nl; ++j) {
            uipar[j] = uipars[i][j];
        }
    }
}

void Evo::h5_write_pop(const std::string& outfilename) const
{
    h5W h5(outfilename);
    // male gametes
    std::vector<v_type> mgams(N, v_type(num_loci));
    // write maternal gametes of males
    for (unsigned i = 0; i < N; ++i) {
        const gam_type& gam = m_pop[i].genotype.mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            mgams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("MlMatGam", mgams);
    // write paternal gametes of males
    for (unsigned i = 0; i < N; ++i) {
        const gam_type& gam = m_pop[i].genotype.pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            mgams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("MlPatGam", mgams);
    // female gametes
    std::vector<v_type> fgams(Np, v_type(num_loci));
    // write maternal gametes of females
    for (unsigned i = 0; i < Np; ++i) {
        const gam_type& gam = f_pop[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            fgams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("FlMatGam", fgams);
    // write paternal gametes of females
    for (unsigned i = 0; i < Np; ++i) {
        const gam_type& gam = f_pop[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            fgams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("FlPatGam", fgams);
    // male offspring gametes
    std::vector<v_type> mogams(N1, v_type(num_loci));
    // write maternal gametes of male offspring
    for (unsigned i = 0; i < N1; ++i) {
        const gam_type& gam = m_offspr[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            mogams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("MloMatGam", mogams);
    // write paternal gametes of male offspring
    for (unsigned i = 0; i < N1; ++i) {
        const gam_type& gam = m_offspr[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            mogams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("MloPatGam", mogams);
    // female offspring gametes
    std::vector<v_type> fogams(Np1, v_type(num_loci));
    // write maternal gametes of female offspring
    for (unsigned i = 0; i < Np1; ++i) {
        const gam_type& gam = f_offspr[i].mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            fogams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("FloMatGam", fogams);
    // write paternal gametes of female offspring
    for (unsigned i = 0; i < Np1; ++i) {
        const gam_type& gam = f_offspr[i].pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            fogams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("FloPatGam", fogams);
    // write members of phenotypes
    // std::vector to hold flt member
    v_type fval(N);
    // w0
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.w0; });
    h5.write_flt("w0", fval);
    // th0
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.th0; });
    h5.write_flt("th0", fval);
    // g0
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.g0; });
    h5.write_flt("g0", fval);
    // ga0
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.ga0; });
    h5.write_flt("ga0", fval);
    // alphw
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.alphw; });
    h5.write_flt("alphw", fval);
    // alphth
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.alphth; });
    h5.write_flt("alphth", fval);
    // beta
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.beta; });
    h5.write_flt("beta", fval);
    // v
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.v; });
    h5.write_flt("v", fval);
    // gf
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.gf; });
    h5.write_flt("gf", fval);
    // mf
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.mf; });
    h5.write_flt("mf", fval);
    // mfcgr
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.mfcgr; });
    h5.write_flt("mfcgr", fval);
    // dq
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.dq; });
    h5.write_flt("dq", fval);
    // q
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.q; });
    h5.write_flt("q", fval);
    // y
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.y; });
    h5.write_flt("y", fval);
    // z
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.z; });
    h5.write_flt("z", fval);
    // dmg
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.dmg; });
    h5.write_flt("dmg", fval);
    // erepr
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.erepr; });
    h5.write_flt("erepr", fval);
    // elor
    std::transform(m_pop.begin(), m_pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.elor; });
    h5.write_flt("elor", fval);
    // std::vector to hold unsigned int member
    ui_type uival(N);
    // age
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.age; });
    h5.write_uint("age", uival);
    // dage
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.dage; });
    h5.write_uint("dage", uival);
    // nInts
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nInts; });
    h5.write_uint("nInts", uival);
    // nRnds
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nRnds; });
    h5.write_uint("nRnds", uival);
    // nAA
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nAA; });
    h5.write_uint("nAA", uival);
    // nOffspr
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nOffspr; });
    h5.write_uint("nOffspr", uival);
    // inum
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.inum; });
    h5.write_uint("inum", uival);
    // yr1
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.yr1; });
    h5.write_uint("yr1", uival);
    // gnum
    std::transform(m_pop.begin(), m_pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.gnum; });
    h5.write_uint("gnum", uival);
    // std::vector to hold int (actually bool) member
    std::vector<int> ival(N);
    // female
    std::transform(m_pop.begin(), m_pop.end(), ival.begin(),
                   [](const ind_type& i) -> int
                   { return i.phenotype.female; });
    h5.write_int("female", ival);
    // alive
    std::transform(m_pop.begin(), m_pop.end(), ival.begin(),
                   [](const ind_type& i) -> int
                   { return i.phenotype.alive; });
    h5.write_int("alive", ival);
    // write learning parameters w and th
    std::vector<v_type> pars(N, v_type(Nl));
    // write w parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = m_pop[i].phenotype.lp.w;
        for (unsigned j = 0; j < Nl; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("w", pars);
    // write th parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = m_pop[i].phenotype.lp.th;
        for (unsigned j = 0; j < Nl; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("th", pars);
    // write expected reproduction per competition group
    // write erepcg parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = m_pop[i].phenotype.lp.erepcg;
        for (unsigned j = 0; j < Nl; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("erepcg", pars);
    // write parameters ncg, wins, losses, and draws
    std::vector<ui_type> uipars(N, ui_type(Nl));
    // write ncg parameters
    for (unsigned i = 0; i < N; ++i) {
        const ui_type& uipar = m_pop[i].phenotype.lp.ncg;
        for (unsigned j = 0; j < Nl; ++j) {
            uipars[i][j] = uipar[j];
        }
    }
    h5.write_uint_arr("ncg", uipars);
    // write wins parameters
    for (unsigned i = 0; i < N; ++i) {
        const ui_type& uipar = m_pop[i].phenotype.lp.wins;
        for (unsigned j = 0; j < Nl; ++j) {
            uipars[i][j] = uipar[j];
        }
    }
    h5.write_uint_arr("wins", uipars);
    // write losses parameters
    for (unsigned i = 0; i < N; ++i) {
        const ui_type& uipar = m_pop[i].phenotype.lp.losses;
        for (unsigned j = 0; j < Nl; ++j) {
            uipars[i][j] = uipar[j];
        }
    }
    h5.write_uint_arr("losses", uipars);
    // write draws parameters
    for (unsigned i = 0; i < N; ++i) {
        const ui_type& uipar = m_pop[i].phenotype.lp.draws;
        for (unsigned j = 0; j < Nl; ++j) {
            uipars[i][j] = uipar[j];
        }
    }
    h5.write_uint_arr("draws", uipars);
}

void Evo::h5_write_hist(const std::string& histfilename) const
{
    h5W h5(histfilename);
    unsigned hlen = stat.size();
    // std::vector to hold unsigned int member
    ui_type uival(hlen);
    // gnum
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.gnum; });
    h5.write_uint("gnum", uival);
    // yr
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.yr; });
    h5.write_uint("yr", uival);
    // cper
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.cper; });
    h5.write_uint("cper", uival);
    // cgrp
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.cgrp; });
    h5.write_uint("cgrp", uival);
    // g
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.g; });
    h5.write_uint("g", uival);
    // cnum
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.cnum; });
    h5.write_uint("cnum", uival);
    // i
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.i; });
    h5.write_uint("i", uival);
    // j
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.j; });
    h5.write_uint("j", uival);
    // yr1i
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.yr1i; });
    h5.write_uint("yr1i", uival);
    // yr1j
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.yr1j; });
    h5.write_uint("yr1j", uival);
    // agei
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.agei; });
    h5.write_uint("agei", uival);
    // agej
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.agej; });
    h5.write_uint("agej", uival);
    // ui
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.ui; });
    h5.write_uint("ui", uival);
    // uj
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.uj; });
    h5.write_uint("uj", uival);
    // irnds
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.irnds; });
    h5.write_uint("irnds", uival);
    // nAA
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.nAA; });
    h5.write_uint("nAA", uival);
    // ndomi
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.ndomi; });
    h5.write_uint("ndomi", uival);
    // ndomj
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.ndomj; });
    h5.write_uint("ndomj", uival);
    // std::vector to hold int member
    std::vector<int> ival(hlen);
    // winij
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.winij; });
    h5.write_int("winij", ival);
    // std::vector to hold flt member
    v_type fval(hlen);
    // qi
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.qi; });
    h5.write_flt("qi", fval);
    // qj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.qj; });
    h5.write_flt("qj", fval);
    // dmgi
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.dmgi; });
    h5.write_flt("dmgi", fval);
    // dmgj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.dmgj; });
    h5.write_flt("dmgj", fval);
    // lij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lij; });
    h5.write_flt("lij", fval);
    // lji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lji; });
    h5.write_flt("lji", fval);
    // pij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.pij; });
    h5.write_flt("pij", fval);
    // pji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.pji; });
    h5.write_flt("pji", fval);
    // wii
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wii; });
    h5.write_flt("wii", fval);
    // wij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wij; });
    h5.write_flt("wij", fval);
    // wjj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wjj; });
    h5.write_flt("wjj", fval);
    // wji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wji; });
    h5.write_flt("wji", fval);
    // thii
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thii; });
    h5.write_flt("thii", fval);
    // thij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thij; });
    h5.write_flt("thij", fval);
    // thjj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thjj; });
    h5.write_flt("thjj", fval);
    // thji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thji; });
    h5.write_flt("thji", fval);
    // elori
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.elori; });
    h5.write_flt("elori", fval);
    // elorj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.elorj; });
    h5.write_flt("elorj", fval);
    // tminyr
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.tminyr; });
    h5.write_flt("tminyr", fval);
    // tm
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.tm; });
    h5.write_flt("tm", fval);
}
