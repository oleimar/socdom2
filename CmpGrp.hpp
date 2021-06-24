#ifndef CMPGRP_HPP
#define CMPGRP_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// The EvoDom program runs evolutionary simulations
// Copyright (C) 2021  Olof Leimar
// See Readme.md for copyright notice

//************************** struct CmpStat *******************************

// This struct stores data on a dominance interactions between two group
// members (i and j), right after a competition-group interaction (i.e. a
// sequence of bouts between i and j); a sequence of structs can be saved as
// the interaction history for a local group

template<typename PhenType>
struct CmpStat {
    using phen_type = PhenType;
    using flt = typename phen_type::flt;
    unsigned gnum;      // local group number
    unsigned yr;        // year
    unsigned cper;      // competition period (0 or 1)
    unsigned cgrp;      // competition group number (in period)
    unsigned g;         // competition group size
    unsigned cnum;      // contest number in competition period
    unsigned i;         // inum for individual; i for short
    unsigned j;         // inum for individual; j for short
    unsigned yr1i;      // year born for i
    unsigned yr1j;      // year born for j
    unsigned agei;      // age of individual i
    unsigned agej;      // age of individual j
    unsigned ui;        // final action by i (A is 1, S is 0)
    unsigned uj;        // final action by j (A is 1, S is 0)
    unsigned irnds;     // number of rounds between i and j
    unsigned nAA;       // number of AA rounds between i and j
    unsigned ndomi;     // number dominated over in competition group
    unsigned ndomj;     // number dominated over in competition group
    int winij;          // outcome (1 if i wins, 0 if draw, -1 if j wins)
    flt qi;             // fighting ability i
    flt qj;             // fighting ability j
    flt dmgi;           // damage for i
    flt dmgj;           // damage for j
    flt lij;            // logit of pij
    flt lji;            // logit of pji
    flt pij;            // prob for i to use action A
    flt pji;            // prob for j to use action A
    flt wii;            // intercept parameter for i
    flt wij;            // intercept parameter for i
    flt wjj;            // intercept parameter for j
    flt wji;            // intercept parameter for j
    flt thii;           // intercept parameter for i
    flt thij;           // intercept parameter for i
    flt thjj;           // intercept parameter for j
    flt thji;           // intercept parameter for j
    flt elori;          // Elo score for i
    flt elorj;          // Elo score for j
    flt tminyr;         // time in current year
    flt tm;             // time (unit is one year)
};


//************************** class CmpGrp *******************************

// This class sets up and simulates the actor-critic learning method for a
// social dominance game.

// The class deals with the interactions in one 'competition group', i.e. a
// group of individuals that have been selected to interact; a competition
// group can be a subset of individuals in a local social group.

template<typename PhenType>
class CmpGrp {
public:
    using phen_type = PhenType;
    using lp_type = typename phen_type::lp_type;
    using vph_type = std::vector<phen_type>;
    using stat_type = CmpStat<PhenType>;
    using vs_type = std::vector<stat_type>;
    using flt = typename phen_type::flt;
    using v_type = std::vector<flt>;
    using ui_type = std::vector<unsigned>;
    using ui_mat = std::vector<ui_type>;
    using bl_type = std::vector<bool>;
    using bl_mat = std::vector<bl_type>;
    using rand_eng = std::mt19937;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_int = std::uniform_int_distribution<unsigned>;
    using rand_norm = std::normal_distribution<flt>;
    CmpGrp(unsigned a_g,
        unsigned numI,
        unsigned a_irndx,
        unsigned a_minrnds,
        unsigned a_domrnds,
        unsigned a_drawrnds,
        unsigned a_yr,
        unsigned a_cper,
        unsigned a_cgrp,
        flt a_V0,
        flt a_cdmg,
        flt a_sigma,
        flt a_sigp,
        flt a_pmax,
        flt a_betelo,
        const v_type& a_V1,
        const vph_type& a_memb,
        bool a_bystl,
        bool a_hist = false);
    const vph_type& Get_memb() const { return memb; }
    const vs_type& Get_stat() const { return stat; }
    void Interact(rand_eng& eng);

private:
    flt Clamp(flt p) const
        {return (p > pmax) ? pmax : ((p < 1 - pmax) ? 1 - pmax : p); }
    void Add_stat(unsigned cnum,
        unsigned i, unsigned j,
        unsigned ui, unsigned uj,
        unsigned irnds, unsigned nAA,
        unsigned ndomi, unsigned ndomj,
        int winij,
        flt lij, flt lji,
        flt pij, flt pji);
    unsigned g;         // pre-mating or mating competition group size
    unsigned numI;      // number of interactions per pair
    unsigned irndx;     // maximum number of rounds for interaction
    unsigned minrnds;   // minimum number of rounds until dominance
    unsigned domrnds;   // number of AS (or SA) rounds until dominance clear
    unsigned drawrnds;  // number of SS rounds until draw
    unsigned yr;        // year
    unsigned cper;      // competition period (0 = pre-mating, 1 = mating)
    unsigned cgrp;      // competition group number (in period)
    flt V0;             // baseline payoff parameter
    flt cdmg;           // fighting cost from damage
    flt sigma;          // SD of error for 'relative quality' observation
    flt sigp;           // SD of error for 'penalty of AA interaction'
    flt pmax;           // maximum value for probability to use A
    flt betelo;         // parameter for Elo score updates
    v_type V1;          // payoffs for mating group
    vph_type memb;      // phenotypes of members of the competition group
    bool bystl;         // whether there is bystander learning
    bool hist;          // whether to collect history
    vs_type stat;       // container for history statistics
};

template<typename PhenType>
CmpGrp<PhenType>::CmpGrp(unsigned a_g,
    unsigned a_numI,
    unsigned a_irndx,
    unsigned a_minrnds,
    unsigned a_domrnds,
    unsigned a_drawrnds,
    unsigned a_yr,
    unsigned a_cper,
    unsigned a_cgrp,
    flt a_V0,
    flt a_cdmg,
    flt a_sigma,
    flt a_sigp,
    flt a_pmax,
    flt a_betelo,
    const v_type& a_V1,
    const vph_type& a_memb,
    bool a_bystl,
    bool a_hist) :
    g{a_g},
    numI{a_numI},
    irndx{a_irndx},
    minrnds{a_minrnds},
    domrnds{a_domrnds},
    drawrnds{a_drawrnds},
    yr{a_yr},
    cper{a_cper},
    cgrp{a_cgrp},
    V0{a_V0},
    cdmg{a_cdmg},
    sigma{a_sigma},
    sigp{a_sigp},
    pmax{a_pmax},
    betelo{a_betelo},
    V1{a_V1},
    memb{a_memb},
    bystl{a_bystl},
    hist{a_hist}
{
    if (hist) {
        stat.reserve(irndx);
    }
}

template<typename PhenType>
void CmpGrp<PhenType>::Interact(rand_eng& eng)
{
    // This code represents a pre-mating (if cper == 0) or mating (if cper ==
    // 1) group interaction between g males
    rand_uni uni(0, 1);
    rand_int uri(0, g - 1);
    rand_norm eps(0, sigma);
    rand_norm eqd(0, sigp);
    unsigned nV1 = V1.size();
    // individuals in memb are supposed to be a stable-sorted random sample of
    // size g; thus there is on average no significance to the ordering of
    // these individuals, but the individuals occur in increasing order with
    // respect to their individual numbers (inum values).
    unsigned npairs = g*(g - 1)/2;
    ui_type ndom(g, 0); // count how many individuals each dominates
    if (npairs > 0) {
        // keep track of certain data between contests for each pair
        ui_mat rndsSS(g, ui_type(g, 0));    // rnds and nSS
        ui_mat nASSA(g, ui_type(g, 0));     // nAS and nSA
        bl_mat dom(g, bl_type(g, false));   // domi and domj
        bl_mat clrdr(g, bl_type(g, false)); // domclear and draw
        // run through contests
        for (unsigned cnum = 0; cnum < npairs*numI; ++cnum) {
            // select random pair to interact
            unsigned i = 0;
            unsigned j = 1;
            // the above values for i and j are OK if npairs == 1
            if (npairs > 1) {
                i = uri(eng);
                j = uri(eng);
                while (j == i) j = uri(eng);
            }
            // make sure i < j (convenient for history data)
            if (i > j) {
                unsigned isave = i;
                i = j;
                j = isave;
            }
            phen_type& mi = memb[i];
            phen_type& mj = memb[j];
            lp_type& lpi = mi.lp;
            lp_type& lpj = mj.lp;
            unsigned inum = mi.inum; // 'actual' individual number for i
            unsigned jnum = mj.inum; // 'actual' individual number for j
            // use 'local copies' of data we want to keep track of
            unsigned rnds = rndsSS[i][j];
            unsigned nSS = rndsSS[j][i];
            unsigned nAS = nASSA[i][j];
            unsigned nSA = nASSA[j][i];
            bool domi = dom[i][j];
            bool domj = dom[j][i];
            bool domclear = clrdr[i][j];
            bool draw = clrdr[j][i];
            flt lij = 0;
            flt lji = 0;
            flt pij = 0.5;
            flt pji = 0.5;
            unsigned ui = 0;
            unsigned uj = 0;
            // run through rounds for this pair
            unsigned irnd = 0;   // count rounds in interaction
            unsigned nAA = 0;    // count AA rounds in interaction
            while (irnd < irndx && !domclear && !draw) {
                // observations of relative quality
                flt xi_ij = mi.y - mj.z + eps(eng);
                flt xi_ji = mj.y - mi.z + eps(eng);
                // logit of probability using A
                lij = mi.l(xi_ij, jnum);
                lji = mj.l(xi_ji, inum);
                // probability using A
                pij = 1/(1 + std::exp(-lij));
                pji = 1/(1 + std::exp(-lji));
                ui = (uni(eng) < Clamp(pij)) ? 1:0;
                uj = (uni(eng) < Clamp(pji)) ? 1:0;
                flt Rij = 0;
                flt Rji = 0;
                if (ui == 1 && uj == 0) {
                    ++nAS;
                    nSA = 0;
                    nSS = 0;
                    Rij = mi.v;
                } else if (ui == 0 && uj == 1) {
                    nAS = 0;
                    ++nSA;
                    nSS = 0;
                    Rji = mj.v;
                } else {
                    nAS = 0;
                    nSA = 0;
                    if (ui == 1 && uj == 1) { // AA round
                        nSS = 0;
                        ++nAA;
                        ++mi.nAA;
                        ++mj.nAA;
                        Rij = mi.v;
                        Rji = mj.v;
                        Rij -= std::exp(-(mi.q - cdmg*mi.dmg) +
                                         (mj.q - cdmg*mj.dmg) + eqd(eng));
                        Rji -= std::exp(-(mj.q - cdmg*mj.dmg) +
                                         (mi.q - cdmg*mi.dmg) + eqd(eng));
                        // accumulate damage
                        flt Di = std::exp(-(mi.q - cdmg*mi.dmg) +
                                           (mj.q - cdmg*mj.dmg));
                        flt Dj = 1/Di;
                        // accumulate damage; NOTE: limit damage to 2000, to
                        // avoid 'damage explosion' for positive cdmg
                        if (Di < 2000 - mi.dmg) {
                            mi.dmg += Di;
                        }
                        if (Dj < 2000 - mj.dmg) {
                            mj.dmg += Dj;
                        }
                    } else { // SS round
                        ++nSS;
                    }
                }
                flt deltaij = Rij - mi.vhat(xi_ij, jnum);
                flt deltaji = Rji - mj.vhat(xi_ji, inum);
                // update learning parameters
                // estimated values
                flt alphwdij = mi.alphw*deltaij;
                lpi.w[inum] += mi.gf*alphwdij;
                lpi.w[jnum] += (1 - mi.gf)*alphwdij;
                flt alphwdji = mj.alphw*deltaji;
                lpj.w[jnum] += mj.gf*alphwdji;
                lpj.w[inum] += (1 - mj.gf)*alphwdji;
                // action preferences
                flt derij = (ui == 1) ? 1 - pij : -pij;
                flt derji = (uj == 1) ? 1 - pji : -pji;
                flt alphthderdij = mi.alphth*derij*deltaij;
                flt alphthderdji = mj.alphth*derji*deltaji;
                lpi.th[inum] += mi.gf*alphthderdij;
                lpi.th[jnum] += (1 - mi.gf)*alphthderdij;
                lpj.th[jnum] += mj.gf*alphthderdji;
                lpj.th[inum] += (1 - mj.gf)*alphthderdji;
                mi.nRnds += 1;
                mj.nRnds += 1;
                ++rnds;
                // check if dominance is established
                if (rnds >= minrnds) {
                    domi = (nAS >= domrnds);
                    domj = (nSA >= domrnds);
                    domclear = domi || domj;
                    draw = (nSS >= drawrnds);
                }
                ++irnd;
            }
            if (irnd > 0) {
                mi.nInts += 1;
                mj.nInts += 1;
                // check for wins, losses, or draws
                if (domi) {
                    ++lpi.wins[jnum];
                    ++lpj.losses[inum];
                } else if (domj) {
                    ++lpi.losses[jnum];
                    ++lpj.wins[inum];
                } else if (draw) {
                    ++lpi.draws[jnum];
                    ++lpj.draws[inum];
                }
                // count number dominated over
                if (domi) {
                    ++ndom[i];
                } else if (domj) {
                    ++ndom[j];
                }
                // update Elo scores, based on win, loss, or draw
                if (draw) {
                    flt peloij = 1/(1 + std::exp(-(mi.elor - mj.elor)));
                    mi.elor -= betelo*(peloij - 0.5);
                    mj.elor += betelo*(peloij - 0.5);
                } else if (domi) {
                    flt peloij = 1/(1 + std::exp(-(mi.elor - mj.elor)));
                    mi.elor += betelo*(1 - peloij);
                    mj.elor -= betelo*(1 - peloij);
                } else if (domj) {
                    flt peloij = 1/(1 + std::exp(-(mi.elor - mj.elor)));
                    mi.elor -= betelo*peloij;
                    mj.elor += betelo*peloij;
                }
                if (hist) {
                    int winij = 0; // 0 is value for draw or not yet decided
                    if (domi) {
                        winij = 1;
                    } else if (domj) {
                        winij = -1;
                    }
                    Add_stat(cnum, i, j, ui, uj, irnd, nAA,
                        ndom[i], ndom[j], winij, lij, lji, pij, pji);
                }
                // bystander learning
                if (bystl && ui != uj) {
                    // other individuals in group are bystanders
                    for (unsigned k = 0; k < g; ++k) {
                        if (k != i && k != j) {
                            phen_type& mk = memb[k];
                            // very simple updating: add or subtract beta from
                            // th depending on if final actions (ui, uj) in
                            // contest were (1, 0) or (0, 1); note that effects
                            // of generalisation on lpk.th[knum] cancel out
                            lp_type& lpk = mk.lp;
                            flt Pij =
                            1/(1 + std::exp(lpk.th[inum] - lpk.th[jnum]));
                            flt reli = std::exp(-mk.y + mi.z + eps(eng));
                            flt relj = std::exp(-mk.y + mj.z + eps(eng));
                            if (ui == 1) { // and uj == 0
                                lpk.th[inum] += -(1 - Pij)*mk.beta*reli;
                                lpk.th[jnum] += (1 - Pij)*mk.beta/relj;
                            } else if (ui == 0) { // and uj == 1
                                lpk.th[inum] += Pij*mk.beta/reli;
                                lpk.th[jnum] += -Pij*mk.beta*relj;
                            }
                        }
                    }
                }
                rndsSS[i][j] = rnds;
                rndsSS[j][i] = nSS;
                nASSA[i][j] = nAS;
                nASSA[j][i] = nSA;
                dom[i][j] = domi;
                dom[j][i] = domj;
                clrdr[i][j] = domclear;
                clrdr[j][i] = draw;
            }
            // ready for next random pair
        }
    }
    // if the interaction is a mating group, accumulate expected reproductive
    // success; the basic assumption is that, for a competition group of size
    // g, there is an amount g*V1 to give based on dominance, in addition to
    // the amount V0 each receives irrespective of dominance; ndom[i] is the
    // number of individuals among the g that individual i has established
    // dominance over (thus, maximal value of ndom[i] is g-1)
    if (cper == 1) {
        if (V0 > 0) {
            // each mating group member gets V0, without competition
            for (unsigned i = 0; i < g; ++i) {
                memb[i].lp.erepcg[0] += V0;
            }
        }
        if (npairs > 0) {
            // compute indices of sorted ndom, from highest to lowest
            ui_type idx(g, 0);
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(),
                             [&ndom](unsigned i1, unsigned i2)
                             { return ndom[i1] > ndom[i2]; });
            if (ndom[idx[0]] == 0) {
                // no dominance hierarchy, construct random positions
                ui_type vpos(g, 0);
                // we should have nV1 >= g
                std::iota(vpos.begin(), vpos.end(), nV1 - g);
                std::shuffle(vpos.begin(), vpos.end(), eng);
                for (unsigned i = 0; i < g; ++i) {
                    phen_type& mi = memb[i];
                    lp_type& lpi = mi.lp;
                    flt v1 = V1[vpos[i]];
                    mi.erepr += V0 + v1;
                    lpi.erepcg[g - 1] += v1;
                    ++lpi.ncg[g - 1];
                }
            } else {
                // apportion V1 values, from V1[nV1 -1] and downwards, in
                // dominance order (NOTE: there is a possible issue with tied
                // positions)
                for (unsigned i = 0; i < g; ++i) {
                    phen_type& mii = memb[idx[i]];
                    lp_type& lpii = mii.lp;
                    flt v1 = V1[nV1 - 1 - i];
                    mii.erepr += V0 + v1;
                    lpii.erepcg[g - 1] += v1;
                    ++lpii.ncg[g - 1];
                }
            }
        } else { // npairs == 0, g == 1
            flt v1 = V1[nV1 - 1];
            memb[0].erepr += V0 + v1;
            memb[0].lp.erepcg[0] += v1;
            ++memb[0].lp.ncg[0];
        }
    }
}

template<typename PhenType>
void CmpGrp<PhenType>::Add_stat(unsigned cnum, unsigned i, unsigned j,
    unsigned ui, unsigned uj, unsigned irnds, unsigned nAA,
    unsigned ndomi, unsigned ndomj, int winij,
    flt lij, flt lji, flt pij, flt pji)
{
    phen_type& mi = memb[i];
    phen_type& mj = memb[j];
    unsigned inum = mi.inum;
    unsigned jnum = mj.inum;
    stat_type st;
    st.gnum = mi.gnum;
    st.yr = yr;
    st.cper = cper;
    st.cgrp = cgrp;
    st.g = g;
    st.cnum = cnum;
    st.i = inum;
    st.j = jnum;
    st.yr1i = mi.yr1;
    st.yr1j = mj.yr1;
    st.agei = mi.age;
    st.agej = mj.age;
    st.ui = ui;
    st.uj = uj;
    st.irnds = irnds;
    st.nAA = nAA;
    st.ndomi = ndomi;
    st.ndomj = ndomj;
    st.winij = winij;
    st.qi = mi.q;
    st.qj = mj.q;
    st.dmgi = mi.dmg;
    st.dmgj = mj.dmg;
    st.lij = lij;
    st.lji = lji;
    st.pij = pij;
    st.pji = pji;
    st.wii = mi.lp.w[inum];
    st.wij = mi.lp.w[jnum];
    st.wjj = mj.lp.w[jnum];
    st.wji = mj.lp.w[inum];
    st.thii = mi.lp.th[inum];
    st.thij = mi.lp.th[jnum];
    st.thjj = mj.lp.th[jnum];
    st.thji = mj.lp.th[inum];
    st.elori = mi.elor;
    st.elorj = mj.elor;
    st.tminyr = 0;
    st.tm = yr;
    stat.push_back(st);
}

#endif // CMPGRP_HPP
