# functions to read data from hdf5 file created by simulation program

# return data table for "one-column" phenotype data
h5_dt <- function(hf_name) {
    require(hdf5r)
    require(data.table)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w0 <- f.h5[["w0"]][]
    th0 <- f.h5[["th0"]][]
    g0 <- f.h5[["g0"]][]
    ga0 <- f.h5[["ga0"]][]
    alphw <- f.h5[["alphw"]][]
    alphth <- f.h5[["alphth"]][]
    beta <- f.h5[["beta"]][]
    v <- f.h5[["v"]][]
    gf <- f.h5[["gf"]][]
    mf <- f.h5[["mf"]][]
    mfcgr <- f.h5[["mfcgr"]][]
    dq <- f.h5[["dq"]][]
    q <- f.h5[["q"]][]
    y <- f.h5[["y"]][]
    z <- f.h5[["z"]][]
    dmg <- f.h5[["dmg"]][]
    erepr <- f.h5[["erepr"]][]
    elor <- f.h5[["elor"]][]
    age <- f.h5[["age"]][] + 1
    dage <- f.h5[["dage"]][] + 1
    nInts <- f.h5[["nInts"]][]
    nRnds <- f.h5[["nRnds"]][]
    nAA <- f.h5[["nAA"]][]
    nOffspr <- f.h5[["nOffspr"]][]
    inum <- f.h5[["inum"]][] + 1
    yr1 <- f.h5[["yr1"]][] + 1
    gnum <- f.h5[["gnum"]][] + 1
    female <- f.h5[["female"]][]
    alive <- f.h5[["alive"]][]
    f.h5$close_all()
    data.table(w0 = w0, th0 = th0, g0 = g0, ga0 = ga0,
               alphw = alphw, alphth = alphth, beta = beta,
               v = v, gf = gf, mf = mf, mfcgr = mfcgr,
               dq = dq, q = q, y = y, z = z,
               dmg = dmg, erepr = erepr, elor = elor,
               age = age, dage = dage,
               nInts = nInts, nRnds = nRnds, nAA = nAA, nOffspr = nOffspr,
               inum = inum, yr1 = yr1, gnum = gnum,
               female = female, alive = alive)
}

# return matrix where each row is a learning parameter w
h5_w <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w <- t(f.h5[["w"]][,])
    f.h5$close_all()
    w
}

# return matrix where each row is a learning parameter th
h5_th <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    th <- t(f.h5[["th"]][,])
    f.h5$close_all()
    th
}

# return matrix where each row is a parameter erepcg
h5_erepcg <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    erepcg <- t(f.h5[["erepcg"]][,])
    f.h5$close_all()
    erepcg
}

# return matrix where each row is a parameter ncg
h5_ncg <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    ncg <- t(f.h5[["ncg"]][,])
    f.h5$close_all()
    ncg
}

# return matrix where each row is a parameter wins
h5_wins <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    wins <- t(f.h5[["wins"]][,])
    f.h5$close_all()
    wins
}

# return matrix where each row is a parameter losses
h5_losses <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    losses <- t(f.h5[["losses"]][,])
    f.h5$close_all()
    losses
}

# return matrix where each row is a parameter draws
h5_draws <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    draws <- t(f.h5[["draws"]][,])
    f.h5$close_all()
    draws
}

# return matrix where each row is a maternal gamete value
h5_mat_gam <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    mat_gam <- t(f.h5[["MatGam"]][,])
    f.h5$close_all()
    mat_gam
}

# return matrix where each row is a paternal gamete value
h5_pat_gam <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    pat_gam <- t(f.h5[["PatGam"]][,])
    f.h5$close_all()
    pat_gam
}


# return data table for learning history data
h5_hdt <- function(hf_name) {
    require(hdf5r)
    require(data.table)
    f.h5 <- H5File$new(hf_name, mode = "r")
    gnum <- f.h5[["gnum"]][] + 1
    yr <- f.h5[["yr"]][] + 1
    cper <- f.h5[["cper"]][]
    cgrp <- f.h5[["cgrp"]][] + 1
    g <- f.h5[["g"]][]
    cnum <- f.h5[["cnum"]][] + 1
    i <- f.h5[["i"]][] + 1
    j <- f.h5[["j"]][] + 1
    yr1i <- f.h5[["yr1i"]][] + 1
    yr1j <- f.h5[["yr1j"]][] + 1
    agei <- f.h5[["agei"]][] + 1
    agej <- f.h5[["agej"]][] + 1
    ui <- f.h5[["ui"]][]
    uj <- f.h5[["uj"]][]
    irnds <- f.h5[["irnds"]][]
    nAA <- f.h5[["nAA"]][]
    ndomi <- f.h5[["ndomi"]][]
    ndomj <- f.h5[["ndomj"]][]
    winij <- f.h5[["winij"]][]
    qi <- f.h5[["qi"]][]
    qj <- f.h5[["qj"]][]
    dmgi <- f.h5[["dmgi"]][]
    dmgj <- f.h5[["dmgj"]][]
    lij <- f.h5[["lij"]][]
    lji <- f.h5[["lji"]][]
    pij <- f.h5[["pij"]][]
    pji <- f.h5[["pji"]][]
    wii <- f.h5[["wii"]][]
    wij <- f.h5[["wij"]][]
    wjj <- f.h5[["wjj"]][]
    wji <- f.h5[["wji"]][]
    thii <- f.h5[["thii"]][]
    thij <- f.h5[["thij"]][]
    thjj <- f.h5[["thjj"]][]
    thji <- f.h5[["thji"]][]
    elori <- f.h5[["elori"]][]
    elorj <- f.h5[["elorj"]][]
    tminyr <- f.h5[["tminyr"]][]
    tm <- f.h5[["tm"]][] + 1
    f.h5$close_all()
    data.table(gnum = gnum, yr = yr,
            cper = cper, cgrp = cgrp, g = g, cnum = cnum,
            i = i, j = j, yr1i = yr1i, yr1j = yr1j,
            agei = agei, agej = agej, ui = ui, uj = uj,
            irnds = irnds, nAA = nAA,
            ndomi = ndomi, ndomj = ndomj, winij = winij,
            qi = qi, qj = qj, dmgi = dmgi, dmgj = dmgj,
            lij = lij, lji = lji, pij = pij, pji = pji,
            wii = wii, wij = wij, wjj = wjj, wji = wji,
            thii = thii, thij = thij, thjj = thjj, thji = thji,
            elori = elori, elorj = elorj, tminyr = tminyr, tm = tm)
}

