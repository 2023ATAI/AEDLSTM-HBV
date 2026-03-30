# ==============================================================================
# ACKNOWLEDGEMENT:
# The HBV model implementation and the associated parameters used in this
# script are adapted from the Python version provided by GloH2O.
# Reference: http://www.gloh2o.org/hbv/
# Modified for integration into the ATAI framework by: Qingliang Li, Cheng Zhang
# ==============================================================================
import torch
import netCDF4 as nc
import os
import numpy as np


def HBVfordata(x, parameters, mid_forcing_list):

    parBETA = parameters['BETA'].astype(np.float32)
    parFC = parameters['FC'].astype(np.float32)
    parK0 = parameters['K0'].astype(np.float32)
    parK1 = parameters['K1'].astype(np.float32)
    parK2 = parameters['K2'].astype(np.float32)
    parLP = parameters['LP'].astype(np.float32)
    parPERC = parameters['PERC'].astype(np.float32)
    parUZL = parameters['UZL'].astype(np.float32)
    parTT = parameters['TT'].astype(np.float32)
    parCFMAX = parameters['CFMAX'].astype(np.float32)
    parCFR = parameters['CFR'].astype(np.float32)
    parCWH = parameters['CWH'].astype(np.float32)
    parPCORR = parameters['PCORR'].astype(np.float32)
    #parSFCF = parameters['SFCF'].astype(np.float32)

    PRECS = 1e-5  # keep the numerical calculation stable

    para_shape = parameters['BETA'].shape
    Qall = np.zeros(para_shape, dtype=np.float32) + 0.001
    SNOWPACK = np.zeros(para_shape, dtype=np.float32) + 0.001
    MELTWATER = np.zeros(para_shape, dtype=np.float32) + 0.001
    SM = np.zeros(para_shape, dtype=np.float32) + 0.001
    SUZ = np.zeros(para_shape, dtype=np.float32) + 0.001
    SLZ = np.zeros(para_shape, dtype=np.float32) + 0.001

    time_dim, nlat, nlon, var_dim = x.shape
    T = x[ :, :, :, 0]
    P = x[ :, :, :, 1]
    ETpot = x[ :, :, :, -1]
    # Convert units: Temp to Celsius, Fluxes from m to mm
    T = T - 273.15
    ETpot *= -1000
    P *= 1000

    variable_lists = {variable: [] for variable in mid_forcing_list}

    for t in range(time_dim):
        #PRECIP = np.multiply(P[ t, :, :], 1)
        PRECIP = np.multiply(P[ t, :, :], parPCORR)
        RAIN = np.multiply(PRECIP, (T[ t, :, :] >= parTT).astype(np.float32))
        SNOW = np.multiply(PRECIP, (T[ t, :, :] < parTT).astype(np.float32))
        #SNOW = SNOW * parSFCF

        SNOWPACK = SNOWPACK + SNOW
        melt = parCFMAX * (T[ t, :, :] - parTT)
        melt = np.clip(melt, 0.0, None)
        melt = np.minimum(melt, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = parCFR * parCFMAX * (parTT - T[ t, :, :])
        refreezing = np.clip(refreezing, 0.0, None)
        refreezing = np.minimum(refreezing, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing
        tosoil = MELTWATER - (parCWH * SNOWPACK)
        tosoil = np.clip(tosoil, 0.0, None)
        MELTWATER = MELTWATER - tosoil

        soil_wetness = (SM / parFC) ** parBETA
        soil_wetness = np.clip(soil_wetness, 0.0, 1.0)
        recharge = (RAIN + tosoil) * soil_wetness

        SM = SM + RAIN + tosoil - recharge
        excess = SM - parFC
        excess = np.clip(excess, 0.0, None)
        SM = SM - excess
        # evapfactor = SM / (parLP* parFC)
        ETact = ETpot[ t, :, :]
        ETact = np.minimum(SM, ETact)
        SM = np.clip(SM - ETact, PRECS, None)

        SUZ = SUZ + recharge + excess
        PERC = np.minimum(SUZ, parPERC)
        SUZ = SUZ - PERC
        Q0 = parK0 * np.clip(SUZ - parUZL, 0.0, None)
        SUZ = SUZ - Q0
        Q1 = parK1 * SUZ
        SUZ = SUZ - Q1
        SLZ = SLZ + PERC
        Q2 = parK2 * SLZ
        SLZ = SLZ - Q2
        Qall = Q0 + Q1 + Q2

        for variable in mid_forcing_list:
            variable_lists[variable].append(locals()[variable])

    mid_data = [np.stack(variable_lists[variable], axis=0) for variable in mid_forcing_list]
    mid = np.stack(mid_data, axis=-1)


    return mid



def load_para_for_pred(cfg):
    parameters = {}
    for file_name in cfg['parameters_names']:
        file_path = os.path.join(cfg['parameters_path'], file_name)
        dataset = nc.Dataset(file_path)
        var_name = file_name.split('.')[0]
        var_values = dataset.variables[var_name][:]
        parameters[var_name] = var_values
        dataset.close()

    return parameters

