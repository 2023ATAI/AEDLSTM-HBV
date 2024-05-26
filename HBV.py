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
    #T[:, :, :] = 0
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

def HBVfortrain(x, parameters):

    

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



    PRECS = 1e-5  # keep the numerical calculation stable


    para_shape = parameters['BETA'].shape
    Qall = np.zeros(para_shape, dtype=np.float32) + 0.001
    SNOWPACK = np.zeros(para_shape, dtype=np.float32) + 0.001
    MELTWATER = np.zeros(para_shape, dtype=np.float32) + 0.001
    SM = np.zeros(para_shape, dtype=np.float32) + 0.001
    SUZ = np.zeros(para_shape, dtype=np.float32) + 0.001
    SLZ = np.zeros(para_shape, dtype=np.float32) + 0.001

    grid_dim, time_dim, var_dim = x.shape
    P = x[ :,:, 0]
    T = x[ :,:, 1]
    T = T -273.15

    ETpot = x[ :,:, -1]
    ETpot *= 1000
    P *= 1000

    Qall_list = []
    # SNOWPACK_list = []
    # MELTWATER_list = []
    # SM_list = []
    SUZ_list = []
    SLZ_list = []

    for t in range(time_dim):
        
        
        PRECIP = np.multiply(P[:,t], parPCORR)
        RAIN = np.multiply(PRECIP, (T[:,t] >= parTT).astype(np.float32))
        SNOW = np.multiply(PRECIP, (T[:,t] < parTT).astype(np.float32))

        SNOWPACK = SNOWPACK + SNOW
        melt = parCFMAX * (T[:,t] - parTT)
        melt = np.clip(melt, 0.0, None)
        melt = np.minimum(melt, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = parCFR * parCFMAX * (parTT - T[:,t])
        refreezing = np.clip(refreezing, 0.0, None)
        refreezing = np.minimum(refreezing, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing
        tosoil = MELTWATER - (parCWH * SNOWPACK)
        tosoil = np.clip(tosoil, 0.0, None)
        MELTWATER = MELTWATER - tosoil

        soil_wetness = (SM / parFC ) ** parBETA
        soil_wetness = np.clip(soil_wetness, 0.0, 1.0)
        recharge = (RAIN + tosoil) * soil_wetness

        SM = SM + RAIN + tosoil - recharge
        excess = SM - parFC
        excess = np.clip(excess, 0.0, None)
        SM = SM - excess
        # evapfactor = SM / (parLP* parFC)
        ETact = ETpot[:,t]
        ETact = np.minimum(SM, ETact)
        SM = np.clip(SM - ETact, PRECS, None)

        SUZ = SUZ + recharge + excess
        PERC = np.minimum(SUZ, parPERC )
        SUZ = SUZ - PERC
        Q0 = parK0 * np.clip(SUZ - parUZL , 0.0, None)
        SUZ = SUZ - Q0
        Q1 = parK1 * SUZ
        SUZ = SUZ - Q1
        SLZ = SLZ + PERC
        Q2 = parK2 * SLZ
        SLZ = SLZ - Q2
        Qall = Q0 + Q1 + Q2

        # SNOWPACK_list.append(SNOWPACK)
        # MELTWATER_list.append(MELTWATER)
        # SM_list.append(SM)
        SUZ_list.append(SUZ)
        SLZ_list.append(SLZ)
        Qall_list.append(Qall)

    # SNOWPACK = np.stack(SNOWPACK_list, axis=1)
    # MELTWATER = np.stack(MELTWATER_list, axis=1)
    # SM = np.stack(SM_list, axis=1)
    SUZ = np.stack(SUZ_list, axis=1)
    SLZ = np.stack(SLZ_list, axis=1)
    Qall = np.stack(Qall_list, axis=1)

    # SNOWPACK_nor = np.clip((SNOWPACK - (-0.1)) / (300 - (-0.1)), 0, 1)
    # MELTWATER_nor = np.clip((MELTWATER - (-1)) / (10 - (-1)), 0, 1)
    # SM_nor = np.clip((SM - 1e-5) / (1000 - 1e-5), 0, 1)
    SUZ_nor = np.clip((SUZ - 0) / (700 - 0), 0, 1)
    SLZ_nor = np.clip((SLZ - (-0.1)) / (150 - (-0.1)), 0, 1)
    Qall_nor = np.clip((Qall - (-0.1)) / (170 - (-0.1)), 0, 1)

    med = np.zeros((grid_dim,time_dim,3), dtype=np.float32)
    # med[..., 1] = SNOWPACK_nor
    # med[..., 2] = MELTWATER_nor
    # med[..., 3] = SM_nor
    med[..., 1] = Qall_nor
    med[..., 2] = SUZ_nor
    med[..., 3] = SLZ_nor

    return med



# def load_parameters_for_HBV(cfg):
#     parameters = {}
#     for file_name in cfg['parameters_names']:
#         file_path = os.path.join(cfg['parameters_path'], file_name)
#         dataset = nc.Dataset(file_path)
#         var_name = file_name.split('.')[0]
#         var_values = dataset.variables[var_name][:]
#         parameters[var_name] = var_values
#         dataset.close()
#     return parameters




def HBVfortest( x, parameters, i, j):


    parBETA = parameters['BETA'][i][j].astype(np.float32)
    parFC = parameters['FC'][i][j].astype(np.float32)
    parK0 = parameters['K0'][i][j].astype(np.float32)
    parK1 = parameters['K1'][i][j].astype(np.float32)
    parK2 = parameters['K2'][i][j].astype(np.float32)
    parLP = parameters['LP'][i][j].astype(np.float32)
    parPERC = parameters['PERC'][i][j].astype(np.float32)
    parUZL = parameters['UZL'][i][j].astype(np.float32)
    parTT = parameters['TT'][i][j].astype(np.float32)
    parCFMAX = parameters['CFMAX'][i][j].astype(np.float32)
    parCFR = parameters['CFR'][i][j].astype(np.float32)
    parCWH = parameters['CWH'][i][j].astype(np.float32)
    parPCORR = parameters['PCORR'][i][j].astype(np.float32)

    PRECS = 1e-5  # keep the numerical calculation stable



    para_shape = parameters['BETA'][i][j].shape
    Qall = np.zeros(para_shape, dtype=np.float32) + 0.001
    SNOWPACK = np.zeros(para_shape, dtype=np.float32) + 0.001
    MELTWATER = np.zeros(para_shape, dtype=np.float32) + 0.001
    SM = np.zeros(para_shape, dtype=np.float32) + 0.001
    SUZ = np.zeros(para_shape, dtype=np.float32) + 0.001
    SLZ = np.zeros(para_shape, dtype=np.float32) + 0.001


    time_dim, var_dim = x.shape
    P = x[:, 0]
    T = x[:, 1]
    T = T - 273.15
    ETpot = x[:, -1]
    ETpot *= 1000
    P *= 1000


    # Qall_list = []
    # SNOWPACK_list = []
    # MELTWATER_list = []
    # SM_list = []
    SUZ_list = []
    SLZ_list = []

    for t in range(time_dim):
        PRECIP = np.multiply(P[t], parPCORR)
        RAIN = np.multiply(PRECIP, (T[t] >= parTT).astype(np.float32))
        SNOW = np.multiply(PRECIP, (T[t] < parTT).astype(np.float32))

        SNOWPACK = SNOWPACK + SNOW
        melt = parCFMAX * (T[t] - parTT)
        melt = np.clip(melt, 0.0, None)
        melt = np.minimum(melt, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = parCFR * parCFMAX * (parTT - T[t])
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
        # evapfactor = SM / (parLP * parFC)
        ETact = ETpot[t]
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



        # Qall_list.append(Qall)
        # SNOWPACK_list.append(SNOWPACK)
        # MELTWATER_list.append(MELTWATER)
        # SM_list.append(SM)
        SUZ_list.append(SUZ)
        SLZ_list.append(SLZ)

    # Qall = np.array(Qall_list)
    # SNOWPACK = np.array(SNOWPACK_list)
    # MELTWATER = np.array(MELTWATER_list)
    # SM = np.array(SM_list)
    SUZ = np.array(SUZ_list)
    SLZ = np.array(SLZ_list)
    
    # Qall_nor = np.clip((Qall - (-0.1)) / (170 - (-0.1)), 0, 1)
    # SNOWPACK_nor = np.clip((SNOWPACK - (-0.1)) / (300 - (-0.1)), 0, 1)
    # MELTWATER_nor = np.clip((MELTWATER - (-1)) / (10 - (-1)), 0, 1)
    # SM_nor = np.clip((SM - 1e-5) / (1000 - 1e-5), 0, 1)
    SUZ_nor = np.clip((SUZ - 0) / (700 - 0), 0, 1)
    SLZ_nor = np.clip((SLZ - (-0.1)) / (150 - (-0.1)), 0, 1)
    

    med = np.zeros((time_dim, 2), dtype=np.float32)
    # med[..., 0] = Qall_nor
    # med[..., 1] = SNOWPACK_nor
    # med[..., 2] = MELTWATER_nor
    # med[..., 3] = SM_nor
    med[..., 0] = SUZ_nor
    med[..., 1] = SLZ_nor


    return med

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

