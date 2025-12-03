from datetime import datetime
import pandas as pd
import configparser
import math


def read_setup_file(setup_file):
    setup = configparser.ConfigParser()
    setup.read(setup_file)
    return setup


def load_input_files_old(event, pavement):
    rain_data = pd.read_csv(rf".\input_files_level\{pavement}\event_{event}\rainfall.dat", sep=" ", header=None)
    rain_data = rain_data.iloc[:, 2]
    pavement_area = 2100  # sq.ft
    area = pavement_area * 0.092903  # m2
    tQin = rain_data * area * 0.0254 / 60
    tQrain = [0] * len(tQin)
    tEmax = [0] * len(tQin)
    return tQin, tQrain, tEmax


def load_input_files(parameters, rainfall_file,  evt_file=None, inflow_file=None, rainfall_unit='in'):
    surface_area = parameters['area']
    dt = parameters['dt']
    rain_data, _ = rainfall_data_treatment(rainfall_file, surface_area, dt, rainfall_unit)
    if evt_file is None:
        evt_data = [0] * len(rain_data)
    if inflow_file is None:
        inflow_data = [0] * len(rain_data)

    return inflow_data, rain_data, evt_data


def initialize_parameters(setup):
    parameters = {}

    # General parameters
    parameters['kc'] = float(setup['GENERAL']['Kc'])
    parameters['df'] = float(setup['GENERAL']['Df'])
    parameters['dtl'] = float(setup['GENERAL']['Dtl'])
    parameters['dg'] = float(setup['GENERAL']['Dg'])
    parameters['l'] = parameters['df'] + parameters['dtl'] + parameters['dg']
    parameters['nf'] = float(setup['GENERAL']['nf'])
    parameters['nt'] = float(setup['GENERAL']['nt'])
    parameters['ng'] = float(setup['GENERAL']['ng'])

    # Ponding zone parameters
    parameters['area'] = float(setup['PONDING_ZONE']['Ap'])
    parameters['hover'] = float(setup['PONDING_ZONE']['Hover'])
    parameters['kweir'] = float(setup['PONDING_ZONE']['Kweir'])
    parameters['wweir'] = float(setup['PONDING_ZONE']['wWeir'])
    parameters['expweir'] = float(setup['PONDING_ZONE']['expWeir'])
    parameters['cs'] = float(setup['PONDING_ZONE']['Cs'])
    parameters['pp'] = float(setup['PONDING_ZONE']['Pp'])
    parameters['flagp'] = float(setup['PONDING_ZONE']['flagp'])

    # Unsaturated zone parameters
    parameters['a'] = float(setup['UNSATURATED_ZONE']['A'])
    #considering the whole pavement depth (L) is unsaturated (empty)
    parameters['husz_ini'] = parameters['l']
    # weighted porosity of entire unsaturated layer considering it starts empty
    parameters['nusz_ini'] = (
        parameters['nf'] * parameters['df'] +
        parameters['nt'] * parameters['dtl'] +
        parameters['ng'] * parameters['dg']
    ) / parameters['l']
    parameters['ks'] = float(setup['UNSATURATED_ZONE']['Ks'])
    parameters['sh'] = float(setup['UNSATURATED_ZONE']['sh'])
    parameters['sw'] = float(setup['UNSATURATED_ZONE']['sw'])
    parameters['sfc'] = float(setup['UNSATURATED_ZONE']['sfc'])
    parameters['ss'] = float(setup['UNSATURATED_ZONE']['ss'])
    parameters['gama'] = float(setup['UNSATURATED_ZONE']['gama'])
    parameters['kf'] = float(setup['UNSATURATED_ZONE']['Kf'])

    # Saturated zone parameters
    parameters['psz'] = float(setup['SATURATED_ZONE']['Psz'])
    parameters['hpipe'] = float(setup['SATURATED_ZONE']['hpipe'])
    parameters['flagsz'] = float(setup['SATURATED_ZONE']['flagsz'])
    parameters['dpipe'] = float(setup['SATURATED_ZONE']['dpipe']) / 1000
    parameters['cd'] = float(setup['SATURATED_ZONE']['Cd'])
    parameters['apipe'] = math.pi * (parameters['dpipe'] / 2) ** 2
    parameters['eta'] = float(setup['SATURATED_ZONE']['eta'])

    # Timestep
    parameters['dt'] = float(setup['TIMESTEP']['dt'])

    return parameters


def modify_parameters(parameters, calibrated_params=None):
    for key, value in calibrated_params.items():
        # Check if the key exists in the parameters dictionary and update it
        if key in parameters:
            parameters[key] = value

    if ("df" or "dtl" or "dg") in calibrated_params.keys():
        parameters["l"] = parameters["df"] + parameters["dtl"] + parameters["dg"]

    return parameters


def cQet(sw, sh, ss, Kc, Emax, A, sEST, dt):
    if sEST <= sh:
        Qet = 0.0
    elif sEST <= sw:
        Qet = A * Emax * Kc * (sEST - sh) / (sw - sh)
    elif sEST <= ss:
        Qet = A * Emax * Kc * (sEST - sw) / (ss - sw)
    else:
        Qet = A * Emax * Kc

    Qet = Qet / (dt * 1000)
    return Qet


def cQover(Kweir, wWeir, hp, Hover, expWeir, Ab, dt, Qin, Qrain):
    Vcheck = hp * Ab + dt * (Qin + Qrain)
    if Vcheck > Hover * Ab:
        Hcheck = Vcheck / Ab
        Qover = Kweir * (Hcheck - Hover) ** expWeir
    else:
        Qover = 0
    return Qover


def cQinfp(Kf, Ab, A, Cs, Pp, flagp, hpEST):
    if flagp == 1:
        Qinfp = 0
    else:
        Qinfp = Kf * ((Ab - A) + Cs * Pp * hpEST)
    return Qinfp


def cQpf(Ks, hp, husz, A, Ab, dt, s, nusz, Qinfp):
    Qpf = min(Ks * A * (hp + husz) / husz, hp * Ab / dt - Qinfp, (1.0 - s) * nusz * husz * A / dt)
    return Qpf


def cQhc(A, ss, sfc, Emax, sEST, Kc, dt):
    s2 = sEST
    den = sfc - ss
    if den == 0:
        den = 0.000001

    Cr = 4 * Emax * Kc / (2.5 * (den) ** 2)
    if ss <= s2 <= sfc:
        Qhc = A * Cr * (s2 - ss) * (sfc - s2)
    else:
        Qhc = 0
    return Qhc


def cQfs(A, Ks, hp, husz, gama, nusz, dt, sfc, sEST):
    if sEST >= sfc:
        Qfs = min((A * Ks * (hp + husz) / husz) * sEST ** gama, (sEST - sfc) * nusz * A * husz / dt)
    else:
        Qfs = 0
    return Qfs


def cQinfsz(Kf, A, Cs, Psz, flagsz, hszEST):
    if flagsz == 1:
        Qinfsz = 0.0
    else:
        Qinfsz = Kf * (A + Cs * Psz * hszEST)
    return Qinfsz


def cQpipe(hpipe, A, nsz, dt, Qinfsz, Apipe, hszEST, Cd, eta):
    if hszEST <= hpipe:
        Qpipe = 0
    else:
        Qpipemax = (hszEST - hpipe) * A * nsz / dt - Qinfsz
        Qpipemax = max(0, Qpipemax)
        Qpipepossible = Cd * Apipe * ((hszEST - hpipe) * 2 * 9.81) ** eta
        Qpipepossible = max(0, Qpipepossible)
        Qpipe = min(Qpipemax, Qpipepossible)
    return Qpipe


def cnsz(hsz, L, Dtl, Dg, nf, nt, ng):
    if Dtl + Dg < hsz <= L:
        nsz = ((ng * Dg + nt * Dtl + nf * (hsz - Dg - Dtl))) / hsz
    elif Dg < hsz <= Dg + Dtl:
        nsz = (ng * Dg + nt * (hsz - Dg)) / hsz
    else:
        nsz = ng
    return nsz


def cnusz(husz, hsz, nusz_ini, ng, Dg, Df):
    if hsz < Dg:
        nusz = (nusz_ini * Df + ng * (Dg - hsz)) / husz
    else:
        nusz = nusz_ini
    return nusz


def read_rainfall_dat_file(rainfall_file):
    rainfall = pd.read_csv(
        rainfall_file,
        sep=r"\s+",
        header=None,
        names=["date_str", "time_str", "rain"]
    )
    rainfall["date"] = pd.to_datetime(rainfall["date_str"] + " " + rainfall["time_str"],
                                  format="%m/%d/%Y %H:%M")
    rainfall_df = rainfall[["date", "rain"]]
    return rainfall_df


def rainfall_data_treatment(rainfall_file, surface_area, dt, rainfall_unit='mm'):
    rainfall = read_rainfall_dat_file(rainfall_file)
    rain_data = rainfall['rain']
    date = rainfall['date']

    rainfall_unit = rainfall_unit.lower()

    # Converts the rainfall to flow in m3/s based on its unit
    if rainfall_unit == 'mm':
        rain_data = ((rain_data / 1000) * surface_area) / dt

    elif rainfall_unit == 'in':
        rain_data = (rain_data * 0.0254 * surface_area) / dt

    else:
        raise ValueError('rainfall_unit must be "mm" or "in"')

    return rain_data, date


def run_model(parameters, rainfall_file, inflow=None, evapotranspiration=None, rainfall_unit='mm'):
    surface_area = parameters['area']
    dt = parameters['dt']

    rain_data, date = rainfall_data_treatment(rainfall_file, surface_area, dt, rainfall_unit)

    if inflow is None:
        inflow = [0] * len(rain_data)

    if evapotranspiration is None:
        evapotranspiration = [0] * len(rain_data)

    results = {
        'date': list(date), 'tt': [], 'tQrain': list(rain_data), 'tQin': list(inflow), 'tQover': [], 'tQpf': [], 'tQinfp': [], 'tQfs': [],
        'tQhc': [], 'tQet': [], 'tQinfsz': [], 'tQpipe': [], 'tQet1': [],
        'tQet2': [], 'thp': [], 'ts': [], 'thsz': [], 'thszEST': [],
        'thusz': [], 'tnsz': [], 'tnusz': [], 'thpEND': [], 'tteta_usz': [],
        'tteta_sz': []
    }

    hp_end = 0
    husz = max(parameters['husz_ini'], 1e-6)
    hsz = max(parameters['hpipe'], 1e-6) if parameters['hpipe'] > 0 else parameters['dpipe']
    nusz = max(parameters['nusz_ini'], 1e-6)
    nsz = max(parameters['ng'], 1e-6)
    s = 0

    for t in range(len(rain_data)):
        qin = inflow[t]
        qrain = rain_data[t]
        emax = evapotranspiration[t]

        qover = cQover(parameters['kweir'], parameters['wweir'], hp_end, parameters['hover'], parameters['expweir'], parameters['area'], parameters['dt'], qin, qrain)
        hp = max(hp_end + parameters['dt'] / max(parameters['area'], 1e-6) * (qrain + qin - qover), 0)

        qinfp = cQinfp(parameters['kf'], parameters['area'], parameters['a'], parameters['cs'], parameters['pp'], parameters['flagp'], hp)
        qpf = min(
            parameters['ks'] * parameters['a'] * (hp + husz) / max(husz, 1e-6),
            hp * parameters['area'] / max(parameters['dt'], 1e-6) - qinfp,
            (1.0 - s) * nusz * husz * parameters['a'] / max(parameters['dt'], 1e-6)
        )

        hp_end = max(hp - parameters['dt'] / max(parameters['area'], 1e-6) * (qpf + qinfp), 0)

        sest = max(min(s + qpf * parameters['dt'] / max((nusz * parameters['a'] * husz), 1e-6), 1), 0)
        qhc = cQhc(parameters['a'], parameters['ss'], parameters['sfc'], emax, sest, parameters['kc'], parameters['dt'])

        qfs = cQfs(parameters['a'], parameters['ks'], hp_end, husz, parameters['gama'], nusz, parameters['dt'], parameters['sfc'], sest)

        sest2 = (sest * nusz * husz + nsz * hsz) / max((nusz * husz + nsz * hsz), 1e-6)

        qet = cQet(parameters['sw'], parameters['sh'], parameters['ss'], parameters['kc'], emax, parameters['a'], sest2, parameters['dt'])
        qet1 = qet * (sest * nusz * husz) / max((sest * nusz * husz + nsz * hsz), 1e-6)
        qet2 = qet - qet1

        hszest = hsz + parameters['dt'] * (qfs - qhc - qet2) / max(parameters['a'], 1e-6) / max(nsz, 1e-6)

        qinfsz = cQinfsz(parameters['kf'], parameters['a'], parameters['cs'], parameters['psz'], parameters['flagsz'], hszest)
        qpipe = cQpipe(parameters['hpipe'], parameters['a'], nsz, parameters['dt'], qinfsz, parameters['apipe'], hszest, parameters['cd'], parameters['eta'])

        hsz = hsz + parameters['dt'] * (qfs - qhc - qinfsz - qpipe - qet2) / max(parameters['a'], 1e-6) / max(nsz, 1e-6)
        husz = max(parameters['l'] - hsz, 1e-6)

        nsz = max(cnsz(hsz, parameters['l'], parameters['dtl'], parameters['dg'], parameters['nf'], parameters['nt'], parameters['ng']), 1e-6)
        nusz = max(cnusz(husz, hsz, parameters['nusz_ini'], parameters['ng'], parameters['dg'], parameters['df']), 1e-6)

        s = max(
            min(
                1.0,
                (s * husz * nusz * parameters['a'] + parameters['dt'] * (qpf + qhc - qfs - qet1)) /
                max(parameters['a'] * husz * nusz, 1e-6)
            ),
            parameters['sh']
        )
        # Save results
        results['tt'].append(t)
        results['tQover'].append(qover)
        results['tQpf'].append(qpf)
        results['tQinfp'].append(qinfp)
        results['tQfs'].append(qfs)
        results['tQhc'].append(qhc)
        results['tQet'].append(qet)
        results['tQinfsz'].append(qinfsz)
        results['tQpipe'].append(qpipe)
        results['tQet1'].append(qet1)
        results['tQet2'].append(qet2)
        results['thp'].append(hp)
        results['ts'].append(s)
        results['thusz'].append(husz)
        results['thsz'].append(hsz)
        results['thszEST'].append(hszest)
        results['tnsz'].append(nsz)
        results['tnusz'].append(nusz)
        results['thpEND'].append(hp_end)
        results['tteta_usz'].append(s * nusz)
        results['tteta_sz'].append(nsz)

    return results

def results_dataframe(results, save=False, filename="water_flow_results.csv"):
    dict_data = {
        'date': results['date'][:len(results['tt'])],
        't': results['tt'],
        'Qrain': results['tQrain'][:len(results['tt'])],
        'Qin': results['tQin'][:len(results['tt'])],
        'Qet': results['tQet'][:len(results['tt'])],
        'hpEND': results['thpEND'][:len(results['tt'])],
        'Qpf': results['tQpf'][:len(results['tt'])],
        'Qover': results['tQover'][:len(results['tt'])],
        'Qfs': results['tQfs'][:len(results['tt'])],
        'Qet_1': results['tQet1'][:len(results['tt'])],
        'Qhc': results['tQhc'][:len(results['tt'])],
        'Qpipe': results['tQpipe'][:len(results['tt'])],
        'Qet_2': results['tQet2'][:len(results['tt'])],
        'teta_usz': results['tteta_usz'][:len(results['tt'])],
        'teta_sz': results['tteta_sz'][:len(results['tt'])],
        'Qinfp': results['tQinfp'][:len(results['tt'])],
        'Qinfsz': results['tQinfsz'][:len(results['tt'])],
        'hp': results['thp'][:len(results['tt'])],
        's': results['ts'][:len(results['tt'])],
        'husz': results['thusz'][:len(results['tt'])],
        'hsz': results['thsz'][:len(results['tt'])],
        'nsz': results['tnsz'][:len(results['tt'])],
        'nusz': results['tnusz'][:len(results['tt'])],
        'hszEST': results['thszEST'][:len(results['tt'])]
    }

    data = pd.DataFrame(dict_data)

    if save:
        data.to_csv(filename, index=False)

    return data


def calculate_water_balance(data, dt):
    Qin_total = data['Qin'].sum()
    Vtotal_in = Qin_total * dt

    Qover_total = data['Qover'].sum()
    Vtotal_over = Qover_total * dt

    Qpipe_total = data['Qpipe'].sum()
    Vtotal_pipe = Qpipe_total * dt

    Qpf_total = data['Qpf'].sum()
    Vtotal_pf = Qpf_total * dt

    Qfs_total = data['Qfs'].sum()
    Vtotal_fs = Qfs_total * dt

    Smax = data['s'].max()
    hmax = data['hp'].max()
    Qpeak_over = data['Qover'].max()
    tpeak = data.loc[data['Qover'] == Qpeak_over, 't'].iloc[0]

    water_balance = pd.DataFrame({
        'Vtotal_in': [Vtotal_in],
        'Vtotal_over': [Vtotal_over],
        'Vtotal_pipe (m3)': [Vtotal_pipe],
        'Vtotal_pf': [Vtotal_pf],
        'Vtotal_fs': [Vtotal_fs],
        'Qpeak_over (L/s)': [Qpeak_over * 1000],
        'Smax (m3)': [Smax],
        'hmax (m)': [hmax],
        'tpeak (min)': [tpeak]
    })

    return water_balance


def run_single_timestep(parameters, qin, qrain, emax,
                        hp_prev, s_prev, hsz_prev, husz_prev,
                        nusz_prev, nsz_prev):
    """
    Run a single time step of the water flow model with given parameters and initial conditions.

    Closely mirrors the original run_model function logic to be applied to the particle filter.
    """
    # Initialize state variables
    hp_end = hp_prev
    hsz = hsz_prev
    husz = husz_prev
    nusz = nusz_prev
    nsz = nsz_prev
    s = s_prev

    # Calculate ponding overflow
    qover = cQover(parameters['kweir'], parameters['wweir'], hp_end,
                   parameters['hover'], parameters['expweir'],
                   parameters['area'], parameters['dt'], qin, qrain)

    # Update ponding depth
    hp = max(hp_end + parameters['dt'] / max(parameters['area'], 1e-6) *
             (qrain + qin - qover), 0)

    # Calculate infiltration
    qinfp = cQinfp(parameters['kf'], parameters['area'], parameters['a'],
                   parameters['cs'], parameters['pp'], parameters['flagp'], hp)

    # Calculate percolation through filtration zone
    qpf = min(
        parameters['ks'] * parameters['a'] * (hp + husz) / max(husz, 1e-6),
        hp * parameters['area'] / max(parameters['dt'], 1e-6) - qinfp,
        (1.0 - s) * nusz * husz * parameters['a'] / max(parameters['dt'], 1e-6)
    )

    # Update ponding depth at end of time step
    hp_end = max(hp - parameters['dt'] / max(parameters['area'], 1e-6) * (qpf + qinfp), 0)

    # Update moisture content
    sest = max(min(s + qpf * parameters['dt'] /
                   max((nusz * parameters['a'] * husz), 1e-6), 1), 0)

    # Calculate heat capacity term
    qhc = cQhc(parameters['a'], parameters['ss'], parameters['sfc'],
               emax, sest, parameters['kc'], parameters['dt'])

    # Calculate free storage flow
    qfs = cQfs(parameters['a'], parameters['ks'], hp_end, husz,
               parameters['gama'], nusz, parameters['dt'],
               parameters['sfc'], sest)

    # Weighted average moisture content
    sest2 = (sest * nusz * husz + nsz * hsz) / max((nusz * husz + nsz * hsz), 1e-6)

    # Calculate evapotranspiration
    qet = cQet(parameters['sw'], parameters['sh'], parameters['ss'],
               parameters['kc'], emax, parameters['a'], sest2, parameters['dt'])

    # Split evapotranspiration
    qet1 = qet * (sest * nusz * husz) / max((sest * nusz * husz + nsz * hsz), 1e-6)
    qet2 = qet - qet1

    # Estimate saturated zone depth
    hszest = hsz + parameters['dt'] * (qfs - qhc - qet2) / max(parameters['a'], 1e-6) / max(nsz, 1e-6)

    # Calculate infiltration to saturated zone
    qinfsz = cQinfsz(parameters['kf'], parameters['a'], parameters['cs'],
                     parameters['psz'], parameters['flagsz'], hszest)

    # Calculate pipe outflow
    qpipe = cQpipe(parameters['hpipe'], parameters['a'], nsz,
                   parameters['dt'], qinfsz, parameters['apipe'],
                   hszest, parameters['cd'], parameters['eta'])

    # Update saturated zone depth
    hsz = hsz + parameters['dt'] * (qfs - qhc - qinfsz - qpipe - qet2) / max(parameters['a'], 1e-6) / max(nsz, 1e-6)

    # Update unsaturated zone depth
    husz = max(parameters['l'] - hsz, 1e-6)

    # Update porosity of saturated zone
    nsz = max(cnsz(hsz, parameters['l'], parameters['dtl'], parameters['dg'],
                   parameters['nf'], parameters['nt'], parameters['ng']), 1e-6)

    # Update porosity of unsaturated zone
    nusz = max(cnusz(husz, hsz, parameters['nusz_ini'],
                     parameters['ng'], parameters['dg'], parameters['df']), 1e-6)

    # Update moisture content
    s = max(min(1.0,
                (s * husz * nusz * parameters['a'] + parameters['dt'] * (qpf + qhc - qfs - qet1)) /
                max(parameters['a'] * husz * nusz, 1e-6)),
            parameters['sh'])

    return {
        'hp': hp_end,
        's': s,
        'hsz': hsz,
        'husz': husz,
        'nusz': nusz,
        'nsz': nsz,
        'Qpipe': qpipe
    }


def run_simulation(params: dict,
                   rainfall_file,
                   inflow=None,
                   evapotranspiration=None,
                   rainfall_unit='mm',
                   verbose: bool = True,
                   plot_outflow: bool = False,
                   output_path: str = None):
    """
    Run the water-flow model for the supplied data series.

    Parameters
    ----------
    params : dict
        Complete parameter dictionary created by `initialize_parameters`
        (and optionally modified by `modify_parameters`).
    qin, qrain, emax : 1-D sequences
        Forcing series of equal length.
    verbose : bool, default True
        Print water balance and runtime summary.

    Returns
    -------
    data : pandas.DataFrame
        Timestep results (see `results_dataframe`).
    wb   : pandas.DataFrame
        One-row water-balance summary.
    """
    start_time = datetime.now()

    # --- run main solver ------------------------------------------------
    results = run_model(params, rainfall_file, inflow, evapotranspiration, rainfall_unit)
    data     = results_dataframe(results, save=False)
    wb       = calculate_water_balance(data, params['dt'])

    if verbose:
        print(wb.to_string())
        print("Elapsed time:", datetime.now() - start_time)

    # Plotting functionality removed - use external plotting if needed
    # if plot_outflow:
    #     plot_rainfall_hydrograph(rainfall_file, data['Qpipe'], rainfall_unit=rainfall_unit, output_path=output_path)

    return data, wb


def run_from_files(pavement: str,
                   event: int,
                   input_folder: str = "input_files",
                   calibrated_parameters: dict | None = None,
                   verbose: bool = True):
    """
    Legacy wrapper that keeps the original behaviour but delegates the
    heavy lifting to `run_simulation`.
    """
    # ---- parameter setup ---------------------------------------------
    setup_file = f"{input_folder}/tc_pf_{pavement}.ini"
    setup      = read_setup_file(setup_file)
    params     = initialize_parameters(setup)

    if calibrated_parameters:
        params = modify_parameters(params, calibrated_parameters)

    # ---- data loading ------------------------------------------------
    qin, qrain, emax = load_input_files_old(event=event, pavement=pavement)

    # ---- run ---------------------------------------------------------
    return run_simulation(params, qin, qrain, emax, verbose=verbose)
