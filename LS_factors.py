import pandas as pd
import numpy as np


def calc_shift(data: pd.DataFrame):
    """
    shift[t] = (1/N_t) * \sum_M ( Y_{M,t} - Y_{M,t-1} )
    """
    shift = np.zeros(len(data))
    shift[0] = np.nan
    maturities = data.columns[1:]
    for t in range(1, len(data)):
        N_t = 0
        for maturity in maturities:
            if not (np.isnan(data.iloc[t][maturity]) or np.isnan(data.iloc[t-1][maturity])):
                shift[t] += data.iloc[t][maturity] - data.iloc[t-1][maturity]
                N_t += 1
        if N_t == 0:
            shift[t] = np.nan
        else:
            shift[t] /= N_t
    return shift


def calc_tilt(data: pd.DataFrame):
    """
    tilt[t] = Slope[t] - Slope[t-1]

    where
    slope[t] = Cov(Y_{M,t}, M) / Var(M)
    """
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    maturity_labels = data.columns[1:]
    tilt = np.zeros(len(data))
    tilt[:] = np.nan

    for t in range(1, len(data)):
        curr_yields = []
        curr_maturities = []
        prev_yields = []
        prev_maturities = []

        for j, maturity_label in enumerate(maturity_labels):
            if (not np.isnan(data.iloc[t][maturity_label]) and
                    not np.isnan(data.iloc[t-1][maturity_label])):
                # at time t
                curr_yields.append(data.iloc[t][maturity_label])
                curr_maturities.append(maturities[j])

                # at time t-1
                prev_yields.append(data.iloc[t-1][maturity_label])
                prev_maturities.append(maturities[j])
        
        curr_slope = np.cov(curr_yields, curr_maturities)[0][1] / np.var(curr_maturities)
        prev_slope = np.cov(prev_yields, prev_maturities)[0][1] / np.var(prev_maturities)
        tilt[t] = curr_slope - prev_slope

    return tilt


def calc_flex(data: pd.DataFrame):
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    mid_maturities = []
    for j in range(len(maturities)):
        if j == 0:
            mid_maturities.append(0)
        else:
            mid_maturities.append(0.5*(maturities[j] + maturities[j-1]))
    
    maturity_labels = data.columns[1:]
    assert len(maturity_labels) == len(maturities)
    flex = np.zeros(len(data))
    flex[:] = np.nan

    for t in range(1, len(data)):
        curr_yield_slopes = []
        prev_yield_slopes = []
        curr_mid_maturities = []
        for j in range(1, len(maturity_labels)):
            # to calculate curr_yield_slope
            Y_Mj_t = data.iloc[t][maturity_labels[j]]
            Y_Mjm1_t = data.iloc[t][maturity_labels[j-1]]

            # to calculate prev_yield_slope
            Y_Mj_tm1 = data.iloc[t-1][maturity_labels[j]]
            Y_Mjm1_tm1 = data.iloc[t-1][maturity_labels[j-1]]

            if not (np.isnan(Y_Mj_t) or np.isnan(Y_Mjm1_t) or np.isnan(Y_Mj_tm1) or np.isnan(Y_Mjm1_tm1)):
                curr_yield_slope = (Y_Mj_t - Y_Mjm1_t) / (maturities[j] - maturities[j-1])
                prev_yield_slope = (Y_Mj_tm1 - Y_Mjm1_tm1) / (maturities[j] - maturities[j-1])
                curr_yield_slopes.append(curr_yield_slope)
                prev_yield_slopes.append(prev_yield_slope)
                curr_mid_maturities.append(mid_maturities[j])
        
        first_term = np.cov(curr_yield_slopes, curr_mid_maturities)[0][1] / np.var(curr_mid_maturities)
        second_term = np.cov(prev_yield_slopes, curr_mid_maturities)[0][1] / np.var(curr_mid_maturities)

        flex[t] = -1*first_term + second_term
    
    return flex
        

if __name__ == "__main__":
    df = pd.read_csv("data/FRB_H15.csv")
    LS_factors = calc_shift(df), calc_tilt(df), calc_flex(df)
    LS_factors = np.stack(LS_factors, axis=1)

    months = np.expand_dims(df["Maturity>>>"].values, axis=1)
    maturity_labels = [df.columns[1:][i] for i in range(len(df.columns[1:]))]

    LS_factors = pd.DataFrame(np.concatenate((months, LS_factors), axis=1),
                              columns=["Holding period", "Shift", "Tilt", "Flex"])
    LS_factors.to_csv("data/LS_factors.csv", index=False)
