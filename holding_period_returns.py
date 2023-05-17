import pandas as pd
import numpy as np


def calc_returns(data: pd.DataFrame):
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    maturity_labels = data.columns[1:]
    assert len(maturities) == len(maturity_labels), "maturities and maturity_labels have different lengths"
    returns = np.zeros((len(data), len(maturity_labels)))
    returns[0, :] = np.nan

    for j in range(len(maturity_labels)):
        N = maturities[j]
        for t in range(1, len(data)):
            C_N_t = data.iloc[t][maturity_labels[j]]
            C_N_tm1 = data.iloc[t-1][maturity_labels[j]]
            if not (np.isnan(C_N_t) or np.isnan(C_N_tm1) or C_N_t == 0):
                coupon_ratio = C_N_tm1 / C_N_t
                forward_return = 1 + C_N_t/200
                den = np.power(forward_return, 2*N - 1/6)
                if den == 0:
                    returns[t, j] = np.nan

                pv_coupons = coupon_ratio * (1 - ( 1/( den ) ))
                pv_principal = 1/den

                P_N_t = pv_coupons + pv_principal
                returns[t, j] = P_N_t + (1/1200)*C_N_tm1 - 1
            else:
                returns[t, j] = np.nan
    
    months = np.expand_dims(data["Maturity>>>"].values, axis=1)
    maturity_labels = [maturity_labels[i] for i in range(len(maturity_labels))]

    returns_df = pd.DataFrame(np.concatenate((months, returns), axis=1),
                              columns=["Holding period",] + maturity_labels)
    returns_df.to_csv("holding_period_returns.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("FRB_H15.csv")
    calc_returns(df)
