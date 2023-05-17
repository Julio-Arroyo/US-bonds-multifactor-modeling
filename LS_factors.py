import pandas as pd
import numpy as np


def calc_shift(data: pd.DataFrame):
    """
    shift[t] = (1/N_t) * \sum_M ( Y_{M,t} - Y_{M,t-1} )
    """
    shift = np.zeros(len(data))
    maturities = data.columns[1:]
    for t in range(1, len(data)):
        N_t = 0
        for maturity in maturities:
            if not np.isnan(data.iloc[t][maturity]) and not np.isnan(data.iloc[t-1][maturity]):
                shift[t] += data.iloc[t][maturity] - data.iloc[t-1][maturity]
                N_t += 1
        assert not N_t == 0, "N_t is zero"
    return shift


if __name__ == "__main__":
    df = pd.read_csv("FRB_H15.csv")
    print(len(calc_shift(df)))
