from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


OUT_PATH = "results/coefficients.csv"


if __name__ == "__main__":
    inputs = pd.read_csv("data/LS_factors.csv")
    targets = pd.read_csv("data/holding_period_returns.csv")

    with open(OUT_PATH, 'w') as f:
        f.write("Maturity,R^2,Shift,Tilt,Flex,Intercept\n")

    for maturity in targets.columns[1:]:
        y = targets[maturity].values
        X = inputs.drop(columns=["Holding period",]).values
        assert len(y) == len(X)

        indices = []
        for i in range(len(y)):
            missing_LS_factors = False
            for j in range(len(X[i])):
                if np.isnan(X[i][j]):
                    print(f"Missing LS factor at time {i}")
                    missing_LS_factors = True
            if np.isnan(y[i]):
                print(f"Missing target at time {i} for maturity {maturity}")
            if not (np.isnan(y[i]) or missing_LS_factors):
                indices.append(i)

        y = y[indices]
        X = X[indices]
        assert len(y) == len(X)

        model = LinearRegression()
        model.fit(X, y)
        R_sq = model.score(X, y)

        # save results to OUT_PATH
        with open(OUT_PATH, 'a') as f:
            f.write(maturity + ',' + str(R_sq) + ',' + ','.join(map(str, model.coef_)) + ',' + str(model.intercept_) + '\n')
