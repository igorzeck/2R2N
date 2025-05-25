## Funções Estatísticas úteis para regressão
## 1. Médias são feitas pelo numpy
import pandas as pd


# Covariância
def cov(x: list, y: list, amostral=True) -> float:
    n = len(x)
    if n != len(y):
        raise Exception("Variável X tem tamanho diferente da Y!")
    return sum([((x[i] - media(x))*(y[i] - media(y)))/(n - (1 if amostral else 0)) for i in range(n)])


# Variância
def var(x: list, amostral=True) -> float:
    return cov(x, x, amostral)


# Regressão linear
def reg_lin(dt: pd.DataFrame, col_x, col_y) -> tuple:
    # Coeficiente angular
    coef_a = cov(dt[col_x], dt[col_y])/var(dt[col_x])

    # Coeficiente independente
    coef_b = media(dt[col_y]) - media(dt[col_x])*coef_a
    return (coef_a, coef_b)