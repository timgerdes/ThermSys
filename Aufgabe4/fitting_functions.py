from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os


class FitFuntion:
    def __init__(self, x_data=None, y_data=None, params=None, x_max=None):
        if params is not None:
            self.params = params
        elif x_data is not None and y_data is not None:
            self.x_data = np.array(x_data)
            self.x_max = self.x_data.max()
            self.y_data = np.array(y_data)
            if hasattr(self, 'bounds'):
                self.params, self.cov = curve_fit(self.func, x_data, y_data, bounds=self.bounds)
            else:
                self.params, self.cov = curve_fit(self.func, x_data, y_data)
        else:
            raise ValueError("Entweder Daten oder Parameter müssen übergeben werden.")

        if x_max is not None:
            self.x_max = x_max

    def func(self, x, *params):
        pass

    def get_params(self):
        return self.params

    def get_covariance(self):
        return self.cov

    def get_data(self):
        return {"x": self.x_data, "y": self.y_data} if hasattr(self, 'x_data') else None

    def get_x_max(self):
        return self.x_max

class LinearFunction(FitFuntion):
    def func(self, x, a, b):
        return a * x + b


class ExponentialFunction(FitFuntion):
    def func(self, x, a, b):
        return a ** x + b


class LogarithmFunction(FitFuntion):
    bounds = ((1e-10, -np.inf), (np.inf, np.inf))
    def func(self, x, a, b):
        # Logarithmus von (x+b) zur Basis a
        return np.log(x + b) / np.log(a)

class QuadraticFunction(FitFuntion):
    def func(self, x, a, b, c):
        return a * x**2 + b * x + c

def read_csv(file):
    return pd.read_csv(os.path.dirname(__file__) + "/" + file, sep=";")


def fit_curves():
    functions = {}

    # Fitten Volumenstrom
    v_dot_data = read_csv("volumenstrom.csv")
    v_dot = np.array(v_dot_data["volumenstrom"]) / 3600  # [m^3/s]
    v_dot_dp = np.array(v_dot_data["druckdifferenz"]) * 100  # [Pa]

    # Massenstrom aus Volumenstrom
    p_0 = 1.013e5  # Druck [Pa]
    t_0 = 293.15  # Temperatur [K]
    r_L = 287  # Gaskonstante für Luft [J/kg*K]

    v_dot_m_dot = (p_0 * v_dot) / (r_L * t_0)

    functions["vdotfromdp"] = LinearFunction(x_data=v_dot_dp, y_data=v_dot)
    functions["dpfromvdot"] = LinearFunction(x_data=v_dot, y_data=v_dot_dp)

    functions["mdotfromdp"] = LinearFunction(x_data=v_dot_dp, y_data=v_dot_m_dot)
    functions["dpfrommdot"] = LinearFunction(x_data=v_dot_m_dot, y_data=v_dot_dp)

    # Fitten Leistung
    p_data = read_csv("leistung.csv")
    p = np.array(p_data["leistung"]) * 1000  # [W]
    p_dp = np.array(p_data["druckdifferenz"]) * 100  # [Pa]

    functions["pelfromdp"] = LinearFunction(x_data=p_dp, y_data=p)
    functions["dpfrompel"] = LinearFunction(x_data=p, y_data=p_dp)

    # Erzeugen: Leistung und Massenstrom
    p_m_dot = functions["mdotfromdp"].func(p_dp, *functions["mdotfromdp"].get_params())
    functions["pelfrommdot"] = LinearFunction(x_data=p_m_dot, y_data=p)
    functions["mdotfrompel"] = LinearFunction(x_data=p, y_data=p_m_dot)

    # Fitten der Temperatur
    dt_data = read_csv("temperatur.csv")
    dt = np.array(dt_data["temperaturanstieg"])  # [°C]
    dt_dp = np.array(dt_data["druckdifferenz"]) * 100  # [Pa]

    functions["dtfromdp"] = QuadraticFunction(x_data=dt_dp, y_data=dt)
    functions["dpfromdt"] = QuadraticFunction(x_data=dt, y_data=dt_dp)

    # Fitten der Enthalphierhöhung
    cp = 1007
    dh = dt * cp
    functions["dhfromdp"] = QuadraticFunction(x_data=dt_dp, y_data=dh)
    functions["dpfromdh"] = QuadraticFunction(x_data=dh, y_data=dt_dp)

    # Speichern der Parameter
    data = []

    for function in functions:
        data.append({"function": function,
                     "params": ",".join(map(str, functions[function].get_params())),
                     "x_max": functions[function].get_x_max()})

    df = pd.DataFrame(data)
    df.to_csv("params.csv", index=False, sep=";")

    return functions


def load_curves():
    functions = {}
    df = read_csv("params.csv")

    for index, row in df.iterrows():
        function = row["function"]
        params = row["params"]
        params = [float(param) for param in params.split(",")]
        x_max = row["x_max"]

        if "vdot" in function:
            model = LinearFunction(params=params, x_max=x_max)
            functions[function] = model
        elif "pel" in function:
            model = LinearFunction(params=params, x_max=x_max)
            functions[function] = model
        elif "dt" in function:
            model = QuadraticFunction(params=params, x_max=x_max)
            functions[function] = model

    return functions
