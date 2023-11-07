import numpy as np
from scipy.constants import sigma
from scipy.optimize import root
import matplotlib.pyplot as plt


def energie_delta(t_thermo: float, t_gas: float, t_env: float, epsilon: float, alpha: float):
    """
    :param t_thermo: Temperatur des Thermoelements [K]
    :param t_gas: Gastemperatur [K]
    :param t_env: Umgebungstemperatur [K]
    :param epsilon: Emissionsgrad [-]
    :param alpha: Wärmeübergangskoeffizient [W/m^2*K]
    :return: Delta der Energiebilanz

    Thermoelement im Gleichgewicht:
    Q_konv = Q_strahl
    alpha * A * (T_gas - T_thermo) = epsilon * sigma * A * (T_thermo^4 - T_env^4)  | A kürzt sich
    alpha * (T_gas - T_thermo) = epsilon * sigma * (T_thermo^4 - T_env^4)

    <=> delta = alpha * (T_gas - T_thermo) - epsilon * sigma * (T_thermo^4 - T_env^4)
        delta = q_konv - q_strahl
    Im Gleichgewicht: delta = 0
    """

    q_konv = alpha * (t_gas - t_thermo)
    q_strahl = epsilon * sigma * (t_thermo ** 4 - t_env ** 4)

    return q_konv - q_strahl


def energie_delta_2(t_des, q_verdampfer: float, t_cooling: float, ua: float, c: float):
    """

    :param t_des: Eingangs- und Ausgangstemperatur [°C]
    :param q_verdampfer: Wärmestrom aus Verdampfer [W]
    :param t_cooling:  Temperatur des Kühlmittels [°C]
    :param ua: Übertragungsfähigkeit [W/K]
    :param c: Wärmekapazitätsstrom [W/K]
    :return: Delta der Energiebilanzen nach Temperaturdifferenz und Wärmeübertrager

    Temperaturdifferenz nach 1. HS:
    Q_hs = C * (T_ein - T_aus)
    muss mit zugeführtem Wärmestrom im Gleichgewicht stehen -> Q_hs = Q_verdampfer
    <=> delta = Q_hs - Q_verdampfer

    Wärmeübertrager:
    Q_uebertrager = k * A * delta_t_m
    Q_uebertrager = UA * delta_t_m
    muss ebenfalls im Gleichgewicht mit zugeführtem Wärmestrom stehen
    -> delta = Q_ubertrager - Q_verdampfer
    """
    t_ein, t_aus = t_des

    q_hs = c * (t_ein - t_aus)

    delta_t_max = t_ein - t_cooling
    delta_t_min = t_aus - t_cooling

    if delta_t_max <= 0.1 or delta_t_min <= 0.1:
        # Vorbeugen von Fehler bei delta = 0, daher Solver mit hohem Wert unzufrieden stellen
        q_uebertrager = 1e5
    else:
        # mittlere logarithmische Temperaturdifferenz
        delta_t_m = (delta_t_max - delta_t_min) / np.log(delta_t_max / delta_t_min)
        q_uebertrager = ua * delta_t_m

    return (q_hs - q_verdampfer), (q_uebertrager - q_verdampfer)


if __name__ == "__main__":
    # Aufgabe 1
    t_gas = 1300  # Gastemperatur [K]
    t_env = 300  # Umgebungstemperatur [K]
    alpha = 100  # Wärmeübergangskoeffizient [W/m^2*K]
    epsilon_arr = np.linspace(0, 1, 100)

    t_thermo = []  # Temperatur des Thermoelements [K]
    t_guess = t_env

    for epsilon in epsilon_arr:
        # fun: Berechnungsformel; x0: Startwert; args: Funktionsargumente
        t_solve = root(fun=energie_delta, x0=t_guess, args=(t_gas, t_env, epsilon, alpha))

        if t_solve.success:
            t_thermo.append(t_solve.x[0])
            t_guess = t_solve.x[0]

    plt.figure(1)
    plt.plot(epsilon_arr, t_thermo)
    plt.xlabel(r"Emissionsgrad $\epsilon$ [-]")
    plt.ylabel("Temperatur [K]")
    plt.title("Aufgabe 1 - Thermoelement")
    plt.show()

    # Aufgabe 2
    q_verdampfer = 50e3  # Wärmestrom [W]
    t_cooling = 5  # Temperatur des Kühlmittels [°C]
    ua = 12000  # Übertragungsfähigkeit (UA = k * A; k: Wärmeübergangskoeffizient) [W/K]
    c = 8000  # Wärmekapazitätsstrom [W/K]

    t_guess = np.array([t_cooling+20, t_cooling+10])

    t_solve = root(fun=energie_delta_2, x0=t_guess, args=(q_verdampfer, t_cooling, ua, c))

    if t_solve.success:
        t_ein = t_solve.x[0]
        t_aus = t_solve.x[1]
        print(f"Eingangstemperatur: {t_ein} °C")
        print(f"Ausgangsstemperatur: {t_aus} °C")
    else:
        print("Keine Lösung gefunden")