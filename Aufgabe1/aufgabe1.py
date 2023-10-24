"""
Aufgabe 1:
a: Berechnung der Nusseltzahl in Abhängigkeit der Reynoldszahl
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

def nusselt_zahl_a(re: float, pr: float, n: float):
    """

    :param re: Reynoldszahl [-]
    :param pr: Prandtlzahl [-]
    :param n: Exponent? [-]
    :return: Nusseltzahl [-]

    Gültigkeit:
    turbulent
    - 0.6 ≤ pr ≤ 160
    - re >= 10000
    """

    # Gültigkeitsbereich prüfen
    if pr < 0.6 or pr > 160:
        nu = 0
    elif re < 10000:
        nu = 3.66
    else:
        nu = 0.023 * re ** (4 / 5) * pr ** n

    return nu


nusselt_zahl_a = np.vectorize(nusselt_zahl_a)


def nusselt_zahl_b(re: float, pr: float):
    """

    :param re: Reynoldszahl [-]
    :param pr: Prandtlzahl [-]
    :return: Nusseltzahl [-]

    Gültigkeit:
    turbulent
    - 0.5 ≤ pr ≤ 2000
    - 3000 ≤ re ≤ 5*10^6
    """

    if pr < 0.5 or pr > 2000:
        nu = 0
    elif re < 3000 or re > 5 * 10 ** 6:
        nu = 3.66
    else:
        f = (0.79 * np.log(re) - 1.64) ** -2
        nu_zaehler = (f / 8) * (re - 1000) * pr
        nu_nenner = 1 + 12.7 * ((f / 8) ** 0.5 * (pr ** (2 / 3) - 1))
        nu = nu_zaehler / nu_nenner

    return nu


nusselt_zahl_b = np.vectorize(nusselt_zahl_b)


def reynoldszahl(v: float, d: float, ny: float):
    """
    Berechnet die Reynoldszahl für eine Rohrströmung
    nach Vorlesungsunterlagen (Einführung)
    :param v: Geschwindigkeit [m/s]
    :param d: Durchmesser [m]
    :param ny: Kinematische Viskosität [m^2/s]
    :return: Reynoldszahl [-]
    """
    re = (v * d) / ny

    return re


reynoldszahl = np.vectorize(reynoldszahl)


def waermeuebergang_koeffizient(nu: float, d: float, lambda_: float):
    """
    Berechnet den Wärmeübergangskoeefizienten
    nach Vorlesungsunterlagen (Einführung)
    :param nu: Nusseltzahl [-]
    :param d: Durchmesser [m]
    :param lambda_: Wärmeleitfähigkeit [W/m K]
    :return: Wärmeübergangskoeffizient [W/m^2K]

    aus nu = (alpha*d)/lambda
    ↔ alpha = (nu*lambda)/d
    """
    alpha = (nu * lambda_) / d

    return alpha


waermeuebergang_koeffizient = np.vectorize(waermeuebergang_koeffizient)

if __name__ == "__main__":
    # Beginn Aufgabe a)
    stoffe = [["Luft", 0.707, "b"],
              ["Wasser", 5.83, "k"],
              ["Ethylenglykol", 151, "m"]]

    re = np.geomspace(10, 2000000, 100)

    plt.figure(1)

    for stoff in stoffe:
        name = stoff[0]
        pr = stoff[1]
        color = stoff[2]

        # Nusseltzahlen berechnen
        nu_a = nusselt_zahl_a(re, pr, 0.4)
        nu_b = nusselt_zahl_b(re, pr)

        # Nusseltzahlen darstellen
        plt.plot(re, nu_a, f"-{color}", label=f"{name} ({pr})")
        plt.plot(re, nu_b, f".{color}")

        # Fehler berechnen
        error = np.abs(nu_a - nu_b)
        max_error = np.max(error)
        max_error_index = np.argmax(error)

        # Fehler anzeigen
        plt.errorbar(re[max_error_index], (nu_a[max_error_index] + nu_b[max_error_index]) / 2, yerr=max_error, fmt=".r",
                     capsize=2)
        plt.text(re[max_error_index] * 0.98, (nu_a[max_error_index] + nu_b[max_error_index]) / 2,
                 f"Fehler: {max_error}", color="r", ha='right', va='bottom', fontsize=8)

    plt.xlabel("Re")
    plt.ylabel("Nu")
    plt.legend()
    plt.title("Aufgabe 1a) Nusseltzahl")
    plt.show()
    # Ende Aufgabe a)

    # Beginn Aufgabe b)
    plt.figure(2)

    v = np.geomspace(0.01, 100)
    d = 0.03
    # Hinzufügen von Wärmeleitfähigkeit und kin. Viskosität
    stoffe[0].extend([26.3e-3, 15.89e-6])  # Luft
    stoffe[1].extend([613e-3, 8.5757e-7])  # Wasser
    stoffe[2].extend([252e-3, 14.1e-6])  # Ethylenglykol

    for stoff in stoffe:
        name = stoff[0]
        pr = stoff[1]
        color = stoff[2]
        lambda_ = stoff[3]
        ny = stoff[4]

        re = reynoldszahl(v, d, ny)

        nu_a = nusselt_zahl_a(re, pr, 0.4)
        nu_b = nusselt_zahl_b(re, pr)

        alpha_a = waermeuebergang_koeffizient(nu_a, d, lambda_)
        alpha_b = waermeuebergang_koeffizient(nu_b, d, lambda_)

        plt.plot(v, alpha_a, f"-{color}", label=f"{name} ({pr})")
        plt.plot(v, alpha_b, f".{color}")

    plt.xlabel("v")
    plt.ylabel("alpha")
    plt.legend()
    plt.title("Aufgabe 1b) Wärmeübertragungskoeffizient")
    plt.show()
    # Ende Aufgabe b)
