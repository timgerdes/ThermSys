import numpy as np
import matplotlib.pyplot as plt


def temperatur_quotient(L: float, x: float, alpha: float, m: float, lambda_: float):
    """

    :param L: Länge der Rippe [m]
    :param x: laufende Koordinate [m]
    :param alpha: Wärmeübergangskoeffizient [W/m^2 K]
    :param m: ??
    :param lambda_: Wärmeleitfähigkeit [W/m K]
    :return: Quotient aus theta / theta_b [-]
    """
    aml = alpha / (m * lambda_)
    zaehler = np.cosh(m * (L - x)) + aml * np.sinh(m * (L - x))
    nenner = np.cosh(m * L) + aml * np.sinh(m * L)

    quotient = zaehler / nenner

    return quotient


temperatur_quotient = np.vectorize(temperatur_quotient)


def temperatur_aus_quotient(quotient: float, t_inf: float, t_b):
    """

    :param quotient: Quotient aus theta und theta_b [-]
    :param t_inf: Umgebungstemperatur [K]
    :param t_b: Bodentemperatur [K]
    :return: Rippentemperatur [K]
    """

    theta_b = t_b - t_inf
    theta = quotient * theta_b

    t = theta + t_inf

    return t


temperatur_aus_quotient = np.vectorize(temperatur_aus_quotient)


def waermestrom_rippe(alpha: float, U: float, lambda_: float, A: float, theta_b: float, m: float, L: float):
    """

    :param alpha: Wärmeübergangskoeffizient [W/m^2 K]
    :param U: Umfang [m]
    :param lambda_: Wärmeleitfähigkeit [W/m K]
    :param A: Fläche [m^2]
    :param theta_b: Temperaturdifferenz t_b - t_inf [K]
    :param m: ?? [1/m]
    :param L: Höhe der Rippe [m]
    :return:
    """

    aml = alpha / (m * lambda_)
    faktor = np.sqrt(alpha * U * lambda_ * A) * theta_b
    zaehler = np.sinh(m * L) + aml * np.cosh(m * L)
    nenner = np.cosh(m * L) + aml * np.sinh(m * L)

    q = faktor * zaehler / nenner

    return q


waermestrom_rippe = np.vectorize(waermestrom_rippe)

if __name__ == "__main__":

    plt.close("all")

    b = 2 * 10 ** -3  # Breite der Grundfläche [m]
    h = 40 * 10 ** -3  # Höhe der Grundfläche [m]
    l = 20 * 10 ** -3  # Länge der Rippe [m]

    A = b * h  # Fläche [m^2]
    U = 2 * (b + h)  # Umfang [m]

    alpha = 5  # Wärmeübergangskoeffizient [W/m^2 K]

    t_inf = 290  # Umgebungstemperatur [K]
    t_b = 370  # Bodentemperatur [K]

    # Name, Wärmeleitfähigkeit [W/m K]
    stoffe = [["Aluminium", 177],
              ["Stahl", 14.9]]

    # m berechnen
    for i in range(2):
        lambda_ = stoffe[i][1]
        m2 = (alpha * U) / (lambda_ * A)
        m = np.sqrt(m2)

        stoffe[i].append(m)

    x = np.linspace(0, l, 100)  # [m]

    # Beginn Aufgabe 2a)
    plt.figure(1)

    for stoff in stoffe:
        name = stoff[0]
        lambda_ = stoff[1]
        m = stoff[2]

        quotient = temperatur_quotient(l, x, alpha, m, lambda_)
        temperatur = temperatur_aus_quotient(quotient, t_inf, t_b)

        plt.plot(x, temperatur, "-", label=f"{name} ({lambda_})")

    plt.xlabel("x")
    plt.ylabel("T")
    plt.legend()
    plt.title("Aufgabe 2a) Temperaturverlauf")
    plt.show()

    # Beginn Aufgabe 2c)

    # Einfluss der Rippenhöhe
    plt.figure(2)

    l = np.linspace(10 * 10 ** -3, 100 * 10 ** -3)  # Bereich der Länge [m]
    theta_b = t_b - t_inf

    for stoff in stoffe:
        name = stoff[0]
        lambda_ = stoff[1]
        m = stoff[2]
        q = waermestrom_rippe(alpha, U, lambda_, A, theta_b, m, l)

        plt.plot(l, q, "-", label=f"{name} ({lambda_})")

    plt.xlabel("L")
    plt.ylabel("Q.")
    plt.legend()
    plt.title("Aufgabe 2c) Einfluss der Rippenhöhe")
    plt.show()

    # Einfluss des Wärmeübergangskoeffizienten
    l = 20 * 10 ** -3  # zurückgesetzt

    alpha = np.linspace(5, 1000)

    for stoff in stoffe:
        name = stoff[0]
        lambda_ = stoff[1]
        m = stoff[2]
        q = waermestrom_rippe(alpha, U, lambda_, A, theta_b, m, l)

        plt.plot(alpha, q, "-", label=f"{name} ({lambda_})")

    plt.xlabel("alpha")
    plt.ylabel("Q.")
    plt.legend()
    plt.title("Aufgabe 2c) Einfluss des Wärmeübergangskoeffizienten")
    plt.show()

    # Aufgabe 2d - Teilung variieren
    alpha = 5  # zurücksetzen
    teilungen = np.linspace(1, 8, 16)

    # beliebiger Stoff
    lambda_ = stoffe[0][1]
    m = stoffe[0][2]

    q_laenge = teilungen * waermestrom_rippe(alpha, U, lambda_, A, theta_b, m, l / teilungen)

    q_flaeche = teilungen * waermestrom_rippe(alpha, U / teilungen, lambda_, A / teilungen, theta_b, m, l)

    plt.scatter(teilungen, q_laenge, label="Länge")
    plt.scatter(teilungen, q_flaeche, label="Fläche")

    plt.xlabel("Teilungen")
    plt.ylabel("Q.")
    plt.legend()
    plt.title("Aufgabe 2d) Einfluss der Teilung")
    plt.show()


