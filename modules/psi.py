import numpy as np
import pandas as pd

def calculate_psi(base_values, new_values, bins=10):
    """
    Calcula PSI usando quantis ou bins pré-definidos.
    base_values: distribuição original
    new_values: nova distribuição
    """

    base_values = np.array(base_values)
    new_values = np.array(new_values)

    # criar bins comuns
    breakpoints = np.percentile(base_values, np.linspace(0, 100, bins + 1))

    base_counts = np.histogram(base_values, bins=breakpoints)[0]
    new_counts = np.histogram(new_values, bins=breakpoints)[0]

    # proporções
    base_perc = base_counts / len(base_values)
    new_perc = new_counts / len(new_values)

    # evitar zeros
    base_perc = np.where(base_perc == 0, 0.0001, base_perc)
    new_perc = np.where(new_perc == 0, 0.0001, new_perc)

    psi = (new_perc - base_perc) * np.log(new_perc / base_perc)

    result = pd.DataFrame({
        "bin": range(1, bins + 1),
        "base_perc": base_perc,
        "new_perc": new_perc,
        "psi_component": psi
    })

    return result, psi.sum()