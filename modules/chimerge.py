import pandas as pd
import numpy as np
from scipy.stats import chi2


def chi2_value(events, non_events):
    """Cálculo do valor de Chi-square para dois bins."""
    total = events + non_events
    expected_events = events.sum() * (total / total.sum())
    expected_non_events = non_events.sum() * (total / total.sum())

    chi = ((events - expected_events) ** 2 / expected_events) + \
          ((non_events - expected_non_events) ** 2 / expected_non_events)
    return chi.sum()


def chimerge_binning(df, feature, target, max_bins=6, confidence=0.95):
    """Implementação do ChiMerge supervisionado."""

    data = df[[feature, target]].copy()
    data = data.sort_values(by=feature)

    # Step 1: start with each distinct value as a separate bin
    grouped = data.groupby(feature)[target].agg(['sum', 'count'])
    grouped['non_events'] = grouped['count'] - grouped['sum']

    bins = grouped.reset_index()

    # Bin boundaries (initial)
    bin_intervals = [[v, v] for v in bins[feature]]

    # Chi critical
    chi_critical = chi2.ppf(confidence, df=1)

    while len(bin_intervals) > max_bins:
        chi_values = []

        # compute chi-square for each adjacent pair
        for i in range(len(bin_intervals) - 1):
            left = bins.iloc[i]
            right = bins.iloc[i + 1]

            chi = chi2_value(
                events=np.array([left['sum'], right['sum']]),
                non_events=np.array([left['non_events'], right['non_events']])
            )
            chi_values.append(chi)

        min_chi_index = np.argmin(chi_values)

        # if smallest chi-square > threshold → stop merging
        if chi_values[min_chi_index] > chi_critical:
            break

        # merge bins
        bins.iloc[min_chi_index, 1:] = bins.iloc[min_chi_index, 1:] + bins.iloc[min_chi_index + 1, 1:]
        bins = bins.drop(bins.index[min_chi_index + 1]).reset_index(drop=True)

        # update intervals
        bin_intervals[min_chi_index][1] = bin_intervals[min_chi_index + 1][1]
        del bin_intervals[min_chi_index + 1]

    # final intervals
    intervals = [(i[0], i[1]) for i in bin_intervals]
    bins["interval"] = intervals

    return intervals, bins
