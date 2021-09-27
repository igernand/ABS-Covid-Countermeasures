"""
Constants of COVID-19 diasease and wealth distribution
"""

import numpy as np




age_severe_probs = [0.0345, 0.02, 0.056, 0.056, 0.14, 0.14, 0.19, 0.19, 0.10]
age_death_probs = [0.0001, 0.00001, 0.0001, 0.0003, 0.001, 0.005, 0.024, 0.095, 0.216]


"""
Wealth distribution - Lorenz Curve
"""

lorenz_curve = [.01, .02, .05, .19, .73]
share = np.min(lorenz_curve)
basic_income = np.array(lorenz_curve) / share
