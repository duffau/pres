import scipy.stats as st
import numpy as np

def gen_spherical_data(n, k, noise_pct_thershold=0.0, seed=None):
        
    X = st.norm().rvs(
        size=(n, k), 
        random_state=seed
    )
    threshold = st.chi2.ppf(0.5, k)

    if noise_pct_thershold > 0.0:
        noise_std = threshold*noise_pct_thershold
        noise = st.norm(loc=0, scale=noise_std).rvs(size=n, random_state=seed)
    else:
        noise = 0.0

    print(f"threshold: {threshold}")
    y = ((X*X).sum(axis=1) + noise < threshold)*2 - 1
    return X, y
