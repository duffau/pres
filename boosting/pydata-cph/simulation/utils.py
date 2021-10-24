import scipy.stats as st

def gen_spherical_data(n, k, noise_pct_thershold=0.0):
    n = 10000
    k = 2
    X = st.norm().rvs(size=(n, k))
    threshold = st.chi2.ppf(0.5, k)

    if noise_pct_thershold > 0.0:
        noise_std = threshold*noise_pct_thershold
        noise = st.norm(loc=0, scale=noise_std).rvs(size=n)
    else:
        noise = 0.0

    print(f"threshold: {threshold}")
    y = ((X*X).sum(axis=1) + noise < threshold)*2 - 1
    print(f"y.shape: {y.shape}")
    print(f"X.shape: {X.shape}")
    return X, y
