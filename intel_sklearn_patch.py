import cpuinfo


def patch_sklearn_if_intel():
    info = cpuinfo.get_cpu_info()
    if 'Intel' in info['brand_raw']:
        from sklearnex import patch_sklearn
        patch_sklearn()
        print("Scikit-learn has been patched for Intel optimization.")
    else:
        print("Non-Intel CPU detected. Standard Scikit-learn will be used.")


patch_sklearn_if_intel()
