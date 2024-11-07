import numpy as np

from .scripts import normalize_accuracy_n_tests


# Area under the curve for performance as function of perturbation
def AUC(perf_list):
    return np.trapz(perf_list)


def mean_AUC(tests):
    return np.mean([AUC(test) for test in tests])


def std_AUC(tests):
    return np.std([AUC(test) for test in tests])


# Coefficient of variation
def CV(test_list):
    '''Given a list of test results, calculate the coefficient of variation'''
    mean = np.mean(test_list)
    std = np.std(test_list)
    res = std / mean
    return res


def mean_CV(tests):
    '''Given a list of test results, calculate the mean coefficient of variation'''
    return np.mean([CV(test) for test in tests.T])


def std_CV(tests):
    '''Given a list of test results, calculate the standard deviation of the coefficient of variation'''
    return np.std([CV(test) for test in tests.T])


def performance_variance(test_list):
    '''Given a list of test results, calculate the variance of the accuracy'''
    np.array(test_list)
    return np.var(test_list)


def mean_performance_variance(tests):
    '''Given a list of test results, calculate the mean variance of the accuracy'''
    return np.mean([performance_variance(test) for test in tests.T])


def std_performance_variance(tests):
    '''Given a list of test results, calculate the standard deviation of the variance of the accuracy'''
    return np.std([performance_variance(test) for test in tests.T])


def robustness_report(tests):
    tests = normalize_accuracy_n_tests(tests)  # type: ignore
    cv = mean_CV(tests)
    cv_std = std_CV(tests)
    auc = mean_AUC(tests)
    auc_std = std_AUC(tests)
    print(f'CV: {cv.round(3)} ± {cv_std.round(3)}')
    print(f'AUC: {auc.round(3)} ± {auc_std.round(3)}')
