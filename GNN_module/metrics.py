import numpy as np

#Area under the curve for performance as function of perturbation
def AUC(perf_list):
    return np.trapz(perf_list)

def mean_AUC(tests):
    return np.mean([AUC(test) for test in tests])

def std_AUC(tests):
    return np.std([AUC(test) for test in tests])


#Coefficient of variation
def CV(test_list):
    mean = np.mean(test_list)
    std = np.std(test_list)
    return std / mean

def mean_CV(tests):
    return np.mean([CV(test) for test in tests.T])

def std_CV(tests):
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
    #Performance variance, CV, AUC. All the mean versions
    perf_var = mean_performance_variance(tests)
    cv = mean_CV(tests)
    auc = mean_AUC(tests)
    print(f'Performance variance: {perf_var}')
    print(f'CV: {cv}')
    print(f'AUC: {auc}')