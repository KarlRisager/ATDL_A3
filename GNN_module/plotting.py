from .scripts import *
import numpy as np
import matplotlib.pyplot as plt




def plot_test(scale, tests, add_std_shading = False,  more_plots_coming=False, repeating = True, normalize = True, titel='', xlabel='Noise scale', ylabel='Accuracy', line_label='Mean Accuracy', std_label='std'):
    if normalize:
        tests = normalize_accuracy_n_tests(tests)
    mean_data = np.mean(tests, axis=0)
    std_data = np.std(tests, axis=0)
    if repeating:
        plt.plot(scale, mean_data, label=line_label)
    else:
        plt.plot(scale, tests, label=line_label)

    if add_std_shading:
        plt.fill_between(scale, mean_data - std_data, mean_data + std_data, alpha=0.2, label=std_label)
    if not(more_plots_coming):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titel)
        plt.legend()
        plt.show()


def plot_n_tests(tests, scale, repeating = True, titels = None, normalize = True, xlabel='Noise scale', ylabel='Accuracy', std_label='std', titel='', line_label='Mean Accuracy'):
    if len(tests.shape) == 1:
        plt.plot(scale, tests)
    elif len(tests.shape) == 2:
        if not repeating:
            for i, test in enumerate(tests):
                if titels is not None:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False, normalize = normalize, line_label=titels[i], xlabel=xlabel, ylabel=ylabel, std_label=std_label, titel=titel)
                else:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False, normalize = normalize, xlabel=xlabel, ylabel=ylabel, line_label=line_label, std_label=std_label, titel=titel)
        else:
            plot_test(scale, tests, add_std_shading=True)
    else:
        for i, test in enumerate(tests):
            if titels is not None:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1, normalize = normalize, line_label=titels[i], xlabel=xlabel, ylabel=ylabel, std_label=std_label, titel=titel)
            else:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1, normalize = normalize, xlabel=xlabel, ylabel=ylabel, line_label=line_label, std_label=std_label, titel=titel)

    
