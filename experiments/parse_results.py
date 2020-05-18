import collections as co
import json
import matplotlib.pyplot as plt
import os
import pathlib
import statistics
import sys
import torch


def _get(filename, metric):
    with open(filename, 'r') as f:
        content = json.load(f)
    if metric == 'accuracy':
        metric_value = content['test_metrics']['accuracy']
    elif metric == 'auroc':
        metric_value = content['test_metrics']['auroc']
    elif metric == 'history':
        metric_value = content['history']
    else:
        raise ValueError
    # NeuralCDE must come after GRU_ODE
    for model_name in ('GRU_ODE', 'GRU_dt', 'GRU_decay', 'GRU_D', 'NeuralCDE', 'ODERNN'):
        if model_name in content['model']:
            break
    else:
        raise RuntimeError
    parameters = content['parameters']
    memory_usage = content['memory_usage']
    return metric_value, model_name, parameters, memory_usage


def plot_history(foldername, metric='accuracy'):
    foldername = pathlib.Path('results') / foldername
    results_for_each_run = co.defaultdict(list)
    for filename in os.listdir(foldername):
        history, model_name, _, _ = _get(foldername / filename, 'history')
        times = []
        values = []
        for entry in history:
            times.append(int(entry['epoch']))
            value = float(entry['val_metrics'][metric])
            if metric == 'accuracy':
                value *= 100
            values.append(value)
        results_for_each_run[model_name].append((times, values))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    model_names = ('NeuralCDE', 'ODERNN', 'GRU_D', 'GRU_dt', 'GRU_ODE')
    assert set(results_for_each_run.keys()) == set(model_names)
    # Ensures the order of plotting
    for colour, model_name in zip(colours, model_names):
        model_results = results_for_each_run[model_name]
        all_times = set()
        for times, _ in model_results:
            all_times.update(times)
        all_times = sorted(list(all_times))
        all_values = [[] for _ in range(len(all_times))]
        for times, values in model_results:
            # The times we measured at should be the same for every run, it's just that some runs finished earlier than
            # others
            assert times == all_times[:len(times)]
            for i, entry in enumerate(values):
                all_values[i].append(entry)
        means = [statistics.mean(entry) for entry in all_values]
        stds = [statistics.stdev(entry) if len(entry) > 1 else 0 for entry in all_values]
        plt.plot(all_times, means, label=model_name, color=colour)
        plt.fill_between(all_times,
                         [mean + 0.2 * std for mean, std in zip(means, stds)],
                         [mean - 0.2 * std for mean, std in zip(means, stds)],
                         color=colour,
                         alpha=0.5)
    plt.title('Validation ' + str(metric) + ' during training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %' if metric == 'accuracy' else str(metric).capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()


def table(foldername, metric):
    assert metric in ('accuracy', 'auroc')
    foldername = pathlib.Path('results') / foldername
    results = co.defaultdict(list)
    parameter_results = {}
    memory_results = {}
    for filename in os.listdir(foldername):
        metric_value, model_name, parameters, memory_usage = _get(foldername / filename, metric)
        results[model_name].append(metric_value)
        parameter_results[model_name] = parameters
        memory_results[model_name] = memory_usage / (1024 ** 2)
    min_result_length = min(len(result) for result in results.values())
    sorted_results = []
    for key, value in results.items():
        sorted_results.append((key, torch.tensor(value)))
    sorted_results.sort(key=lambda x: -x[1].mean())
    print("Num samples: " + str(min_result_length))
    for key, value in sorted_results:
        value = value[:min_result_length]
        print("{:9}: min: {:.3f} mean: {:.3f} median: {:.3f} max: {:.3f} std:{:.3f} | mem: {:.3f}MB param: {} "
              "".format(key, value.min(), value.mean(), value.median(), value.max(), value.std(),
                        memory_results[key], parameter_results[key]))


if __name__ == '__main__':
    assert len(sys.argv) == 3
    foldername = sys.argv[1]
    metric = sys.argv[2]
    table(foldername, metric)
