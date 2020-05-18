import csv
import math
import os
import pathlib
import torch
import urllib.request
import zipfile

from . import common


here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsis'
loc_Azip = base_loc / 'training_setA.zip'
loc_Bzip = base_loc / 'training_setB.zip'


def download():
    if not os.path.exists(loc_Azip):
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                   str(loc_Azip))
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                   str(loc_Bzip))

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(str(base_loc))
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(str(base_loc))
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(base_loc / folder):
                if os.path.exists(base_loc / filename):
                    raise RuntimeError
                os.rename(base_loc / folder / filename, base_loc / filename)


def _process_data(static_intensity, time_intensity):
    X_times = []
    X_static = []
    y = []
    for filename in os.listdir(base_loc):
        if filename.endswith('.psv'):
            with open(base_loc / filename) as file:
                time = []
                label = 0.0
                reader = csv.reader(file, delimiter='|')
                reader = iter(reader)
                next(reader)  # first line is headings
                prev_iculos = 0
                for line in reader:
                    assert len(line) == 41
                    *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                    iculos = int(iculos)
                    if iculos > 72:  # keep at most the first three days
                        break
                    for iculos_ in range(prev_iculos + 1, iculos):
                        time.append([float('nan') for value in time_values])
                    prev_iculos = iculos
                    time.append([float(value) for value in time_values])
                    label = max(label, float(sepsislabel))
                unit1 = float(unit1)
                unit2 = float(unit2)
                unit1_obs = not math.isnan(unit1)
                unit2_obs = not math.isnan(unit2)
                if not unit1_obs:
                    unit1 = 0.
                if not unit2_obs:
                    unit2 = 0.
                hospadmtime = float(hospadmtime)
                if math.isnan(hospadmtime):
                    hospadmtime = 0.  # this only happens for one record
                static = [float(age), float(gender), unit1, unit2, hospadmtime]
                if static_intensity:
                    static += [unit1_obs, unit2_obs]
                if len(time) > 2:
                    X_times.append(time)
                    X_static.append(static)
                    y.append(label)
    final_indices = []
    for time in X_times:
        final_indices.append(len(time) - 1)
    maxlen = max(final_indices) + 1
    for time in X_times:
        for _ in range(maxlen - len(time)):
            time.append([float('nan') for value in time_values])

    X_times = torch.tensor(X_times)
    X_static = torch.tensor(X_static)
    y = torch.tensor(y)
    final_indices = torch.tensor(final_indices)

    times = torch.linspace(1, X_times.size(1), X_times.size(1))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X_times, y, final_indices, append_times=True,
                                                   append_intensity=time_intensity)
    if static_intensity:
        X_static_ = X_static[:, :-2]
        X_static_ = common.normalise_data(X_static_, y)
        X_static = torch.cat([X_static_, X_static[:, -2:]], dim=1)
    else:
        X_static = common.normalise_data(X_static, y)
    train_X_static, val_X_static, test_X_static = common.split_data(X_static, y)
    train_coeffs = (*train_coeffs, train_X_static)
    val_coeffs = (*val_coeffs, val_X_static)
    test_coeffs = (*test_coeffs, test_X_static)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(static_intensity, time_intensity, batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('sepsis' + ('_staticintensity' if static_intensity else '_nostaticintensity') + ('_timeintensity' if time_intensity else '_notimeintensity'))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d'], tensors['train_static']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d'], tensors['val_static']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d'], tensors['test_static']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(static_intensity, time_intensity)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3], train_static=train_coeffs[4],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         val_static=val_coeffs[4],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         test_static=test_coeffs[4],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader
