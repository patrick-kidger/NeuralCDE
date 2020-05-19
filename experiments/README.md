# Experiments

Every experiment is contained in an easy to run Python file.

A lot of repositories tend to use command-line interfaces for running scripts - we do more or less the same thing here, with the Python interface. We find this is usually easier to understand, and easier to query the results afterwards.

### Requirements
We used:
+ Ubuntu 18.04.4 LTS
+ Python 3.7
+ PyTorch 1.3.1
+ torchaudio 0.3.2
+ torchdiffeq 0.0.1
+ Sklearn 0.22.1
+ sktime 0.3.1
+ tqdm 4.42.1

But more recent versions are likely to work as well. Installing these through `conda` or `pip` should be straightforward as usual.

### Example

Train and evaluate a Neural CDE model on the speech commands dataset, return a result object, and check how well we did:
```python
import speech_commands
result = speech_commands.main(model_name='ncde', hidden_channels=90, 
                              hidden_hidden_channels=40, num_hidden_layers=4)
print(result.keys())  # things we can inspect
print(result.test_metrics.accuracy)
```
`hidden_channels` is the size of the hidden state. `hidden_hidden_channels` and `num_hidden_layers` describe the size of the feedforward network parameterising the vector fields of the Neural CDE and ODE-RNN models.

### First time

The first time a particular dataset is run, it will be downloaded automatically.

The first time a particular dataset/model combination is run, it may take a reasonable amount of time to preprocess and save the dataset into the appropriate format. This is just an implementation thing; there's some points which are unnecessarily serial - we just didn't try to parallelise this aspect of it since it only happens once.

### Usage

There are three main files, `uea.py`, `sepsis.py`, `speech_commands.py`. Each one has two functions, `main()` and `run_all()`.

`main()` trains and evaluates a single model and reports the results.
`run_all()` trains and evaluates every model with the specific hyperparameters we selected, five times each.

#### run_all

If you want to repeat our results exactly as in the paper, then:
```python
import uea
import sepsis
import speech_commands

# group = 1 corresponds to 30% mising rate
uea.run_all(group=1, device='cuda', dataset_name='CharacterTrajectories')
# group = 2 corresponds to 50% mising rate
uea.run_all(group=2, device='cuda', dataset_name='CharacterTrajectories')
# group = 3 corresponds to 70% mising rate
uea.run_all(group=3, device='cuda', dataset_name='CharacterTrajectories')

sepsis.run_all(intensity=True, device='cuda')
sepsis.run_all(intensity=False, device='cuda')

speech_commands.run_all(device='cuda')
```

You might notice that you can use other UEA datasets, not just CharacterTrajectories! Give those a try as well if you like. :) (Valid dataset names can be found in [datasets/uea.py::valid_dataset_names](./datasets/uea.py).)

Obviously running everything serially in one script like this will take forever (split it up over multiple devices by using separate scripts and specify different `device`s), but you get the idea.

#### main

+ If you want to run a single model, then the `main()` functions take the following *mandatory* keyword-only arguments: `model_name`, `hidden_channels`, `hidden_hidden_channels`, `num_hidden_layers`.

    + Valid values for `model_name` are `"ncde"`, `"odernn"`, `"gruode"`, `"dt"`, `"decay"`, corresponding to a Neural CDE, ODE-RNN, GRU-ODE, GRU-dt, and GRU-D respectively.

    + `hidden_channels` is the size of the hidden state.

    + `hidden_hidden_channels` and `num_hidden_layers` describe the size of the feedforward network parameterising the vector fields of the Neural CDE and ODE-RNN models, and are ignored for the GRU-ODE, GRU-dt, GRU-D models.

+ Every `main()` also takes the following *optional* arguments: `device`, `max_epochs`, `dry_run`, specifying the PyTorch device to run on, the maximum number of epochs to train for (early stopping may kick in before this though), and whether to save the result to disk or not.

+ `uea` has a *mandatory* argument `dataset_name`, which should be set to any one of the UEA datasets as discussed above, and should be a string. It has an  *optional* argument `missing_rate` to set the missing rate of the data, which should be a a float between 0 and 1.

+ `sepsis` has a *mandatory* argument `intenstity`, which should a boolean specifying whether to use observational intensity or not. It has an *optional* argument `pos_weight`, which describes how much to weight the positive samples in the loss function. (Useful because this is an imbalanced classification problem.)
