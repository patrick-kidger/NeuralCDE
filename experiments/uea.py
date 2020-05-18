import common
import datasets


def main(dataset_name, missing_rate=0.3,                                          # dataset parameters
         device='cuda', max_epochs=1000, *,                                       # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 32
    lr = 0.001 * (batch_size / 32)

    # Need the intensity data to know how long to evolve for in between observations, but the model doesn't otherwise
    # use it because of use_intensity=False below.
    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False

    (times, train_dataloader, val_dataloader,
     test_dataloader, num_classes, input_channels) = datasets.uea.get_data(dataset_name, missing_rate, device,
                                                                           intensity=intensity_data,
                                                                           batch_size=batch_size)

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes

    make_model = common.make_model(model_name, input_channels, output_channels, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))
    return common.main(name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model,
                       num_classes, max_epochs, lr, kwargs, step_mode=False)


def run_all(group, device, dataset_name, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    if group == 1:
        missing_rate = 0.3
    elif group == 2:
        missing_rate = 0.5
    elif group == 3:
        missing_rate = 0.7
    else:
        raise ValueError
    model_kwargs = dict(ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        dt=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(dataset_name, missing_rate, device, model_name=model_name, **model_kwargs[model_name])
