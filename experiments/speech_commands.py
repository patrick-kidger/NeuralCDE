import common
import datasets


def main(device='cuda', max_epochs=200, *,                                        # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 1024
    lr = 0.00005 * (batch_size / 32)

    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False
    times, train_dataloader, val_dataloader, test_dataloader = datasets.speech_commands.get_data(intensity_data,
                                                                                                 batch_size)
    input_channels = 1 + (1 + intensity_data) * 20

    make_model = common.make_model(model_name, input_channels, 10, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    name = None if dry_run else 'speech_commands'
    num_classes = 10
    return common.main(name, times, train_dataloader, val_dataloader, test_dataloader, device, new_make_model,
                       num_classes, max_epochs, lr, kwargs, step_mode=True)


def run_all(device, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    model_kwargs = dict(ncde=dict(hidden_channels=90, hidden_hidden_channels=40, num_hidden_layers=4),
                        odernn=dict(hidden_channels=128, hidden_hidden_channels=64, num_hidden_layers=4),
                        dt=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(device, model_name=model_name, **model_kwargs[model_name])
