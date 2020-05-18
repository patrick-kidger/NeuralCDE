import pathlib
import sys
import torch
import torchdiffeq

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..'))

import controldiffeq


class _GRU(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super(_GRU, self).__init__()

        assert (input_channels % 2) == 1, "Input channels must be odd: 1 for time, plus 1 for each actual input, " \
                                          "plus 1 for whether an observation was made for the actual input."

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.use_intensity = use_intensity

        gru_channels = input_channels if use_intensity else (input_channels - 1) // 2
        self.gru_cell = torch.nn.GRUCell(input_size=gru_channels, hidden_size=hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, use_intensity={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.use_intensity)

    def evolve(self, h, time_diff):
        raise NotImplementedError

    def _step(self, Xi, h, dt, half_num_channels):
        observation = Xi[:, 1: 1 + half_num_channels].max(dim=1).values > 0.5
        if observation.any():
            Xi_piece = Xi if self.use_intensity else Xi[:, 1 + half_num_channels:]
            Xi_piece = Xi_piece.clone()
            Xi_piece[:, 0] += dt
            new_h = self.gru_cell(Xi_piece, h)
            h = torch.where(observation.unsqueeze(1), new_h, h)
            dt += torch.where(observation, torch.tensor(0., dtype=Xi.dtype, device=Xi.device), Xi[:, 0])
        return h, dt

    def forward(self, times, coeffs, final_index, z0=None):
        interp = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        half_num_channels = (self.input_channels - 1) // 2

        # change cumulative intensity into intensity i.e. was an observation made or not, which is what is typically
        # used here
        X[:, 1:, 1:1 + half_num_channels] -= X[:, :-1, 1:1 + half_num_channels]

        # change times into delta-times
        X[:, 0, 0] -= times[0]
        X[:, 1:, 0] -= times[:-1]

        batch_dims = X.shape[:-2]

        if z0 is None:
            z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device)

        X_unbound = X.unbind(dim=1)
        h, dt = self._step(X_unbound[0], z0, torch.zeros(*batch_dims, dtype=X.dtype, device=X.device),
                           half_num_channels)
        hs = [h]
        time_diffs = times[1:] - times[:-1]
        for time_diff, Xi in zip(time_diffs, X_unbound[1:]):
            h = self.evolve(h, time_diff)
            h, dt = self._step(Xi, h, dt, half_num_channels)
            hs.append(h)
        out = torch.stack(hs, dim=1)

        final_index_indices = final_index.unsqueeze(-1).expand(out.size(0), out.size(2)).unsqueeze(1)
        final_out = out.gather(dim=1, index=final_index_indices).squeeze(1)

        return self.linear(final_out)


class GRU_dt(_GRU):
    def evolve(self, h, time_diff):
        return h


class GRU_D(_GRU):
    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super(GRU_D, self).__init__(input_channels=input_channels,
                                    hidden_channels=hidden_channels,
                                    output_channels=output_channels,
                                    use_intensity=use_intensity)
        self.decay = torch.nn.Linear(1, hidden_channels)

    def evolve(self, h, time_diff):
        return h * torch.exp(-self.decay(time_diff.unsqueeze(0)).squeeze(0).relu())


class _ODERNNFunc(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(_ODERNNFunc, self).__init__()

        layers = [torch.nn.Linear(hidden_channels, hidden_hidden_channels)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, x):
        return self.sequential(x)


class ODERNN(_GRU):
    def __init__(self, input_channels, hidden_channels, output_channels, hidden_hidden_channels, num_hidden_layers,
                 use_intensity):
        super(ODERNN, self).__init__(input_channels=input_channels,
                                     hidden_channels=hidden_channels,
                                     output_channels=output_channels,
                                     use_intensity=use_intensity)
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.func = _ODERNNFunc(hidden_channels, hidden_hidden_channels, num_hidden_layers)

    def extra_repr(self):
        return "hidden_hidden_channels={}, num_hidden_layers={}".format(self.hidden_hidden_channels,
                                                                        self.num_hidden_layers)

    def evolve(self, h, time_diff):
        t = torch.tensor([0, time_diff.item()], dtype=time_diff.dtype, device=time_diff.device)
        out = torchdiffeq.odeint_adjoint(func=self.func, y0=h, t=t, method='rk4')
        return out[1]
