from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # Create two neural networks:
        f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_1
        )
        f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_2
        )
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f

    def forward(self, ob_no):
        #  Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        pred = self.f(ob_no).detach()
        target = self.f_hat(ob_no)
        error = pred - target
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        #  Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        self.optimizer.zero_grad()
        error = self.forward(ob_no)
        loss = (error ** 2).mean()
        loss.backward()
        self.optimizer.step()
