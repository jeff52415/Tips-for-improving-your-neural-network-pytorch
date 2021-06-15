import math

from torch.optim import Optimizer

## modify from source : https://github.com/Tony-Y/pytorch_warmup#radam-warmup


class BaseWarmup:
    """Base class for all warmup schedules
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer: Optimizer, warmup_params, last_step=-1):
        self.optimizer = optimizer
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.step()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, step=None):
        """Dampen the learning rates.
        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group["lr"] *= omega

    def warmup_factor(self, step, **params):
        raise NotImplementedError


class ExponentialWarmup(BaseWarmup):
    """Exponential warmup schedule.
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Effective warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = [dict(warmup_period=warmup_period) for _ in range(group_count)]
        super().__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return 1.0 - math.exp(-(step + 1) / warmup_period)
