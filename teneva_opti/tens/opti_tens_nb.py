import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    Noisy Bandit (NB).
    We use the implementation from the nevergrad (v. 0.8.0) package [1]
    with default parameters.

    The NB method is based on a simple approach related to the estimation of
    the upper confidence limit and is implemented in the nevergrad [1].

    Links:
    [1] https://github.com/facebookresearch/nevergrad/blob/
    7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/
    optimizerlib.py#L1120
"""


class OptiTensNb(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'nb')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.NoisyBandit)
