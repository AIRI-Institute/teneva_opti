import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    One-Plus-One (OPO).
    We use the implementation from the nevergrad (v. 0.8.0) package [1]
    with default parameters.

    The OPO method is based on a simple stochastic approach using at each
    step a random mutation of the current candidate for the optimum, while
    on average only one index of the multi-index changes at each step.

    Links:
    [1] https://github.com/facebookresearch/nevergrad/blob/
    7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/
    optimizerlib.py#L401
"""


class OptiTensOpo(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'opo')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.OnePlusOne)
