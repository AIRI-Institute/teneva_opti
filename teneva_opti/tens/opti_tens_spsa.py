import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    Simultaneous Perturbation Stochastic Approximation (SPSA) [1, 2].
    We use the implementation from the nevergrad (v. 0.8.0) package [3]
    with default parameters.

    The SPSA method is based on the classical gradient descent formula,
    while for approximate calculation of the gradient at each step of the
    algorithm, only one additional calculation of the objective function is
    performed at a point randomly shifted by a small amount relative to the
    current point. This method turns out to be much more economical in terms
    of the number of requests to the objective function than many other
    approaches that use finite differences to estimate the gradient (in this
    case, the number of additional calculations of the objective function
    coincides with its dimension).

    Links:
    [1] Spall, J.C., 1992. Multivariate stochastic approximation using a
    simultaneous perturbation gradient approximation. IEEE transactions on
    automatic control, 37(3), pp.332-341.
    [2] https://en.wikipedia.org/wiki/
    Simultaneous_perturbation_stochastic_approximation
    [3] https://github.com/facebookresearch/nevergrad/blob/
    7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/
    optimizerlib.py#L1332
"""


class OptiTensSpsa(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'spsa')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.SPSA)
