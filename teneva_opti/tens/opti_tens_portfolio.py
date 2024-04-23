import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    Portfolio built from Covariance matrix adaptation evolution strategy
    (CMA-ES) [1, 2, 3] and Differential Evolution (DE) [4, 5].
    We use the implementation from the nevergrad (v. 0.8.0) package [6]
    (see also CMA-ES [7] and DE [8]) with default parameters.

    The CMA-ES method is based on sampling a set of candidates for the
    optimum from a normal distribution, followed by updating the
    distribution parameters (mean value and covariance matrix) based on
    the objective function values calculated for the candidates, while the
    mean value and elements of the covariance matrix are changed so that
    the sampling probability for the current best solution has risen.

    The DE method is based on the heuristic evolution of a set of candidates
    for an optimum (a population of agents). At each step of the algorithm,
    for each of the agents, three other three agents are randomly selected,
    and taking into account the location of these three agents in the
    function's domain, a new candidate is formed according to a certain
    formula. If the new candidate corresponds to a better value of the
    objective function than the considered agent, then the new candidate
    replaces the considered agent in the population.

    Links:
    [1] Hansen, N. and Ostermeier, A., 2001. Completely derandomized
    self-adaptation in evolution strategies. Evolutionary computation,
    9(2), pp.159-195.
    [2] https://en.wikipedia.org/wiki/CMA-ES
    [3] https://github.com/CMA-ES/pycma
    [4] Storn, R. and Price, K., 1997. Differential evolution–a simple
    and efficient heuristic for global optimization over continuous
    spaces. Journal of global optimization, 11, pp.341-359.
    [5] https://en.wikipedia.org/wiki/Differential_evolution
    [6] https://github.com/facebookresearch/nevergrad/blob/
    7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/
    optimizerlib.py#L1706
    [7] https://github.com/facebookresearch/nevergrad/blob/7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/optimizerlib.py#L571
    [8] https://github.com/facebookresearch/nevergrad/blob/7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/differentialevolution.py#L302
"""


class OptiTensPortfolio(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'portfolio')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.Portfolio)
