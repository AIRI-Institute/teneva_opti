import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    Particle Swarm Optimization (PSO) [1, 2].
    We use the implementation from the nevergrad (v. 0.8.0) package [3]
    with default parameters.

    The PSO method is based on the idea of a controlled movement of a set of
    candidates for the optimum (a swarm of particles) over the domain of the
    function. At each step of the algorithm, the displacement of each of the
    particles in the swarm is determined by its speed, which is calculated
    taking into account the distance of the particle from the current best
    solution found by this particle and the entire swarm in the aggregate.
    This method is heuristic, and its effectiveness is usually justified by
    the successful exploration-exploitation balance in the method.

    Links:
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm
    optimization. In Proceedings of ICNN'95-international conference on
    neural networks (Vol. 4, pp. 1942-1948). IEEE.
    [2] https://en.wikipedia.org/wiki/Particle_swarm_optimization
    [3] https://github.com/facebookresearch/nevergrad/blob/
    7d1e2d2a15b89130206f28d86f5de2bf321d0636/nevergrad/optimization/
    optimizerlib.py#L1261
"""


class OptiTensPso(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'pso')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.PSO)
