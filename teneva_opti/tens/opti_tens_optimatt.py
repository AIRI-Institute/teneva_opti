import teneva


from teneva_opti import OptiTens


DESC = """
    Optima-TT optimizer.
    We use the implementation from the teneva (v. >=0.14.6) package [1]
    with default parameters. The method is based on the TT-format, see [2].


    Links:
    [1] https://github.com/AndreiChertkov/teneva
    [2] Optimization of functions given in the tensor train format
    https://arxiv.org/pdf/2209.14808.pdf
"""


class OptiTensOptimatt(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'optimatt')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'r0': {
                'desc': 'Initial TT-rank',
                'kind': 'int',
                'dflt': 1
            },
            'dr_max': {
                'desc': 'Maximum TT-rank increment',
                'kind': 'int',
                'dflt': 2
            },
            'eps': {
                'desc': 'TT-cross convergence parameter',
                'kind': 'float',
                'form': '.1e',
                'dflt': 1.E-8
            },
        }

    def _optimize(self):
        Y = teneva.rand(self.n, r=self.r0, seed=self.seed)
        Y = teneva.cross(self.target, Y, e=self.eps, m=self.bm.budget_m-2,
            dr_max=self.dr_max)
        Y = teneva.truncate(Y, e=self.eps)

        i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
        self.target(i_min)
        self.target(i_max)
