from ttopt import TTOpt


from teneva_opti import OptiTens


DESC = """
    TTOpt optimizer.
    We use the implementation from the ttopt (v. 0.6.2) package [1]
    with default parameters. The method is based on the TT-format, see [2].

    Links:
    [1] https://github.com/AndreiChertkov/ttopt
    [2] TTOpt: A maximum volume quantized tensor train-based optimization and
    its application to reinforcement learning
    https://openreview.net/forum?id=Kf8sfv0RckB
"""


class OptiTensTtopt(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'ttopt')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'rank': {
                'desc': 'TT-rank',
                'kind': 'int',
                'dflt': 4
            },
            'fs_opt': {
                'desc': 'Transformation option',
                'kind': 'float',
                'form': '.1f',
                'dflt': 1.
            },
            'quan': {
                'desc': 'Allow quantization of modes',
                'kind': 'bool',
                'dflt': True
            },
        }

    def _optimize(self):
        tto = TTOpt(f=self.target, d=self.d_inner, n=self.n_inner,
            evals=1.E+99, is_func=False, is_vect=True)
        tto.optimize(rank=self.rank, seed=self.seed,
            fs_opt=self.fs_opt, is_max=self.is_max)
