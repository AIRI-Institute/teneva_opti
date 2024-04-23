import os


TENEVA_OPTI_JAX_CPU = os.environ.get('TENEVA_OPTI_JAX_CPU', 'True') == 'True'
if TENEVA_OPTI_JAX_CPU:
    import jax
    # jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platform_name', 'cpu')
    jax.default_device(jax.devices('cpu')[0])


from protes import protes
from protes import protes_general


from teneva_opti import OptiTens


DESC = """
    PROTES optimizer.
    We use the implementation from the PROTES (v. 0.3.4) package [1]
    with default parameters. The method is based on the TT-format, see [2].

    Links:
    [1] https://github.com/anabatsh/PROTES
    [2] PROTES: Probabilistic optimization with tensor sampling
    https://arxiv.org/pdf/2301.12162.pdf;
    https://neurips.cc/virtual/2023/poster/71674
"""


class OptiTensProtes(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'protes')
        super().__init__(*args, **kwargs)
        self.set_desc(DESC)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'k': {
                'desc': 'Batch size',
                'kind': 'int',
                'dflt': 100
            },
            'k_top': {
                'desc': 'Number of selected candidates',
                'kind': 'int',
                'dflt': 10
            },
            'k_gd': {
                'desc': 'Number of gradient lifting iters',
                'kind': 'int',
                'dflt': 1
            },
            'lr': {
                'desc': 'Learning rate for gradient lifting',
                'kind': 'float',
                'form': '.1e',
                'dflt': 5.E-2
            },
            'r': {
                'desc': 'TT-rank of the inner prob tensor',
                'kind': 'int',
                'dflt': 5
            },
            'quan': {
                'desc': 'Allow quantization of modes',
                'kind': 'bool',
                'dflt': True
            },
        }

    def _optimize(self):
        if self.is_n_equal:
            protes(self.target, self.d_inner, self.n0_inner, 1.E+99,
                k=self.k, k_top=self.k_top, k_gd=self.k_gd, lr=self.lr,
                r=self.r, seed=self.seed, is_max=self.is_max)
        else:
            protes_general(self.target, self.n_inner, 1.E+99,
                k=self.k, k_top=self.k_top, k_gd=self.k_gd, lr=self.lr,
                r=self.r, seed=self.seed, is_max=self.is_max)
