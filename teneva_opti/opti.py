import numpy as np
import os


from .utils import Log
from .utils import get_identity_str
from .utils import path
from teneva_opti import __version__


class Opti:
    def __init__(self, bm, m=1.E+4, seed=0, name='optimizer',
                 fold='result_demo', machine='', m_cache=None, with_cache=True,
                 log=True, log_info=False, log_file=False, log_with_desc=True,
                 with_inp_tens_mix=False):
        self.bm = bm
        self.m = int(m)
        self.seed = seed
        self.name = name
        self.fold = fold
        self.machine = machine
        self.with_cache = with_cache

        self.bm.set_cache(self.with_cache)
        self.bm.set_budget(self.m, m_cache=(m_cache or self.m))
        if not self.bm.is_prep:
            self.bm.prep()

        if self.bm.is_opti_max and self.bm.is_opti_min:
            msg = 'Benchmark should has one only one task (min or max)'
            raise NotImplementedError(msg)

        self.log = Log(self.fpath('log') if log_file else None, log, log_info)
        self.log_with_desc = log_with_desc

        self.bm.set_log(self.log, prefix=self.name,
            cond=('max' if self.is_max else 'min'),
            with_max=self.is_max, with_min=self.is_min)

        self.with_inp_tens_mix = with_inp_tens_mix
        if self.with_inp_tens_mix:
            rand = np.random.default_rng(42)
            self.inp_tens_mix = rand.permutation(np.arange(self.d))
        else:
            self.inp_tens_mix = None

        self.err = ''
        self.set_desc('optimizer_description')
        self.set_opts_dflt()
        self.init()

    @property
    def args(self):
        """Dict with values of optimizers's arguments (i.e., main params)."""
        return self.build_dict(self.args_info)

    @property
    def args_info(self):
        """Dict with info about optimizers's arguments."""
        return {
            'm': {
                'desc': 'Computation budget',
                'kind': 'int',
                'form': '.2e'
            },
            'seed': {
                'desc': 'Random seed',
                'kind': 'int'
            },
            'name': {
                'desc': 'Optimizer name',
                'kind': 'str'
            },
            'fold': {
                'desc': 'Folder with results',
                'kind': 'str'
            },
            'machine': {
                'desc': 'Used machine',
                'kind': 'str',
                'info_skip_if_none': True
            },
            'with_cache': {
                'desc': 'Use cache',
                'kind': 'bool'
            },
            'with_inp_tens_mix': {
                'desc': 'Random permutation',
                'kind': 'bool',
                'info_skip_if_none': True
            },
        }

    @property
    def d(self):
        """Get the dimension of the problem."""
        return self.bm.d

    @property
    def dict(self):
        """Return the dict with full info and history of requests."""
        return {
            'args': self.args,
            'opts': self.opts,
            'prps': self.prps,
            'hist': self.hist,
            'bm': self.bm.dict}

    @property
    def hist(self):
        """Dict with history values."""
        return self.build_dict(self.hist_info)

    @property
    def hist_info(self):
        """Dict with info about optimizer's history parameters."""
        return {
            'err': {
                'desc': 'Error message',
                'kind': 'str',
                'info_skip_if_none': True
            },
            'time_full': {
                'desc': 'Total work time (sec)',
                'kind': 'float',
                'form': '-10.3e'
            },
        }

    @property
    def i_opt(self):
        """Get the found optimum multi-index."""
        return self.bm.i_max if self.is_max else self.bm.i_min

    @property
    def identity(self):
        """Get a list of arg names that define the optimizer."""
        return ['m', 'seed']

    @property
    def is_bm_func(self):
        """Check if BM relates to function (i.e., continuous function)."""
        return not self.bm.is_tens

    @property
    def is_bm_tens(self):
        """Check if BM relates to tensor (i.e., discrete function)."""
        return not self.bm.is_func

    @property
    def is_func(self):
        """Check if the optimizer is continuous."""
        return not self.is_tens

    @property
    def is_max(self):
        """Get true if we solve maximization problem."""
        return self.bm.is_opti_max

    @property
    def is_min(self):
        """Get true if we solve minimization problem."""
        return self.bm.is_opti_min

    @property
    def is_n_equal(self):
        """Check if all the mode sizes are the same."""
        return self.bm.is_n_equal

    @property
    def is_tens(self):
        """Check if the optimizer is discrete."""
        return not self.is_func

    @property
    def n(self):
        """Get the mode sizes of the discrete problem."""
        return self.bm.n

    @property
    def n0(self):
        """Get the mode size int value of the problemif it is constant."""
        return self.bm.n0

    @property
    def name_class(self):
        """Get name of the class."""
        return self.__class__.__name__

    @property
    def opts(self):
        """Dict with values of optimizer's options (i.e., addit. parameters)."""
        return self.build_dict(self.opts_info)

    @property
    def opts_info(self):
        """Dict with info about optimizer's options."""
        return {}

    @property
    def prps(self):
        """Dict with values of optimizer's properties."""
        return self.build_dict(self.prps_info)

    @property
    def prps_info(self):
        """Dict with info about optimizer's properties."""
        return {
            'version': {
                'desc': 'Package version',
                'kind': 'str'
            },
            'name_class': {
                'desc': 'Optimizer class name',
                'kind': 'str'
            },
            'is_func': {
                'desc': 'Optimizer is continuous',
                'kind': 'bool',
                'info_skip': 'is_tens'
            },
            'is_tens': {
                'desc': 'Optimizer is discrete',
                'kind': 'bool',
                'info_skip': 'is_func'
            },
            'is_bm_func': {
                'desc': 'The problem is continuous',
                'kind': 'bool',
                'info_skip': 'is_bm_tens'
            },
            'is_bm_tens': {
                'desc': 'The problem is discrete',
                'kind': 'bool',
                'info_skip': 'is_bm_func'
            },
            'is_max': {
                'desc': 'The problem with maximization',
                'kind': 'bool',
                'info_skip': 'is_min'
            },
            'is_min': {
                'desc': 'The problem with minimization',
                'kind': 'bool',
                'info_skip': 'is_max'
            },
        }

    @property
    def time_full(self):
        """Get the full work time."""
        return self.bm.time_full

    @property
    def version(self):
        """Get the version of the package."""
        return __version__

    @property
    def x_opt(self):
        """Get the found optimum point."""
        return self.bm.x_max if self.is_max else self.bm.x_min

    @property
    def y_opt(self):
        """Get the found optimum value."""
        return self.bm.y_max if self.is_max else self.bm.y_min

    def build_dict(self, info):
        """Build a dictionary with class variables."""
        res = {}
        for name, opts in info.items():
            if not hasattr(self, name):
                raise ValueError(f'Variable "{name}" does not exist')
            res[name] = getattr(self, name, None)
        return res

    def fpath(self, kind):
        id = get_identity_str(self)
        id_bm = get_identity_str(self.bm)
        return os.path.join(self.fold, self.bm.name, kind, id_bm, id, self.name)

    def info(self, footer=''):
        """Get a detailed description of the optimizer as text."""
        text = self.info_prefix()

        text += self.info_section('Description')
        if self.log_with_desc:
            text += self.info_desc()

        text_section = ''
        for name, opt in self.args_info.items():
            text_section += self.info_var(name, opt, with_name=True)
        if text_section:
            text += self.info_section('Arguments') + text_section

        text_section = ''
        for name, opt in self.opts_info.items():
            text_section += self.info_var(name, opt, with_name=True)
        if text_section:
            text += self.info_section('Options') + text_section

        text_section = ''
        for name, opt in self.prps_info.items():
            text_section += self.info_var(name, opt, skip_none=True)
        if text_section:
            text += self.info_section('Properties') + text_section

        return text + footer + '=' * 78 + '\n'

    def info_desc(self):
        text = '.' * 78 + '\n'
        desc = f'    {self.desc.strip()}'
        text += desc.replace('            ', '    ')
        text += '\n'
        text += '.' * 78 + '\n'
        return text

    def info_prefix(self):
        text = '*' * 78 + '\n' + 'OPTI: '
        text += self.name + ' ' * max(0, 34-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-4d} | '
        n = 0 if self.n is None else np.mean(self.n)
        text += '<MODE SIZE> = ' + (f'{n:-7.1f}' if n<9999 else f'{n:-7.1e}')
        return text + '\n'

    def info_section(self, name):
        text = '-' * 41 + '|             >'
        text += ' ' * max(0, 22-len(name)) + name
        return text + '\n'

    def info_var(self, name, opt, with_name=False, skip_none=False):
        kind = opt.get('kind', 'str')
        form = opt.get('form', None)

        def is_none(v, with_bool=False):
            if v is None:
                return True
            if isinstance(v, str) and v == '':
                return True
            if isinstance(v, bool) and not v and with_bool:
                return True
            return False

        v = getattr(self, name, None)

        if skip_none and is_none(v):
            return ''

        cond = opt.get('info_skip')
        if cond:
            if cond is True:
                return ''
            elif isinstance(cond, str):
                v_ref = getattr(self, opt['info_skip'], False)
                if not is_none(v_ref, with_bool=True):
                    return ''

        cond = opt.get('info_skip_if_none')
        if cond:
            if cond is True:
                if v is None or v == '' or v == False:
                    return ''
            elif isinstance(cond, str):
                v_ref = getattr(self, cond, None)
                if is_none(v_ref, with_bool=True):
                    return ''

        def build(v):
            if v is None:
                return 'NONE'
            elif isinstance(v, (list, np.ndarray)):
                if form:
                    v = [('{:' + form + '}').format(v_) for v_ in v]
                if self.d > 3:
                    v = f'[{v[0]}, {v[1]}, <...>, {v[-1]}]'
                return f'{v}'
            elif form:
                return ('{:' + form + '}').format(v)
            elif kind == 'bool':
                return 'YES' if v else 'no'
            elif kind == 'int':
                return f'{v}'
            elif kind == 'float':
                return f'{v:.6f}'
            else:
                return f'{v}'

        text = opt['desc']
        text += f' [{name}]' if with_name else ''
        text += ' ' * max(0, 40-len(text)) + ' : '
        text += build(v)
        if opt.get('list') and isinstance(v, (int, float, str)):
            text += f', ..., ' + build(v)

        if opt.get('info_add'):
            v = getattr(self, opt['info_add'], None)
            if v is not None:
                # TODO: do it more accurate
                text += f'   [real: {build(v)}]'

        return text + '\n'

    def init(self):
        """Inner method for initialization."""
        return

    def load(self, fpath=None):
        """Load configuration and optimization result from npz file."""
        fpath = path(fpath or self.fpath('data'), 'npz')
        return np.load(fpath, allow_pickle=True).get('data').item()

    def run(self, with_err=True):
        """Run the optimization process."""
        self.bm.init()
        self.log.info(self.info() + '\n' + self.bm.info())

        self.err = ''

        try:
            self._optimize()
        except Exception as e:
            self.err = f'Optimization with "{self.name}" is failed [{e}]'
            self.log.err(self.err) if with_err else self.log.wrn(self.err)

        self.log.info(self.bm.info_history())

    def render(self, fpath=None, with_wrn=True):
        """Run the "render" method for the used benchmark."""
        if self.bm.with_render:
            return self.bm.render(fpath or self.fpath('render'))
        elif with_wrn:
            self.log.wrn(f'Render is not supported for BM "{self.bm.name}"')

    def save(self, fpath=None):
        """Save configuration and optimization result to the npz file."""
        fpath = path(fpath or self.fpath('data'), 'npz')
        np.savez_compressed(fpath, data=self.dict)

    def set_desc(self, desc=''):
        """Set text description of the optimizer."""
        self.desc = desc

    def set_opts(self, **kwargs):
        """Set values for some of options specific to the optimizer."""
        for name, value in kwargs.items():
            if not name in self.opts_info.keys():
                raise ValueError(f'Option "{name}" does not exist')
            setattr(self, name, value)

    def set_opts_dflt(self):
        """Set default values for options specific to the optimizer."""
        for name, opt in self.opts_info.items():
            if not 'dflt' in opt:
                raise ValueError(f'Option "{name}" has not default value')
            if hasattr(self, name):
                raise ValueError(f'Invalid option name "{name}" (conflict)')
            setattr(self, name, opt['dflt'])

    def show(self, fpath=None, with_wrn=True):
        """Run the "show" method for the used benchmark."""
        if self.bm.with_show:
            return self.bm.show(fpath or self.fpath('show'))
        elif with_wrn:
            self.log.wrn(f'Show is not supported for BM "{self.bm.name}"')

    def target(self, inp):
        """Get the value(s) of the used benchmark for index or point."""
        raise NotImplementedError

    def target_func(self, X):
        """Get the value(s) of the used benchmark for point."""
        return self.bm.get_poi(X)

    def target_tens(self, I):
        """Get the value(s) of the used benchmark for index."""
        I = np.asanyarray(I, dtype=int)

        is_batch = len(I.shape) == 2
        if not is_batch:
            I = I.reshape(1, -1)

        if self.inp_tens_mix is not None:
            I = I[:, self.inp_tens_mix]

        y = self.bm.get(I)

        return y if is_batch or y is None else y[0]

    def _optimize(self):
        """Inner function which perform optimization process."""
        raise NotImplementedError()
