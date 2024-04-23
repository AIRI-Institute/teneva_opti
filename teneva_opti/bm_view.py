from copy import deepcopy as copy
import numpy as np


class BmView:
    def __init__(self, data=None, bm=None):
        self.bms = []
        self.op_seed_list = []
        self.y_all_list = []
        self.y_opt_list = []
        self.t_list = []
        self.is_init = False

        if data is not None:
            self.init_from_data(data)
        if bm is not None:
            self.init_from_bm(bm)

    @property
    def is_group(self):
        return len(self.bms) > 0

    @property
    def t_best(self):
        return np.min(self.t_list)

    @property
    def t_mean(self):
        return np.mean(self.t_list)

    @property
    def t_wrst(self):
        return np.max(self.t_list)

    @property
    def y_opt_best(self):
        y = self.y_opt_list
        return np.max(y) if self.is_max else np.min(y)

    @property
    def y_opt_mean(self):
        return np.mean(self.y_opt_list)

    @property
    def y_opt_wrst(self):
        y = self.y_opt_list
        return np.min(y) if self.is_max else np.max(y)

    def add(self, bm):
        self.bms.append(bm)
        if len(self.bms) == 1:
            self.init_from_bm(bm)

        self.op_seed_list.append(bm.op_seed)
        self.y_all_list.append(copy(bm.y_list))
        self.y_opt_list.append(copy(bm.y_opt))
        self.t_list.append(bm.t)
        if bm.is_fail:
            self.is_fail = True

    def get(self, kind='mean', is_time=False):
        if self.is_fail:
            return None
        if self.is_group:
            if kind == 'best':
                return self.t_best if is_time else self.y_opt_best
            elif kind == 'mean':
                return self.t_mean if is_time else self.y_opt_mean
            elif kind == 'wrst':
                return self.t_wrst if is_time else self.y_opt_wrst
            else:
                raise ValueError('Invalid kind')
        else:
            return self.t if is_time else self.y_opt

    def get_lists(self):
        if self.is_fail:
            return {'best': None, 'mean': None, 'wrst': None, 'skip': True}

        Y = []
        # m = np.max([len(y_list) for y_list in self.y_all_list])
        m = self.bm_prps['budget_m']
        for y_list in self.y_all_list:
            if len(y_list) < m:
                y_list = y_list + [y_list[-1]]*(m - len(y_list))
            Y.append(y_list)

        Y = np.array(Y)
        if self.is_max:
            Y = np.maximum.accumulate(Y, axis=1)
        else:
            Y = np.minimum.accumulate(Y, axis=1)

        return {
            'max': np.max(Y, axis=0),
            'avg': np.mean(Y, axis=0),
            'min': np.min(Y, axis=0),
            'skip': False}

    def get_opt_str(self, opts, pretty=True):
        res = []
        for id, v in opts.items():
            if v is None or v == '' or v is False:
                continue
            elif isinstance(v, bool):
                res.append(f'{id}')
            else:
                res.append(f'{id}: {v}' if pretty else f'{id}-{v}')
        return ('; ' if pretty else '__').join(res)

    def init_from_bm(self, bm):
        self.bm_args = copy(bm.bm_args)
        self.op_args = copy(bm.op_args)

        self.bm_opts = copy(bm.bm_opts)
        self.op_opts = copy(bm.op_opts)

        self.bm_prps = copy(bm.bm_prps)
        self.op_prps = copy(bm.op_prps)

        self.bm_hist = copy(bm.bm_hist)
        self.op_hist = copy(bm.op_hist)

        self.d = bm.d
        self.n = copy(bm.n)

        self.bm_name = bm.bm_name
        self.op_name = bm.op_name

        self.bm_seed = bm.bm_seed
        self.op_seed = bm.op_seed

        self.is_max = bm.is_max
        self.is_min = bm.is_min

        self.y_opt = bm.y_opt

        self.t = bm.t

        self.y_list = copy(bm.y_list)
        self.y_list_full = copy(bm.y_list_full)

        self.is_fail = bm.is_fail

        self.is_init = True

    def init_from_data(self, data):
        self.bm_args = copy(data['bm']['args'])
        self.op_args = copy(data['args'])

        self.bm_opts = copy(data['bm']['opts'])
        self.op_opts = copy(data['opts'])

        self.bm_prps = copy(data['bm']['prps'])
        self.op_prps = copy(data['prps'])

        self.bm_hist = copy(data['bm']['hist'])
        self.op_hist = copy(data['hist'])

        self.d = self.bm_args['d']
        self.n = self.bm_args['n']

        self.bm_name = self.bm_args['name']
        self.op_name = self.op_args['name']

        self.bm_seed = self.bm_args['seed']
        self.op_seed = self.op_args['seed']

        self.is_max = self.bm_prps['is_opti_max']
        self.is_min = self.bm_prps['is_opti_min']

        self.y_opt = self.bm_hist['y_max' if self.is_max else 'y_min']

        self.t = self.bm_hist['time_full']

        self.y_list = copy(self.bm_hist['y_list'])
        self.y_list_full = copy(self.bm_hist['y_list_full'])

        self.is_fail = True if self.op_hist['err'] else False

        self.is_init = True

    def info_table(self, prec=2, value_best=None, kind='mean', is_time=False,
                   prefix='    & ', fail='FAIL', best_cmd='fat', postfix='',
                   prefix_comment_inner='%       > ', with_comment=False):
        form = '{:-10.' + str(prec) + 'e}'

        if self.is_fail:
            v = fail
        else:
            v = self.get(kind, is_time)
            v = form.format(v).strip()

        if value_best is not None:
            value_best = form.format(value_best).strip()

        text = ''

        if with_comment:
            text += prefix_comment_inner
            text += self.info_text_bm(with_prefix=False) + '\n'
            text += prefix_comment_inner
            text += self.info_text_op(with_prefix=False) + '\n'

        text += prefix

        if v == value_best and best_cmd:
            text += '\\' + best_cmd + '{'+ v + '}'
        else:
            text += v

        return text + postfix

    def info_text(self, prec=5, prec_time=1, len_max=21):
        text = ''
        text += '\n' + self.info_text_bm(len_max)
        text += '\n' + self.info_text_op(len_max)

        task = 'max' if self.is_max else 'min'

        form = '{:-14.' + str(prec) + 'e}'
        form_time = '{:-.' + str(prec_time) + 'e}'

        if self.is_group:
            if self.is_fail:
                text += '\n          ***FAIL***'
            else:
                text += '\n'
                text += '  > BEST >> '
                v = form.format(self.y_opt_best)
                text += f'{task}: {v}   '
                v = form_time.format(self.t_best)
                text += f'[time: {v}]'

                text += '\n'
                text += '  > MEAN >> '
                v = form.format(self.y_opt_mean)
                text += f'{task}: {v}   '
                v = form_time.format(self.t_mean)
                text += f'[time: {v}]'

                text += '\n'
                text += '  > WRST >> '
                v = form.format(self.y_opt_wrst)
                text += f'{task}: {v}   '
                v = form_time.format(self.t_wrst)
                text += f'[time: {v}]'

        else:
            text += '\n'
            text += '  >>>>>> '
            if self.is_fail:
                text += ' ***FAIL***'
            else:
                v = form.format(self.y_opt)
                text += f'{task}: {v}   '
                v = form_time.format(self.t)
                text += f'[time: {v}]'

        return text

    def info_text_bm(self, len_max=21, with_prefix=True):
        text = ''
        name = self.bm_name[:(len_max-1)]
        if with_prefix:
            pref = '- BM      > ' if self.is_group else '- BM   > '
        else:
            pref = ''
        args = copy(self.bm_args)
        del args['name']
        text += pref + name + ' ' * max(0, len_max-len(name))

        text_args = ' [' + self.get_opt_str(args, pretty=True) + ']'

        if len(text + text_args) > 79:
            return text + '\n ' + text_args
        else:
            return text + text_args

    def info_text_op(self, len_max=21, with_prefix=True):
        text = ''
        name = self.op_name[:(len_max-1)]
        if with_prefix:
            pref = '- OPTI    > ' if self.is_group else '- OPTI > '
        else:
            pref = ''
        args = copy(self.op_args)
        del args['fold']
        del args['machine']
        del args['with_cache']
        if self.is_group:
            args['SEEDS'] = len(self.op_seed_list)
            del args['seed']
        text += pref + name + ' ' * max(0, len_max-len(name))

        text_args = ' [' + self.get_opt_str(args, pretty=True) + ']'

        if len(text + text_args) > 79:
            return text + '\n ' + text_args
        else:
            return text + text_args

    def is_better(self, value, kind='mean', is_time=False):
        if value is None:
            return True

        v = self.get(kind, is_time)
        if v is None:
            return False

        if is_time:
            return v < value
        else:
            return (value < v) if self.is_max else (value > v)

    def is_same(self, bm, skip_op_seed=True):
        if not self.is_init:
            return True

        bm1 = self
        bm2 = bm

        if bm1.bm_name !=bm2.bm_name:
            return False

        if bm1.op_name !=bm2.op_name:
            return False

        for id in bm1.bm_args.keys():
            if not id in bm2.bm_args:
                return False
            if bm1.bm_args[id] != bm2.bm_args[id]:
                return False

        for id in bm2.bm_args.keys():
            if not id in bm1.bm_args:
                return False
            if bm1.bm_args[id] != bm2.bm_args[id]:
                return False

        for id in bm1.op_args.keys():
            if id == 'seed' and skip_op_seed:
                continue
            if not id in bm2.op_args:
                return False
            if bm1.op_args[id] != bm2.op_args[id]:
                if id in ['fold', 'machine']:
                    continue
                return False

        for id in bm2.op_args.keys():
            if id == 'seed' and skip_op_seed:
                continue
            if not id in bm1.op_args:
                return False
            if bm1.op_args[id] != bm2.op_args[id]:
                if id in ['fold', 'machine']:
                    continue
                return False

        return True
