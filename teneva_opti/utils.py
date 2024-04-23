import numpy as np
import os


class Log:
    def __init__(self, fpath=None, with_log=True, with_log_info=True,
                 is_file_add=False):
        self.fpath = path(fpath, 'txt')
        self.with_log = with_log
        self.with_log_info = with_log_info
        self.is_file_add = is_file_add
        self.is_file_new = True

    def __call__(self, text, force_log=False):
        if self.with_log or force_log:
            print(text)
        if self.fpath:
            opt = 'w' if self.is_file_new and not self.is_file_add else 'a+'
            with open(self.fpath, opt) as f:
                f.write(text + '\n')
            self.is_file_new = False

    def err(self, content=''):
        raise ValueError(content)

    def info(self, content=''):
        if self.with_log_info:
            self(content)

    def prc(self, content=''):
        self(f'\n.... {content}')

    def res(self, content=''):
        self(f'DONE {content}')

    def wrn(self, content=''):
        self(f'WRN ! {content}', force_log=True)


def get_identity(obj):
    res = {}

    for id in obj.identity:
        v = getattr(obj, id)

        if v is None:
            msg = f'Identity "{id}" is not set'
            raise ValueError(msg)

        if isinstance(v, (list, np.ndarray)):
            if len(v) == 0:
                msg = f'List for identity "{id}" is empty'
                raise ValueError(msg)
            if isinstance(v[0], float):
                msg = 'Float identity is not supported'
                raise NotImplementedError(msg)
            if len(set(v)) > 1:
                msg = 'List-like identity is not supported'
                raise NotImplementedError(msg)
            v = v[0]

        if isinstance(v, float):
            msg = 'Float identity is not supported'
            raise NotImplementedError(msg)

        res[id] = v

    return res


def get_identity_str(obj, pretty=False):
    res = []
    for id, v in get_identity(obj).items():
        if isinstance(v, bool):
            res.append(f'{id}')
        else:
            res.append(f'{id}: {v}' if pretty else f'{id}-{v}')
    return ('; ' if pretty else '__').join(res)


def path(fpath=None, ext=None):
    if not fpath:
        return

    fold = os.path.dirname(fpath)
    if fold:
        os.makedirs(fold, exist_ok=True)

    if ext and not fpath.endswith('.' + ext):
        fpath += '.' + ext

    return fpath
