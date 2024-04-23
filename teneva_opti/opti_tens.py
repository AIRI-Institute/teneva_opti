import nevergrad as ng
import numpy as np


from teneva_opti import Opti


class OptiTens(Opti):
    @property
    def d_inner(self):
        if self.with_quan:
            return self.d * int(np.log2(self.n0))
        return self.d

    @property
    def is_tens(self):
        return True

    @property
    def n_inner(self):
        if self.with_quan:
            return np.array([2]*self.d_inner, dtype=int)
        return self.n

    @property
    def n0_inner(self):
        if self.with_quan:
            return 2
        return self.n0

    @property
    def prps_info(self):
        return {**super().prps_info,
            'with_quan': {
                'desc': 'With quantization of tensor modes',
                'kind': 'bool',
                'info_skip_if_none': 'quan'
            }
        }

    @property
    def with_quan(self):
        if not self.opts.get('quan'):
            return False
        if not self.is_n_equal:
            return False
        if self.n0 == 2:
            return False
        if 2**int(np.log2(self.n0)) != self.n0:
            return False
        return True

    def target(self, I):
        if self.with_quan:
            I = self.unquantize(I)
        return self.target_tens(I)

    def unquantize(self, I_qtt):
        if len(I_qtt.shape) == 1:
            is_many = False
            I_qtt = I_qtt.reshape(1, -1)
        else:
            is_many = True

        d = self.d
        q = int(np.log2(self.n0))
        n = [2] * q
        m = I_qtt.shape[0]

        I = np.zeros((m, d), dtype=I_qtt.dtype)
        for k in range(d):
            I_qtt_curr = I_qtt[:, q*k:q*(k+1)].T
            I[:, k] = np.ravel_multi_index(I_qtt_curr, n, order='F')

        return I if is_many else I[0, :]

    def _optimize_ng_helper(self, solver):
        if not self.is_n_equal:
            raise NotImplementedError

        parametrization = ng.p.TransitionChoice(
            range(self.n0), repetitions=self.d)
        parametrization.random_state.seed(self.seed)

        optimizer = solver(parametrization=parametrization,
            budget=None, num_workers=1)

        while True:
            x = optimizer.ask()
            i = np.array(x.value, dtype=int)
            y = self.target(i)
            if y is None or self.bm.m == self.bm.budget_m-1:
                break
            optimizer.tell(x, -y if self.is_max else y)

        # We call for the final recommendation:
        x = optimizer.provide_recommendation()
        i = np.array(x.value, dtype=int)
        for _ in range(2):
            # We repeat it to stop the Bm
            y = self.bm.get(i, skip_cache=True)
