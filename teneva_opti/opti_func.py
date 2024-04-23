from teneva_opti import Opti


class OptiFunc(Opti):
    @property
    def is_func(self):
        return True

    def target(self, X):
        return self.target_func(X)
