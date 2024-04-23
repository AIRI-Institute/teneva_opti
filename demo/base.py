"""Basic examples for benchmark optimization."""
from teneva_bm import *
from teneva_opti import *


def demo(steps=250, m=1.E+3):
    bm = BmQuboFixKnap100()

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.set_opts(rank=5)
    opti.run()
    opti.save()

    print('\n\n')

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.save()

    print('\n\n')

    opti = OptiTensProtes(bm, m, seed=42, log_info=True, log_file=True)
    opti.run()
    opti.save()

    print('\n\n')

    bm = BmQuboFixKnap100()
    opti = OptiTensProtes(bm, m*2, seed=42, log_info=True, log_file=True)
    opti.run()
    opti.save()

    print('\n\n')

    bm = BmAgentSwimmer(steps=steps)

    opti = OptiTensTtopt(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.render()
    opti.show()
    opti.save()

    print('\n\n')

    opti = OptiTensProtes(bm, m, log_info=True, log_file=True)
    opti.run()
    opti.render()
    opti.show()
    opti.save()

    print('\n\n')

    print('Demo for result loading (we present y_list below):')
    data = opti.load()
    print(data['bm']['hist']['y_list'][:25])
    print('...')
    print(data['bm']['hist']['y_list'][-25:])


if __name__ == '__main__':
    demo()
