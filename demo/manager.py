"""Basic examples for manager usage."""
from teneva_bm import *
from teneva_opti import *


OPTIS = [OptiTensProtes, OptiTensTtopt, OptiTensPso]
TASKS = []


for Opti in OPTIS:
    for seed in [0, 1]:
        TASKS.append({
            'bm': BmQuboMvc,
            'bm_args': {'d': 55, 'pcon': 3, 'seed': 99},
            'opti': Opti,
            'opti_args': {'m': 1.E+3, 'seed': seed},
        })


for Bm in [BmAgentPendInv, BmAgentSwimmer]:
    for Opti in OPTIS:
        TASKS.append({
            'bm': Bm,
            # We set "long" seed to check the log with auto-line break:
            'bm_args': {'steps': 250, 'seed': 12345678},
            'opti': Opti,
            'opti_args': {'m': 1.E+3, 'seed': 12345},
        })


def demo(fold='result_demo_manager', with_calc=True):
    if with_calc:
        oman = OptiManager(TASKS, fold=fold)
        oman.run()

    oman = OptiManager(fold=fold, load=True)
    oman.filter_by_bm(arg='name', value='QuboMvc')
    oman.sort_by_op(arg='name', values=['pso', 'protes', 'ttopt'])
    oman.join_by_op_seed()

    print('\n\nLoaded result for QuboMvc:\n')
    oman.show_text()

    oman.show_table('\n\nTable for mean:')
    oman.show_table('\n\nTable for best:', kind='best')
    oman.show_table('\n\nTable for wrst:', kind='wrst')

    oman.show_table('\n\nTable for mean time:', prec=1, is_time=True)
    oman.show_table('\n\nTable for best time:', kind='best', is_time=True)
    oman.show_table('\n\nTable for wrst time:', kind='wrst', is_time=True)

    oman.show_plot(f'{fold}/QuboMvc')

    oman.reset()
    oman.filter_by_bm(arg='name', value='AgentSwimmer')
    oman.sort_by_op(arg='name', values=['protes', 'ttopt', 'pso'])

    print('\n\n\n\nLoaded result for AgentSwimmer:\n')
    oman.show_text()
    oman.show_table('\n\nTable:')
    oman.show_plot(f'{fold}/AgentSwimmer')


if __name__ == '__main__':
    demo()
