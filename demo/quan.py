"""Benchmark optimization example with quantization.

We are looking for a global minimum for benchmark "BmHsFunc001" using a
gradient-free TTOpt and PROTES optimizerers based on the tensor train (TT)
decomposition (see https://github.com/AndreiChertkov/ttopt and
https://github.com/anabatsh/PROTES). Since TTOpt and PROTES works only for the
multidimensional case (d > 2), we perform quantization for the used 2D
benchmark. The mode size factor for the benchmark ("q"; "n = 2^q") and
budget ("m") are given as function "demo" arguments.

"""
from teneva_bm import *
from teneva_opti import *


def demo(q=15, m=1.E+4):
    bm = BmHsFunc001(n=2**q)

    opti = OptiTensTtopt(bm, m, log_info=True)
    opti.run()

    print('\n\n\n\n')

    opti = OptiTensProtes(bm, m, log_info=True)
    opti.run()


if __name__ == '__main__':
    demo()
