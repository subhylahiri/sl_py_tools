# -*- coding: utf-8 -*-
"""Testing tools for working with graphs and plotting them
"""
# %%
import numpy as np
import sl_py_tools.numpy_tricks.markov as ma
import sl_py_tools.graph_tricks as gt
import sl_py_tools.graph_plots as gp
np.set_printoptions(precision=2, suppress=True)
# %%
par = np.random.rand(2, 5)
top = ma.TopologyOptions(serial=True)
graph = gt.param_to_graph(par, node_keys=[-1, -1, -1, 1, 1, 1], topology=top)
# %%
gp.GraphPlots(graph)
# %%
