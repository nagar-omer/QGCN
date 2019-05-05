from torch import Tensor
from torch.utils.data import Dataset
from collections import Counter

from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from multi_graph import MultiGraph
from parameters import BilinearDatasetParams, DATA_INPUT_DIR, PKL_DIR, FEATURES_PKL_DIR, DEG, IN_DEG, OUT_DEG, \
    AidsDatasetTestParams
import os
import pandas as pd
import networkx as nx
import pickle
import numpy as np


class BilinearDataset(Dataset):
    def __init__(self, params: BilinearDatasetParams):
        self._params = params
        self._logger = PrintLogger("logger")
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
        self.init_ftrs()
        self._src_file_path = os.path.join(self._base_dir, DATA_INPUT_DIR, params.DATASET_FILENAME)
        self._multi_graph, self._labels, self._label_to_idx, self._idx_to_label = self._build_multi_graph(params.PERCENTAGE)

        self._data, self._idx_to_name = self._build_data()

    @property
    def label_count(self):
        return Counter([v[3] for name, v in self._data.items()])

    def label(self, idx):
        return self._data[self._idx_to_name[idx]][3]

    @property
    def len_features(self):
        return self._data[self._idx_to_name[0]][2].shape[1]

    def init_ftrs(self):
        self._deg, self._in_deg, self._out_deg, self._is_ftr, self._ftr_meta = False, False, False, False, {}
        for ftr in self._params.FEATURES:
            if ftr == DEG:
                self._deg = True
            elif ftr == IN_DEG:
                self._in_deg = True
            elif ftr == OUT_DEG:
                self._out_deg = True
            else:
                self._ftr_meta[ftr[0]] = ftr[1]
        if len(self._ftr_meta) > 0:
            self._ftr_path = os.path.join(self._base_dir, FEATURES_PKL_DIR, self._params.DATASET_NAME)
            if not os.path.exists(self._ftr_path):
                os.mkdir(self._ftr_path)
            self._is_ftr = True

    """
    build multi graph according to csv 
    each community is a single graph, no consideration to time
    """
    def _build_multi_graph(self, split):
        path_pkl = os.path.join(self._base_dir, PKL_DIR, self._params.DATASET_NAME + "_split_" +
                                str(self._params.PERCENTAGE) + "_mg.pkl")
        if os.path.exists(path_pkl):
            return pickle.load(open(path_pkl, "rb"))
        multi_graph_dict = {}
        labels = {}
        label_to_idx = {}
        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._src_file_path)
        stop = data_df.shape[0] * split

        for index, edge in data_df.iterrows():
            if index > stop:
                break
            # write edge to dictionary
            graph_id = str(edge[self._params.GRAPH_NAME_COL])
            src = str(edge[self._params.SRC_COL])
            dst = str(edge[self._params.DST_COL])
            multi_graph_dict[graph_id] = multi_graph_dict.get(graph_id, []) + [(src, dst)]
            label = edge[self._params.LABEL_COL]
            label_to_idx[label] = len(label_to_idx) if label not in label_to_idx else label_to_idx[label]
            labels[graph_id] = label_to_idx[label]

        mg = MultiGraph(self._params.DATASET_NAME, graphs_source=multi_graph_dict,
                        directed=self._params.DIRECTED, logger=self._logger)
        idx_to_label = [l for l in sorted(label_to_idx, key=lambda x: label_to_idx[x])]
        mg.suspend_logger()
        pickle.dump((mg, labels, label_to_idx, idx_to_label), open(path_pkl, "wb"))
        mg.wake_logger()
        return mg, labels, label_to_idx, idx_to_label

    """
    returns a vector x for gnx 
    basic version returns degree for each node
    """
    def _gnx_vec(self, gnx_id, gnx: nx.Graph, node_order):
        final_vec = []
        if self._deg:
            degrees = gnx.degree(gnx.nodes)
            final_vec.append(np.matrix([degrees[d] for d in node_order]).T)
        if self._in_deg:
            degrees = gnx.in_degree(gnx.nodes)
            final_vec.append(np.matrix([degrees[d] for d in node_order]).T)
        if self._out_deg:
            degrees = gnx.out_degree(gnx.nodes)
            final_vec.append(np.matrix([degrees[d] for d in node_order]).T)
        if self._is_ftr:
            name = str(gnx_id)
            gnx_dir_path = os.path.join(self._ftr_path, name)
            if not os.path.exists(gnx_dir_path):
                os.mkdir(gnx_dir_path)
            raw_ftr = GraphFeatures(gnx, self._ftr_meta, dir_path=gnx_dir_path, is_max_connected=False,
                                    logger=PrintLogger("logger"))
            raw_ftr.build(should_dump=True)  # build features
            final_vec.append(FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm))
        return np.hstack(final_vec)

    def _degree_matrix(self, gnx, nodelist):
        degrees = gnx.degree(gnx.nodes)
        return np.diag([degrees[d] for d in nodelist])

    """
    builds a data dictionary
    { ... graph_name: ( A = Adjacency_matrix, x = graph_vec, label ) ...}  
    """
    def _build_data(self):
        if os.path.exists(os.path.join(PKL_DIR, self._params.id + "_data.pkl")):
            return pickle.load(open(os.path.join(PKL_DIR, self._params.id + "_data.pkl"), "rb"))
        data = {}
        idx_to_name = []

        for gnx_id, gnx in zip(self._multi_graph.graph_names(), self._multi_graph.graphs()):
            # if gnx.number_of_nodes() < 5:
            #     continue
            node_order = list(gnx.nodes)
            idx_to_name.append(gnx_id)
            A = nx.adjacency_matrix(gnx, nodelist=node_order)
            D = self._degree_matrix(gnx, nodelist=node_order)
            gnx_vec = self._gnx_vec(gnx_id, gnx, node_order)
            data[gnx_id] = (A, D, gnx_vec, self._labels[gnx_id])

        pickle.dump((data, idx_to_name), open(os.path.join(PKL_DIR, self._params.id + "_data.pkl"), "wb"))
        return data, idx_to_name

    def __getitem__(self, index):
        gnx_id = self._idx_to_name[index]
        A, D, x, label = self._data[gnx_id]
        return Tensor(A.todense()), Tensor(D), Tensor(x), label

    def __len__(self):
        return len(self._idx_to_name)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datset_sampler import ImbalancedDatasetSampler

    ds = BilinearDataset(AidsDatasetTestParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        sampler=ImbalancedDatasetSampler(ds)
    )
    p = []
    for i, (A, D, x, l) in enumerate(dl):
        print(i, A, D, x, l)
        p.append(l.item())
    e = 0
