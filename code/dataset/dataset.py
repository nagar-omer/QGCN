from scipy.stats import zscore
from torch import Tensor
from torch.utils.data import Dataset
from collections import Counter

from dataset.dataset_external_data import ExternalData
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from multi_graph import MultiGraph
from params.parameters import BilinearDatasetParams, DATA_INPUT_DIR, PKL_DIR, FEATURES_PKL_DIR, DEG, IN_DEG, OUT_DEG
import os
import pandas as pd
import networkx as nx
import pickle
import numpy as np


class BilinearDataset(Dataset):
    def __init__(self, params: BilinearDatasetParams, external_data: ExternalData = None):
        self._params = params
        self._logger = PrintLogger("logger")
        # path to base directory
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")

        self._external_data = external_data
        # init ftr_meta dictionary and other ftr attributes
        self._init_ftrs()
        self._src_file_path = os.path.join(self._base_dir, DATA_INPUT_DIR, params.DATASET_FILENAME)
        self._multi_graph, self._labels, self._label_to_idx, self._idx_to_label = self._build_multi_graph()

        self._data, self._idx_to_name = self._build_data()

    @property
    def all_labels(self):
        return self._idx_to_label

    @property
    def label_count(self):
        return Counter([v[4] for name, v in self._data.items()])

    def label(self, idx):
        return self._data[self._idx_to_name[idx]][4]

    @property
    def len_features(self):
        return self._data[self._idx_to_name[0]][2].shape[1]

    def _init_ftrs(self):
        self._deg, self._in_deg, self._out_deg, self._is_ftr, self._ftr_meta = False, False, False, False, {}
        self._is_external_data = False if self._external_data is None else True
        # params.FEATURES contains string and list of two elements (matching to key: value)
        # should Deg/In-Deg/Out-Deg be calculated
        for ftr in self._params.FEATURES:
            if ftr == DEG:
                self._deg = True
            elif ftr == IN_DEG:
                self._in_deg = True
            elif ftr == OUT_DEG:
                self._out_deg = True
            else:
                self._ftr_meta[ftr[0]] = ftr[1]

        # add directories for pickles
        if len(self._ftr_meta) > 0:
            self._ftr_path = os.path.join(self._base_dir, FEATURES_PKL_DIR, self._params.DATASET_NAME)
            if not os.path.exists(self._ftr_path):
                os.mkdir(self._ftr_path)
            self._is_ftr = True

    """
    build multi graph according to csv 
    each community is a single graph, no consideration to time
    """
    def _build_multi_graph(self):
        path_pkl = os.path.join(self._base_dir, PKL_DIR, self._params.DATASET_NAME + "_split_" +
                                str(self._params.PERCENTAGE) + "_mg.pkl")
        if os.path.exists(path_pkl):
            return pickle.load(open(path_pkl, "rb"))
        multi_graph_dict = {}
        labels = {}
        label_to_idx = {}
        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._src_file_path)
        stop = data_df.shape[0] * self._params.PERCENTAGE

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
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._in_deg:
            degrees = gnx.in_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._out_deg:
            degrees = gnx.out_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._is_external_data and self._external_data.is_value:
            final_vec.append(np.matrix([self._external_data.value_feature(gnx_id, d) for d in node_order]))
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

    def _z_score_all_data(self, data):
        all_data_values_vec = []                # stack all vectors for all graphs
        key_to_idx_map = []                     # keep ordered list (g_id, num_nodes) according to stack order

        # stack
        for g_id, (A, D, gnx_vec, embed_vec, label) in data.items():
            all_data_values_vec.append(gnx_vec)
            key_to_idx_map.append((g_id, gnx_vec.shape[0]))  # g_id, number of nodes ... ordered
        all_data_values_vec = np.vstack(all_data_values_vec)

        # z-score data
        z_scored_data = zscore(all_data_values_vec, axis=0)

        # rebuild data to original form -> split stacked matrix according to <list: (g_id, num_nodes)>
        new_data_dict = {}
        start_idx = 0
        for g_id, num_nodes in key_to_idx_map:
            new_data_dict[g_id] = (data[g_id][0], data[g_id][1], z_scored_data[start_idx: start_idx+num_nodes],
                                   data[g_id][3], data[g_id][4])
            start_idx += num_nodes

        return new_data_dict

    """
    builds a data dictionary
    { ... graph_name: ( A = Adjacency_matrix, x = graph_vec, label ) ...}  
    """
    def _build_data(self):
        ext_data_id = "None" if not self._is_external_data else "_embed_ftr_" + str(self._external_data.embed_headers)\
                                                                + "_value_ftr_" + str(self._external_data.value_headers)
        pkl_path = os.path.join(self._base_dir, PKL_DIR, self._params.id + ext_data_id + "_data.pkl")
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path, "rb"))
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
            embed_vec = [self._external_data.embed_feature(gnx_id, d) for d in node_order] \
                if self._is_external_data and self._external_data.is_embed else None
            data[gnx_id] = (A, D, gnx_vec, embed_vec, self._labels[gnx_id])

        data = self._z_score_all_data(data)
        pickle.dump((data, idx_to_name), open(pkl_path, "wb"))
        return data, idx_to_name

    def __getitem__(self, index):
        gnx_id = self._idx_to_name[index]
        A, D, x, embed, label = self._data[gnx_id]
        embed = 0 if embed is None else Tensor(embed).long()
        return Tensor(A.todense()), Tensor(D), Tensor(x), embed, label

    def __len__(self):
        return len(self._idx_to_name)


if __name__ == "__main__":
    from params.protein_params import ProteinDatasetTrainParams
    from torch.utils.data import DataLoader
    from dataset.datset_sampler import ImbalancedDatasetSampler
    ds = BilinearDataset(ProteinDatasetTrainParams())
    # ds = BilinearDataset(AidsDatasetTestParams())
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
