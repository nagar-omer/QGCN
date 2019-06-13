from torch.nn.functional import relu, softmax, cross_entropy
from torch.optim import Adam
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from bilinear_model import LayeredBilinearModule
from dataset.dataset_model import BilinearDataset
from dataset.dataset_external_data import ExternalData
from feature_calculators import FeatureMeta
from multi_class_bilinear_activator import BilinearMultiClassActivator
from params.parameters import BilinearDatasetParams, BilinearActivatorParams, BilinearLayerParams, LinearLayerParams, \
    LayeredBilinearModuleParams, DEG, CENTRALITY, BFS, NORM_REDUCED, ExternalDataParams


# ---------------------------------------------------  PROTEIN ---------------------------------------------------------
class CoilRagAllExternalDataParams(ExternalDataParams):
    def __init__(self):
        super().__init__()
        self.GRAPH_COL = "g_id"
        self.NODE_COL = "node"
        self.FILE_NAME = "COIL-RAG_external_data_all.csv"
        self.EMBED_COLS = []
        self.VALUE_COLS = ["histo1", "histo10", "histo11", "histo12", "histo13", "histo14", "histo15", "histo16",
                           "histo17", "histo18", "histo19", "histo2", "histo20", "histo21", "histo22", "histo23",
                           "histo24", "histo25", "histo26", "histo27", "histo28", "histo29", "histo3", "histo30",
                           "histo31", "histo32", "histo33", "histo34", "histo35", "histo36", "histo37", "histo38",
                           "histo39", "histo4", "histo40", "histo41", "histo42", "histo43", "histo44", "histo45",
                           "histo46", "histo47", "histo48", "histo49", "histo5", "histo50", "histo51", "histo52",
                           "histo53", "histo54", "histo55", "histo56", "histo57", "histo58", "histo59", "histo6",
                           "histo60", "histo61", "histo62", "histo63", "histo64", "histo7", "histo8", "histo9"]


class CoilRagDatasetAllParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "coil_rag_train"
        self.DATASET_FILENAME = "COIL-RAG_all.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, CENTRALITY, BFS]


# ----------------------------------------------------------------------------------------------------------------------


class CoilRagBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 100         # out cols
        self.ACTIVATION_FUNC = softmax
        self.ACTIVATION_FUNC_ARGS = {"dim": 2}


class CoilRagLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__(in_dim, out_dim, dropout)
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class CoilRagLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.EMBED_DIMS = []
        self.NORM = NORM_REDUCED
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 1e-2

        if layer_dim is None:
            self.NUM_LAYERS = 2
            self.LINEAR_PARAMS_LIST = [
                CoilRagLinearLayerParams(in_dim=ftr_len, out_dim=50, dropout=self.DROPOUT),
                CoilRagLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
                CoilRagLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
                CoilRagLinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = CoilRagBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                          self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class CoilRagBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        self.LOSS = cross_entropy  # f.factor_loss  #
        self.BATCH_SIZE = 16
        self.EPOCHS = 500
        self.DATASET = "COIL_RAG - MultiClass"


if __name__ == '__main__':
    ext_train = ExternalData(CoilRagAllExternalDataParams())
    aids_train_ds = BilinearDataset(CoilRagDatasetAllParams(), external_data=ext_train)

    activator = BilinearMultiClassActivator(LayeredBilinearModule(CoilRagLayeredBilinearModuleParams(
        ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
        CoilRagBilinearActivatorParams(), aids_train_ds)
    activator.train()
