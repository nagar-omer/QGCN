from torch import sigmoid, tanh
from torch.optim import Adam
from torch.nn import functional
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from bilinear_activator import BilinearActivator
from bilinear_model import LayeredBilinearModule
from dataset.dataset import BilinearDataset
from feature_calculators import FeatureMeta
from params.parameters import BilinearDatasetParams, BilinearActivatorParams, BilinearLayerParams, LinearLayerParams, \
    LayeredBilinearModuleParams, DEG, CENTRALITY, BFS, NORM_REDUCED

# ------------------------------------------------------  REFAEL -------------------------------------------------------


class RefaelDatasetParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Refael_Binary_18_12"
        self.DATASET_FILENAME = "Refael_18_12_18_Binary.csv"
        self.SRC_COL = "SourceID"
        self.DST_COL = "DestinationID"
        self.GRAPH_NAME_COL = "Community"
        self.LABEL_COL = "target"
        self.PERCENTAGE = 1
        self.DIRECTED = True
# ----------------------------------------------------------------------------------------------------------------------


class RefaelBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 1           # out cols
        self.ACTIVATION_FUNC = sigmoid
        self.ACTIVATION_FUNC_ARGS = {}


class RefaelLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__(in_dim, out_dim, dropout)
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = tanh
        self.DROPOUT = dropout


class RefaelLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.NORM = NORM_REDUCED
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0

        self.NUM_LAYERS = 2
        self.LINEAR_PARAMS_LIST = [
            RefaelLinearLayerParams(in_dim=ftr_len, out_dim=50, dropout=self.DROPOUT),
            RefaelLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
            RefaelLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
            RefaelLinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
        ]
        self.BILINEAR_PARAMS = RefaelBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                         self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class RefaelBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        self.LOSS = functional.binary_cross_entropy_with_logits  # f.factor_loss  #
        self.BATCH_SIZE = 16
        self.EPOCHS = 100


if __name__ == '__main__':
    refael_train_ds = BilinearDataset(RefaelDatasetParams())
    activator = BilinearActivator(LayeredBilinearModule(RefaelLayeredBilinearModuleParams(ftr_len=refael_train_ds.len_features)),
                                  RefaelBilinearActivatorParams(), refael_train_ds)
    activator.train()
