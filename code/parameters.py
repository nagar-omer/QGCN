from torch import sigmoid, tanh
import torch
from torch.nn.functional import binary_cross_entropy, relu
from torch.optim import Adam, SGD
from torch.nn import functional
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from feature_calculators import FeatureMeta

CODE_DIR = "code"
DATA_INPUT_DIR = "dataset_input"
PKL_DIR = "pkl"
FEATURES_PKL_DIR = os.path.join(PKL_DIR, "features")
NORM_REDUCED = "_REDUCED_"
NORM_REDUCED_SYMMETRIC = "_REDUCED_SYMMETRIC_"

DEG = "_DEGREE_"
IN_DEG = "_IN_DEGREE_"
OUT_DEG = "_OUT_DEGREE_"
CENTRALITY = ["betweenness_centrality", FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})]
BFS = ["bfs_moments", FeatureMeta(BfsMomentsCalculator, {"bfs"})]


class FactorLoss:
    def __init__(self):
        self._begin_low_limit = 0  # 0 .. 1
        self._end_low_limit = 0.3  # 0.5 .. 1
        self._interval = 1e-4

        self._curr_start = self._begin_low_limit
        self._curr_epoch = 0

    def factor_loss(self, output, target, jump=False):
        # scale = 1 - self._curr_start
        # shift = 1 - scale
        # loss = -((target * torch.log(output * scale + shift)) + ((1 - target) * torch.log((1 - output) * scale + shift)))
        loss = -((target * torch.log(output)) + ((1 - target) * torch.log((1 - output))))

        # if jump and self._curr_start + self._interval < self._end_low_limit:
        #     self._curr_start = self._curr_start + self._interval
        return loss


class BilinearDatasetParams:
    def __init__(self):
        self.DATASET_NAME = "Refael_Binary_18_12"
        self.DATASET_FILENAME = "Refael_18_12_18_Binary.csv"
        self.SRC_COL = "SourceID"
        self.DST_COL = "DestinationID"
        self.GRAPH_NAME_COL = "Community"
        self.LABEL_COL = "target"
        self.PERCENTAGE = 1
        self.DIRECTED = True
        self.FEATURES = [DEG, IN_DEG, OUT_DEG, CENTRALITY, BFS]

    @property
    def id(self):
        attributes = ["DATASET_NAME", "PERCENTAGE", "DIRECTED", "FEATURES"]

        attr_str = []
        for attr in attributes:
            if attr == "FEATURES":
                attr_str.append(attr + "_" + str([k[0] if type(k) is list else k for k in self.FEATURES]))
            else:
                attr_str.append(attr + "_" + str(getattr(self, attr)))
        return "_".join(attr_str)

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


# ------------------------------------------------------  AIDS ---------------------------------------------------------


class AidsDatasetTrainParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_train"
        self.DATASET_FILENAME = "AIDS_train.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, CENTRALITY, BFS]


class AidsDatasetDevParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_dev"
        self.DATASET_FILENAME = "AIDS_dev.csv"


class AidsDatasetTestParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_test"
        self.DATASET_FILENAME = "AIDS_test.csv"
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------  MUTAGEN ---------------------------------------------------------


class MutagenDatasetTrainParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_train"
        self.DATASET_FILENAME = "Mutagenicity_train.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, CENTRALITY, BFS]


class MutagenDatasetDevParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_dev"
        self.DATASET_FILENAME = "Mutagenicity_dev.csv"


class MutagenDatasetTestParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_test"
        self.DATASET_FILENAME = "Mutagenicity_test.csv"
# ----------------------------------------------------------------------------------------------------------------------


class BilinearLayerParams:
    def __init__(self, in_col_dim, ftr_len):
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 1           # out cols
        self.ACTIVATION_FUNC = sigmoid


class LinearLayerParams:
    def __init__(self, in_dim, out_dim, dropout=0.3):
        self.NORM = NORM_REDUCED
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class BilinearModuleParams:
    def __init__(self, ftr_len=6):
        self.LINEAR_PARAMS = LinearLayerParams(in_dim=ftr_len, out_dim=50)
        self.BILINEAR_PARAMS = BilinearLayerParams(self.LINEAR_PARAMS.COL_DIM, self.LINEAR_PARAMS.ROW_DIM)
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0


class LayeredBilinearModuleParams:
    def __init__(self, ftr_len=6, layer_dim=None):
        self.DROPOUT = 0.1
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0

        self.NUM_LAYERS = len(layer_dim) if layer_dim else 2
        if layer_dim:
            self.LINEAR_PARAMS_LIST = []
            self.LINEAR_PARAMS_LIST.append(LinearLayerParams(in_dim=ftr_len, out_dim=layer_dim[0][1], dropout=self.DROPOUT))
            for in_dim, out_dim in layer_dim[1:]:
                self.LINEAR_PARAMS_LIST.append(LinearLayerParams(in_dim=in_dim, out_dim=out_dim, dropout=self.DROPOUT))
        else:
            self.LINEAR_PARAMS_LIST = [
                LinearLayerParams(in_dim=ftr_len, out_dim=50, dropout=self.DROPOUT),
                LinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
                LinearLayerParams(in_dim=100, out_dim=200, dropout=self.DROPOUT),
                LinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = BilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                   self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class BilinearActivatorParams:
    def __init__(self):
        f = FactorLoss()
        self.DEV_SPLIT = 0.2
        self.TEST_SPLIT = 0.6
        self.LOSS = functional.binary_cross_entropy_with_logits  # f.factor_loss  #
        self.BATCH_SIZE = 4
        self.EPOCHS = 250
