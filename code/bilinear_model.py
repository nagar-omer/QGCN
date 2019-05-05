from torch.nn import Module, Linear, Dropout
from torch.nn.functional import softmax
import torch
from parameters import LinearLayerParams, BilinearLayerParams, BilinearModuleParams, NORM_REDUCED, \
    NORM_REDUCED_SYMMETRIC, LayeredBilinearModuleParams
import numpy as np

"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""


class LinearLayer(Module):
    def __init__(self, params: LinearLayerParams):
        super(LinearLayer, self).__init__()
        # useful info in forward function
        self._norm = params.NORM
        self._linear = Linear(params.ROW_DIM, params.COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._dropout = Dropout(p=params.DROPOUT) if params.DROPOUT else None

    def forward(self, A, D, x0):
        if self._norm == NORM_REDUCED:
            # D^-0.5 A D^-0.5
            D_squrt = np.power(D[0, :, :], np.ones((D.shape[1], D.shape[2])) - np.identity(D.shape[1]) * -1.5).float()
            AD = torch.mm(A[0, :, :], D_squrt)
            DAD = torch.mm(D_squrt, AD)
            Adj = DAD
        if self._norm == NORM_REDUCED_SYMMETRIC:
            # D^-0.5 [A + A.T + I] D^-0.5
            pass
        else:       # no normalization
            Adj = A[0, :, :]

        # Dropout layer
        x0 = self._dropout(x0) if self._dropout else x0
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        x = self._linear(torch.mm(Adj, x0[0, :, :]))
        x1 = self._activation(x)
        return x1.unsqueeze(dim=0)


class BilinearLayer(Module):
    def __init__(self, params: BilinearLayerParams):
        super(BilinearLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(params.LEFT_LINEAR_ROW_DIM, params.LEFT_LINEAR_COL_DIM)
        self._right_linear = Linear(params.RIGHT_LINEAR_ROW_DIM, params.RIGHT_LINEAR_COL_DIM)
        self._activation = params.ACTIVATION_FUNC

    def forward(self, A, x0, x1):
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.mm(torch.t(x1[0, :, :]), A[0, :, :])
        W2_x1_A = self._left_linear(torch.t(x1_A))
        W2_x1_A_x0 = torch.mm(torch.t(W2_x1_A), x0[0, :, :])
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        x2 = self._activation(W2_x1_A_x0_W3)
        return x2


class BilinearModule(Module):
    def __init__(self, params: BilinearModuleParams):
        super(BilinearModule, self).__init__()
        self._linear_layer = LinearLayer(params.LINEAR_PARAMS)
        self._bilinear_layer = BilinearLayer(params.BILINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LR, params.OPTIMIZER, params.WEIGHT_DECAY)

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, A, D, x0):
        x1 = self._linear_layer(A, D, x0)
        x2 = self._bilinear_layer(A, x0, x1)
        return x2


class LayeredBilinearModule(Module):
    """
    first linear layer is executed numerous times
    """
    def __init__(self, params: LayeredBilinearModuleParams):
        super(LayeredBilinearModule, self).__init__()
        self._num_layers = params.NUM_LAYERS
        self._linear_layers = []
        # create linear layers
        for i in range(params.NUM_LAYERS):
            self._linear_layers.append(LinearLayer(params.LINEAR_PARAMS_LIST[i]))

        self._bilinear_layer = BilinearLayer(params.BILINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LR, params.OPTIMIZER, params.WEIGHT_DECAY)

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, A, D, x0):
        x1 = x0.clone()
        for i in range(self._num_layers):
            x1 = self._linear_layers[i](A, D, x1)
        x2 = self._bilinear_layer(A, x0, x1)
        return x2


if __name__ == "__main__":
    from dataset import RefaelDataset
    from torch.utils.data import DataLoader
    from parameters import RefaelDatasetParams
    ds = RefaelDataset(RefaelDatasetParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
    )

    module = LayeredBilinearModule(LayeredBilinearModuleParams(ftr_len=ds.len_features))
    # module = BilinearModule(BilinearModuleParams())
    for i, (_A, _D, _x0, _l) in enumerate(dl):
        _x2 = module(_A, _D, _x0)
        e = 0
