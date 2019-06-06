from torch.nn import Module, Linear, Dropout, Embedding, ModuleList
import torch
from params.parameters import LinearLayerParams, BilinearLayerParams, NORM_REDUCED, \
    NORM_REDUCED_SYMMETRIC, LayeredBilinearModuleParams

"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""


class LinearLayer(Module):
    def __init__(self, params: LinearLayerParams):
        super(LinearLayer, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.ROW_DIM, params.COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._dropout = Dropout(p=params.DROPOUT) if params.DROPOUT else None
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0) if self._dropout else x0
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.mm(A[0, :, :], x0[0, :, :])
        self._sync()

        x = self._linear(Ax)
        x1 = self._activation(x)
        return x1.unsqueeze(dim=0)


class BilinearLayer(Module):
    def __init__(self, params: BilinearLayerParams):
        super(BilinearLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(params.LEFT_LINEAR_ROW_DIM, params.LEFT_LINEAR_COL_DIM)
        self._right_linear = Linear(params.RIGHT_LINEAR_ROW_DIM, params.RIGHT_LINEAR_COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._activation_args = params.ACTIVATION_FUNC_ARGS
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def forward(self, A, x0, x1):
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.mm(torch.t(x1[0, :, :]), A[0, :, :])
        self._sync()
        W2_x1_A = self._left_linear(torch.t(x1_A))
        W2_x1_A_x0 = torch.mm(torch.t(W2_x1_A), x0[0, :, :])
        self._sync()
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        x2 = self._activation(W2_x1_A_x0_W3, **self._activation_args)
        return x2


class LayeredBilinearModule(Module):
    """
    first linear layer is executed numerous times
    """

    def __init__(self, params: LayeredBilinearModuleParams):
        super(LayeredBilinearModule, self).__init__()

        # add embedding layers
        self._is_embed = False if params.EMBED_VOCAB_DIMS is None else True
        if self._is_embed:
            # embeddings are added to ftr vector -> update dimensions of relevant weights
            params.LINEAR_PARAMS_LIST[0].ROW_DIM += sum(params.EMBED_DIMS)
            params.BILINEAR_PARAMS.RIGHT_LINEAR_ROW_DIM += sum(params.EMBED_DIMS)

            # add embedding layers
            self._embed_layers = []
            for vocab_dim, embed_dim in zip(params.EMBED_VOCAB_DIMS, params.EMBED_DIMS):
                self._embed_layers.append(Embedding(vocab_dim, embed_dim))
            self._embed_layers = ModuleList(self._embed_layers)

        self._norm = params.NORM
        self._num_layers = params.NUM_LAYERS
        self._linear_layers = []
        # create linear layers
        for i in range(params.NUM_LAYERS):
            self._linear_layers.append(LinearLayer(params.LINEAR_PARAMS_LIST[i]))
        self._linear_layers = ModuleList(self._linear_layers)

        self._bilinear_layer = BilinearLayer(params.BILINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LR, params.OPTIMIZER, params.WEIGHT_DECAY)

        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu
        for layer in self._linear_layers:
            layer.gpu_device(is_gpu)
        self._bilinear_layer.gpu_device(is_gpu)

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _fix_shape(self, input):
        return input if len(input.shape) == 3 else input.unsqueeze(dim=0)

    def forward(self, A, D, x0, embed):
        if self._is_embed:
            list_embed = []
            for i, embedding in enumerate(self._embed_layers):
                list_embed.append(embedding(embed[0, :, i]).unsqueeze(dim=0))
            x0 = torch.cat([x0] + list_embed, dim=2)

        # A, D, x0 = self._fix_shape(A), self._fix_shape(D), self._fix_shape(x0)
        if self._norm == NORM_REDUCED:
            # D^-0.5 A D^-0.5
            D_squrt = torch.sqrt(D[0, :, :]).float()
            self._sync()
            AD = torch.mm(A[0, :, :], D_squrt)
            self._sync()
            DAD = torch.mm(D_squrt, AD)
            self._sync()
            Adj = DAD.unsqueeze(dim=0)
            self._sync()
        if self._norm == NORM_REDUCED_SYMMETRIC:
            # D^-0.5 [A + A.T + I] D^-0.5
            pass
        else:  # no normalization
            Adj = A
            self._sync()

        x1 = x0.clone()
        self._sync()
        for i in range(self._num_layers):
            x1 = self._linear_layers[i](Adj, x1)
        x2 = self._bilinear_layer(Adj, x0, x1)
        return x2


if __name__ == "__main__":

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
