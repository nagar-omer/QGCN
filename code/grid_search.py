import os
from time import strftime, gmtime
import ast
from torch.optim import Adam, SGD
from bilinear_activator import BilinearActivator
from bilinear_model import LayeredBilinearModule
from parameters import RefaelDatasetParams, LayeredBilinearModuleParams, BilinearActivatorParams, BFS, CENTRALITY, \
    OUT_DEG, IN_DEG, DEG
from dataset import BilinearDataset
from itertools import product
import csv
import numpy as np


class GridSearch:
    def __init__(self):
        pass

    def _all_configurations(self):
        """
        set grid parameters here
        """
        data_split = [0.5, 0.75, 1]
        optimizer = [Adam, SGD]
        lrs = [1e-3, 1e-2, 1e-1, 1]
        dropout = [0, 0.1, 0.2, 0.3, 0.4]
        regularization = [0]
        input_vec = [[DEG], [IN_DEG, OUT_DEG], [DEG, IN_DEG, OUT_DEG, CENTRALITY, BFS]]
        layers_config = [[[None, 10]], [[None, 30], [30, 10]], [[None, 30], [30, 50], [50, 10]]]
        batch_size = [2, 4, 8, 16]

        configurations = list(product(*[data_split, optimizer, lrs, dropout, regularization, input_vec, layers_config,
                                        batch_size]))

        # prepare param objects
        for split, optimizer, lr, dropout, regularization, input_vec, layers_config, batch_size in configurations:
            # str for configuration
            config_str = "|".join([str(split), str(optimizer), str(lr), str(dropout), str(regularization),
                                   str(input_vec), str(layers_config), str(batch_size)])
            # dataset
            ds_params = RefaelDatasetParams()
            ds_params.FEATURES = input_vec
            ds_params.PERCENTAGE = split
            dataset = BilinearDataset(ds_params)

            # model
            model_params = LayeredBilinearModuleParams(ftr_len=dataset.len_features, layer_dim=layers_config)
            model_params.DROPOUT = dropout
            model_params.WEIGHT_DECAY = regularization
            model_params.LR = lr
            model_params.OPTIMIZER = optimizer

            # activator
            activator_params = BilinearActivatorParams()
            activator_params.BATCH_SIZE = batch_size
            yield dataset, model_params, activator_params, config_str

    def _check_configuration(self, dataset: BilinearDataset, model_params: LayeredBilinearModuleParams,
                             activator_params: BilinearActivatorParams):

        model = LayeredBilinearModule(model_params)
        activator = BilinearActivator(model, activator_params, dataset)
        activator.train(show_plot=False)

        return activator.accuracy_train_vec, activator.auc_train_vec, activator.loss_train_vec, \
               activator.accuracy_dev_vec, activator.auc_dev_vec, activator.loss_dev_vec, \
               activator.accuracy_test_vec, activator.auc_test_vec, activator.loss_test_vec

    def go(self):
        time = strftime("%m%d%H%M%S", gmtime())
        out_res = open(os.path.join("grid_results", "grid_" + time + ".txt"), "wt")
        out_res.write("line0: config, line1: acc_train, line2: auc_train, line3: loss_train, line4: acc_dev, "
                      "line5: auc_dev, line6: loss_dev, line7: acc_test, line8: auc_test, line9: loss_test\n")
        for dataset, model_params, activator_params, config_str in self._all_configurations():
            print(config_str)
            acc_train, auc_train, loss_train, acc_dev, auc_dev, loss_dev, acc_test, auc_test, loss_test = \
                self._check_configuration(dataset, model_params, activator_params)
            out_res.write(config_str + "\n"
                          + str(acc_train) + "\n"
                          + str(auc_train) + "\n"
                          + str(loss_train) + "\n"
                          + str(acc_dev) + "\n"
                          + str(auc_dev) + "\n"
                          + str(loss_dev) + "\n"
                          + str(acc_test) + "\n"
                          + str(auc_test) + "\n"
                          + str(loss_test) + "\n")

    def parse(self, res_file):
        file_name = sorted(os.listdir("grid_results"))[res_file] if type(res_file) is int else res_file
        res_file = open(os.path.join("grid_results", file_name), "rt")
        res_file.readline()                 # skip header

        res_list = [["best_train_acc", "best_train_auc", "best_train_loss", "std_train_loss", "best_dev_acc", "best_dev_auc",
                     "best_dev_loss", "std_dev_loss", "best_test_acc", "best_test_auc", "best_test_loss", "std_test_loss", "data_split",
                     "optimizer", "lrs", "dropout", "regularization", "layers_config", "batch_size",
                     'DEGREE', 'IN_DEGREE', 'OUT_DEGREE', "betweenness_centrality",  "bfs_moments"]]

        while True:
            config = res_file.readline()
            acc_train = res_file.readline()
            auc_train = res_file.readline()
            loss_train = res_file.readline()
            acc_dev = res_file.readline()
            auc_dev = res_file.readline()
            loss_dev = res_file.readline()
            acc_test = res_file.readline()
            auc_test = res_file.readline()
            loss_test = res_file.readline()

            if not loss_test:
                break

            std_loss_train = np.std(ast.literal_eval(loss_train))
            std_loss_dev = np.std(ast.literal_eval(loss_dev))
            std_loss_test = np.std(ast.literal_eval(loss_test))

            config = config.replace("\n", "").split('|')
            ftrs = [str(True) if ftr in config[5] else str(False) for ftr in
                    ['DEGREE', 'IN_DEGREE', 'OUT_DEGREE', "betweenness_centrality", "bfs_moments"]]
            del config[5]
            best_result_idx = np.argmax(ast.literal_eval(auc_dev))

            acc_train = ast.literal_eval(acc_train)[best_result_idx]
            auc_train = ast.literal_eval(auc_train)[best_result_idx]
            loss_train = ast.literal_eval(loss_train)[best_result_idx]
            acc_dev = ast.literal_eval(acc_dev)[best_result_idx]
            auc_dev = ast.literal_eval(auc_dev)[best_result_idx]
            loss_dev = ast.literal_eval(loss_dev)[best_result_idx]
            acc_test = ast.literal_eval(acc_test)[best_result_idx]
            auc_test = ast.literal_eval(auc_test)[best_result_idx]
            loss_test = ast.literal_eval(loss_test)[best_result_idx]

            config_line = [str(acc_train)] + [str(auc_train)] + [str(loss_train)] + [str(std_loss_train)] + \
                          [str(acc_dev)] + [str(auc_dev)] + [str(loss_dev)] + [str(std_loss_dev)] + \
                          [str(acc_test)] + [str(auc_test)] + [str(loss_test)] + [str(std_loss_test)] + \
                           config + ftrs

            res_list.append(config_line)

        with open(os.path.join("grid_results", file_name.strip(".txt") + "_analyzed.csv"), "wt") as f:
            writer = csv.writer(f)
            writer.writerows(res_list)


if __name__ == "__main__":
    # GridSearch().go()
    GridSearch().parse(0)


