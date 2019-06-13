import sys
import os

sys.path.insert(0, os.path.join("..", "..", "graph-measures"))
sys.path.insert(0, os.path.join("..", "..", "graph-measures", "features_algorithms"))
sys.path.insert(0, os.path.join("..", "..", "graph-measures", "graph_infra"))
sys.path.insert(0, os.path.join("..", "..", "graph-measures", "features_infra"))
sys.path.insert(0, os.path.join("..", "..", "graph-measures", "features_meta"))
sys.path.insert(0, os.path.join("..", "..", "graph-measures", "features_algorithms", "vertices"))
sys.path.insert(0, os.path.join("..", "..", "graphs-package", "features_processor"))
sys.path.insert(0, os.path.join("..", "..", "graphs-package", "multi_graph"))
sys.path.insert(0, os.path.join("..", "..", "graphs-package", "temporal_graphs"))
sys.path.insert(0, os.path.join("..", "..", "graphs-package", "features_processor", "motif_variations"))
sys.path.insert(0, os.path.join("..", "..", "graphs-package"))

import csv
from params.mutagen_params import MutagenAllExternalDataParams, MutagenDatasetAllParams, MutagenBilinearActivatorParams, \
    MutagenLayeredBilinearModuleParams
from params.parameters import DEG
from bilinear_activator import BilinearActivator
from bilinear_model import LayeredBilinearModule
from bilinear_model_gpu import LayeredBilinearModuleGPU
from dataset.dataset_model import BilinearDataset
from dataset.dataset_external_data import ExternalData
from multi_class_bilinear_activator import BilinearMultiClassActivator
from params.aids_params import AidsDatasetAllParams, AidsLayeredBilinearModuleParams, AidsBilinearActivatorParams, \
    AidsAllExternalDataParams
import pickle
import numpy as np
from bokeh.plotting import figure, show

from params.protein_params import ProteinAllExternalDataParams, ProteinDatasetAllParams, ProteinBilinearActivatorParams, \
    ProteinLayeredBilinearModuleParams
from params.web_params import WebAllExternalDataParams, WebDatasetAllParams, WebBilinearActivatorParams, \
    WebLayeredBilinearModuleParams
from scipy import stats


def get_activator_aids(dev_split, test_split, topological_ftrs=True):
    data_name = "Aids"
    ext_train = ExternalData(AidsAllExternalDataParams())
    ds_params = AidsDatasetAllParams()
    if not topological_ftrs:
        ds_params.FEATURES = []
    ds = BilinearDataset(ds_params, external_data=ext_train)
    activator_params = AidsBilinearActivatorParams()
    activator_params.TEST_SPLIT = test_split
    activator_params.DEV_SPLIT = dev_split
    module_params = AidsLayeredBilinearModuleParams(ftr_len=ds.len_features, embed_vocab_dim=ext_train.len_embed())
    return data_name, BilinearActivator(LayeredBilinearModule(module_params), activator_params, ds)


def get_activator_mutagen(dev_split, test_split, topological_ftrs=True):
    data_name = "Mutagen"
    ext_train = ExternalData(MutagenAllExternalDataParams())
    ds_params = MutagenDatasetAllParams()
    if not topological_ftrs:
        ds_params.FEATURES = [DEG]
    ds = BilinearDataset(ds_params, external_data=ext_train)
    activator_params = MutagenBilinearActivatorParams()
    activator_params.TEST_SPLIT = test_split
    activator_params.DEV_SPLIT = dev_split
    module_params = MutagenLayeredBilinearModuleParams(ftr_len=ds.len_features, embed_vocab_dim=ext_train.len_embed())
    return data_name, BilinearActivator(LayeredBilinearModule(module_params), activator_params, ds)


def get_activator_protein(dev_split, test_split, topological_ftrs=True):
    data_name = "Protein"
    ext_train = ExternalData(ProteinAllExternalDataParams())
    ds_params = ProteinDatasetAllParams()
    if not topological_ftrs:
        ds_params.FEATURES = []
    ds = BilinearDataset(ds_params, external_data=ext_train)
    activator_params = ProteinBilinearActivatorParams()
    activator_params.TEST_SPLIT = test_split
    activator_params.DEV_SPLIT = dev_split
    module_params = ProteinLayeredBilinearModuleParams(ftr_len=ds.len_features, embed_vocab_dim=ext_train.len_embed())
    return data_name, BilinearMultiClassActivator(LayeredBilinearModule(module_params), activator_params, ds)


def get_activator_web(dev_split, test_split):
    data_name = "Web"
    ext_train = ExternalData(WebAllExternalDataParams())
    ds = BilinearDataset(WebDatasetAllParams(), external_data=ext_train)
    activator_params = WebBilinearActivatorParams()
    activator_params.TEST_SPLIT = test_split
    activator_params.DEV_SPLIT = dev_split
    module_params = WebLayeredBilinearModuleParams(ftr_len=ds.len_features, embed_vocab_dim=ext_train.len_embed())
    return data_name, BilinearMultiClassActivator(LayeredBilinearModuleGPU(module_params), activator_params, ds)


def accuracy_as_function_of_train(activator_func, train_sizes=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), num_runs=3):
    res = []
    for train_size in train_sizes:
        dev_size, test_size = (1 - train_size) * 0.25, (1 - train_size) * 0.75
        print("train_sizes:",  train_size, "dev_size:", dev_size, "test_size:", test_size)
        for _ in range(num_runs):
            data_name, activator = activator_func(dev_size, test_size)
            activator.train(show_plot=False)
            res.append((train_size, activator.accuracy_train_vec, activator.auc_train_vec, activator.loss_train_vec,
                        activator.accuracy_dev_vec, activator.auc_dev_vec, activator.loss_dev_vec,
                        activator.accuracy_test_vec, activator.auc_test_vec, activator.loss_test_vec))
    pickle.dump(res, open(data_name + "_acc_as_f_of_train.pkl", "wb"))


def accuracy_as_function_of_external_data(activator_func, dev, test, num_runs=3):
    res = []

    for _ in range(num_runs):
        data_name, activator = activator_func(dev, test)
        activator.train(show_plot=False)
        res.append(("with", activator.accuracy_train_vec, activator.auc_train_vec, activator.loss_train_vec,
                    activator.accuracy_dev_vec, activator.auc_dev_vec, activator.loss_dev_vec,
                    activator.accuracy_test_vec, activator.auc_test_vec, activator.loss_test_vec))
    for _ in range(num_runs):
        data_name, activator = activator_func(dev, test, topological_ftrs=False)
        activator.train(show_plot=False)
        res.append(("without", activator.accuracy_train_vec, activator.auc_train_vec, activator.loss_train_vec,
                    activator.accuracy_dev_vec, activator.auc_dev_vec, activator.loss_dev_vec,
                    activator.accuracy_test_vec, activator.auc_test_vec, activator.loss_test_vec))

    pickle.dump(res, open(data_name + "_acc_as_f_of_external.pkl", "wb"))


def pick_best_csv(names):

    plt_table = [["data", "repeat", "external_data", "method", "best_epoch", "section", "measure", "score"]]

    for name in names:
        res = pickle.load(open(name + "_acc_as_f_of_external.pkl", "rb"))
        new_res = {"with": {"max_dev_auc": [],
                            "max_dev_acc": [],
                            "min_dev_loss": [],
                            "last_epoch": [],
                            "vec": []},
                   "without": {"max_dev_auc": [],
                               "max_dev_acc": [],
                               "min_dev_loss": [],
                               "last_epoch": [],
                               "vec": []}
                   }

        for external, accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, \
                accuracy_test_vec, auc_test_vec, loss_test_vec in res:

            new_res[external]["max_dev_auc"].append(np.argmax(auc_dev_vec))
            new_res[external]["max_dev_acc"].append(np.argmax(accuracy_dev_vec))
            new_res[external]["min_dev_loss"].append(np.argmin(loss_dev_vec))
            new_res[external]["last_epoch"].append(len(accuracy_train_vec) - 1)
            new_res[external]["vec"].append((accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec,
                                             auc_dev_vec, loss_dev_vec, accuracy_test_vec, auc_test_vec, loss_test_vec))

        for external, result in new_res.items():
            for i in range(len(result["max_dev_auc"])):

                for method in ["max_dev_auc", "max_dev_acc", "min_dev_loss", "last_epoch"]:
                    accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, accuracy_test_vec, auc_test_vec, loss_test_vec = \
                    result["vec"][i]

                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "train", "acc",
                                      accuracy_train_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "train", "auc",
                                      auc_train_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "train", "loss",
                                      loss_train_vec[result[method][i]]])

                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "dev", "acc",
                                      accuracy_dev_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "dev", "auc",
                                      auc_dev_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "dev", "loss",
                                      loss_dev_vec[result[method][i]]])

                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "test", "acc",
                                      accuracy_test_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "test", "auc",
                                      auc_test_vec[result[method][i]]])
                    plt_table.append([name, i, True if external == "with" else False, method, result[method][i], "test", "loss",
                                      loss_test_vec[result[method][i]]])

    with open("best_result_by_dev.csv", "wt") as csv_best:
        writer = csv.writer(csv_best)
        writer.writerows(plt_table)


def plot_accuracy_(name):
    res = pickle.load(open(name + "_acc_as_f_of_external.pkl", "rb"))
    new_res = {"with": {"acc": [], "vec": []},
               "without": {"acc": [], "vec": []}}

    for external, accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, \
            accuracy_test_vec, auc_test_vec, loss_test_vec in res:

        new_res[external]["acc"].append(accuracy_test_vec[np.argmax(auc_dev_vec)])
        new_res[external]["vec"].append((accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec,
                                         auc_dev_vec, loss_dev_vec, accuracy_test_vec, auc_test_vec, loss_test_vec))
    plt_table = []
    for external, result in new_res.items():
        best = np.argmax(result["acc"])
        print("best_acc_test " + external + "external data - " + name, result["acc"][best] * 100)
        x_axis = list(range(len(accuracy_train_vec)))

        acc_train = result["vec"][best][0]
        acc_dev = result["vec"][best][3]
        acc_test = result["vec"][best][6]

        p = figure(plot_width=600, plot_height=250, title="accuracy - " + name + " " + external + " external data",
                   x_axis_label="train size", y_axis_label="accuracy rate")

        color1, color2, color3 = ("green", "black", "blue")
        p.line(x_axis, acc_train, line_color=color1, legend="train")
        p.line(x_axis, acc_dev, line_color=color2, legend="dev")
        p.line(x_axis, acc_test, line_color=color3, legend="test")
        p.legend.location = "bottom_right"
        show(p)

        plt_table += [[name + " " + external + " external data - " + "train"] + acc_train,
                      [name + " " + external + " external data - " + "dev"] + acc_dev,
                      [name + "  " + external + " external data - " + "test"] + acc_test]

    with open(name + "_external_vs_pure_topological_plot_vectors.csv", "wt") as csv_err:
        writer = csv.writer(csv_err)
        writer.writerows(plt_table)


def pick_best_csv_as_function_of_train_csv(names):
    res_all = {}
    count = {}

    plt_table = [["data", "train_size", "repeat", "external_data", "method", "best_epoch", "section", "measure", "score"]]
    for name in names:
        res_all[name] = pickle.load(open(name + "_acc_as_f_of_train.pkl", "rb"))

    for name, res in res_all.items():

        for train_size, accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, accuracy_test_vec, auc_test_vec, loss_test_vec in res:

            count[(name, str(train_size))] = count.get((name, str(train_size)), 0) + 1

            repeat = count[(name, str(train_size))]
            result = {
                "max_dev_auc": np.argmax(auc_dev_vec),
                "max_dev_acc": np.argmax(accuracy_dev_vec),
                "min_dev_loss": np.argmin(loss_dev_vec),
                "last_epoch": len(accuracy_train_vec) - 1
            }

            for method in ["max_dev_auc", "max_dev_acc", "min_dev_loss", "last_epoch"]:
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "train", "acc",
                     accuracy_train_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "train", "auc",
                     auc_train_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "train", "loss",
                     loss_train_vec[result[method]]])

                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "dev", "acc",
                     accuracy_dev_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "dev", "auc",
                     auc_dev_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "dev", "loss",
                     loss_dev_vec[result[method]]])

                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "test", "acc",
                     accuracy_test_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "test", "auc",
                     auc_test_vec[result[method]]])
                plt_table.append(
                    [name, train_size, repeat, True, method, result[method], "test", "loss",
                     loss_test_vec[result[method]]])

    with open("best_pick_as_function_of_train_size.csv", "wt") as csv_acc:
        writer = csv.writer(csv_acc)
        writer.writerows(plt_table)


def plot_accuracy_as_function_of_train(names):
    res_all = {}
    for name in names:
        res_all[name] = pickle.load(open(name + "_acc_as_f_of_train.pkl", "rb"))

    for name, res in res_all.items():
        new_res = {}
        for train_size, accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, \
        accuracy_test_vec, auc_test_vec, loss_test_vec in res:
            if train_size not in new_res:
                new_res[train_size] = []
            new_res[train_size].append(accuracy_test_vec[np.argmax(auc_dev_vec)])
        res_all[name] = new_res

    vec_dict = {}
    x_axis = []
    for name, new_res in res_all.items():
        x_axis = []
        best_vec = []
        std_err_vec = []
        for train_size, acc_test_list in new_res.items():
            acc_test_list = list(sorted(acc_test_list))
            std_err = stats.sem(acc_test_list[-3:])
            best = acc_test_list[-1]
            best_vec.append(best)
            x_axis.append(train_size)
            std_err_vec.append(std_err)
        vec_dict[name] = (best_vec, std_err_vec)

    p1 = figure(plot_width=600, plot_height=250, title="accuracy as function of train size - ",
                x_axis_label="train size", y_axis_label="accuracy rate")
    p2 = figure(plot_width=600, plot_height=250, title="std-error as function of train size - ",
                x_axis_label="train size", y_axis_label="std_err")
    colors1 = ("green", "black", "blue")
    colors2 = ("red", "orange", "yellow")

    acc_table = [["data", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]]
    err_table = [["data", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]]

    for i, (name, (best_vec, std_err_vec)) in enumerate(vec_dict.items()):
        acc_table.append([name] + best_vec)
        err_table.append([name] + std_err_vec)
        p1.line(x_axis, best_vec, line_color=colors1[i], legend=name)
        p2.line(x_axis, std_err_vec, line_color=colors2[i], legend=name)

    with open("acc_as_function_of_train.csv", "wt") as csv_acc:
        writer = csv.writer(csv_acc)
        writer.writerows(acc_table)

    with open("std_err_as_function_of_train.csv", "wt") as csv_err:
        writer = csv.writer(csv_err)
        writer.writerows(err_table)

    p1.legend.location = "bottom_left"
    show(p1)
    show(p2)

#
# def plot_accuracy_as_function_of_train(name):
#     res = pickle.load(open(name + "_acc_as_f_of_train.pkl", "rb"))
#     new_res = {}
#     for train_size, accuracy_train_vec, auc_train_vec, loss_train_vec, accuracy_dev_vec, auc_dev_vec, loss_dev_vec, \
#     accuracy_test_vec, auc_test_vec, loss_test_vec in res:
#         if train_size not in new_res:
#             new_res[train_size] = []
#         new_res[train_size].append(accuracy_test_vec[np.argmax(auc_dev_vec)])
#
#     x_axis = []
#     best_vec = []
#     # mid_vec = []
#     # worst_vec = []
#     std_err_vec = []
#     for train_size, acc_test_list in new_res.items():
#         acc_test_list = list(sorted(acc_test_list))
#         std_err = stats.sem(acc_test_list[-3:])
#         best = acc_test_list[-1]
#         # mid = acc_test_list[-2]
#         # worst = acc_test_list[-3]
#         best_vec.append(best)
#         # mid_vec.append(mid)
#         # worst_vec.append(worst)
#         x_axis.append(train_size)
#         std_err_vec.append(std_err)
#
#     p = figure(plot_width=600, plot_height=250, title=name + " - accuracy as function of train size - ",
#                x_axis_label="train size", y_axis_label="accuracy rate")
#     color1, color2 = ("green", "red")
#     p.line(x_axis, best_vec, line_color=color1, legend="best_accuracy")
#     p.line(x_axis, std_err_vec, line_color=color2, legend="std_err")
#     p.legend.location = "bottom_left"
#     show(p)


if __name__ == "__main__":
    # accuracy_as_function_of_train(get_activator_web, num_runs=5)
    # accuracy_as_function_of_train(get_activator_mutagen, num_runs=4)
    # accuracy_as_function_of_train(get_activator_protein, num_runs=4)
    # accuracy_as_function_of_train(get_activator_aids)
    # plot_accuracy_as_function_of_train("Aids")
    # accuracy_as_function_of_external_data(get_activator_mutagen, dev=0.125, test=0.75, num_runs=3)
    # accuracy_as_function_of_external_data(get_activator_aids, dev=0.1153, test=0.5388, num_runs=3)
    # accuracy_as_function_of_external_data(get_activator_protein, dev=0.333, test=0.333, num_runs=3)
    # plot_accuracy_as_function_of_train(["Aids", "Mutagen", "Protein"])
    pick_best_csv(["Aids", "Mutagen", "Protein"])
    # pick_best_csv_as_function_of_train_csv(["Aids", "Mutagen", "Protein"])
    plot_accuracy_("Protein")

