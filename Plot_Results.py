import numpy as np
from matplotlib import pyplot as plt
import warnings
from sklearn import metrics
from matplotlib.patches import Ellipse
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.patches import Polygon, Rectangle
from matplotlib.patches import Patch
from matplotlib import pylab

# from Image_Results import Image_Results, Sample_Images

warnings.filterwarnings('ignore')

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'SAA-ADCA-MSA', 'ECO-ADCA-MSA', 'COA-ADCA-MSA', 'MAOA-ADCA-MSA', 'SEMAO-ADCA-MSA']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        Conv_Graph = Fitness[i]
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    for n in range(No_of_Dataset):
        lw = 3
        cls = ['GNN', 'ResNet-50', 'EfficientNetV2-ConvNeXt', 'Distributed CNN-LightGBM', 'QEGT-SGINA']
        Actual = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path = "./Results/ROC_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def Plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    kfold = [1, 2, 3, 4]
    Algorithm = ['GNN', 'ResNet-50', 'EfficientNetV2-ConvNeXt', 'Distributed CNN-LightGBM', 'QEGT-SGINA']
    Classifier = ['CNN', 'Mobilenet', 'Inception Net', 'cGAN', 'ULD-CCO-DR-AcGAN-NAF']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Algorithm Comparison of No.of Epochs')
            term_max = np.max(Graph)
            cap_height = term_max * 0.05
            X = np.arange(len(kfold))
            width = 0.18
            colors = ['#FFB703', '#FB8500', '#E63946', '#D00000', '#9D0208']
            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 0] - cap_height, 0), width=width,
                               color=colors[0], label=Algorithm[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 1] - cap_height, 0), width=width,
                               color=colors[1], label=Algorithm[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 2] - cap_height, 0), width=width,
                               color=colors[2], label=Algorithm[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 3] - cap_height, 0), width=width,
                               color=colors[3], label=Algorithm[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width,
                               color=colors[4], label=Algorithm[4]))

            plt.xticks(X + 0.40, ['20', '40', '60', '80'], fontname="Arial",
                       fontsize=14, fontweight='bold', color='#14213d')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')

            dot_markers = [plt.Line2D([2], [2], marker='s', color='w',
                                      markerfacecolor=color, markersize=10) for color in colors]

            plt.legend(dot_markers, Algorithm, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), fontsize=9,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()

                    color = bar.get_facecolor()
                    maxv = np.max(Graph)

                    # Pentagon vertices (flat bottom, point at top)
                    pentagon = [
                        (x + w * 0.15, h),  # bottom-left
                        (x + w * 0.85, h),  # bottom-right
                        (x + w * 1.00, h + w * maxv / 6.0),  # right upper
                        (x + w * 0.50, h + w * maxv / 4),  # top peak
                        (x + w * 0.00, h + w * maxv / 6.0)  # left upper
                    ]

                    poly = Polygon(pentagon, closed=True, color=color)
                    ax.add_patch(poly)

            ax.set_ylim(0, np.max(Graph) * 1.3)
            path = "./Results/Dataset_%s_%s_Alg_method.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            # # ------------------------------- Method ----------------------------------
            # fig = plt.figure(figsize=(10, 6))
            # ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            # fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of No. of Epochs')
            # term_max = np.max(Graph)
            # cap_height = term_max * 0.05
            # X = np.arange(len(kfold))
            # width = 0.18
            # colors = ['#c2457a', '#d66b97', '#e8a8c2', '#a33b67', '#7c2d50']
            # bars = []
            # bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 5] - cap_height, 0), width=width,
            #                    color=colors[0], label=Classifier[0]))
            # bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 6] - cap_height, 0), width=width,
            #                    color=colors[1], label=Classifier[1]))
            # bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 7] - cap_height, 0), width=width,
            #                    color=colors[2], label=Classifier[2]))
            # bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 8] - cap_height, 0), width=width,
            #                    color=colors[3], label=Classifier[3]))
            # bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width,
            #                    color=colors[4], label=Classifier[4]))
            #
            # plt.xticks(X + 0.40, ['20', '40', '60', '80'], fontname="Arial",
            #            fontsize=14, fontweight='bold', color='#14213d')
            # plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            # plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            #
            # dot_markers = [plt.Line2D([2], [2], marker='s', color='w',
            #                           markerfacecolor=color, markersize=10) for color in colors]
            #
            # plt.legend(dot_markers, Classifier, loc='upper center',
            #            bbox_to_anchor=(0.5, 1.15), fontsize=9,
            #            frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            # for group in bars:
            #     for bar in group:
            #         x = bar.get_x()
            #         w = bar.get_width()
            #         h = bar.get_height()
            #
            #         color = bar.get_facecolor()
            #         maxv = np.max(Graph)
            #
            #         # Pentagon vertices (flat bottom, point at top)
            #         pentagon = [
            #             (x + w * 0.15, h),  # bottom-left
            #             (x + w * 0.85, h),  # bottom-right
            #             (x + w * 1.00, h + w * maxv / 6.0),  # right upper
            #             (x + w * 0.50, h + w * maxv / 4),  # top peak
            #             (x + w * 0.00, h + w * maxv / 6.0)  # left upper
            #         ]
            #
            #         poly = Polygon(pentagon, closed=True, color=color)
            #         ax.add_patch(poly)
            #
            # ax.set_ylim(0, np.max(Graph) * 1.3)
            # path = "./Results/Dataset_%s_%s_mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            # plt.savefig(path)
            # plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    # Algorithm = ['BatchSize', 'POA-DR-AcGAN-NAF', 'WPA-DR-AcGAN-NAF', 'MOA-DR-AcGAN-NAF', 'CCO-DR-AcGAN-NAF', 'ULD-CCO-DR-AcGAN-NAF']
    Classifier = ['BatchSize', 'GNN', 'ResNet-50', 'EfficientNetV2-ConvNeXt', 'Distributed CNN-LightGBM', 'QEGT-SGINA']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 4, 8, 9, 12, 15]).astype(int)
    Table_Terms = [0, 4, 8, 9, 12, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = ['4', '8', '16', '32', '48']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------Dataset -   ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)

            # Table = PrettyTable()
            # Table.add_column(Classifier[0], Kfold)
            # for j in range(len(Classifier) - 1):
            #     Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            # print('--------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
            #       '---------------------------------------')
            # print(Table)


def plot_Seg_Tabel():
    Eval_all = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['TERMS', 'SAA-ADCA-MSA', 'ECO-ADCA-MSA', 'COA-ADCA-MSA', 'MAOA-ADCA-MSA', 'SEMAO-ADCA-MSA']
    Methods = ['TERMS', 'Unet', 'Unet3+', 'ResUnet', 'DCA-MSA', 'SEMAO-ADCA-MSA']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 6):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Statistics[0::3])
            Table.add_column(Algorithm[1], stats[i, 0, 0::3])
            Table.add_column(Algorithm[2], stats[i, 1, 0::3])
            Table.add_column(Algorithm[3], stats[i, 2, 0::3])
            Table.add_column(Algorithm[4], stats[i, 3, 0::3])
            Table.add_column(Algorithm[5], stats[i, 4, 0::3])
            print('--------------------------------------------------Algorithm -  ', Terms[i - 4],
                  'Comparison for Segmentation of dataset', n + 1, '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Methods[0], Statistics[0::3])
            Table.add_column(Methods[1], stats[i, 5, 0::3])
            Table.add_column(Methods[2], stats[i, 6, 0::3])
            Table.add_column(Methods[3], stats[i, 7, 0::3])
            Table.add_column(Methods[4], stats[i, 8, 0::3])
            Table.add_column(Methods[5], stats[i, 4, 0::3])
            print('-------------------------------------------------- method - ', Terms[i - 4],
                  'Comparison for Segmentation of dataset', n + 1, '--------------------------------------------------')
            print(Table)


def plot_seg_results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    Algorithm = ['SAA-ADCA-MSA', 'ECO-ADCA-MSA', 'COA-ADCA-MSA', 'MAOA-ADCA-MSA', 'SEMAO-ADCA-MSA']
    Classifier = ['Unet', 'Unet3+', 'ResUnet', 'DCA-MSA', 'SEMAO-ADCA-MSA']
    LossFun = ['Binary Cross-Entropy', 'Categorical Cross-Entropy', 'Sparse Categorical Cross-Entropy', 'Hinge Loss',
               'Focal Loss']
    Epoch = np.asarray(LossFun)
    Graph_Terms = [0, 1, 2]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :5, :, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Loss Function')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Epoch))
            width = 0.18
            bars = []
            colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', 'k']
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 0] - cap_height, 0), width=width, color=colors[0],
                               label=Algorithm[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 1] - cap_height, 0), width=width, color=colors[1],
                               label=Algorithm[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 2] - cap_height, 0), width=width, color=colors[2],
                               label=Algorithm[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 3] - cap_height, 0), width=width, color=colors[3],
                               label=Algorithm[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width, color=colors[4],
                               label=Algorithm[4]))
            plt.xticks(X + 0.30, ['Binary \n Cross-Entropy', 'Categorical \n Cross-Entropy',
                                  'Sparse \n Categorical \n Cross-Entropy', 'Hinge Loss', 'Focal Loss'],
                       fontname="Arial",
                       fontsize=14,
                       fontweight='bold', color='#14213d')
            plt.xlabel('Loss Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='*', color='w', markerfacecolor=color, markersize=18) for
                           color
                           in colors]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 12})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph[0, 4])
                    triangle = [
                        (x, h),  # left top of bar
                        (x + w, h),  # right top
                        (x + w / 2, h + w * cap_height + 0.02)  # taller triangle point
                    ]
                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)
            ax.set_ylim(0, max * 1.3)
            path = "./Results/Dataset_%s_%s_Seg_Alg_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Loss Function')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Epoch))
            width = 0.18
            bars = []
            colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', 'k']
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 5] - cap_height, 0), width=width, color=colors[0],
                               label=Classifier[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 6] - cap_height, 0), width=width, color=colors[1],
                               label=Classifier[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 7] - cap_height, 0), width=width, color=colors[2],
                               label=Classifier[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 8] - cap_height, 0), width=width, color=colors[3],
                               label=Classifier[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width, color=colors[4],
                               label=Classifier[4]))
            plt.xticks(X + 0.30, ['Binary \n Cross-Entropy', 'Categorical \n Cross-Entropy',
                                  'Sparse \n Categorical \n Cross-Entropy', 'Hinge Loss', 'Focal Loss'],
                       fontname="Arial",
                       fontsize=14,
                       fontweight='bold', color='#14213d')
            plt.xlabel('Loss Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='*', color='w', markerfacecolor=color, markersize=18) for
                           color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 12})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph[0, 4])
                    triangle = [
                        (x, h),  # left top of bar
                        (x + w, h),  # right top
                        (x + w / 2, h + w * cap_height + 0.02)  # taller triangle point
                    ]
                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)
            ax.set_ylim(0, max * 1.3)
            path = "./Results/Dataset_%s_%s_Seg_Mod_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def ProposedPlotResults():
    eval = np.load('Evaluate_Opti.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 4, 5, 6, 7]
    Algorithm = ['DCA-MSA', 'SEMAO-ADCA\n-MSA']
    Classifier = ['cGAN', 'ULD-CCO-DR- \n AcGAN-NAF']
    Kfold = ['Adam', 'SGD', 'RMSprop', 'Adagrad']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.10, 0.22, 0.7, 0.7])
            ax.set_facecolor('#C0C0C0')
            fig.canvas.manager.set_window_title(Terms[j] + ' Algorithm Comparison of Optimizer')
            X = np.arange(len(Kfold))
            plt.bar(X + 0.00, Graph[:, 3], color='crimson', width=0.20, label=Algorithm[0])
            plt.bar(X + 0.20, Graph[:, 4], color='k', width=0.20, label=Algorithm[1])
            plt.xticks(X + 0.10, ['Adam', 'SGD', 'RMSprop', 'Adagrad'], fontsize=14, fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('Optimizer', fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            colors = ['crimson', 'k']
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Algorithm, loc='best', bbox_to_anchor=(0.98, 0.7), fontsize=12,
                       frameon=False, ncol=1, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            plt.grid(which='major', color='w', linestyle='-', linewidth=2)
            plt.minorticks_on()
            plt.grid(which='major', color='w', linestyle='-', linewidth=2)
            path = "./Results/Dataset_%s_%s_Prop_method_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            # # ---------------- Method -----------------
            # fig = plt.figure(figsize=(10, 6))
            # ax = fig.add_axes([0.10, 0.22, 0.7, 0.7])
            # ax.set_facecolor('#C0C0C0')
            # fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Optimizer')
            # X = np.arange(len(Kfold))
            # plt.bar(X + 0.00, Graph[:, 8], color='m', width=0.20, label=Classifier[0])
            # plt.bar(X + 0.20, Graph[:, 4], color='k', width=0.20, label=Classifier[1])
            # plt.xticks(X + 0.10, ['Adam', 'SGD', 'RMSprop', 'Adagrad'], fontsize=14, fontname="Arial",
            #            fontweight='bold', color='k')
            # plt.xlabel('Optimizer', fontsize=14, fontname="Arial", fontweight='bold', color='k')
            # plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['left'].set_visible(False)
            # plt.gca().spines['bottom'].set_visible(True)
            # colors = ['m', 'k']
            # dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
            #                in colors]
            # plt.legend(dot_markers, Classifier, loc='best', bbox_to_anchor=(0.98, 0.7), fontsize=12,
            #            frameon=False, ncol=1, prop={'weight': 'bold', 'size': 12})
            # plt.tight_layout()
            # plt.grid(which='major', color='w', linestyle='-', linewidth=2)
            # plt.minorticks_on()
            # plt.grid(which='major', color='w', linestyle='-', linewidth=2)
            # path = "./Results/Dataset_%s_%s_Prop_Mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            # plt.savefig(path)
            # plt.show()


if __name__ == '__main__':
    # plotConvResults()
    # Plot_Results()
    # Plot_ROC_Curve()
    # Table()
    plot_seg_results()
    plot_Seg_Tabel()
    ProposedPlotResults()
    # Image_Results()
    # Sample_Images()
