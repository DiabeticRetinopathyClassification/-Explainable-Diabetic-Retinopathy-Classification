import os
import pandas as pd
from numpy import matlib
from Global_Vars import Global_Vars
# from Image_Results import *
from Plot_Results import *
import cv2 as cv


No_of_Dataset = 3


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    image = cv.resize(image, (512, 512))
    return image


# Read Dataset 1
an = 1
if an == 1:
    Image = []
    Target = []
    path = './Dataset/Dataset_1/colored_images'
    in_dir = os.listdir(path)
    for i in range(len(in_dir)):
        fold = path + '/' + in_dir[i]
        out_dir = os.listdir(fold)
        for j in range(len(out_dir)):
            files = fold + '/' + out_dir[j]
            img = ReadImage(files)
            Image.append(img)
            Target.append(i)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_1.npy', index)
    np.save('Image_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Read Dataset 2
an = 1
if an == 1:
    Image = []
    Target = []
    path = './Dataset/Dataset_2'
    in_dir = os.listdir(path)
    for i in range(len(in_dir)):
        fold = path + '/' + in_dir[i]
        out_dir = os.listdir(fold)
        for j in range(len(out_dir)):
            files = fold + '/' + out_dir[j]
            img = ReadImage(files)
            Image.append(img)
            Target.append(i)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_2.npy', index)
    np.save('Image_2.npy', Shuffled_Datas)
    np.save('Target_2.npy', Shuffled_Target)

# Read Dataset 3
an = 1
if an == 1:
    Image = []
    Target = []
    path = './Dataset/Dataset_3'
    in_dir = os.listdir(path)
    for i in range(len(in_dir)):
        fold = path + '/' + in_dir[i]
        out_dir = os.listdir(fold)
        for j in range(len(out_dir)):
            files = fold + '/' + out_dir[j]
            img = ReadImage(files)
            Image.append(img)
            Target.append(i)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_3.npy', index)
    np.save('Image_3.npy', Shuffled_Datas)
    np.save('Target_3.npy', Shuffled_Target)

# Generate Ground Truth
an = 1
if an == 1:
    for n in range(No_of_Dataset):
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = []
        for i in range(len(Images)):
            print(i)
            image = Images[i]
            img = np.zeros(image.shape, dtype=np.uint8)
            max_val = np.max(image)
            thresh = max_val - (max_val * 0.2)
            index = np.where(image >= thresh)
            img[index[0], index[1]] = 255
            GT.append(img)
        np.save('Ground_Truth_' + str(n + 1) + '.npy', GT)

# Segmentation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        Eval = np.zeros((5, 16))
        Eval[0, :], Seg_Img_1 = Model_CNNSeg(Image, GT)
        Eval[1, :], Seg_Img_2 = Model_Unet(Image, GT)
        Eval[2, :], Seg_Img_3 = Model_ResUnet(Image, GT)
        Eval[3, :], Seg_Img_4 = Model_DenseUnet(Image, GT)
        Eval[4, :], Seg_Img_5 = Model_MobileUnetPlusPlus(Image, GT)
        np.save('Proposed_' + str(n + 1) + '.npy', Seg_Img_5)
        np.save('Evaluate_Seg_all.npy', Eval)

# Optimization for Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron, Learning Rate, Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("POA...")
        [bestfit1, fitness1, bestsol1, time] = POA(initsol, fname, xmin, xmax, Max_iter)

        print("WPA...")
        [bestfit2, fitness2, bestsol2, time1] = WPA(initsol, fname, xmin, xmax, Max_iter)

        print("MOA...")
        [bestfit3, fitness3, bestsol3, time2] = MOA(initsol, fname, xmin, xmax, Max_iter)

        print("CCO...")
        [bestfit4, fitness4, bestsol4, time3] = CCO(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feature = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Feat = Feature
        EVAL = []
        Epoch = [20, 40, 60, 80]
        for act in range(len(Epoch)):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_GAN(Train_Data, Train_Target, Test_Data, Test_Target, Epoch[act],
                                             sol)  # With optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epoch[act])
            Eval[6, :], pred2 = Model_MobileNet(Train_Data, Train_Target, Test_Data,
                                                Test_Target, Epoch[act])
            Eval[7, :], pred3 = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target,
                                                Epoch[act])
            Eval[8, :], pred4 = Model_GAN(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epoch[act])  # Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        np.save('Evaluate_all.npy', EVAL)  # Save Eval all

# plotConvResults()
# Plot_Results()
# Plot_ROC_Curve()
# Table()
# plot_seg_results()
# ProposedPlotResults()
# Image_Results()
# Sample_Images()
