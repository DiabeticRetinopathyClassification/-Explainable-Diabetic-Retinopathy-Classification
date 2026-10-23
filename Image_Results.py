import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

No_of_Dataset = 3


def Image_Results():
    I = [[1060, 1340, 1371, 2175, 2081], [62, 117, 339, 343, 376], [66, 225, 307, 310, 766]]
    for n in range(2, No_of_Dataset):
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        GT = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        UNET = np.load('Yolov3_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        RESUNET = np.load('Deeplabv3_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        Mobile = np.load('Yolov4_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        PROPOSED = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)[:800]
        for i in range(len(I[n])):
            # plt.subplot(2, 3, 1)
            # plt.title('Original')
            # plt.imshow(Images[I[n][i]])
            # plt.subplot(2, 3, 2)
            # plt.title('GroundTruth')
            # plt.imshow(GT[I[n][i]])
            # plt.subplot(2, 3, 3)
            # plt.title('Ref 1')
            # plt.imshow(UNET[I[n][i]])
            # plt.subplot(2, 3, 4)
            # plt.title('Ref 2')
            # plt.imshow(RESUNET[I[n][i]])
            # plt.subplot(2, 3, 5)
            # plt.title('Ref 3')
            # plt.imshow(Mobile[I[n][i]])
            # plt.subplot(2, 3, 6)
            # plt.title('PROPOSED')
            # plt.imshow(PROPOSED[I[n][i]])
            # plt.tight_layout()
            # plt.show()
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " Original -"+str(i+1)+".png", Images[I[n][i]])
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " Groundtruth -"+str(i+1)+".png", GT[I[n][i]])
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " unet -"+str(i+1)+".png", UNET[I[n][i]])
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " unet3plus -"+str(i+1)+".png", RESUNET[I[n][i]])
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " Resunet -"+str(i+1)+".png", Mobile[I[n][i]])
            # cv2.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " DDCA-Med SA -"+str(i+1)+".png", PROPOSED[I[n][i]])
            cv.imwrite("./Results/Image Results/Dataset -"+str(n+1)+ " Proposed -"+str(i+1)+".png", PROPOSED[I[n][i]])


def Sample_Images():
    for n in range(No_of_Dataset):
        Orig = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        ind = [10, 50, 100, 115, 145, 155]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images ")
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    # Sample_Images()
