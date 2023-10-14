from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def loss_chart(losses):

    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(losses)

    if len(losses.T) == 4:
        plt.plot(losses.T[0], label='Discriminator Total Loss')
        plt.plot(losses.T[1], label='Discriminator Real Loss')
        plt.plot(losses.T[2], label='Discriminator Fake Loss')
        plt.plot(losses.T[3], label='Generator Loss')

    elif len(losses.T) == 2:
        plt.plot(losses.T[0], label='Discriminator Loss')
        plt.plot(losses.T[1], label='Generator Loss')

    plt.title("Training Losses")
    plt.legend()

def view_samples(samples, epoch):

    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """

    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):  # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((9, 9)), cmap='Greys_r')


def view_process(samples, epoch_num=300):

    # 指定要查看的轮次
    if epoch_num == 300:
        epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250]  # 一共300轮，不要越界
    elif epoch_num == 120:
        epoch_idx = [0, 5, 10, 20, 30, 45, 60, 80, 95 ,100] # 一共120轮，不要越界

    show_imgs = []

    for i in epoch_idx:
        show_imgs.append(samples[i][1])

    # 指定图片形状
    rows, cols = 10, 25
    fig, axes = plt.subplots(figsize=(30, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    idx = range(0, epoch_num, int(epoch_num / rows))

    for sample, ax_row in zip(show_imgs, axes):
        for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
            ax.imshow(img.reshape((9, 9)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

def adv_sample_legalization(gan_ae):

    # 加载标准化模型
    My_StandardScaler = joblib.load('CSE-CIC-IDS2018_StandardScaler_01')

    #数据预处理
    scaler = My_StandardScaler

    def MaxMinNormalization(x):     # 数组数据归一化处理 —> 灰度图像范围（0,255）
        Max = max(x)
        Min = min(x)
        for i in range(0, len(x)):
            if Max != Min:
                x[i] = (x[i] - Min) / (Max - Min)
        return x

    gan_ae = MaxMinNormalization(gan_ae)
    gan_ae = scaler.inverse_transform([gan_ae[:78]])
    gan_ae = (np.around(gan_ae[:78], decimals=2)).tolist()
    print(gan_ae)
