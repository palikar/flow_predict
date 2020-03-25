import matplotlib
import matplotlib.pyplot as plt
import os
import sys

plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})


def constant_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [1,1.3,1.5,0.9,1.23,1.32]
    data['without'] = [1,1.3,1.5,0.9,1.23,1.32]

    box_data.append([1,1.3,1.5,0.9,1.23,1.32])
    box_data.append([1,1.3,1.5,0.9,1.23,1.32])

    labels.append('with pressure')
    labels.append('without pressure')

    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=17)

    plt.figure(figsize=(17, 10))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times, alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR')
    plt.title('Constant model (single image performance)', fontsize=17)

    plt.show()


def speed_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [1,1.3,1.5,0.9,1.23,1.32]
    data['without'] = [1,1.3,1.5,0.9,1.23,1.32]

    box_data.append([1,1.3,1.5,0.9,1.23,1.32])
    box_data.append([1,1.3,1.5,0.9,1.23,1.32])

    labels.append('with pressure')
    labels.append('without pressure')

    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=17)

    plt.figure(figsize=(17, 10))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times, alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR')
    plt.title('Constant model (single image performance)', fontsize=17)

    plt.show()


def fluid_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [1,1.3,1.5,0.9,1.23,1.32]
    data['without'] = [1,1.3,1.5,0.9,1.23,1.32]

    box_data.append([1,1.3,1.5,0.9,1.23,1.32])
    box_data.append([1,1.3,1.5,0.9,1.23,1.32])

    labels.append('with pressure')
    labels.append('without pressure')

    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=17)

    plt.figure(figsize=(17, 10))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times, alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR')
    plt.title('Constant model (single image performance)', fontsize=17)

    plt.show()


def main():

    constant_models()

    # speed_models()
    
    # fluid_models()

if __name__ == '__main__':
    main()
