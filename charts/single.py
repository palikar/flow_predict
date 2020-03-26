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

    data['with'] = [42.4267183875451 , 43.78148944993617]
    data['without'] = [37.73239422055144 , 42.99703106930973 , 43.02076006161946]

    box_data.append(data['with'])
    box_data.append(data['without'])

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

    plt.ylabel('PSNR', fontsize=16)
    plt.title('Constant model (single image performance)', fontsize=17)

    # plt.show()
    plt.savefig('single_const_psnr.png')


def speed_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [32.70363132431171 , 32.37544378610233 , 33.22313390003679, 33.80816838419148 , 33.1802077172184 , 33.873630834872216 , 33.95285390084521 ,33.67447951182013 , 33.179886956867975 , 34.17147778053133]
    data['without'] = [32.038006486935075 , 32.192804717704554 , 32.184300070511554, 32.48270450488618 , 32.311327037629844 , 32.19408204987984 , 32.35434794736937 , 32.15279210676768 , 32.04192581940871 , 32.10279251816919]

    box_data.append(data['with'])
    box_data.append(data['without'])

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

    plt.ylabel('PSNR', fontsize=17)
    plt.title('Inflow speed model (single image performance)', fontsize=17)

    # plt.show()
    plt.savefig('single_speed_psnr.png')
    

def fluid_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [40.21078091833494 , 41.00434320329602 , 40.051247189433234 , 40.80200386644962 , 42.235433361144615 , 39.74997269237842 , 41.39466838557988 , 42.59324688714353 , 41.08656559139025 , 39.13305476736238 , 44.14030779137574 , 37.84617269863215]

    data['without'] = [44.75477512310624 , 45.09586318757287 , 47.97932840785287 , 43.976377566130566 , 43.42009224407885]

    box_data.append(data['with'])
    box_data.append(data['without'])

    labels.append('with pressure')
    labels.append('without pressure')

    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=17)

    plt.figure(figsize=(17, 10))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels, fontsize=17)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times, alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR', fontsize=17)
    plt.title('Viscosity-density model (single image performance)', fontsize=17)

    # plt.show()

    plt.savefig('single_fluid_psnr.png')



def main():

    constant_models()

    speed_models()
    
    fluid_models()
    
    print('Done!')


if __name__ == '__main__':
    main()
