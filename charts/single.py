import matplotlib
import matplotlib.pyplot as plt
import os
import sys

plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})

matplotlib.rc('xtick', labelsize=19)
matplotlib.rc('ytick', labelsize=24)
matplotlib.rcParams['xtick.labelsize'] = 32
matplotlib.rcParams['ytick.labelsize'] = 32


def constant_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [42.4267183875451 , 43.78148944993617, 38.40345812572034 , 31.116301500139375 , 30.37560326297156 , 29.362363717887963 , 34.59130832566979 , 38.37602182790361 , 33.13829763767398 , 41.07864349729163]
    data['without'] = [37.73239422055144 , 42.99703106930973 , 43.02076006161946, 42.380817879629305 , 33.832192800572635 , 37.29558487343366 , 35.618496045124644 , 40.663238902192695 , 37.53103078541188 , 29.66568572886583 , 38.50229110243776]

    box_data.append(data['with'])
    box_data.append(data['without'])

    labels.append('with pressure')
    labels.append('without pressure')

    plt.figure(figsize=(12, 7))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times,  s=[90]*len(times), alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR', fontsize=36)
    plt.title('Constant model', fontsize=36)

    # plt.show()
    plt.savefig('single_const_psnr.eps')


def speed_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [33.97098612536543 , 33.09168803458099 , 33.92103725434269 , 33.358489909948055 , 33.965947654564765 , 32.914941686462726 , 33.229815073301175 , 32.452859691353765 , 32.991500322533646 , 34.54424296545302]
    
    data['without'] = [31.578030727496312, 31.634777717271934, 30.622411383150723, 31.947708632439785, 31.523502668755157, 31.32136801097808 , 31.247603430043284,31.821169342440722, 31.638146874797222 , 30.740445801233186]

    box_data.append(data['with'])
    box_data.append(data['without'])

    labels.append('with pressure')
    labels.append('without pressure')

    plt.figure(figsize=(12, 7))

    x = [i for i in range(1, len(data.keys())+1)]

    
    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times,  s=[90]*len(times), alpha=0.5, linewidths=None)

    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR', fontsize=36)
    plt.title('Inflow speed model', fontsize=36)

    # plt.show()

    plt.savefig('single_speed_psnr.eps')
    

def fluid_models():

    data = {}
    labels = []
    box_data = []

    data['with'] = [41.00434320329602 ,40.80200386644962 , 42.235433361144615 , 39.74997269237842 , 41.39466838557988 , 42.59324688714353 , 41.08656559139025 , 39.13305476736238 , 44.14030779137574 , 37.84617269863215]

    data['without'] = [44.75477512310624 , 45.09586318757287 , 47.97932840785287 , 43.976377566130566 , 43.42009224407885, 49.93111731579327 , 48.062160856770774 , 47.53947080459642 , 48.03952387801668 , 47.21545498004758 ]

    box_data.append(data['with'])
    box_data.append(data['without'])

    labels.append('with pressure')
    labels.append('without pressure')

    plt.figure(figsize=(12, 7))

    x = [i for i in range(1, len(data.keys())+1)]

    plt.boxplot(box_data, labels=labels)

    for (_, times), i  in zip(data.items(), x):
        plt.scatter([i]*len(times), times,  s=[90]*len(times), alpha=0.5, linewidths=None)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylabel('PSNR', fontsize=36)
    plt.title('Viscosity-density model', fontsize=36)

    plt.savefig('single_fluid_psnr.eps')



def main():

    constant_models()

    speed_models()
    
    fluid_models()
    
    print('Done!')


if __name__ == '__main__':
    main()
