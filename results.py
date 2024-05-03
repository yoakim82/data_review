import json
import matplotlib.pyplot as plt
import numpy as np


def flatten(plots):
    ret = []
    for p in plots:
        for l in p:
            ret.append(l)
    return ret


def box(ax, plots, labels, title, color='blue', xlabel="Test split", ylabel="Detection Accuracy", ylim=[0, 1], set_indx=0):


    bplot1 = ax.boxplot(plots, patch_artist=True,
                        positions=np.array(np.arange(len(plots)))*2.0-0.4+set_indx*0.3, widths=0.3)
    for patch in bplot1['boxes']:
        patch.set_facecolor(color)
    plt.title(title)
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    ax.grid(True)

    return bplot1


def create_box_plot_from_file(file_paths, colors, exp_names):

    multi_plots_mAP50 =[]
    multi_labels_mAP50 = []
    multi_plots_mAP =[]
    multi_labels_mAP = []
    source = []

    plt.figure(figsize=(24, 6))

    for i, file_path in enumerate(file_paths):

        #print(f"File {file_path} processed")
        plots = []
        labels = []
        plots_mAP = []
        labels_mAP = []
        splits = ["test", "4k_drone-vs-birds", "field"]
        disp_names = ['Synthetic', 'Drone-vs-bird subset', 'Custom field test']


        mAP50_values_multirotor = dict()
        mAP_values_multirotor = dict()
        mAP50_values_bird = dict()
        mAP_values_bird = dict()
        mAP50_values = dict()
        mAP_values = dict()

        for s in splits:
            mAP50_values_multirotor[s] = []
            mAP_values_multirotor[s] = []
            mAP50_values_bird[s] = []
            mAP_values_bird[s] = []
            mAP50_values[s] = []
            mAP_values[s] = []


        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                train = data['train_set']
                split = data['test_split']
                if split in splits:
                    results = data['results']
                    for i, result in enumerate(results):
                        if i == 0:
                            mAP50_values_multirotor[split].append(result['mAP50'])
                            mAP_values_multirotor[split].append(result['mAP'])
                        elif i == 1:
                            mAP50_values_bird[split].append(result['mAP50'])
                            mAP_values_bird[split].append(result['mAP'])
                        else:
                            mAP50_values[split].append(result['mAP50'])
                            mAP_values[split].append(result['mAP'])


        plot_splits = splits#["test"]

        # Creating a box plot
        # collect results for all splits, for one class
        for split in plot_splits:

            plots.append(mAP50_values_multirotor[split])
            labels.append(f'{split}')
            plots_mAP.append(mAP_values_multirotor[split])
            labels_mAP.append(f'{split}')

            #plt.boxplot([mAP50_values[split], mAP50_values_multirotor[split], mAP50_values_bird[split]], labels=[f'{split}/All', f'{split}/Multirotor', f'{split}/Bird'])
        multi_plots_mAP50.append(plots)
        multi_plots_mAP.append(plots_mAP)
        multi_labels_mAP.append(labels_mAP)
        multi_labels_mAP50.append(labels)
        source.append(train)

    dpi = 96
    fig, axis = plt.subplots(2, figsize=(1200/dpi, 800/dpi), dpi=dpi)
    # join plots from multiple files


    mAP50_elements = []
    #mAP50_labels = []
    set_indx = 0
    plt.sca(axis[0])
    for (p, l, s, clr) in zip(multi_plots_mAP50, multi_labels_mAP50, source, colors):
        mAP50_elements.append(box(axis[0], p, l, title=f'Spread of mAP50 Parameter for Multirotor', color=clr, set_indx=set_indx))
        set_indx += 1
        #mAP50_labels.append(s)

    axis[0].legend([element["boxes"][0] for element in mAP50_elements], exp_names)

    display_labels = []
    plt.xticks(np.arange(0, len(disp_names) * 2, 2), disp_names)

    mAP_elements = []
    set_indx = 0
    plt.sca(axis[1])
    for (p, l, s, clr) in zip(multi_plots_mAP, multi_labels_mAP, source, colors):
        mAP_elements.append(box(axis[1], p, l, title=f'Spread of mAP Parameter for Multirotor', color=clr, set_indx=set_indx))
        set_indx += 1


    plt.xticks(np.arange(0, len(disp_names) * 2, 2), disp_names)
    #fig2, ax2 = plt.subplots()
    #ax2.legend([element["boxes"][0] for element in mAP_elements], source, loc='upper right')

    axis[1].legend([element["boxes"][0] for element in mAP_elements], exp_names)
    multi_plots = []
    multi_labels = []
    source = []

    # plots = []
    # labels = []
    # for split in plot_splits:
    #
    #     plots.append(mAP50_values_bird[split])
    #     labels.append(f'{split}')
    #
    #     #plt.boxplot([mAP50_values[split], mAP50_values_multirotor[split], mAP50_values_bird[split]], labels=[f'{split}/All', f'{split}/Multirotor', f'{split}/Bird'])
    #
    # bplot1 = plt.boxplot(plots, labels=labels, patch_artist=True)
    # colors = ['pink', 'lightblue']
    # for patch in bplot1['boxes']:
    #     patch.set_facecolor(colors[0])
    #
    # plt.title(f'Spread of mAP50 Parameter for Bird')
    # plt.xlabel('Parameters')
    # plt.ylabel('Detection Accuracy')
    # plt.ylim([0,1])
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 3)
    # plots = []
    # labels = []
    # for split in plot_splits:
    #
    #     plots.append(mAP50_values[split])
    #     labels.append(f'{split}')
    #
    #     #plt.boxplot([mAP50_values[split], mAP50_values_multirotor[split], mAP50_values_bird[split]], labels=[f'{split}/All', f'{split}/Multirotor', f'{split}/Bird'])
    # plt.boxplot(plots, labels=labels)
    # plt.title('Spread of mAP50 Parameter')
    # plt.xlabel('Parameters')
    # plt.ylabel('Detection Accuracy')
    # plt.ylim([0,1])
    # plt.grid(True)
    #
    #
    # #plt.savefig(f"{file_path}_mAP50.png")
    #
    # plt.subplot(2, 3, 4)
    # for split in plot_splits:
    #     plt.boxplot([mAP_values[split], mAP_values_multirotor[split], mAP_values_bird[split]], labels=[f'All', f'Multirotor', f'Bird'])
    # plt.title('Spread of mAP Parameter')
    # plt.xlabel('Parameters')
    # plt.ylim([0, 1])
    # plt.ylabel('Detection Accuracy')
    # plt.grid(True)

    plt.tight_layout()

    fig.show()
    #fname = f"file_paths[0]
    fig.savefig(f"{results_dir}/boxplot.png", dpi=dpi)

# Replace 'file_path.json' with the path to your JSON file
file_path = 'results/nerf_batches.txt'

files = ['results/new_batches.txt', 'results/nerf_batches.txt', 'results/small_batches.txt', 'results/small_nerf_batches.txt']
files = ['results/new_batches.txt', 'results/nerf_batches.txt']
files = ['results/small_batches.txt', 'results/small_nerf_batches.txt']
#files = ['results/seg_small_batches.txt', 'results/seg_small_nerf_batches.txt']
#files = ['results/seg_small_batches.txt', 'results/vtol_nerf_batches_640.txt']

results_dir = 'clean_results'
files = [f'{results_dir}/seg_small_batches.txt', f'{results_dir}/seg_small_nerf_batches.txt', f'{results_dir}/multi_nerf_batches.txt', f'{results_dir}/vtol_nerf_batches.txt']
exp_names = ['Synthetic only', 'Synthetic + NeRF finetuning', 'Synthetic + NeRF tuning w perturbations', 'Synthetic + NeRF finetuning tailored scene']
clrs = ["lightblue", "lightgreen", "magenta", "orange"]

#for f in files:
create_box_plot_from_file(files, colors=clrs, exp_names=exp_names)
