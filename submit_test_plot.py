import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def output_data(slide_id):
    labels = ['background', 'Acinar', "Lepidic", "Papillary", "Solid", "Complex gland", "Micropapillary"]
    colors1 = ['#000000', '#00F5FF', '#EE82EE', '#00FF7F', '#FF69B4', '#FF3030', '#87CEFF']
    save_path1 = "/media/lingang/e1/Kim/fenji/codes1/Figures_pred/model_binary/resnet/demo_results2/Figures/"
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    #filenames=os.listdir(save_path)
    filenames=[slide_id]
    for filename in filenames:
        data = np.load('./submit_data/' + filename + '.npy')
        print('start plot')
        values = np.array([0, 1, 2, 3, 4, 5, 6])
        a = [colors1[int(values[j])] for j in range(len(values))]
        cmap = mcolors.ListedColormap(a)
        labels1 = [labels[int(values[j])] for j in range(len(values))]
        plt.figure()
        im=plt.imshow(data,cmap=cmap)

        colors = [im.cmap(im.norm(int(value))) for value in values]
        if len(values) > 1:
            patches = [mpatches.Patch(color=colors[j + 1], label=labels1[int(values[j + 1])]) for j in range(len(values) - 1)]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
        plt.xticks([])
        plt.yticks([])
        #plt.savefig(save_path1 + filename + "_pred.png")
        plt.show()
        plt.close()

output_data("150352A1A2A3-MPA80-PPA15-APA5_001")