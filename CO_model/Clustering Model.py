from sklearn.manifold import TSNE
import numpy as np
import csv
import pandas as pd
from sklearn.mixture import GaussianMixture
import time
import functools
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

data=pd.read_csv('CO_data\\spectra data.csv')
X_tsne = TSNE(n_components=2, random_state=15).fit_transform(data)

def time_cost(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        t1 = time.time()
        print(args[0], '：%.2fs' % (t1 - t0))
        return func(*args, **kwargs), t1 - t0

    return wrapper


@time_cost
def cluster_function(model_name, model, data):
    y_pred = model.fit_predict(data)
    return y_pred

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)


if __name__ == '__main__':

    model_list = {
        "GaussianMixture": GaussianMixture(n_components=4, covariance_type="spherical",
                                           tol=1e-4, max_iter=100, verbose=1)
    }
    n_clusters = 4
    fig = plt.figure(figsize=(8, 8))
    idx_gauss = [[] for i in range(n_clusters)]

    i = 1
    for model in model_list:
        fig.add_subplot(1, 1, i)
        result = cluster_function(model, model_list[model], data)  # 先聚类后降维
        color_map = {'0': '#77CAED', '1': '#DBB2E7', '2': '#9EE1B0', '3': '#F5CE89'}
        colors = []
        for j in range(len(result[0])):
            for k, v in color_map.items():
                class_k = int(k)
                if result[0][j] == class_k:
                    colors.append(v)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', c=colors, s=10)
        plt.title("{}".format(model))
        plt.text(.99, .01, ('%.2fs' % (result[1])).lstrip('0'), transform=plt.gca().transAxes,
                 horizontalalignment='right')
        if model == 'GaussianMixture':
            difference = result[0].max() - result[0].min()
            equal_df = difference / (n_clusters - 1)
            for num, value in enumerate(result[0]):
                id = num + 1
                if value == result[0].min() + value * equal_df:
                    idx_gauss[value].append(id)

    plt.savefig(f'CO_data/tSNE-IR.png')
    # plt.show()

    with open('CO_data\\idx_gauss_IR.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in idx_gauss:
            writer.writerow(row)

    free_energy = pd.read_csv(r'CO_data/free energy.csv')
    free_energy = free_energy.values
    free_energy_data = [[] for i in range(len(free_energy))]
    for i in range(len(free_energy)):
        for j in range(len(free_energy[i])):
            free_energy_data[i].append(free_energy[i][j])

    with open('CO_data\\idx_gauss_IR.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            rows[i][j] = int(rows[i][j])

    free_energy_clusters=[]
    k = 0
    for i in range(len(free_energy_data)):
        free_energy_data[i][0]=k+1
        for j in range(len(rows)):
            if free_energy_data[i][0] in rows[j]:
                free_energy_data[i].append(j + 1)
                copy_right = copy.deepcopy(free_energy_data[i])
                free_energy_clusters.append(copy_right)
                k+=1

    free_energy_clusters = np.array(free_energy_clusters)

    n_fe = [[] for i in range(n_clusters)]
    for a in range(n_clusters):
        cluster = a + 1
        for b in range(len(free_energy_clusters)):
            if int(free_energy_clusters[b][-1]) == cluster:
                n_fe[a].append(free_energy_clusters[b][-2])

    fig, axs = plt.subplots(1, 2, figsize=(60, 30))

    x = free_energy_clusters[:, 2].astype(int).tolist()
    y = free_energy_clusters[:, 1].tolist()
    # sns.violinplot(x, y, ax=axs[0])
    df_1 = pd.DataFrame({'x': x, 'y': y})
    # Grouping data by x values
    fig_1 = [df_1[df_1['x'] == i]['y'] for i in range(1, 5)]
    bp_1 = axs[0].boxplot(fig_1, boxprops=dict(linewidth=8), whiskerprops=dict(linewidth=10),
                    flierprops=dict(marker='o', markersize=12), medianprops=dict(linewidth=8,color='black'), patch_artist=True)

    colors = ['#77CAED', '#DBB2E7', '#9EE1B0', '#F5CE89']
    for patch, color in zip(bp_1['boxes'], colors):
        patch.set_facecolor(color)

    # axs[0,0].scatter(x,y)
    axs[0].set_title('free_energy', fontsize=80)

    ######
    ddec6 = pd.read_csv(r'CO_data\DDEC06_charge.csv',encoding='gbk')
    ddec6 = ddec6.values
    ddec6_data = [[] for i in range(len(ddec6))]
    for i in range(len(ddec6)):
        for j in range(len(ddec6[i])):
            ddec6_data[i].append(ddec6[i][j])

    with open('CO_data\\idx_gauss_IR.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            rows[i][j] = int(rows[i][j])

    ddec6_clusters = []
    k=0
    for i in range(len(ddec6_data)):
        ddec6_data[i][0]=k+1
        for j in range(len(rows)):
            if int(ddec6_data[i][0]) in rows[j]:
                ddec6_data[i].append(j + 1)
                copy_right = copy.deepcopy(ddec6_data[i])
                ddec6_clusters.append(copy_right)
                k+=1

    ddec6_clusters = np.array(ddec6_clusters)

    n_ddec = [[] for i in range(n_clusters)]
    for a in range(n_clusters):
        cluster = a + 1
        for b in range(len(ddec6_clusters)):
            if int(ddec6_clusters[b][-1]) == cluster:
                n_ddec[a].append(ddec6_clusters[b][-2])

    x = ddec6_clusters[:, 2].astype(int).tolist()
    y = ddec6_clusters[:, 1].tolist()
    # sns.violinplot(x, y, ax=axs[1])
    df_2 = pd.DataFrame({'x': x, 'y': y})
    # Grouping data by x values
    fig_2 = [df_2[df_2['x'] == i]['y'] for i in range(1, 5)]
    bp_2 = axs[1].boxplot(fig_2, boxprops=dict(linewidth=8), whiskerprops=dict(linewidth=10),
                          flierprops=dict(marker='o', markersize=12), medianprops=dict(linewidth=8,color='black'),
                          patch_artist=True)

    colors = ['#77CAED', '#DBB2E7', '#9EE1B0', '#F5CE89']
    for patch, color in zip(bp_2['boxes'], colors):
        patch.set_facecolor(color)

    axs[1].set_title('ddec6', fontsize=80)

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
        ax.tick_params(axis='both', which='major', labelsize=75)

    fig.suptitle('CO_clustering results', fontsize=90)
    plt.savefig('CO_data\\CO_clustering_results.png',dpi=300)

    plt.show()
    plt.close()

    X_tsne_new = [[] for i in range(len(X_tsne))]
    for i in range(len(X_tsne)):
        add = result[0][i] + 1
        X_tsne_new[i].append(X_tsne[i][0])
        X_tsne_new[i].append(X_tsne[i][1])
        X_tsne_new[i].append(add)

    X_tsne_new = pd.DataFrame(X_tsne_new, columns=['x', 'y', 'clusters'])
    n_fe = pd.DataFrame(n_fe)
    n_ddec = pd.DataFrame(n_ddec)
    free_energy_clusters = pd.DataFrame(free_energy_clusters, columns=['id', 'value', 'id_clusters'])
    ddec6_clusters = pd.DataFrame(ddec6_clusters, columns=['id', 'value', 'id_clusters'])

    with pd.ExcelWriter(f"CO_data\\CO_results.xlsx") as writer:
        X_tsne_new.to_excel(writer, sheet_name='X_tsne', index=False)
        n_fe.to_excel(writer, sheet_name='n_fe', index=False)
        n_ddec.to_excel(writer, sheet_name='n_ddec', index=False)
        free_energy_clusters.to_excel(writer, sheet_name='free energy', index=False)
        ddec6_clusters.to_excel(writer, sheet_name='ddec6', index=False)

    spec_data = [[] for i in range(4)]
    data=data.values.tolist()
    for i in range(len(data)):
        data_id = i + 1
        if data_id in rows[0]:
            spec_data[0].append(data[i])
        if data_id in rows[1]:
            spec_data[1].append(data[i])
        if data_id in rows[2]:
            spec_data[2].append(data[i])
        if data_id in rows[3]:
            spec_data[3].append(data[i])

    for i in range(4):
        spec_ = pd.DataFrame(spec_data[i])
        with pd.ExcelWriter(f'CO_data\\spec_{i + 1}.xlsx') as writer:
            spec_.to_excel(writer, index=False)

