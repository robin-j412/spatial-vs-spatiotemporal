import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

tex_witdh_in_pt = 506.45905

def set_size(width, fraction=1, subplots=(1, 1), fact=1.):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in * fact)


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
}

plt.rcParams.update(tex_fonts)

def r2_raw(y_true, y_pred):
    r2_ = np.array([[r2_score(obs, pred) for obs, pred in zip(y_true[i], y_pred[i])] for i in range(5)])
    return r2_

def raw_score_on_lit_areas(y_true, y_pred):
    all_r2s = []
    all_mae = []
    all_dis = []
    all_card = []
    for obss, preds in zip(y_true, y_pred):
        indexes = np.where(obss.mean(axis=0) > 0)[0]
        r2s = []
        mae = []
        dis = []
        for obs, pred in zip(obss[:, indexes], preds[:, indexes]):
            r2s.append(r2_score(obs, pred))
            mae.append(np.abs(obs - pred).mean())
            dis.append(np.abs(obs - pred).std())

        all_r2s.append(r2s)
        all_mae.append(mae)
        all_dis.append(dis)
        all_card.append(len(indexes))

    all_r2s = np.array(all_r2s)
    all_mae = np.array(all_mae)
    all_dis = np.array(all_dis)
    all_card = np.array(all_card)

    return all_r2s, all_mae, all_dis, all_card


#----------Metrics for global results ----------------#

def mae_global(y_true, y_pred):
    raw_error = np.abs(y_true - y_pred).mean(axis=-1)

    error = raw_error.mean(axis=1).mean().astype('float32')
    std = raw_error.mean(axis=1).std().astype('float32')
    # print(error.shape)
    return error, std

def dis_global(y_true, y_pred):
    dis = np.abs(y_true - y_pred).std(axis=-1)

    error = dis.mean(axis=1).mean().astype('float32')
    std = dis.mean(axis=1).std().astype('float32')

    return error, std

def r2_global(y_true, y_pred):
    r2_ = r2_raw(y_true, y_pred)

    r2_mean = r2_.mean(axis=1).mean()  # .mean()
    r2_std = r2_.mean(axis=1).std()  # .std()

    return r2_mean, r2_std

def global_scores(observations_d, prediction_d):

    for metric_name, metric in zip(['MAE', 'DIS', 'R2'], [mae_global, dis_global, r2_global]):
        print(f'\t--------- {metric_name} ----------')
        for (key, pred), (_, true) in zip(prediction_d.items(), observations_d.items()):
            pred = np.moveaxis(pred, -1, 1) #metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            true = np.moveaxis(true, -1, 1) #metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            error, std = metric(true, pred)
            print(f'\t\t{key} :', error, '+-', std)


#----------Metrics for Reference year results --------#

def mae_ref_year(y_true, y_pred, year=10):
    raw_error = np.abs(y_true - y_pred).mean(axis=-1)

    error = raw_error[:, year].mean().astype('float32')
    std = raw_error[:, year].std().astype('float32')
    # print(error.shape)
    return error, std

def dis_ref_year(y_true, y_pred, year=10):
    dis = np.abs(y_true - y_pred).std(axis=-1)

    error = dis[:, year].mean().astype('float32')
    std = dis[:, year].std().astype('float32')

    return error, std

def r2_ref_year(y_true, y_pred, year=10):
    r2_ = r2_raw(y_true, y_pred)

    r2_mean = r2_[:, year].mean()
    r2_std = r2_[:, year].std()

    return r2_mean, r2_std

def ref_year_score(observations_d, prediction_d, ref_year):
    for metric_name, metric in zip(['MAE', 'DIS', 'R2'], [mae_ref_year, dis_ref_year, r2_ref_year]):
        print(f'\t--------- {metric_name} {ref_year} ----------')
        for (key, pred), (_, true) in zip(prediction_d.items(), observations_d.items()):
            pred = np.moveaxis(pred, -1, 1)  # metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            true = np.moveaxis(true, -1, 1)  # metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            error, std = metric(true, pred, year=ref_year)
            print(f'\t\t{key} :', error, '+-', std)


#----------Metrics for Per year results (plot) -------#

def mae_per_year(y_true, y_pred):  # for plots

    # variations over years

    raw_error = np.abs(y_true - y_pred).mean(axis=-1)

    error = raw_error.mean(axis=0).astype('float32')
    std = raw_error.std(axis=0).astype('float32')

    return error, std

def dis_per_year(y_true, y_pred):
    dis = np.abs(y_true - y_pred).std(axis=-1)

    error = dis.mean(axis=0).astype('float32')
    std = dis.std(axis=0).astype('float32')

    return error, std

def r2_per_year(y_true, y_pred):  # for plots
    # Average R2 score for each year

    r2_ = r2_raw(y_true, y_pred)

    r2_per_year = r2_.mean(axis=0)
    r2_std_per_year = r2_.std(axis=0)

    return r2_per_year, r2_std_per_year

def plot_per_year(observations_dictionary, predictions_dictionary, data_name, filename):
    fig, axes = plt.subplots(1, 3, dpi=100, constrained_layout=True, figsize=set_size(tex_witdh_in_pt, subplots=(1, 3), fact=1.5))
    timerange = np.arange(2000, 2000 + predictions_dictionary['baseline0'].shape[-1], 1).astype(int)

    for (name, fold_true), (_, fold_pred) in zip(observations_dictionary.items(), predictions_dictionary.items()):

        fold_true = np.moveaxis(fold_true, -1, 1)
        fold_pred = np.moveaxis(fold_pred, -1, 1)

        mae, std_mae = mae_per_year(fold_true, fold_pred)
        dis, std_dis = dis_per_year(fold_true, fold_pred)
        r2, std_r2 = r2_per_year(fold_true, fold_pred)

        for i, (score, std, metric_name) in enumerate(
                [[mae, std_mae, 'MAE'], [dis, std_dis, 'DIS'], [r2, std_r2, 'R2']]):
            axes[i].errorbar(timerange, score, std, fmt='o', ms=2, elinewidth=0.5, label=f'{name}')
            axes[i].set_xticks(timerange[::4])
            axes[i].tick_params(axis='x')
            axes[i].set_title(metric_name)

            if metric_name == 'R2':
                axes[i].set_ylim(0, 1)

    fig.supxlabel('Time (years)')
    fig.suptitle(f'{data_name}')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right',
               alignment='left',
               labelspacing=0.1,
               handlelength=0.5,
               ncols=len(predictions_dictionary))
    fig.savefig(filename)

#----------Metrics for global evaluation on lit areas ----#

def mae_global_on_lit_areas(y_true, y_pred):
    all_r2s, all_mae, all_dis, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_mae.mean(axis=1).mean().astype('float32')
    std = all_mae.mean(axis=1).std().astype('float32')

    return error, std

def dis_global_on_lit_areas(y_true, y_pred):
    all_r2s, all_mae, all_dis, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_dis.mean(axis=1).mean().astype('float32')
    std = all_dis.mean(axis=1).std().astype('float32')

    return error, std

def r2_global_on_lit_areas(y_true, y_pred):
    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    r2_mean = all_r2s.mean(axis=1).mean()
    r2_std = all_r2s.mean(axis=1).std()

    return r2_mean, r2_std

def global_lit_areas_scores(observations_d, prediction_d):

    for metric_name, metric in zip(['MAE', 'DIS', 'R2'], [mae_global_on_lit_areas, dis_global_on_lit_areas, r2_global_on_lit_areas]):
        print(f'\t--------- {metric_name} ----------')
        for (key, pred), (_, true) in zip(prediction_d.items(), observations_d.items()):
            pred = np.moveaxis(pred, -1, 1) #metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            true = np.moveaxis(true, -1, 1) #metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            error, std = metric(true, pred)
            print(f'\t\t{key} :', error, '+-', std)


#----------Per year metrics --------------------------#

def mae_per_year_on_lit_areas(y_true, y_pred):
    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_mae.mean(axis=0).astype('float32')
    std = all_mae.std(axis=0).astype('float32')

    return error, std

def dis_per_year_on_lit_areas(y_true, y_pred):
    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_std.mean(axis=0).astype('float32')
    std = all_mae.std(axis=0).astype('float32')

    return error, std

def r2_per_year_on_lit_areas(y_true, y_pred):
    # std = moyenne des 5 fold + variation par année

    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    r2_per_year = all_r2s.mean(axis=0)
    r2_std_per_year = all_r2s.std(axis=0)

    return r2_per_year, r2_std_per_year

def plot_per_year_on_lit_areas(observations_dictionary, predictions_dictionary, data_name, filename):
    fig, axes = plt.subplots(1, 3, dpi=100, constrained_layout=True, figsize=set_size(tex_witdh_in_pt, subplots=(1, 3), fact=1.5))
    timerange = np.arange(2000, 2000 + predictions_dictionary['baseline0'].shape[-1], 1).astype(int)

    for (name, fold_true), (_, fold_pred) in zip(observations_dictionary.items(), predictions_dictionary.items()):

        fold_true = np.moveaxis(fold_true, -1, 1)
        fold_pred = np.moveaxis(fold_pred, -1, 1)

        mae, std_mae = mae_per_year_on_lit_areas(fold_true, fold_pred)
        dis, std_dis = dis_per_year_on_lit_areas(fold_true, fold_pred)
        r2, std_r2 = r2_per_year_on_lit_areas(fold_true, fold_pred)

        for i, (score, std, metric_name) in enumerate(
                [[mae, std_mae, 'MAE'], [dis, std_dis, 'DIS'], [r2, std_r2, 'R2']]):
            axes[i].errorbar(timerange, score, std, fmt='o', ms=2, elinewidth=0.5, label=f'{name}')
            axes[i].set_xticks(timerange[::4])
            axes[i].tick_params(axis='x')
            axes[i].set_title(metric_name)

            if metric_name == 'R2':
                axes[i].set_ylim(0, 1)

    fig.supxlabel('Time (years)')
    fig.suptitle(f'{data_name}')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right',
               alignment='left',
               labelspacing=0.1,
               handlelength=0.5,
               ncols=len(predictions_dictionary))
    fig.savefig(f'{filename}.pdf')



#----------Metrics for ref year evaluation on lit areas ----#

def dis_ref_year_on_lit_areas(y_true, y_pred, year=10):
    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_std[:, year].mean().astype('float32')
    std = all_std[:, year].std().astype('float32')

    return error, std

def mae_ref_year_on_lit_areas(y_true, y_pred, year=10):
    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    error = all_mae[:, year].mean().astype('float32')
    std = all_mae[:, year].std().astype('float32')

    return error, std

def r2_ref_year_on_lit_areas(y_true, y_pred, year=10):
    # std = moyenne des années + variation par fold

    all_r2s, all_mae, all_std, _ = raw_score_on_lit_areas(y_true, y_pred)

    r2_mean = all_r2s[:, year].mean()
    r2_std = all_r2s[:, year].std()

    return r2_mean, r2_std

def ref_year_lit_areas_score(observations_d, prediction_d, ref_year):
    for metric_name, metric in zip(['MAE', 'DIS', 'R2'], [mae_ref_year_on_lit_areas, dis_ref_year_on_lit_areas, r2_ref_year_on_lit_areas]):
        print(f'\t--------- {metric_name} {ref_year} ----------')
        for (key, pred), (_, true) in zip(prediction_d.items(), observations_d.items()):
            pred = np.moveaxis(pred, -1, 1)  # metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            true = np.moveaxis(true, -1, 1)  # metrics funcs take data in the fomrat (n_folds, n_timesteps, n_examples)
            error, std = metric(true, pred, year=ref_year)
            print(f'\t\t{key} :', error, '+-', std)

#----------- Metric to compute the best model on average  -----#

def compare_with_threshold(true, temp, spat, threshold):
    fold_score = []
    fold_number_of_examples = []
    for fold_true, fold_temp, fold_spat in zip(true, temp, spat):
        score = []
        number_of_examples = []
        for yf_true, yf_temp, yf_spat in zip(fold_true, fold_temp, fold_spat):
            # compute on non zeros ntls
            if threshold > 0:
                non_zeros_ntls = np.where(np.abs(yf_temp - yf_spat) > threshold)[0]
            else:
                non_zeros_ntls = np.where(yf_true != 0)[0]

            yf_true = yf_true[non_zeros_ntls]
            yf_temp = yf_temp[non_zeros_ntls]
            yf_spat = yf_spat[non_zeros_ntls]

            error_temp = np.abs(yf_true - yf_temp)
            error_spat = np.abs(yf_true - yf_spat)

            all_models = np.array([error_temp, error_spat])

            count_temp = np.where(all_models.argmin(axis=0) == 0)[0].shape[0]
            count_spat = np.where(all_models.argmin(axis=0) == 1)[0].shape[0]
            total = all_models.shape[1]

            score.append(count_temp)
            number_of_examples.append(total)

        fold_score.append(score)
        fold_number_of_examples.append(number_of_examples)

    fold_score = np.array(fold_score)
    fold_number_of_examples = np.array(fold_number_of_examples)

    return fold_score, fold_number_of_examples

def compare(true, temp, spat, threshold_list=[0, 0.1, 0.5, 1, 2, 4, 7]):

    for threshold in threshold_list:

        fold_scores, fold_numbers = compare_with_threshold(true, temp, spat, threshold)

        global_score = (fold_scores / fold_numbers).mean(axis=-1).mean()
        global_std = (fold_scores / fold_numbers).mean(axis=-1).std()

        print(f"For a threshold of {threshold}")
        print(f"\tthere are on average {fold_numbers.mean(axis=-1).mean():.4f} +- {fold_numbers.mean(axis=-1).std():.4f} examples")
        print(f"\tTemp model is better {100*global_score:.2f} % +- {100*global_std}")

def per_year_compare(true, temp, spat, threshold_list=[0, 0.1, 0.5, 1, 2, 4, 7]):

    for threshold in threshold_list:

        fold_scores, fold_numbers = compare_with_threshold(true, temp, spat, threshold)

        global_score = (fold_scores / fold_numbers).mean(axis=0)
        global_std = (fold_scores / fold_numbers).std(axis=0).std()

        fig, axes = plt.subplots(1, 3, dpi=100, constrained_layout=True,
                                 figsize=set_size(tex_witdh_in_pt, subplots=(1, 3), fact=1.5))
        timerange = np.arange(2000, 2000 + predictions_dictionary['baseline0'].shape[-1], 1).astype(int)

def compute_evo_table(observation_d, prediction_d, results_path):

    os.makedirs(results_path, exist_ok=True)
    for (name, y_true), (_, y_pred) in zip(observation_d.items(), prediction_d.items()):
        res = np.zeros((y_true.shape[-1], y_true.shape[-1]))
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=set_size(width=tex_witdh_in_pt, subplots=(1, 1), fact=1.5))
        for i in range(0, y_true.shape[-1]):
            for j in range(0, y_true.shape[-1]):
                if j > i:
                    evo_true = y_true[:, :, j] - y_true[:, :, i]
                    evo_pred = y_pred[:, :, j] - y_pred[:, :, i]

                    res[i, j] = np.array([r2_score(true, pred) for true, pred in zip(evo_true, evo_pred)]).mean().round(2)

                if i == j:
                    res[i, j] = np.array([r2_score(true, pred) for true, pred in zip(y_true[:, :, i], y_pred[:, :, i])]).mean().round(2)

        cmp = ConfusionMatrixDisplay(matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)(res), display_labels=np.arange(2000, 2021))

        pd.DataFrame(res).to_csv(results_path + f'cm_r2_evo_{name}.csv')
        # Deactivate default colorbar
        cmp.plot(ax=ax, values_format='.2g', colorbar=False, im_kw={'norm':matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)})
        cmp.ax_.set_xlabel('End Year')
        cmp.ax_.set_ylabel('Start Year')
        # Adding custom colorbar
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True), cmap='viridis'), cax=cax)
        fig.suptitle(f'Evolution R2 with {name} Model')
        fig.savefig(results_path + f'cm_r2_evo_{name}.pdf')

        plt.show()

def compute_evo_table_on_lit_areas(observation_d, prediction_d, results_path):
    os.makedirs(results_path, exist_ok=True)
    for (name, y_true), (_, y_pred) in zip(observation_d.items(), prediction_d.items()):
        res = np.zeros((y_true.shape[-1], y_true.shape[-1]))
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=set_size(width=tex_witdh_in_pt, subplots=(1, 1), fact=1.5))
        for i in range(0, y_true.shape[-1]):
            for j in range(0, y_true.shape[-1]):
                if j > i:
                    evo_true = y_true[:, :, j] - y_true[:, :, i]
                    evo_pred = y_pred[:, :, j] - y_pred[:, :, i]

                    #res[i, j] = np.array([r2_score(true, pred) for true, pred in zip(evo_true, evo_pred)]).mean().round(2)
                    r2 = 0
                    for true, pred in zip(evo_true, evo_pred):
                        index = np.where(true != 0)[0]
                        true = true[index]
                        pred = pred[index]
                        r2 += r2_score(true, pred)


                elif i == j:
                    r2 = 0
                    for true, pred in zip(y_true[:, :, i], y_pred[:, :, i]):
                        index = np.where(true != 0)[0]
                        true = true[index]
                        pred = pred[index]
                        r2 += r2_score(true, pred)
                else:
                    r2 = 0

                res[i, j] = r2 / y_true.shape[0]

        cmp = ConfusionMatrixDisplay(matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)(res), display_labels=np.arange(2000, 2021),)


        # Deactivate default colorbar
        cmp.plot(ax=ax, values_format='.2g', colorbar=False, im_kw={'norm':matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)})
        cmp.ax_.set_xlabel('End Year')
        cmp.ax_.set_ylabel('Start Year')
        # Adding custom colorbar
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True), cmap='viridis'), cax=cax)
        fig.suptitle(f'Evolution R2 with {name} Model')
        fig.savefig(results_path + f'cm_r2_evo_{name}.pdf')

        plt.show()





