import argparse

import matplotlib.pyplot as plt
import numpy as np

from data import EXPE, EX_ZONE, PS

from metrics import global_scores, ref_year_score, global_lit_areas_scores, ref_year_lit_areas_score, compare
from metrics import plot_per_year, plot_per_year_on_lit_areas
from metrics import compute_evo_table, compute_evo_table_on_lit_areas

def load_fold_results(expe, num):
    predictions_test = []
    predictions_ex = []
    observations_test = []
    observations_ex = []

    path_to_obs = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/data/'
    path_to_pred = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/models/{expe}/{expe}_{num}/'

    for i in range(5):
        predictions_test.append(np.load(path_to_pred + f'{i}/predictions.npy'))
        predictions_ex.append(np.load(path_to_pred + f'{i}/ex_zone_pred.npy'))

        observations_test.append(np.load(path_to_obs + f'fold_{i}/viirs.npy'))
        observations_ex.append(np.load(path_to_obs + f'ex_zone/viirs.npy'))

    observations_test = np.array(observations_test)
    observations_ex = np.array(observations_ex)

    predictions_test = np.array(predictions_test)
    predictions_ex = np.array(predictions_ex)

    return observations_test, predictions_test, observations_ex, predictions_ex

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-on_test", action='store_true')
    parser.add_argument("-on_zanzibar", action='store_true')
    parser.add_argument("-on_test_evo", action='store_true')
    parser.add_argument("-on_zanzibar_evo", action='store_true')
    parser.add_argument("-plot", action='store_true')
    parser.add_argument("-plot_evo", action='store_true')
    parser.add_argument("-all", action='store_true')
    parser.add_argument("-ref_year")
    parser.add_argument("-timestep")

    args = parser.parse_args()

    if args.all:
        args.on_test = True
        args.on_zanzibar = True
        args.on_test_evo = True
        args.on_zanzibar_evo = True

    expes = ['baseline', 'spat', 'temp_only', 'res_cnn', 'effnet']
    nums = [0, 0, 1, 0, 'gbs']

    # expes = ['baseline', 'effnet']
    # nums = [0, 0]

    results_saving_path = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/results/'

    observations_dictionary = {}
    observations_ex_dictionary = {}
    predictions_dictionary = {}
    predictions_ex_dictionary = {}

    evo_observations_dictionary = {}
    evo_observations_ex_dictionary = {}
    evo_predictions_dictionary = {}
    evo_predictions_ex_dictionary = {}

    timestep = int(args.timestep)

    for expe, num in zip(expes, nums):

        observations_test, predictions_test, observations_ex, predictions_ex = load_fold_results(expe, num)

        predictions_dictionary[expe+f'{num}'] = predictions_test
        predictions_ex_dictionary[expe+f'{num}'] = predictions_ex
        observations_dictionary[expe+f'{num}'] = observations_test
        observations_ex_dictionary[expe+f'{num}'] = observations_ex

        evo_observations = observations_test[:, :, timestep:] - observations_test[:, :, :-timestep]
        evo_predictions = predictions_test[:, :, timestep:] - predictions_test[:, :, :-timestep]
        evo_observations_ex = observations_ex[:, :, timestep:] - observations_ex[:, :, :-timestep]
        evo_predictions_ex = predictions_ex[:, :, timestep:] - predictions_ex[:, :, :-timestep]

        evo_predictions_dictionary[expe+f'{num}'] = evo_predictions
        evo_predictions_ex_dictionary[expe+f'{num}'] = evo_predictions_ex
        evo_observations_dictionary[expe+f'{num}'] = evo_observations
        evo_observations_ex_dictionary[expe+f'{num}'] = evo_observations_ex

    ref_year = int(args.ref_year)
    if args.on_test:
        print(f'============ ON TEST ==========\n')
        print(f'======== GLOBAL SCORES ========\n')
        global_scores(observations_d=observations_dictionary, prediction_d=predictions_dictionary)
        print('\n')
        print(f'======== REF YEAR {ref_year} SCORES ========\n')
        ref_year_score(observations_d=observations_dictionary, prediction_d=predictions_dictionary, ref_year=ref_year)
        print('\n')
        print(f'======== GLOBAL LIT AREAS SCORES ========\n')
        global_lit_areas_scores(observations_d=observations_dictionary, prediction_d=predictions_dictionary)
        print('\n')
        print(f'======== REF YEAR {ref_year} LIT AREAS SCORES ========\n')
        ref_year_lit_areas_score(observations_d=observations_dictionary, prediction_d=predictions_dictionary, ref_year=ref_year)
        print('================================\n')
    if args.on_zanzibar:
        print(f'============ ON ZANZIBAR ======\n')
        print(f'======== GLOBAL SCORES ========\n')
        global_scores(observations_d=observations_ex_dictionary, prediction_d=predictions_ex_dictionary)
        print('\n')
        print(f'======== REF YEAR {ref_year} SCORES ========\n')
        ref_year_score(observations_d=observations_ex_dictionary, prediction_d=predictions_ex_dictionary, ref_year=ref_year)
        print('\n')
        print(f'======== GLOBAL LIT AREAS SCORES ========\n')
        global_lit_areas_scores(observations_d=observations_ex_dictionary, prediction_d=predictions_ex_dictionary)
        print('\n')
        print(f'======== REF YEAR {ref_year} LIT AREAS SCORES ========\n')
        ref_year_lit_areas_score(observations_d=observations_ex_dictionary, prediction_d=predictions_ex_dictionary, ref_year=ref_year)
        print('================================\n')
    if args.on_test_evo:
        print(f'============ ON TEST EVO ==========\n')
        print(f'======== GLOBAL SCORES ========\n')
        global_scores(observations_d=evo_observations_dictionary, prediction_d=evo_predictions_dictionary)
        print('\n')
        print(f'======== REF EVO {ref_year}, {ref_year+timestep} SCORES ========\n')
        ref_year_score(observations_d=evo_observations_dictionary, prediction_d=evo_predictions_dictionary, ref_year=ref_year)
        print('\n')
        print(f'======== GLOBAL LIT AREAS SCORES ========\n')
        global_lit_areas_scores(observations_d=evo_observations_dictionary, prediction_d=evo_predictions_dictionary)
        print('\n')
        print(f'======== REF YEAR {ref_year}, {ref_year+timestep}  LIT AREAS SCORES ========\n')
        ref_year_lit_areas_score(observations_d=evo_observations_dictionary, prediction_d=evo_predictions_dictionary, ref_year=ref_year)
        print('================================\n')
    if args.on_zanzibar_evo:
        print(f'============ ON ZANZIBAR EVO ==========\n')
        print(f'======== GLOBAL SCORES ========\n')
        global_scores(observations_d=evo_observations_ex_dictionary, prediction_d=evo_predictions_ex_dictionary)
        print('\n')
        print(f'======== REF EVO {ref_year}, {ref_year+timestep} SCORES ========\n')
        ref_year_score(observations_d=evo_observations_ex_dictionary, prediction_d=evo_predictions_ex_dictionary, ref_year=ref_year)
        print('\n')
        print(f'======== GLOBAL LIT AREAS SCORES ========\n')
        global_lit_areas_scores(observations_d=evo_observations_ex_dictionary, prediction_d=evo_predictions_ex_dictionary)
        print('\n')
        print(f'======== REF EVO {ref_year}, {ref_year+timestep} LIT AREAS SCORES ========\n')
        ref_year_lit_areas_score(observations_d=evo_observations_ex_dictionary, prediction_d=evo_predictions_ex_dictionary, ref_year=ref_year)
        print('================================\n')

    compare(np.moveaxis(observations_test, -1, 1),
            np.moveaxis(predictions_dictionary['baseline0'], -1, 1),
            np.moveaxis(predictions_dictionary['effnetgbs'], -1, 1))

    print('evo compare')
    compare(np.moveaxis(evo_observations, -1, 1),
            np.moveaxis(evo_predictions_dictionary['baseline0'], -1, 1),
            np.moveaxis(evo_predictions_dictionary['effnetgbs'], -1, 1))

    print('Ex zone compare')
    compare(np.moveaxis(observations_ex, -1, 1),
            np.moveaxis(predictions_ex_dictionary['baseline0'], -1, 1),
            np.moveaxis(predictions_ex_dictionary['effnetgbs'], -1, 1))

    print('Evo ex zone compare')
    compare(np.moveaxis(evo_observations_ex, -1, 1),
            np.moveaxis(evo_predictions_ex_dictionary['baseline0'], -1, 1),
            np.moveaxis(evo_predictions_ex_dictionary['effnetgbs'], -1, 1))
    if args.plot:
        plot_per_year(observations_dictionary=observations_dictionary,
                      predictions_dictionary=predictions_dictionary,
                      data_name='Test Area, All examples', filename=results_saving_path + 'plot_on_test_all.pdf')
        plt.show()
        plot_per_year(observations_dictionary=observations_ex_dictionary,
                      predictions_dictionary=predictions_ex_dictionary,
                      data_name='Excluded Zone, All Examples',
                      filename=results_saving_path + 'plot_on_exzone_all.pdf')
        plt.show()
        plot_per_year_on_lit_areas(observations_dictionary=observations_dictionary,
                                   predictions_dictionary=predictions_dictionary,
                                   data_name='Excluded Zone, Lit Examples',
                                   filename=results_saving_path + 'plot_on_test_litareas.pdf')
        plt.show()
        plot_per_year_on_lit_areas(observations_dictionary=observations_ex_dictionary,
                                   predictions_dictionary=predictions_ex_dictionary,
                                   data_name='Excluded Zone, Lit Examples',
                                   filename=results_saving_path + 'plot_on_exzone_litareas.pdf')
        plt.show()

    if args.plot_evo:
        plot_per_year(observations_dictionary=evo_observations_dictionary,
                      predictions_dictionary=evo_predictions_dictionary,
                      data_name='Evo Test Area, All examples', filename=results_saving_path + 'plot_on_test_all_evo.pdf')
        plt.show()
        plot_per_year(observations_dictionary=evo_observations_ex_dictionary,
                      predictions_dictionary=evo_predictions_ex_dictionary,
                      data_name='Evo Excluded Zone, All Examples',
                      filename=results_saving_path + 'plot_on_exzone_all_evo.pdf')
        plt.show()
        plot_per_year_on_lit_areas(observations_dictionary=evo_observations_dictionary,
                                   predictions_dictionary=evo_predictions_dictionary,
                                   data_name='Evo Excluded Zone, Lit Examples',
                                   filename=results_saving_path + 'plot_on_test_litareas_evo.pdf')
        plt.show()
        plot_per_year_on_lit_areas(observations_dictionary=evo_observations_ex_dictionary,
                                   predictions_dictionary=evo_predictions_ex_dictionary,
                                   data_name='Evo Excluded Zone, Lit Examples',
                                   filename=results_saving_path + 'plot_on_exzone_litareas_evo.pdf')
        plt.show()


    compute_evo_table(observations_dictionary, predictions_dictionary, results_path=results_saving_path + 'all_test_examples/')

    compute_evo_table(observations_ex_dictionary, predictions_ex_dictionary, results_path=results_saving_path + 'all_exzone_examples/')

    compute_evo_table_on_lit_areas(observations_dictionary, predictions_dictionary, results_path=results_saving_path + 'all_test_lit_examples/')

    compute_evo_table_on_lit_areas(observations_ex_dictionary, predictions_ex_dictionary,
                                   results_path=results_saving_path + 'all_exzone_lit_examples/')

    print('tot')




