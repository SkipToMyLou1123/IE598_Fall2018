__author__ = 'Yijun Lou, Changjie Ma, Yuxin Sun, Xiaoyu Yuan'
__copyright__ = "Copyright 2018, The Group Project of IE598"
__credits__ = ["Yijun Lou", "Changjie Ma", "Yuxin Sun", "Xiaoyu Yuan"]
__license__ = "University of Illinois, Urbana Champaign"
__version__ = "1.2.0"
__maintainer__ = "Yijun Lou"
__email__ = "ylou4@illinois.edu"

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from copy import deepcopy as dcp
from functools import partial

class Moody_CLF:

    def __init__(self):
        '''
        Initialize class with empty result dictionary.
        '''
        self.result = {}
        pass

    def parse_data(self, path = 'MLF_GP1_CreditScore.csv'):
        '''
        Parse data of Moody Credit Score from a certain path.
        :param path: string (default='MLF_GP1_CreditScore.csv'), path of the file
        :return:
        '''
        self.data_set = pd.read_csv(path)
        self.attr_table = self.data_set.iloc[:, :-2]
        self.inv_grd = self.data_set.iloc[:, -2]
        self.rating = self.data_set.iloc[:, -1]

    def make_training_test(self, test_size=0.25, random_state=None):
        '''
        Make training and test sets.
        :param test_size: float, int or None, optional (default=0.25), proportion of the dataset to include in the test split.
        :param random_state: int, RandomState instance or None, optional (default=None).
        :return:
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.attr_table, self.inv_grd, test_size=test_size, random_state=random_state, stratify=self.inv_grd)

    def initialize(self, path = 'MLF_GP1_CreditScore.csv', test_size=0.25, random_state=None):
        '''
        Initialize the class. First parse data from certain path and make training and test sets and initialize model settings.
        :param path: string (default='MLF_GP1_CreditScore.csv'), path of the file.
        :param test_size: float, int or None, optional (default=0.25), proportion of the dataset to include in the test split.
        :param random_state: int, RandomState instance or None, optional (default=None).
        :return:
        '''
        self.parse_data(path)
        self.make_training_test(test_size, random_state)

        # n_components = list(range(1, self.attr_table.shape[1]+1,1))
        n_components = [1,2,3,4,5,6]
        # C = np.logspace(-4, 4, 50)
        C = np.logspace(-4, 4, 10)
        penalty = ['l1', 'l2']
        kernel = ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]
        kernel = ["linear", "poly", "rbf"]

        # TODO: FILL THE DICTIONARY BELOW AND RUN THIS SCRIPT
        # Initialize the process dictionary, pipeline will be generated based on this dictionary.
        self.process_dict = {
            'preprocessing': {
                'standard_scaler': StandardScaler(),
                'do_nothing': None,
            },
            'decomposition': {
                'pca': PCA(),
                #'lda': LDA(),
                #'kpca': KernelPCA(),
                'do_nothing': None,
            },
            'model': {
                'logistic': LogisticRegression(),
            }
        }

        # Initialize the parameters dictionary, parameters that pass into models of pipeline will be generated based on this dictionary.
        self.parameters_dict = {
            'pca': {
                'n_components': n_components,
            },
            'lda': {
                'n_components': n_components,
            },
            'kpca': {
                'n_components': n_components,
                'kernel': kernel,
            },
            'logistic': {
                'C': C,
                'penalty': penalty,
            }
        }

    def run(self):
        '''
        Run model in single processor.
        :return:
        '''
        self.make_pipelines()
        self.show_result()

    def run_mp(self, processes=4):
        '''
        Run model in multiple processor.
        :return:
        '''
        self.do_task_mp(processes=processes)
        self.show_result()

    def make_pipelines(self):
        '''
        Core algorithm, coming soon.
        :return:
        '''
        for k1, v1 in self.process_dict['preprocessing'].items():
            steps = []
            parameters = {}
            k1_list = []
            if k1 != 'do_nothing':
                steps.append((k1, v1))
                if k1 in self.parameters_dict:
                    for k, v in self.parameters_dict[k1].items():
                        key = k1 + "__" + k
                        parameters[key] = v
                        k1_list.append(key)
            for k2, v2 in self.process_dict['decomposition'].items():
                k2_list = []
                if k2 != 'do_nothing':
                    steps.append((k2, v2))
                    if k2 in self.parameters_dict:
                        for k, v in self.parameters_dict[k2].items():
                            key = k2 + "__" + k
                            parameters[key] = v
                            k2_list.append(key)
                for k3, v3 in self.process_dict['model'].items():
                    k3_list = []
                    if k3 in self.parameters_dict:
                        for k, v in self.parameters_dict[k3].items():
                            key = k3 + "__" + k
                            parameters[key] = v
                            k3_list.append(key)
                    steps.append((k3, v3))
                    # r_key = ((k1 + "+") if k1 != 'do_nothing' else "") + ((k2 + "+") if k2 != 'do_nothing' else "")  + k3
                    r_key = k1 + "+" + k2 + "+" + k3
                    self.result[r_key] = {}
                    # print("steps", steps)
                    # print("parameters", parameters)
                    self.run_estimator(r_key, Pipeline(steps=dcp(steps)), dcp(parameters))
                    steps.remove((k3, v3))
                    for key in k3_list:
                        del parameters[key]
                if k2 != 'do_nothing':
                    steps.remove((k2, v2))
                    for key in k2_list:
                        del parameters[key]
            if k1 != 'do_nothing':
                steps.remove((k1, v1))
                for key in k1_list:
                    del parameters[key]

    def run_estimator(self, key, pipe, parameters, scoring=None, cv=10):
        '''
        Fit the current pipeline.
        :param key: string, combination of names in a pipeline.
        :param pipe: Pipeline Object, the pipeline that will be fitted.
        :param parameters: dict, parameters of each part of pipeline.
        :param scoring: string, callable, list/tuple, dict or None, default: None.
        :param cv: int, cross-validation generator or an iterable, optional.
        :return:
        '''
        print("Generating estimator %s" % key)
        self.clf = GridSearchCV(estimator=pipe, param_grid=parameters, scoring=scoring, cv=cv)
        print("Fitting model")
        self.clf.fit(self.attr_table, self.inv_grd)
        print("Saving result")
        self.record_model(key, parameters)
        print("-" * 60 + "\n")

    def make_pipelines_mp(self):
        '''
        Core algorithm, coming soon.
        :return:
        '''
        ret_list = []
        for k1, v1 in self.process_dict['preprocessing'].items():
            steps = []
            parameters = {}
            k1_list = []
            if k1 != 'do_nothing':
                steps.append((k1, v1))
                if k1 in self.parameters_dict:
                    for k, v in self.parameters_dict[k1].items():
                        key = k1 + "__" + k
                        parameters[key] = v
                        k1_list.append(key)
            for k2, v2 in self.process_dict['decomposition'].items():
                k2_list = []
                if k2 != 'do_nothing':
                    steps.append((k2, v2))
                    if k2 in self.parameters_dict:
                        for k, v in self.parameters_dict[k2].items():
                            key = k2 + "__" + k
                            parameters[key] = v
                            k2_list.append(key)
                for k3, v3 in self.process_dict['model'].items():
                    k3_list = []
                    if k3 in self.parameters_dict:
                        for k, v in self.parameters_dict[k3].items():
                            key = k3 + "__" + k
                            parameters[key] = v
                            k3_list.append(key)
                    steps.append((k3, v3))
                    # r_key = ((k1 + "+") if k1 != 'do_nothing' else "") + ((k2 + "+") if k2 != 'do_nothing' else "") + k3
                    r_key = k1 + "+" + k2 + "+" + k3
                    self.result[r_key] = {}
                    ret_list.append([r_key, Pipeline(steps=dcp(steps)), dcp(parameters)])
                    steps.remove((k3, v3))
                    for key in k3_list:
                        del parameters[key]
                if k2 != 'do_nothing':
                    steps.remove((k2, v2))
                    for key in k2_list:
                        del parameters[key]
            if k1 != 'do_nothing':
                steps.remove((k1, v1))
                for key in k1_list:
                    del parameters[key]
        return ret_list

    def run_estimator_mp(self, iter_list, scoring=None, cv=10):
        '''
        TODO
        :param iter_list:
        :param scoring:
        :param cv:
        :return:
        '''
        key = iter_list[0]
        pipe = iter_list[1]
        parameters = iter_list[2]
        print("Generating estimator %s" % key)
        self.clf = GridSearchCV(estimator=pipe, param_grid=parameters, scoring=scoring, cv=cv)
        # print("Fitting model")
        self.clf.fit(self.attr_table, self.inv_grd)
        # print("Saving result")
        ret = self.record_model(key, parameters)
        # print("-" * 60 + "\n")
        return ret

    def do_task_mp(self, processes=4):
        '''
        TODO
        :return:
        '''
        pipeline_list = self.make_pipelines_mp()
        pool = mp.Pool(processes=processes)
        ret_list = pool.map(self.run_estimator_mp, pipeline_list)
        pool.close()
        pool.join()

        for item in ret_list:
            key = item[0]
            best_score = item[1]
            self.result[key]['best_score'] = best_score
            self.result[key]['best_parameters_set'] = {}
            for p, v in item[2:]:
                self.result[key]['best_parameters_set'][p] = v

    def record_model(self, key, parameters):
        '''
        Save best score and parameters for current model.
        :param key: string, combination of names in a pipeline.
        :param parameters: dict, parameters of each part of pipeline.
        :return: list, with [key, score, all pairs of parameters with its best value in order...], length of this list is different based on number of parameters.
        '''
        # 输出best score
        ret = [key]
        print("Best score: %0.3f" % self.clf.best_score_)
        self.result[key]['best_score'] = self.clf.best_score_
        ret.append(self.clf.best_score_)
        self.result[key]['best_parameters_set'] = {}
        # 输出最佳的分类器的参数
        best_parameters = self.clf.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            self.result[key]['best_parameters_set'][param_name] = best_parameters[param_name]
            ret.append((param_name, best_parameters[param_name]))
        return ret

    def show_result(self, save=True, file_name='result.csv'):
        '''
        Print result of each pipeline.
        :param save: bool, save as csv file if the value is true.
        :param file_name: string, save file using this name.
        :return:
        '''
        best_score = 0
        best_estimator = ""
        estimator_list = []
        best_score_list = []
        params_list = []
        max_no_params = 0
        for k, v in self.result.items():

            estimator_name = k.replace('do_nothing+', '')
            estimator_score = v['best_score']

            print("Estimator Name: %s" % (estimator_name))
            print("Best Score: %0.3f" % (estimator_score))

            if save:
                estimator_list.append(k)
                best_score_list.append(estimator_score)

            if v['best_score'] > best_score:
                best_estimator = estimator_name
                best_score = estimator_score

            print("Best parameters set:")
            params_count = 0
            estimator_params_list = []
            for p, v in self.result[k]['best_parameters_set'].items():
                params_count +=1
                print("\t%s: %r" % (p, v))
                if save:
                    estimator_params_list.append(p + ": " + str(v))
            if save:
                params_list.append(estimator_params_list)
            if params_count > max_no_params:
                max_no_params = params_count
            print("*" * 60 + "\n")

        if save:
            self.save_result(estimator_list, best_score_list, params_list, max_no_params, file_name)
        print("BEST ESTIMATOR IS %s WITH SCORE %0.3f" % (best_estimator, best_score))

    def save_result(self, estimator_list, best_score_list, params_list, max_no_params, file_name='result.csv'):
        cwd = os.getcwd()
        full_path = cwd + "/" + file_name
        print("Save result as %s" % full_path)

        data = {
            'preprocessing': [],
            'decomposition': [],
            'classifier': [],
        }

        for i in range(0, max_no_params):
            col_name = "parameter_" + str(i+1)
            data[col_name] = []

        data['best_score'] = []

        for i in range(0, len(estimator_list)):

            curr_estimator = estimator_list[i]
            p1, p2, p3 = curr_estimator.split("+")
            data['preprocessing'].append(p1)
            data['decomposition'].append(p2)
            data['classifier'].append(p3)

            curr_score = best_score_list[i]
            data['best_score'].append(curr_score)

            curr_params_set = params_list[i]
            for j in range(0, max_no_params):
                col_name = "parameter_" + str(j + 1)
                data[col_name].append(curr_params_set[j] if j < len(curr_params_set) else '')

        df = pd.DataFrame.from_dict(data)
        df.to_csv(file_name)

if __name__ == '__main__':
    my_clf = Moody_CLF()
    my_clf.initialize()
    my_clf.run_mp()
