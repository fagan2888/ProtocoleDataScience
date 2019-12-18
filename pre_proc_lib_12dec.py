'''
LIB for DS pre-processing from data matrix
'''
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
OPERATIONS
- read dataset
- save it into a pandas dataframe
- feature engineering
-- separate categorical numerical variables
-- numerical variable normalization
-- heteroskedasticity
-- one_hot encoding for categorical variables
--


REFERENCES
https://www.kaggle.com/parasjindal96/how-to-normalize-dataframe-pandas

'''

''' ===========================
    ### BUILD ALL FUNCTIONS ###
    ===========================
NB: functions tested one by one in a side Terminal
to have a TDD approach with Libs call and function_tests
'''

''' ========================
    ## PRE-PROC FUNCTIONS ##
    ======================== '''

class PreProcLib():
    '''
    Purpose: lib for reading, heteroskedasticity test,
    multi-colinearity,
    normality error test, one-hot-encoding of categorical features
    '''
    THRESHOLD_OUTLIERS = 3
    THRESHOLD_VIF = 100
    NUMBER_OF_CLASSES = 6#4
    SUB_STRING_TO_REMOVE = 'RET' # in case of ATP.csv
    STRING_SEPARATOR = '-'

    def __init__(self, data_file, target_choosen=None,
        list_cols_to_drop=None, list_nominal_cols=None):
        self.data_file = data_file
        self.dataframe = pd.DataFrame()
        self.list_cols_to_drop = list_cols_to_drop
        self.numeric_columns = []
        self.np_outliers = []
        self.list_nominal_cols = list_nominal_cols
        self.best_of_flag = 'no'#'yes'#'no'
        self.list_of_accepted_targets = ['score', 'minutes']
        if target_choosen:
            self.target_choosen = target_choosen
        else:
            self.target_choosen = 'minutes'#'score'
        if self.target_choosen == 'score':
            self.best_of_flag = 'yes'

    def __enter__(self):
        a = 1 # to fill

    def __exit__(self):
        b = 2 # to fill

    def read_csv_data_into_dataframe(self):
        '''
        output: a pandas dataframe
        '''
        try:
            self.dataframe = pd.read_csv(self.data_file, encoding='utf-8')
            self.dataframe.head()
            # return dataframe
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_column_names(self):
        try:
            # first row sets column names
            if not self.dataframe.empty:
                dataframe_cols = self.dataframe.columns
                print(dataframe_cols)
            else:
                print('PreProcLib instance has no dataframe yet!')
                raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_numerical_features(self):
        '''
        output: list of columns for numerical features
        set by hand here to save time
        '''
        try:
            if not self.dataframe.empty:
                self.numeric_columns = \
                    self.dataframe.select_dtypes(include=[np.number]).columns
            else:
                print('PreProcLib instance has no dataframe yet!')
                raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_ordinal_features(self):
        '''
        output: list of columns for ordinal features
        set by hand here to save time
        '''
        return None

    def get_nominal_features(self):
        '''
        output: list of columns for nominal features
        set by hand here to save time
        '''
        return None

    def plot_pairwise_numerical_features(self):
        '''
        produces a matrix of plots with histograms of features in diagonal
        '''
        return None


    def normalize_numerical_variables(self):
        '''
        Good FE practice to set numercial variables on same scale
        REF:
        https://www.kaggle.com/parasjindal96/how-to-normalize-dataframe-pandas
        TEST:
        preproc = PreProcLib()
        for column in preproc.numeric_columns:
            print(column)
            print(preproc.dataframe[column].values)
        preproc.normalize_numerical_variables()
        for column in preproc.numeric_columns:
            print(column)
            print(preproc.dataframe[column].values)
        '''
        try:
            # 1- get numerical column
            self.get_numerical_features()
            print('normalize_numerical_variables self.numeric_columns %s'%\
                str(self.numeric_columns))

            # 2- apply scaling normalization transfo to these
            for column in self.numeric_columns:
                if column not in self.list_of_accepted_targets:
                    self.dataframe[column] = \
                        (self.dataframe[column] - \
                         self.dataframe[column].min()) / \
                        (self.dataframe[column].max() - \
                         self.dataframe[column].min())
        except Exception as e:
            print(e.__str__())
            raise
        '''
        except ZeroDivisionError:
            pass
        except ValueError as valerr:
            # except ValueError, valerr: # Deprecated?
            print(valerr)
            raise # Raises the exception just caught
        except Exception as e:
            print("error in normalize_numerical_variables %s"%e.__str__())
        finally: # Optional
            pass # Clean up
        '''

    def nominal_features_one_hot_encoder(self):
        from sklearn.preprocessing import OneHotEncoder
        '''
        encoder = OneHotEncoder(handle_unknown='ignore')
        df_data = dataframe.data
        encoder.fit(df_data)
        dataframe.data = df_data
        '''
        try:
            print('list_nominal_cols %s'%str(self.list_nominal_cols))
            for col_name in self.list_nominal_cols:
                # Get one hot encoding of columns
                one_hot = pd.get_dummies(self.dataframe[col_name])

                # 'Grass' -> 'surface_Grass'
                one_hot_cols = one_hot.columns

                for col in one_hot_cols:
                    if col != col_name:
                        one_hot.rename(columns={col: col_name+'_'+col}, inplace=True)
                print('one_hot %s'%str(one_hot))

                # Drop column col_name as it is now encoded
                self.dataframe = self.dataframe.drop([col_name], axis = 1)

                # Join the encoded dataframe
                self.dataframe = self.dataframe.join(one_hot)
                if {col_name}.issubset(self.dataframe.columns):
                    self.dataframe = self.dataframe.drop([col_name], axis = 1)
            print('dataframe columns after nominal_features_one_hot_encoder %s'%\
                self.dataframe.columns)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def find_outliers_from_dataframe(self):
        '''
        Called by : exclude_outliers_from_dataframe
        Algo :
            - find outliers with Z-score function applied on rows
        Generic function? Yes
        '''
        from scipy import stats
        import numpy as np

        try:
            zscore_array = np.abs(stats.zscore(self.dataframe))
            print(zscore_array)
            threshold = self.THRESHOLD_OUTLIERS
            print(np.where(zscore_array > self.THRESHOLD_OUTLIERS))
            # z[row_i][col_j]>3 => dataframe[row_i][col_j] is an outlier under Z-score thresholding rule
            # voir le format de np_outliers
            self.np_outliers = np.where(zscore_array > self.THRESHOLD_OUTLIERS)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def exclude_outliers_from_dataframe(self):
        '''
        Algo:
            - find outliers with Z-score function applied on rows
            - exclude from dataframe any row having at least an element is an outlier
        '''
        ''' list 2D(row, col) of indexes for outlier data '''
        array_of_ouliers_indexes = self.find_outliers_from_dataframe()
        return array_of_ouliers_indexes
        # for row, col in array_of_ouliers_indexes:

    def keep_dataframe_selected_columns_by_indexes(self, selected_columns_list,
        dataframe):
        ''' keep only columns selected in dataframe
            input: dataframe isnot necessarily self.dataframe
        '''
        new_dataframe = dataframe[selected_columns_list]
        return new_dataframe

    def assess_multi_collinearity(self, target_column_name):
        '''
        Rferences:
        - Multicollinearity occurs when indep vars in a regression model are correlated.
        read multicollinearity_towardsdatascience
        - stats.stackexchange on VIF

        test if exists coeffs ak / Fj = a0+a1F1+..+aPFp; p !=j ?

        use Variance Inflation Factor

        Exclude target column from analysis here
        Algo:
            - Xdf=self.dataframe.drop(target_colmun_name)
            - apply VIF onto Xdf
            - new self.dataframe=vif_xdf and target
        '''

        # A FINIR!

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        Xdf = drop([target_column_name], axis=1, inplace=False)
        df_cols = Xdf.columns
        index_features = np.arange(Xdf.shape[1])
        dropped = True
        while dropped:
            dropped = False

            ''' get data matrix values (besides target) '''
            column_values = Xdf[df_cols[index_features]].values

            ''' build vif list for each data column '''
            vif_list = [variance_inflation_factor(column_values, ix) \
                for ix in np.arange(column_values.shape[1])]

            ''' get index of max value vif '''
            index_of_max_vif = vif_list.index(max(vif_list))

            ''' thresholding '''
            if max(vif_list) > self.THRESHOLD_VIF:
                print('dropping \'' + Xdf[df_cols[index_features]].columns[index_of_max_vif] +
                      '\' at index: ' +
                      str(index_of_max_vif))
                index_features = np.delete(index_features, index_of_max_vif)
                dropped = True

        print('Remaining features:')
        print(Xdf.columns[index_features])
        self.dataframe = Xdf  # method to finish!!
        # return None

    def drop_useless_cols_dataframe(self, list_useless_cols):
        '''
        useless in the scope of our targets
        '''
        try:
            ''' -1- init df '''
            df_usefull_cols = self.dataframe
            print(df_usefull_cols.columns)

            ''' -2- iteratively drops unwanted columns '''
            for column in list_useless_cols:
                df_usefull_cols.drop([column], axis=1, inplace=True)

            return df_usefull_cols
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def drop_row_with_any_nan_col(self):
        ''' Purpose: remove in place rows having 1+ NaN value(s) '''
        try:
            self.dataframe.dropna(how='any', inplace=True)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    ''' -------------------------------------
        # sub-functions for building target #
        ------------------------------------- '''
    def _read_format_class(self, string_value):
        '''
        atomic string operation:
        example : 4-6 6-8 6-3 7-5 1-0 RET -> len(df[].split('-'))-1
        is string_to_remove not in
        else len(df[].split('-'))-2
        to build more generic method without hard-coded RET

        Purpose: 1 raw target data to 1 processed target data
        '''
        sub_string_to_remove = self.SUB_STRING_TO_REMOVE
        string_separator = self.STRING_SEPARATOR
        try:
            if self.target_choosen == 'score':
                format_target_values =\
                    len(string_value.split(string_separator))-1 if \
                        sub_string_to_remove not in string_value else\
                            len(string_value.split(string_separator))-2
            elif self.target_choosen == 'minutes':
                if isinstance(string_value, str):
                    if string_value.isdigit():
                        format_target_values =\
                            int(min([int(string_value) / 90, 1]))

                    else:
                        format_target_values = 0 # CF FOR A BETTER REPLACING VALUE
                elif isinstance(string_value, (int, float)):
                    format_target_values = min([int(string_value / 90), 1])
            else:
                pass
            return format_target_values
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def _change_score_format(self, dataframe_score):
        '''
        apply to all values of column score
        dataframe unchanged
        '''
        return dataframe_score.apply(self._read_format_class)

    def _extract_only_N_sets_data(self, dataframe):
        '''
        Purpose: to filter rows based on a specific value on a column
        related to target
        Exple for ATP dataset: get only 3-sets or 5-sets games; N=best_of arg
        Called by:
        '''
        # 0- set best_of number
        best_of = self.NUMBER_OF_CLASSES - 1

        # 1- exclude NaN rows wr best_of col
        df_no_Nan = dataframe.dropna(subset=['best_of'])

        # 2- filter rows where matching best_of
        df_best_of_sets_match = df_no_Nan[(df_no_Nan['best_of'] == best_of)]

        # 3- filter score in 0 to best_of: outliers filtering
        df_number_sets_played = \
            self._change_score_format(df_best_of_sets_match['score'])
        print('df_number_sets_played %s'%df_number_sets_played)
        df_best_of_sets_no_outliers = \
            df_best_of_sets_match[(df_number_sets_played<=best_of)]

        return df_best_of_sets_no_outliers

    def _change_target_format_string_to_int(self, sub_dataframe):
        '''
        Pupose: apply to all values of target column of sub_dataframe
        dataframe unchanged
        '''
        try:
            return sub_dataframe.apply(self._read_format_class) #DEFINE FUNCTION TO APPLY
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def _change_target_format_numeric_to_class(self, sub_dataframe):
        '''
        Purpose: from numeric to class on sub_dataframe
        '''
        try:
            return sub_dataframe.apply(self._read_format_class)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    '''
    ###### end of sub-functions for target #####
    '''

    def build_target_options(self, target_label, target_transfo_type,
        best_of_flag=None):
        '''
        Purpose: build target dataframe as per transformation type
          given target_type, build corresponding target
          and drop corresponding column
        -target_best_of: number of played sets out of 5-set best_of
        -target_age_classes: age class of winner any surface, any round
        Need to make it generic!

        Difficulty: from raw target data to processed target in a generic manner
        '''
        print('build_target_options target_label: %s'%str(target_label))
        print('build_target_options self.dataframe[target_label]: %s'%str(self.dataframe[target_label]))
        try:
            new_dataframe = self.dataframe
            if target_transfo_type=='string_to_int':
                # 1- transformation type1
                target = self._change_target_format_string_to_int(self.dataframe[target_label])
            elif target_transfo_type=='numeric_to_class':
                # 2- transformation type2
                if best_of_flag and best_of_flag=='yes':
                    # best_of N sets -> extract_only_N_sets_data
                    self.dataframe = self._extract_only_N_sets_data(self.dataframe)
                    print('build_target_options best_of_flag case self.dataframe %s'%\
                        str(self.dataframe))

                print('build_target_options %s'%\
                    str(self.dataframe[target_label]))
                target = self._change_target_format_numeric_to_class(self.dataframe[target_label])

            print(target[:100])
            print('type of target %s'%type(target))

            # 3- drop target column out from dataframe
            new_dataframe = self.dataframe.drop([target_label], axis=1)

            return target, new_dataframe
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    ''' =================================
        ## PLOT OPERATIONS ##
        ================================='''
    def make_plots(self, a_model, plot_type):
        # from ML in Python - Wiley
        plt.figure()

        plt.plot(a_model.alphas_, a_model.mse_path_, ':')
        # alphas_:
        if plot_type == 'alphas':
            plt.plot(a_model.alphas_, a_model.mse_path_.mean(axis=1), \
                label='Average MSE Accross folds', linewidth=2)
            plt.axvline(a_model.alpha_, linestyle='--', label='CV Estimate of Best alpha')
        #elif plot_type == 'mse_list':

        plt.show()

    def build_X_y_train_test(self, data, target):
        ''' Purpose: to build XTrain, XTest, yTrain, yTest '''
        try:
            #-1- convert data and target into proper format numpy array accepted by LinReg
            # https://stackoverflow.com/questions/29512130/convert-pandas-dataframe-to-numpy-for-sklearn
            # with sklearn.LinearReg all variables are expected to be float!
            X = np.array(data)
            Y = np.array(target)
            print('X.shape[0] %s'%str(X.shape[0]))
            print('Y.shape[0] %s'%str(Y.shape[0]))

            #-2- train, test CV datasets
            from sklearn.cross_validation import train_test_split
            xTrain, xTest, yTrain, yTest = train_test_split(data,
                                                        target,
                                                        test_size=0.3,
                                                        random_state=321)
            # -3- check consistency before / after
            print('yTrain.shape() %s'%str(yTrain.shape))
            print('data_best_of %s'%str(data[:11]))
            print('target_best_of %s'%str(target[:10]))
            print(yTrain[:11])
            print(xTrain[:10][:])

            return xTrain, xTest, yTrain, yTest
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def overall_preproc_operations(self, dataframe, target_tag):
        ''' combine all pre-proc operations '''
        print('overall_preproc_operations target_tag: %s'%str(target_tag))
        # 2019-11-24 keep round
        list_cols_to_drop = ['tourney_id',
                         'tourney_level',
                         'tourney_name',
                         'tourney_date',
                         'winner_name',
                         'loser_name',
                         'winner_seed',
                         'loser_seed',
                         'winner_entry',
                         'loser_entry',
                         'winner_id',
                         'loser_id',
                         'winner_ioc',
                         'loser_ioc',
                         'round',
                         'w_ace',
                         'w_df',
                         'w_svpt',
                         'w_1stIn',
                         'w_1stWon',
                         'w_2ndWon',
                         'w_SvGms',
                         'w_bpSaved',
                         'w_bpFaced',
                         'l_ace',
                         'l_df',
                         'l_svpt',
                         'l_1stIn',
                         'l_1stWon',
                         'l_2ndWon',
                         'l_SvGms',
                         'l_bpSaved',
                         'l_bpFaced'
            ]
        if 'best_of' in target_tag and self.target_choosen=='score':
            list_cols_to_drop.append('minutes')
        elif 'best_of' in target_tag and self.target_choosen=='minutes':
            #list_cols_to_drop.append('best_of')
            list_cols_to_drop.append('score')
        # !!! actually don't drop tourney_name but change values into one-hot !!!
        # - winner_ioc to be changed into ordinal
        # - touney_level exluded as prediction to be for any tourney_level
        # - round convert to ordinal TO DO
        # - minutes: a posteriori variable and strongly correlated to score
        # - ace, srv_pencent, 1stin, 2ndin, w_SvGms, l_SvGms : all a posteriori variables

        self.dataframe = self.drop_useless_cols_dataframe(list_cols_to_drop)
        print('number of dataframe rows pre-proc step2: %s'%\
            str(self.dataframe.shape[0]))

        ''' 1. pre-proc: transform nominal features '''
        self.list_nominal_cols = ['surface','winner_hand','loser_hand']
        self.nominal_features_one_hot_encoder() #self.dataframe upgraded
        print('number of non NaN dataframe per column pre-proc step3: %s'%\
            str(self.dataframe.count()))
        print('number of dataframe rows pre-proc step3: %s'%\
            str(self.dataframe.shape[0]))

        ''' 2. pre-proc: drop row with any col having a NaN '''
        # this is rough, but missing values imputation would take a bit if time
        self.drop_row_with_any_nan_col()
        print('number of non NaN dataframe per colmumn pre-proc step3: %s'%\
            str(self.dataframe.count()))
        print('number of dataframe rows pre-proc step4: %s'%\
            str(self.dataframe.shape[0]))
        print('new_tennis_df columns %s'%str(self.dataframe.columns))

        ''' 3. pre-proc: detect and exclude outliers '''
        '''
        exclude_outliers_from_data_frame(tennis_dataframe)
        '''
        ''' 4. pre-proc: multi-colinearity '''
        '''
        not applied here due to lack of time
        '''
        ''' 5. pre_proc normalize numerical features
        '''
        for column in self.numeric_columns:
            print('column %s'%column)
            print('column values BEFORE %s'%str(self.dataframe[column].values))
        self.normalize_numerical_variables()
        for column in self.numeric_columns:
            print('column %s'%column)
            print('column values AFTER %s'%str(self.dataframe[column].values))

        self.normalize_numerical_variables()
        print('self.dataframe after normalizing_numerical variables %s'%\
            str(self.dataframe))

        return self.dataframe

    def build_dataset_target(self, dataframe, target_tag):
        ''' from initial dataframe to final target and data '''
        try:
            if 'best_of' in target_tag:
                # 1- all pre_proc_operations besides target df operations
                self.dataframe = self.overall_preproc_operations(dataframe, 'best_of')

                # 2- target operations
                target_label = self.target_choosen #'target_best_of'
                print('build_dataset_target %s'%str(target_label))
                # need to make sure pre-filtered rows based on condition on target related column well applied if needed!
                target, data = \
                    self.build_target_options(target_label,
                    target_transfo_type='numeric_to_class', best_of_flag=self.best_of_flag)
                print('number of dataframe data rows pre-proc: %s'%str(data.shape[0]))
                print('target.shape %s'%target.shape)

                return target, data
            return None
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def full_pre_proc_from_file_to_X_y_train_test(self, dataset_file):
        '''
        Purpose: manages all operations for pre_proc and building X,y, Train, Test arrays
        '''
        self.data_file = dataset_file

        # 1- build dataframe
        # dataframe = PREPROCinstance.read_csv_data_into_dataframe()
        dataframe = self.read_csv_data_into_dataframe()
        print('step1: self.dataframe.columns: %s'%self.dataframe.columns)

        # 2- build data and target as per preprocessing pipeline wr target_label
        # target, data = PREPROCinstance.build_dataset_target(dataframe, 'best_of')
        target, data = self.build_dataset_target(dataframe, 'best_of')

        # 3- compute XT, yTs
        # xTrain, xTest, yTrain, yTest = PREPROCinstance.build_X_y_train_test(data, target)
        xTrain, xTest, yTrain, yTest = self.build_X_y_train_test(data, target)
        print('yTest %s'%str(yTest))

        return data, target, xTrain, xTest, yTrain, yTest

#====================
# main part
PREPROCinstance = PreProcLib('ATP.csv')
#data, target, xTrain, xTest, yTrain, yTest = PREPROCinstance.full_pre_proc_from_file_to_X_y_train_test('ATP.csv')
