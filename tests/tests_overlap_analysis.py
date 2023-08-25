"""
The `test_data` folder should have the two files -
1. train_data_initial_invoice_NA_20210629_updated_05_05_2022.csv.
2. !UNSPSC Comparison v905 vs v23.xlsx.

The `validation_data` folder should have the two files -
1. label_frequency.csv.
2. unspsc_title_similarity.csv.

Do NOT change these files or the contents of these files,
they are necessary to be consistent with the hardcoded unit tests.
"""

import os
import unittest
import pandas as pd

from helper import OverlapAnalysisHelpers
from overlap_analysis import OverlapAnalysis

class TestOAMethods(unittest.TestCase):

    '''
    Overlap analysis unit tests.
    '''

    selected_labels = [25202100, 30111700]

    test_data_dir = './test_data'
    validation_data_dir = './validation_data'

    validation_label_frequency_df = pd.read_csv(os.path.join(validation_data_dir, 'label_frequency.csv'))
    validation_unspsc_title_similarity_df = pd.read_csv(os.path.join(validation_data_dir, 'unspsc_title_similarity.csv'))

    test_label_frequency_df = None
    test_unspsc_title_similarity_df = None

    oa = OverlapAnalysis(
        oa_dir_path = test_data_dir,
        dataset_path = 'train_data_initial_invoice_NA_20210629_updated_05_05_2022.csv',
        selected_labels = selected_labels
    )

    helper = OverlapAnalysisHelpers(oa.title_dfs)


    def __init__(self):
        self.oa.dataset_df = self.helper.preprocess_lids(self.oa.dataset_df)


    def test_generate_label_frequency(self):
        self.test_label_frequency_df = \
            self.oa.label_frequency_df = \
            self.oa.generate_label_frequency()

        assert not False in (
            self.test_label_frequency_df \
                == self.validation_label_frequency_df.drop(['Unnamed: 0'], axis = 1)
        ).all()


    def test_generate_interest_pairs(self):
        self.test_unspsc_title_similarity_df = self.oa.helper.generate_interest_pairs(
            os.path.join(self.test_data_dir, 'unspsc_title_similarity.csv'),
            self.test_label_frequency_df
        )
        assert not False in (
            self.test_unspsc_title_similarity_df.reset_index(drop = True) \
                == self.validation_unspsc_title_similarity_df.drop(['Unnamed: 0'], axis = 1
        ).reset_index(drop = True)).all()
