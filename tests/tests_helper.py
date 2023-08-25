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

class TestHelperMethods(unittest.TestCase):

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


    def test_get_unspsc_label_title(self):
        assert self.helper.get_unspsc_label_title(self.selected_labels[0]) == 'Flight instrumentation'


    def test_unigrams_in_common(self):
        assert self.helper.unigrams_in_common(
            self.oa.dataset_df.PartDescription1.iloc[0], self.oa.dataset_df.PartDescription2.iloc[0]
        ) \
            == (1.0, 8, ',S,OTHER,HACKER,BAG,HIGH,80,MATL,RAWPACKAGING')


    def test_remove_stopwords(self):
        assert self.helper.remove_stopwords(
            self.oa.dataset_df.lid_preprocessed.iloc[0]
        ) \
            == 'matl rawpackaging bag hacker high matl rawpackaging bag hacker high plaster matl raw packaging bag hacker high'


    def test_remove_special_chars(self):
        assert self.helper.remove_special_chars(
            self.oa.dataset_df.PartDescription1.iloc[0]
        ) \
            == 'MATL RAWPACKAGING  OTHER  BAG  80  HACKER HIGH S'


    def test_reduce_by_unigrams(self):
        assert self.helper.reduce_by_unigrams(
            self.oa.dataset_df.PartDescription1.iloc[0]
        ) \
            == 'MATL RAWPACKAGING OTHER BAG 80 HACKER HIGH S'
