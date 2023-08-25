"""
Helper methods that are used in preprocessing text for overlap analysis.
"""

import json
import os
import re
import string

import pandas as pd

from nltk.corpus import stopwords
from wordlist.wordlist import populate_custom_stopwords

from rapidfuzz import fuzz

from dateparser.search import search_dates


class OverlapAnalysisHelpers:

    """
    Helper methods that are used in preprocessing text for overlap analysis.
    """

    title_dfs = pd.DataFrame(),
    stop_words = [],
    preprocess_timestamps = False,
    preprocess_lower = True,
    preprocess_stopwords = True,
    preprocess_special_characters = True,
    preprocess_digits = True,
    preprocess_whitespace = True


    def __init__(self, title_dfs: pd.DataFrame, preprocess_timestamps = False):

        self.title_dfs = title_dfs
        self.stop_words = list(set(stopwords.words('english')))
        self.stop_words.extend(populate_custom_stopwords())
        self.preprocess_timestamps = preprocess_timestamps




    def get_unspsc_label_title(self, unspsc_code: int) -> str: # train_data_initial_invoice

        '''
        Takes in 8-digit UNSPSC code (str or numeric type),
        refers to external CSV and returns label title if found.
        Returns 'Unknown' if invalid UNSPSC code is passed in
        or external CSV does not have a match for a semantically valid code.
        '''

        title = 'Unknown'

        for source_df in self.title_dfs:
            source_df = self.title_dfs[source_df]
            if len(source_df[source_df.Code == unspsc_code].Title):
                title = source_df[source_df.Code == unspsc_code].Title.values[0]

        return title



    def unigrams_in_common(
        self,
        lid_1: str,
        lid_2: str
    ) -> [float, int, list]:

        '''
        Custom method that returns -
        1. average similarity score (common count divided by total count).
        2. absolute similarity score (common count).
        3. list of common unigrams.
        between two preprocessed LIDs.
        '''

        unigrams_1 = list(set(lid_1.strip().split(' ')))
        unigrams_2 = list(set(lid_2.strip().split(' ')))

        common_unigrams = list(set(unigrams_1).intersection(unigrams_2))

        absolute_unigram_score = len(common_unigrams)
        average_unigram_score = absolute_unigram_score / ((len(unigrams_1) + len(unigrams_2)) / 2)

        return average_unigram_score, absolute_unigram_score, ','.join(common_unigrams)



    def remove_stopwords(self, text):

        '''
        Custom function to remove stopwords including custom stopwords.
        '''

        return " ".join([word for word in str(text).split() if word not in self.stop_words])



    def replace_timestamps(self, lid_string):

        '''
        If a timestamp is found in a LID, replace the first instance with placeholder,
        replace any further occurrence(s) with a single whitespace.
        '''

        placeholder = '<timestamp>'

        found_timestamps = search_dates(lid_string)
        if not found_timestamps:
            return lid_string

        found_timestamps = [item[0] for item in search_dates(lid_string)]
        lid_string = lid_string.replace(found_timestamps[0], placeholder)

        for timestamp in found_timestamps:
            lid_string = lid_string.replace(timestamp, ' ')

        return lid_string



    def preprocess_lids(self, dataset_df: pd.DataFrame) -> None: # oa_2; sampling_6.4m

        '''
        Add two columns to self.dataset_df;
        one that uses simple concatenation of all line item descriptions
        and another that uses special sanitization to reduce string to only common unigrams.
        '''

        print("preprocessing LIDs.")
        # standard preprocessing; replacing nan rows with empty characters
        dataset_df['PartDescription1'] = dataset_df['PartDescription1'].fillna('')
        dataset_df['PartDescription2'] = dataset_df['PartDescription2'].fillna('')
        dataset_df['InvoiceDescription'] = dataset_df['InvoiceDescription'].fillna('')
        dataset_df['PODescription'] = dataset_df['PODescription'].fillna('')

        # creating two new columns; first where the individual
        # columns are simply concatenated one after another.
        dataset_df['lid_concatenated'] = \
            dataset_df['PartDescription1'] + ' ' + \
            dataset_df['PartDescription2'] + ' ' + \
            dataset_df['InvoiceDescription'] + ' ' + \
            dataset_df['PODescription']

        # the second column uses a more involved method of
        # reducing strings to their unique unigrams
        dataset_df['lid_reduced'] = dataset_df['lid_concatenated'].apply(self.reduce_by_unigrams)

        # the following lines of code can be disabled one each,
        # depending upon the amount of preprocessing that is done on text before overlap analysis.
        # please note that the preserving the provided order of preprocessing is important.
        # switching it up might bring variations in quality of results.

        # lower string
        if self.preprocess_lower:
            dataset_df['lid_preprocessed'] = dataset_df['lid_concatenated'].str.lower()

        # remove stopwords
        if self.preprocess_stopwords:
            dataset_df['lid_preprocessed'] = dataset_df['lid_preprocessed'] \
                .apply(self.remove_stopwords)

        # replace all punctuation with a single space
        if self.preprocess_special_characters:
            dataset_df['lid_preprocessed'] = dataset_df['lid_preprocessed'] \
                .apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), ' ' , text))

        # replace all (recognizable) timestamp values with a placeholder '<timestamp>'.
        if self.preprocess_timestamps:
            dataset_df['lid_preprocessed'] = dataset_df['lid_preprocessed'] \
                .progress_apply(self.replace_timestamps)

        # remove all numbers and words containing digits
        if self.preprocess_digits:
            dataset_df['lid_preprocessed'] = dataset_df['lid_preprocessed'] \
                .apply(lambda text: re.sub(r'\w*\d\w*', '', text))

        # replace multiple spaces with a single space
        if self.preprocess_whitespace:
            dataset_df['lid_preprocessed'] = dataset_df['lid_preprocessed'] \
                .apply(lambda text: re.sub(' +', ' ', text))

        # TODO: NER and remove peoples' / companies' names.
        # TODO: stemming / lemmatization to handle {include, includes, including, included} etc.

        print("done preprocessing LIDs.")

        return dataset_df



    def compute_similarity(
        self,
        lid_1: str,
        lid_2: str,
        similarity_metric: str = 'spacy'
    ) -> float: # overlap_analysis

        '''
        Takes in pair of strings and uses a predefined method to calculate similarity.
        '''

        if similarity_metric == 'spacy':

            return self.get_spacy_similarity(lid_1, lid_2)



    def remove_special_chars(self, lid_string: str, specials: str = ',.-') -> str:

        '''
        Replaces parameterized (a specified list of) special characters in a string
        with a single whitespace.
        '''

        for special in specials:
            lid_string = lid_string.replace(special, ' ')

        return lid_string



    def reduce_by_unigrams(
        self,
        # list of LID strings: PartDescription1, PartDescription, InvoiceDescription, PODescription
        lid_string: str
    ) -> str: # print_uni, unigramsInCommon, oa_2; sampling_6.4m

        '''
        Splits a string by a list of provided special characters,
        then reduces strings to unique unigrams.
        '''

        if isinstance(lid_string, float):
            return ''

        unigrams = []

        lid_string = self.remove_special_chars(lid_string)
        for word in lid_string.split():
            if word not in unigrams:
                unigrams.append(word)

        return ' '.join(unigrams)



    def update_unigram_count_json(self, unigrams_list: dict, unigram_count_json_path: str):

        '''
        Takes in a list of strings of comma separated unigrams, parses through each string to get unigrams
        and populates the frequency of each into a JSON file.
        '''

        if os.path.isfile(unigram_count_json_path):
            with open(unigram_count_json_path) as json_file:
                unigram_count_dict = json.load(json_file)
        else:
            unigram_count_dict = {
                'single_row': {},
                'independent': {}
            }

        for unigrams_string in unigrams_list:

            if ',' in unigrams_string:
                if unigrams_string in unigram_count_dict['single_row']:
                    unigram_count_dict['single_row'][unigrams_string] += 1
                else:
                    unigram_count_dict['single_row'][unigrams_string] = 1

            else:
                if unigrams_string in unigram_count_dict['independent']:
                    unigram_count_dict['independent'][unigrams_string] += 1
                else:
                    unigram_count_dict['independent'][unigrams_string] = 1

        with open(unigram_count_json_path, 'w') as json_file:
            json.dump(unigram_count_dict, json_file)



    def generate_interest_pairs(
        self,
        unspsc_title_similarity_path: str,
        label_frequency_df: pd.DataFrame
    ) -> pd.DataFrame:

        '''
        Uses self.label_frequency_df to list all possible pairs of classes
        and computes similarity between them to populate self.unspsc_title_similarity_df.
        '''

        def get_spacy_similarity(lid_1: str, lid_2: str) -> float:

            '''
            Uses spacy's builtin similarity scores.
            '''

            return self.spacy_nlp(lid_1).similarity(self.spacy_nlp(lid_2))



        def get_rapidfuzz_similarity(lid_1: str, lid_2: str) -> float:

            '''
            Uses rapidfuzz's's builtin similarity scores.
            '''

            return round(fuzz.token_sort_ratio(lid_1, lid_2))



        if os.path.isfile(unspsc_title_similarity_path):
            print("label similarity file found.")
            return pd.read_csv(unspsc_title_similarity_path)

        print("label similarity file not found. generating.")

        index = pd.MultiIndex.from_product(
            [label_frequency_df.label, label_frequency_df.label],
            names = ["label_1", "label_2"]
        )
        crossed_labels = pd.DataFrame(index = index).reset_index()

        index = pd.MultiIndex.from_product([
                label_frequency_df.label_title, label_frequency_df.label_title
            ],
            names = ["label_title_1", "label_title_2"]
        )
        crossed_titles = pd.DataFrame(index = index).reset_index()

        unspsc_title_similarity_df = pd.concat([crossed_labels, crossed_titles], axis = 1)
        unspsc_title_similarity_df = unspsc_title_similarity_df[
            unspsc_title_similarity_df.label_1 != unspsc_title_similarity_df.label_2
        ]

        # drop (b, a) values when (a, b) already exist
        # unspsc_title_similarity_df = unspsc_title_similarity_df.sort_values('label_1')
        unspsc_title_similarity_df = unspsc_title_similarity_df[
            unspsc_title_similarity_df['label_1'] < unspsc_title_similarity_df['label_2']
        ]

        unspsc_title_similarity_df['titles_similarity_score'] = \
            unspsc_title_similarity_df[['label_title_1', 'label_title_2']] \
            .progress_apply(lambda x: get_rapidfuzz_similarity(*x), axis = 1)

        unspsc_title_similarity_df.to_csv(unspsc_title_similarity_path)

        print("label similarity file generated and saved as CSVfile to: " \
            + unspsc_title_similarity_path)

        return unspsc_title_similarity_df
