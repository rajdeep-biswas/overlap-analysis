Overlap analysis is an experiment that was performed with aim of understanding various class interactions, and how any preprocessing performed on data might affect the accuracy of DAR on said dataset. It was demonstrated that by following a certain series of textual preprocessing, the overlap percentage of ~95% of a given set of class pairs will go down, while the other 5% can go up. Recommendations are in the form of an ordered series of preprocessing steps and cleaning out a custom set of stopwords.  

This experiment is intended to be run iteratively. The following set of classes (bottom 200 in terms of frequency) were used to get the initial set of custom stopwords. 

All you need to do is run the following code -

```python3
from overlap_analysis import OverlapAnalysis
oa = OverlapAnalysis(oa_dir_path = '../Data/OA/')
oa.run_overlap_analysis()
```

You can also optionally pass in selected labels of your choice, only on which you'd like to run a more constrained overlap analysis -

```python3
oa = OverlapAnalysis(oa_dir_path = '../Data/OA/', selected_labels = [23151800, 23151700, 72110000, 72120000, 23271700, 23271800, 23301500, 23121600, 40101700, 42241600])

# this example is of top 10 labels that have highest title similarity
```

#### Make sure to have these files (under `oa_dir_path`) before running the code -
1. `train_data_initial_invoice_sample.csv`
2. `UNSPSC Comparison v905 vs v23.xlsx`  

Each of these files are linked at the runbook (linked above).

Within a directory `Data/OA` (this can be easily changed by modifying the `oa_dir_path` value at the constructor call).  
`Data/` should be at the same level as `src/`.

Directory structure -
```shell
src/
data/
-- dataset_csv (input)
-- label_titles_csv (input)
-- overlap_scores/ (output)
    -- [overlap_11101700_12141700.csv, # overlap_classA_classB
        ...
        ]
-- label_frequency_csv (intermediate)
-- label_similarity_csv (intermediate)
```
