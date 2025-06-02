from merge_datasets import merge
from training import training_model

merged_df = merge.merging()
training_model.train(merged_df)
