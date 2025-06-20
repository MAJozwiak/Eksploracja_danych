from merge_datasets import merge
from training import training_model
from training import metrics


merged_df = merge.merging()
model, history = training_model.train(merged_df)
metrics(history)






