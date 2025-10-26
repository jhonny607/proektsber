import pandas as pd
from preprocessor import Preprocessor
from module import TrainModel

df = pd.read_csv('train.csv')
preprocessor = Preprocessor(df)
preprocessor_run = preprocessor.run_pipeline()

test_cols = ['price_doc','timestamp']
feature_cols = [col for col in preprocessor_run.columns if col not in test_cols]
X = preprocessor_run[feature_cols]
y = preprocessor_run['price_doc']
trainer = TrainModel(X,y)
trainer.train()
