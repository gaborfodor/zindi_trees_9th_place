import os
import pandas as pd
from pathlib import Path



os.makedirs('subs', exist_ok=True)

models_dir = Path("./output/models")
exps = [
    'e4_2',
    'e4_5',
    'e5_1',
    'e5_2',
    'cfg_3',
]

fs = []
for exp in exps:
    fs += list(models_dir.glob(f"{exp}/*/test_pred_df.csv"))

print(os.listdir(models_dir))

test_preds = pd.concat([pd.read_csv(f) for f in fs])


# Simple blend
subm = test_preds.groupby('ImageId').pred_Target.mean().reset_index()
subm.columns = ["ImageId" , "Target"]
subm.to_csv(f'subs/e5_{len(fs)}.csv', index=False)

print(subm.head())




