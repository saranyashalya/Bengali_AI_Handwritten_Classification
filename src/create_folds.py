import pandas as pd
import pyarrow
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("/Users/egovsar/Personal/Bengali_AI_Handwritten_Grapheme_Classification/input/bengaliai-cv19/train.csv")
    print(df.head())
    df.loc[:,'kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)
    
    X = df.image_id.values
    y = df[["grapheme_root","vowel_diacritic","consonant_diacritic"]].values
    
    mskf = MultilabelStratifiedKFold(n_splits= 5)
    
    for fold , (train_, val_) in enumerate(mskf.split(X,y)):
        print("Train: ", train_, "val: ", val_)
        df.loc[val_,"kfold"] = fold
    
    print(df.kfold.value_counts())
    df.to_csv("/Users/egovsar/Personal/Bengali_AI_Handwritten_Grapheme_Classification/input/train_folds.csv", index = False)
    p1 = pd.read_parquet("/Users/egovsar/Personal/Bengali_AI_Handwritten_Grapheme_Classification/input/bengaliai-cv19/train_image_data_0.parquet")
    print(p1.head())