from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
import argparse
import joblib
import os
import pandas as pd
from src.utils import compute_metrics

def main(args):
    # Load data
    data = {}
    data['train'] = pd.read_csv(os.path.join(args.data_folder, "train.csv"))
    data['test'] = pd.read_csv(os.path.join(args.data_folder, "test.csv"))
    
    cls = Pipeline([('tfidf', TfidfVectorizer(min_df=args.min_tf, max_features=args.max_f, analyzer='char_wb', ngram_range=(args.ngram,args.ngram))),
                        ('rf', RandomForestClassifier(verbose = args.verbose*1, n_jobs=-1, n_estimators=200)),
                        ], verbose=args.verbose)

    cls.fit(list(data['train']['sequence']), list(data['train']['label'].values))
    #cls = joblib.load(os.path.join('models', "model_tfidf.pkl"))
    y_pred = cls.predict(data['test']['sequence'])
    y_true = data['test']['label'].values
    names = list(data['test']['sequence_name'].values)
    compute_metrics(y_true, y_pred, path_results='results/',  method = "TF-IDF", names=names)
    
    joblib.dump(cls, os.path.join('models', "model_tfidf.pkl"), compress = 1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="data", 
        help="data folder name")
    parser.add_argument("-pr", "--path_results", type=str, default="results",
        help = "folder to save the model")
    parser.add_argument("-v", "--verbose", type=bool, default=True,
        help = "verbose arg of the pipeline")
    parser.add_argument("-mtf", "--min_tf", type=int, default=0.05,
        help = "Minimum number of count for a word to be taken into account in tf idf")
    parser.add_argument("-mf", "--max_f", type=int,default=1500, 
	    help = "Maximum term frequency across the corpus")
    parser.add_argument("--ngram", type=int,default=3, 
	help = "number of characters in each n-gram")
    
    main(parser.parse_args())