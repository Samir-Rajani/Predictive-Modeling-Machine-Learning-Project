import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42

def main(train_file, test_file):
    train_df = pd.read_csv(train_file)
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]

    print("Initializing...")    
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=RANDOM_STATE))
    ])
    svm_param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear'],
    }
    
    f1_scorer = make_scorer(f1_score)
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring=f1_scorer)
    svm_grid.fit(X, y)
    
    print("Best SVM parameters:", svm_grid.best_params_)
    print("Best SVM F1 score:", svm_grid.best_score_)
    
    final_model = svm_grid.best_estimator_
    final_model.fit(X, y)
    
    test_df = pd.read_csv(test_file)
    X_test = test_df 
    
    predictions = final_model.predict(X_test)
    
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')
    
    print("Predictions saved to predictions.txt")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python final_model.py <train_csv_file> <test_csv_file>")
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)
