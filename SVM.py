import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('train.csv', index_col=0)
df_test = pd.read_csv('test.csv', index_col=0)
df_val = pd.read_csv('val.csv', index_col=0)

# Load SBERT model
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Get SBERT embeddings
def get_sbert_embeddings(sentences):
    embeddings = sbert_model.encode(sentences)
    return embeddings

def prepare_data(df):

   X_0, X_1, y = [], [], []
   
   selected_cols = ['referent' if i % 2 == 0 else 'wrong_referent' for i in range(len(df))]
   for i, row in tqdm(df.iterrows(), total=len(df)):

      ref = row[selected_cols[i]]
      label = 1 if i % 2 == 0 else 0
      sent1, sent2, pro = row[['sentence1', 'sentence2', 'pronoun']]
      sent2 = sent2.replace(pro, 'The ' + ref)
      
      sent1_arr = [get_sbert_embeddings(t) for t in sent1.split()]   
      sent2_arr = [get_sbert_embeddings(t) for t in sent2.split()]   

      emb1_add = np.sum(sent1_arr, axis=0)
      emb2_add = np.sum(sent2_arr, axis=0)

      s = sent1 + '. ' + sent2 + '.'
      # print(s)
      
      method_full = get_sbert_embeddings(s)
      method_add =  emb1_add + emb2_add
      
      X_0.append(method_full)
      X_1.append(method_add)
       
      y.append(label)
      
   return X_0, X_1, y
    
def run_SVM(X_train, y_train, X_val, y_val, X_test, y_test):
    # Step 4: Hyperparameter Tuning with Validation Set
    param_grid = {
        'C': [0.1, 1, 10],           # Regularization parameter
        'kernel': ['linear', 'rbf']  # Kernel type
    }

    svm_classifier = svm.SVC()
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    # Get the best SVM model with tuned hyperparameters
    best_svm = grid_search.best_estimator_

    # Step 8: Model Evaluation on Validation Set (optional)
    y_val_pred = best_svm.predict(X_val)

    # Step 9: Model Evaluation on Test Set
    y_test_pred = best_svm.predict(X_test)

    # Evaluate the model's performance on the validation set
    validation_report = classification_report(y_val, y_val_pred, digits=5)
    print("Validation Report:")
    print(validation_report)

    # Evaluate the model's performance on the test set
    test_report = classification_report(y_test, y_test_pred, digits=5)
    print("\nTest Report:")
    print(test_report)
    
    print(best_svm)

X_0, X_1, y_1 = prepare_data(df_train)
X_00, X_11, y_11 = prepare_data(df_val)
X_000, X_111, y_111 = prepare_data(df_test)

print('METHOD FULL')
run_SVM(X_0, y_1, X_00, y_11, X_000, y_111)

print('METHOD ADD')
run_SVM(X_1, y_1, X_11, y_11, X_111, y_111)