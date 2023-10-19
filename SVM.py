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
    
def tune_and_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    # Define the hyperparameters grid
    param_grid = {
        'C': [0.1, 1, 10],           # Regularization parameter
        'kernel': ['linear', 'rbf']  # Kernel type
    }

    best_svm = None
    best_score = 0

    # Perform hyperparameter tuning and evaluation in a loop
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            # Initialize SVM classifier
            svm_classifier = SVC(C=C, kernel=kernel, gamma='scale', probability=True, cache_size=2000)

            # Train the SVM classifier
            svm_classifier.fit(X_train, y_train)

            # Evaluate on the validation set
            y_val_pred = svm_classifier.predict(X_val)
            score = accuracy_score(y_val, y_val_pred)

            # Update the best model if this configuration has a higher accuracy
            if score > best_score:
                best_score = score
                best_svm = svm_classifier

    # Evaluate the best model on the test set
    y_test_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Return the best model, its test set accuracy, and F1-score
    return best_svm, accuracy, f1
    
X_0, X_1, y_1 = prepare_data(df_train)
X_00, X_11, y_11 = prepare_data(df_val)
X_000, X_111, y_111 = prepare_data(df_test)

# Define the range for methods
methods = range(0, 1)

# Create a dictionary to store the best models and their metrics
best_models = {}

for method in methods:
    print(f'METHOD {method}')
    
    # Prepare the data for the current method
    X_train = globals()[f'X_{method}']
    y_train = y_1
    X_val = globals()[f'X_{method}{method}']
    y_val = y_11
    X_test = globals()[f'X_{method}{method}{method}']
    y_test = y_111
    
    # Tune and evaluate the SVM for the current method
    best_svm, test_accuracy, test_f1 = tune_and_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test)

    # Print the best SVM model parameters
    print("Best SVM Model Parameters:")
    print(best_svm)

    # Print the test set accuracy and F1-score
    print(f"Best SVM Model - Test Accuracy: {test_accuracy:.5f}")
    print(f"Best SVM Model - Test F1-Score: {test_f1:.5f}")

    # Save the best model
    joblib.dump(best_svm, f'method{method}.pkl')
    
