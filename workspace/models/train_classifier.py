import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score
from sklearn.model_selection  import GridSearchCV
import warnings

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
       Function:
       load data from database
       Args:
       database_filepath: the path of the database
       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
       """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    return X, Y



def tokenize(text):
    """
    Function: tokenize the text
    Args:  source string
    Return:
    clean_tokens(str list): clean string list
    
    """
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    #token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    #sterm and lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    Pipleine 1: Random Forest Classifier
    """
    pipeline_RF = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
      ])
    
    parameters_RF = {
        'vect__max_df': [1.0,2.0],
        'tfidf__use_idf': [True, False],
        'clf__estimator__criterion': ['gini'],
        'clf__estimator__n_estimators': [10,20],
        'clf__estimator__min_samples_split':[2,5]
        }

    cv_RF = GridSearchCV(pipeline_RF, param_grid = parameters_RF, verbose=True,cv=3,n_jobs = -1)
    return cv_RF



def get_results(Y_test, model, X_test):
    y_pred=model.predict(X_test)
    report = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for colnm in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[colnm], y_pred[:,num], average='weighted')
        report.at[num+1, 'Category'] = colnm
        report.at[num+1, 'f_score'] = f_score
        report.at[num+1, 'precision'] = precision
        report.at[num+1, 'recall'] = recall
        num += 1
    print('Aggregated f_score:', report['f_score'].mean())
    print('Aggregated precision:', report['precision'].mean())
    print('Aggregated recall:', report['recall'].mean())
    print('Accuracy:', np.mean(Y_test.values == y_pred))
    return report   


def save_model(model, model_filepath):
    
    # Create a pickle file for the model
    with open (model_filepath, 'wb') as f:
        
        pickle.dump(model, f)


def main():

    
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        get_results(Y_test, model,X_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()