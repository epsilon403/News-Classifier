from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import chromadb
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re

def etl_task():
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'http', 'https', 'amp', 'co', 'new', 'get', 'like', 'would'}
    stop_words.update(custom_stopwords)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        return " ".join(filtered_words)
    
    df_train = pd.read_csv('cleaned_ag_news_train.csv')
    df_test = pd.read_csv('cleaned_ag_news_test.csv')
    
    df_train['cleaned_text'] = df_train['text'].apply(clean_text)
    df_test['cleaned_text'] = df_test['text'].apply(clean_text)
    
    df_train.to_csv('cleaned_ag_news_train.csv', index=False)
    df_test.to_csv('cleaned_ag_news_test.csv', index=False)
    print("ETL completed")

def embedding_task():
    df_train = pd.read_csv('cleaned_ag_news_train.csv')
    df_test = pd.read_csv('cleaned_ag_news_test.csv')
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    embeddings_train = model.encode(df_train['cleaned_text'].tolist(), show_progress_bar=True)
    embeddings_test = model.encode(df_test['cleaned_text'].tolist(), show_progress_bar=True)
    
    joblib.dump(embeddings_train, 'embeddings_train.pkl')
    joblib.dump(embeddings_test, 'embeddings_test.pkl')
    joblib.dump(model, 'models/sentence_transformer.pkl')
    print("Embeddings created")

def storage_task():
    client = chromadb.PersistentClient(path='./chromadb_data')
    collection_train = client.get_or_create_collection(name='news_train')
    collection_test = client.get_or_create_collection(name='news_test')
    
    df_train = pd.read_csv('cleaned_ag_news_train.csv')
    df_test = pd.read_csv('cleaned_ag_news_test.csv')
    
    embeddings_train = joblib.load('embeddings_train.pkl')
    embeddings_test = joblib.load('embeddings_test.pkl')
    
    batch_size = 5000
    datasets = [
        (df_train, embeddings_train, collection_train, 'train'),
        (df_test, embeddings_test, collection_test, 'test')
    ]
    
    for df, embeddings, collection, name in datasets:
        embeddings_list = embeddings.tolist()
        documents_list = df['cleaned_text'].tolist()
        metadatas_list = [{'label': label} for label in df['label']]
        ids_list = [str(i) for i in range(len(df))]
        total_items = len(df)
        
        for i in range(0, total_items, batch_size):
            collection.add(
                embeddings=embeddings_list[i:i+batch_size],
                documents=documents_list[i:i+batch_size],
                metadatas=metadatas_list[i:i+batch_size],
                ids=ids_list[i:i+batch_size]
            )
        print(f"Stored {total_items} items in {name} collection")

def training_task():
    df_train = pd.read_csv('cleaned_ag_news_train.csv')
    embeddings_train = joblib.load('embeddings_train.pkl')
    
    svc = LinearSVC()
    svc.fit(embeddings_train, df_train['label'])
    
    joblib.dump(svc, 'models/vc_model.pkl')
    print("Model trained and saved")

def evaluation_task():
    df_test = pd.read_csv('cleaned_ag_news_test.csv')
    embeddings_test = joblib.load('embeddings_test.pkl')
    svc = joblib.load('models/vc_model.pkl')
    
    preds = svc.predict(embeddings_test)
    
    accuracy = accuracy_score(df_test['label'], preds)
    report = classification_report(df_test['label'], preds)
    
    print(f"Accuracy: {accuracy}")
    print(report)

with DAG(
    dag_id='news_classifier_pipeline',
    start_date=datetime(2025, 12, 12),
    schedule="@weekly",
    catchup=False,
    tags=['ml', 'pipeline']
) as dag:
    
    etl = PythonOperator(
        task_id='etl',
        python_callable=etl_task
    )
    
    embedding = PythonOperator(
        task_id='embedding',
        python_callable=embedding_task
    )
    
    storage = PythonOperator(
        task_id='storage',
        python_callable=storage_task
    )
    
    training = PythonOperator(
        task_id='training',
        python_callable=training_task
    )
    
    evaluation = PythonOperator(
        task_id='evaluation',
        python_callable=evaluation_task
    )
    
    etl >> embedding >> storage >> training >> evaluation
