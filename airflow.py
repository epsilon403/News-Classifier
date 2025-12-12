from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import chromadb
import joblib
from sklearn.svm import SVC

def load_data_split():
    client = chromadb.PersistentClient(path='./chromadb_data')
    collection_train = client.get_or_create_collection(name='news_train')
    collection_test = client.get_or_create_collection(name='news_test')
    all_data = collection_train.get(
        include=['embeddings', 'metadatas', 'documents']    
    )
    return all_data

def train_model():
    data = load_data_split()
    embeddings_train = data['embeddings']
    labels_train = [meta['label'] for meta in data['metadatas']]
    
    model = SVC()
    model.fit(embeddings_train, labels_train)
    
    joblib.dump(model, 'svc_model.pkl')
    print("Model trained and saved successfully")

with DAG(
    dag_id='training_model',
    start_date=datetime(2025, 12, 12),
    schedule="@weekly",
    catchup=False,
    tags=['training', 'ml']
) as dag:
    task1 = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
