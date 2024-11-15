from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from airflow.models import XCom
import logging
import numpy as np

# Define the DAG and default arguments
default_args = {
    'owner': 'suvarna',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Define task functions
def load_and_preprocess_data():
    try:
        # Load the dataset
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # Split the dataset into features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Train/test split and scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Return arrays as lists
        return XCom.push(key="data", value=(X_train, X_test, y_train, y_test))
    
    except Exception as e:
        logging.error(f"Error in loading or preprocessing data: {e}")
        raise e


def train_and_evaluate_model(model_type, **kwargs):
    try:
        # Retrieve data from XCom and convert back to NumPy arrays
        ti = kwargs['ti']
        X_train = np.array(ti.xcom_pull(key='X_train', task_ids='load_and_preprocess_data'))
        X_test = np.array(ti.xcom_pull(key='X_test', task_ids='load_and_preprocess_data'))
        y_train = np.array(ti.xcom_pull(key='y_train', task_ids='load_and_preprocess_data'))
        y_test = np.array(ti.xcom_pull(key='y_test', task_ids='load_and_preprocess_data'))

        # Train and evaluate the model (same as before)
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type")

        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"{model_type.title()} Model Accuracy: {accuracy:.4f}")

        # Push model to XCom
        ti.xcom_push(key=model_type, value=model)

        return model
    except Exception as e:
        logging.error(f"Error in training or evaluating {model_type} model: {e}")
        raise e

from airflow.utils.dates import days_ago

# Define the DAG
with DAG('ml_pipeline_dag', start_date=days_ago(1)) as dag:
    load_preprocess_task = PythonOperator(
        task_id='load_and_preprocess_data',
        python_callable=load_and_preprocess_data,
        do_xcom_push=True    
    )

    with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='A simple ML pipeline in Airflow',
    schedule_interval='@daily',
    start_date=datetime(2024, 11, 14),
    catchup=False
) as dag:

        # Task 1: Load and preprocess data
        load_preprocess_task = PythonOperator(
            task_id='load_and_preprocess_data',
            python_callable=load_and_preprocess_data
        )

        # Task 2: Train logistic regression model
        train_log_reg_task = PythonOperator(
            task_id='train_logistic_regression',
            python_callable=train_and_evaluate_model,
            op_args=['logistic'],
        )

        # Task 3: Train random forest model
        train_rf_task = PythonOperator(
            task_id='train_random_forest',
            python_callable=train_and_evaluate_model,
            op_args=['random_forest'],
        )


        

        # Task dependencies
        load_preprocess_task >> [train_log_reg_task, train_rf_task]
        [train_log_reg_task, train_rf_task] 
