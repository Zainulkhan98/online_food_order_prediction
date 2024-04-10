Terminal Commands

To insall zenml integration mlflow

zenml integration install mlflow -y

To register mlflow tracker

zenml experiment-tracker register mlflow_tracker_ev --flavor=mlflow

To deploy

zenml model-deployer register mlflow_ev --flavor=mlflow

zenml stack register mlflow_stack_ev -a default -o default -d mlflow -e mlflow_tracker_ev --set

To see the run over MLflow Dashboard

mlflow ui --backend-store-uri "<tracking_uri>"
