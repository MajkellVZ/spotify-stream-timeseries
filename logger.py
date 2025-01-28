import mlflow

class MlflowLogger:
   def __init__(self, experiment_name='forecasting', tracking_uri="http://127.0.0.1:5000"):
       self.mlflow = mlflow
       self.mlflow.set_tracking_uri(uri=tracking_uri)
       self.mlflow.set_experiment(experiment_name)
       self.active_run = None

   def start_run(self, run_name=None):
       self.active_run = self.mlflow.start_run(run_name=run_name)
       return self.active_run

   def log_param(self, key, value):
       if not self.active_run:
           self.start_run()
       self.mlflow.log_param(key, value)

   def log_params(self, params_dict):
       if not self.active_run:
           self.start_run()
       self.mlflow.log_params(params_dict)

   def log_metric(self, key, value):
       if not self.active_run:
           self.start_run()
       self.mlflow.log_metric(key, value)

   def log_metrics(self, metrics_dict):
       if not self.active_run:
           self.start_run()
       self.mlflow.log_metrics(metrics_dict)

   def log_model(self, model, model_name):
       if not self.active_run:
           self.start_run()
       self.mlflow.sklearn.log_model(model, model_name)

   def register_model(self, model_name):
       if not self.active_run:
           self.start_run()
       self.mlflow.register_model(f"runs:/{self.active_run.info.run_id}/{model_name}", model_name)

   def log_artifact(self, path):
       if not self.active_run:
           self.start_run()
       self.mlflow.log_artifact(path)

   def end_run(self):
       if self.active_run:
           self.mlflow.end_run()
           self.active_run = None

   def __enter__(self):
       self.start_run()
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.end_run()