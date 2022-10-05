import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logger")

import mlflow


import docker
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"



###
def port_mapping_dict_from_str(port_mapping_str):
    if port_mapping_dict_from_str:
        host_port, container_port = port_mapping_str.split(":")
        return {container_port: host_port}

def check_port_mapping(container_port_mapping, target_port):
    """Checks if given container_port_mapping has already the target_port in use

    Args:
        container_port_mapping (dict): the container port dict
        target_port (str): the target port

    Returns:
        Bool: True if already in use, else False
    """
    port_mapping_list = list(container_port_mapping.items())
    if len(port_mapping_list) > 0:
        return any([True if ((_mapping[1] is not None) and (target_port == _mapping[1][0]['HostPort'])) else False for _mapping in port_mapping_list])
    else:
        return False



def find_best_model(
    mlflow_tracking_uri: str,
    mlflow_experiment_name:str, 
    mlflow_metric_selector:str):

    mlflow_model_uri_overwrite = os.getenv("MODEL_OVERWRITE")
    if mlflow_model_uri_overwrite is not None:
        logger.warn(f"mlflow model uri overwriten: {mlflow_model_uri_overwrite} ... does not search for best model")
        return mlflow_model_uri_overwrite
    
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    logger.info(f"mlflow on: {mlflow_tracking_uri}")
    mlflow_experiment =  mlflow.get_experiment_by_name(name=mlflow_experiment_name)

    if mlflow_experiment is None:
        msg = f"Could not find experiment with name '{mlflow_experiment_name}' on server '{mlflow_uri}'"
        logger.error(msg)
        raise Exception(msg)
        return None

    mlflow.set_experiment(experiment_name=mlflow_experiment_name)
    logger.info(f"mlflow experiment activated: {mlflow_experiment_name} (id={mlflow_experiment.experiment_id})")
    
    mlflow_runs = mlflow.list_run_infos(experiment_id=mlflow_experiment.experiment_id, order_by=[mlflow_metric_selector])
    mlflow_runs_successful = [run for run in mlflow_runs if run.status == 'FINISHED']
    mlflow_best_run = mlflow_runs_successful[0]
    logger.info(f"best run by '{mlflow_metric_selector}' has run_id '{mlflow_best_run.run_id}'")
    mlflow_model_uri = f"s3://mlflow/{mlflow_best_run.experiment_id}/{mlflow_best_run.run_id}/artifacts/model"

    # check generated artifacts root by run info
    mlflow_best_run_obj = mlflow.get_run(mlflow_best_run.run_id)
    if not mlflow_best_run_obj.info.artifact_uri in mlflow_model_uri:
        msg = f"generated model uri '{mlflow_model_uri}' does not contain the artifact root path '{mlflow_best_run_obj.info.artifact_uri}' from run with id '{mlflow_best_run_obj.info.run_id}'"
        logger.error(msg)
        raise Exception(msg)
        return None

    logger.info(f"generated 'model_uri' to deploy: '{mlflow_model_uri}' for run_id '{mlflow_best_run.run_id}'")
    return mlflow_model_uri
    

def deploy_model(model_url):
    ai_service_tag = "ai-service:latest"
    ai_service_container_name = "someName"
    ai_service_port = "7476:7476"
    ai_service_network = "mlflow_network"
    force_replace = True

    if model_url is None:
        msg = f"'model_url' is None. Can not deploy! Aborting..."
        logger.error(msg)
        raise Exception(msg)
        return None

    logger.info(f"model to deploy: {model_url}")

    client = docker.from_env()
    logger.info(f"currently '{len(client.containers.list())}' docker containers are running on host")

    previous_ai_service_containers = [container for container in client.containers.list(all=True) if check_port_mapping(container.ports,"7476") and container.image==client.images.get(ai_service_tag) and container.name==ai_service_container_name]
    
    if len(previous_ai_service_containers) > 0:
        logger.warning(f"currently '{len(previous_ai_service_containers)}' '{ai_service_tag}' containers are running on host")

        if not force_replace:
            msg = f"cannot proceed since running container will block procedure! Aborting..."
            logger.error(msg)
            raise Exception(msg)
            return None
        else:
            for _container in previous_ai_service_containers:
                logger.warning(f"shutting down container '{_container.name}({_container.short_id})', since flag 'force_replace' is set to True")
                _container.remove(force=True)

    logger.info(f"Building AI Service image: {ai_service_tag} ...")
    image, err = client.images.build(
        path="./deployment_container",
        dockerfile="Dockerfile",
        tag=ai_service_tag,
        rm=True
    )
    logger.info(f"Building AI Service image successful")
    
    container_environment = [
            f"MODEL_URL={model_url}",
            f"MODEL_PORT={ai_service_port.split(':')[1]}"
    ]
    logger.info(f"Prepared deployment config: {container_environment}")
    logger.info(f"Running container '{ai_service_container_name}' with image '{ai_service_tag}' on port '{ai_service_port}' in network '{ai_service_network}'")
    _container = client.containers.run(
        image=ai_service_tag,
        auto_remove=False,
        detach=True,
        name=ai_service_container_name,
        ports=port_mapping_dict_from_str(ai_service_port),
        environment=container_environment,
        network=ai_service_network
        )
    logger.info(f"successfully started container '{_container.name}({_container.short_id})'!")
    return f"{_container.name}-{_container.short_id}"


if __name__ == '__main__':
    mlflow_model_uri = find_best_model(mlflow_tracking_uri="http://localhost:5002", mlflow_experiment_name="wine", mlflow_metric_selector="metric.rmse ASC")
    data = deploy_model(mlflow_model_uri)