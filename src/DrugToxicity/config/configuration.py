
from src.DrugToxicity.constant import *
from src.DrugToxicity.utils.common import read_yaml,create_directories 
from src.DrugToxicity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            final_data_file=Path(config.final_data_file)
        )

        return data_ingestion_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            data_path=Path(config.data_path)
        )
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            features_path=Path(config.features_path),
            labels_path=Path(config.labels_path),
            selector_path=Path(config.selector_path),
            scaler_path=Path(config.scaler_path)
        )
    


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([Path(config.root_dir)])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            model_bundle_path=Path(config.model_bundle_path)
        )
    



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            model_bundle_path=Path(config.model_bundle_path),
            metric_file_name=Path(config.metric_file_name),
            roc_plot_path=Path(config.roc_plot_path),
            pr_plot_path=Path(config.pr_plot_path)
        )