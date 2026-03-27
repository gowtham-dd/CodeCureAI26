from src.DrugToxicity.config.configuration import ConfigurationManager
from src.DrugToxicity.components.Data_Validation import DataValidation
from src.DrugToxicity import logger  
import pandas as pd

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
    
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()

            validator = DataValidation(config=data_validation_config)
            is_valid  = validator.validate_dataset()

            if not is_valid:
                raise ValueError(
                    "Data validation failed — check "
                    f"{data_validation_config.STATUS_FILE} for details."
                )

            print("Data validation passed successfully")

        except Exception as e:
            print(f"Error during data validation: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
