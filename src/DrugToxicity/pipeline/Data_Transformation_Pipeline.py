

from src.DrugToxicity.config.configuration import ConfigurationManager
from src.DrugToxicity.components.Data_Transformation import DataTransformation
from src.DrugToxicity import logger


STAGE_NAME="Data Ingestion stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            transform_config = config.get_data_transformation_config()
            transformer = DataTransformation(config=transform_config)
            transformer.transform()
        except Exception as e:
            print(f"Error during transformation: {e}")
            raise e




if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataTransformationPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
