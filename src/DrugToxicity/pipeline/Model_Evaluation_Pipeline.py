from src.DrugToxicity.config.configuration import ConfigurationManager
from src.DrugToxicity.components.Model_Evaluation import ModelEvaluation
from src.DrugToxicity import logger  
import pandas as pd


STAGE_NAME = "Data Validation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_model_evaluation_config()
            evaluator = ModelEvaluation(config=eval_config)

            if evaluator.evaluate():
                logger.info("New evaluation performed and saved.")
            else:
                logger.info("Using existing evaluation results.")

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
