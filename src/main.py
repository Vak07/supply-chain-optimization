from src.pipelines.ingestion_pipeline import run_ingestion_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.pipelines.evaluation_pipeline import run_evaluation_pipeline

def main():
    # Step 1: Ingestion
    print("Running Data Ingestion Pipeline...")
    demand_data, inventory_data = run_ingestion_pipeline()

    # Step 2: Training
    print("Running Training Pipeline for LSTM...")
    run_training_pipeline()

    # Step 3: Evaluation
    print("Running Evaluation Pipeline...")
    run_evaluation_pipeline()

if __name__ == "__main__":
    main()
