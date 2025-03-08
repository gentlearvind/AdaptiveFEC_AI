from data_generator import DataGenerator
from model_trainer import ModelTrainer

def main():
    # Generate data
    data_gen = DataGenerator(num_data_points=100, file_name="impairments.csv")
    data_gen.generate_data()

    # Create and evaluate models
    model_trainer = ModelTrainer(file_name="impairments.csv")
    accuracies = model_trainer.create_model()

    # Plot accuracies
    model_trainer.plot_accuracies(accuracies)

    # Print confusion matrices
    model_trainer.print_confusion_matrices(accuracies)

if __name__ == "__main__":
    main()