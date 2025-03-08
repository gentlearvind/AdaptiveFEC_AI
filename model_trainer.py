import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ModelTrainer:
    def __init__(self, file_name="impairments.csv"):
        self.file_name = file_name

    def create_model(self):
        data = pd.read_csv(self.file_name)

        # Encode categorical variables
        label_encoder_modulation = LabelEncoder()
        label_encoder_fec = LabelEncoder()

        data['modulation_format'] = label_encoder_modulation.fit_transform(data['modulation_format'])
        data['FEC_algorithm'] = label_encoder_fec.fit_transform(data['FEC_algorithm'])

        # Split the data into features and target for modulation format prediction
        X_modulation = data.drop(columns=['modulation_format', 'FEC_algorithm'])
        y_modulation = data['modulation_format']

        # Split the data into training and testing sets for modulation format prediction
        X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(X_modulation, y_modulation, test_size=0.2, random_state=42)

        # Standardize the features for modulation format prediction
        scaler_mod = StandardScaler()
        X_train_mod = scaler_mod.fit_transform(X_train_mod)
        X_test_mod = scaler_mod.transform(X_test_mod)

        # Train and evaluate RandomForestClassifier for modulation format prediction
        rf_mod = RandomForestClassifier(random_state=42)
        rf_mod.fit(X_train_mod, y_train_mod)
        y_pred_mod_rf = rf_mod.predict(X_test_mod)
        accuracy_mod_rf = accuracy_score(y_test_mod, y_pred_mod_rf)

        # Train and evaluate MLPClassifier for modulation format prediction with increased max_iter
        mlp_mod = MLPClassifier(random_state=42, max_iter=1000)
        mlp_mod.fit(X_train_mod, y_train_mod)
        y_pred_mod_mlp = mlp_mod.predict(X_test_mod)
        accuracy_mod_mlp = accuracy_score(y_test_mod, y_pred_mod_mlp)

        # Split the data into features and target for FEC algorithm prediction
        X_fec = data.drop(columns=['modulation_format', 'FEC_algorithm'])
        y_fec = data['FEC_algorithm']

        # Split the data into training and testing sets for FEC algorithm prediction
        X_train_fec, X_test_fec, y_train_fec, y_test_fec = train_test_split(X_fec, y_fec, test_size=0.2, random_state=42)

        # Standardize the features for FEC algorithm prediction
        scaler_fec = StandardScaler()
        X_train_fec = scaler_fec.fit_transform(X_train_fec)
        X_test_fec = scaler_fec.transform(X_test_fec)

        # Train and evaluate RandomForestClassifier for FEC algorithm prediction
        rf_fec = RandomForestClassifier(random_state=42)
        rf_fec.fit(X_train_fec, y_train_fec)
        y_pred_fec_rf = rf_fec.predict(X_test_fec)
        accuracy_fec_rf = accuracy_score(y_test_fec, y_pred_fec_rf)

        # Train and evaluate MLPClassifier for FEC algorithm prediction with increased max_iter
        mlp_fec = MLPClassifier(random_state=42, max_iter=1000)
        mlp_fec.fit(X_train_fec, y_train_fec)
        y_pred_fec_mlp = mlp_fec.predict(X_test_fec)
        accuracy_fec_mlp = accuracy_score(y_test_fec, y_pred_fec_mlp)

        return {
            'accuracy_mod_rf': accuracy_mod_rf,
            'accuracy_mod_mlp': accuracy_mod_mlp,
            'accuracy_fec_rf': accuracy_fec_rf,
            'accuracy_fec_mlp': accuracy_fec_mlp,
            'y_test_mod': y_test_mod,
            'y_pred_mod_rf': y_pred_mod_rf,
            'y_pred_mod_mlp': y_pred_mod_mlp,
            'y_test_fec': y_test_fec,
            'y_pred_fec_rf': y_pred_fec_rf,
            'y_pred_fec_mlp': y_pred_fec_mlp
        }

    def plot_accuracies(self, accuracies):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([0, 1])
        y = np.array([0, 1])
        
        xpos, ypos = np.meshgrid(x, y)
        
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        
        zpos = np.zeros_like(xpos)
        
        dx = dy = 0.5
        
        dz = [accuracies['accuracy_mod_rf'], accuracies['accuracy_mod_mlp'], accuracies['accuracy_fec_rf'], accuracies['accuracy_fec_mlp']]

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=['r', 'g', 'b', 'y'])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Prediction Type')
        
        ax.set_zlabel('Accuracy')
        
        ax.set_xticks([0.25, 0.75])
        
        ax.set_xticklabels(['RandomForest', 'MLP'])
        
        ax.set_yticks([0.25, 0.75])
        
        ax.set_yticklabels(['Modulation', 'FEC'])
        
        # Save the plot to an image file
        plt.savefig("accuracy_comparison.png")
        
        plt.show()

    def print_confusion_matrices(self, accuracies):
        cm_mod_rf = confusion_matrix(accuracies['y_test_mod'], accuracies['y_pred_mod_rf'])
        cm_mod_mlp = confusion_matrix(accuracies['y_test_mod'], accuracies['y_pred_mod_mlp'])
        cm_fec_rf = confusion_matrix(accuracies['y_test_fec'], accuracies['y_pred_fec_rf'])
        cm_fec_mlp = confusion_matrix(accuracies['y_test_fec'], accuracies['y_pred_fec_mlp'])

        with open("confusion_matrix.txt", "w") as f:
            f.write("Confusion Matrix for RandomForest (Modulation Format):\n")
            f.write(np.array2string(cm_mod_rf))
            f.write("\n\nConfusion Matrix for MLP (Modulation Format):\n")
            f.write(np.array2string(cm_mod_mlp))
            f.write("\n\nConfusion Matrix for RandomForest (FEC Algorithm):\n")
            f.write(np.array2string(cm_fec_rf))
            f.write("\n\nConfusion Matrix for MLP (FEC Algorithm):\n")
            f.write(np.array2string(cm_fec_mlp))

        print("Confusion matrices have been written to confusion_matrix.txt")