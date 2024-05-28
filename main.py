import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score,roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import label_binarize
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException,File,UploadFile
from fastapi.staticfiles import StaticFiles
import asyncio
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
from pathlib import Path
import logging
import numpy as np

df= None
X_train= None
X_test= None 
y_train= None 
y_test = None
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_logger")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")



def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


@app.get("/")
def read_root():
    html_path = Path("static") / "index.html"
    return FileResponse(html_path)

csv_file_path = 'car_eval_dataset.csv'

# @app.on_event("startup")
# async def startup_event():
#     global df
#     # Read the CSV file during startup
#     try:
#         df = pd.read_csv(csv_file_path)
#     except FileNotFoundError:
#         df = None

@app.get("/load-data")
def load_data():
    global df
    if df is None:
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            return JSONResponse(content={"message": "CSV file not found"}, status_code=404)
    return JSONResponse(content={"message": "CSV file loaded successfully", "data": df.to_dict(orient="records")})
    

@app.get("/prep-for-training")
def prep_for_training():
    global df, X_train, X_test, y_train, y_test
    try:
        if df is not None:
            encoded_data = pd.get_dummies(df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

            features = encoded_data.drop('class', axis=1)
            target = encoded_data['class']

            X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
            X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
            

            logger.info("Data preparation completed successfully")

            return JSONResponse(content={"message": "Preparation for training completed successfully"})
        else:
            return JSONResponse(content={"message": "No data loaded. Upload a CSV file first."}, status_code=400)
    except Exception as e:
        logger.error(f"An unexpected error occurred during preparation for training: {str(e)}")
        return JSONResponse(content={"message": f"An unexpected error occurred during preparation for training: {str(e)}"}, status_code=500)

@app.get("/perform-logistic-regression")
def perform_logistic_regression():
    try:
        global df, X_train, X_test, y_train, y_test

        logistic_model = LogisticRegression(max_iter=10000)
        logistic_model.fit(X_train, y_train)

        weight_file = "model_weights_logistic.joblib"
        joblib.dump(logistic_model, weight_file)

        loaded_model = joblib.load(weight_file)
        test_predictions = loaded_model.predict(X_test)

        accuracy = np.sum(y_test.values == test_predictions) / len(y_test.values)

        print(f"Logistic accuracy: {accuracy}")

        predictions_df = pd.DataFrame({'Actual Output': y_test, 'Predicted Output': test_predictions})
        predictions_df.to_csv('logistic_regression_predictions.csv', index=False)

        message = f"Logistic Regression applied successfully. Model weights saved in 'model_weights_logistic.joblib'. Accuracy: {accuracy}"
        return JSONResponse(content={"message": message, "accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": f"Something wrong happened: {e}"})


@app.get("/confusion-matrix")
def get_confusion_matrix(model_name : str):
    global df, X_train, X_test, y_train, y_test

    if model_name == "LogisticRegression":
        loaded_model = joblib.load("model_weights_logistic.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)

        plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title='Confusion Matrix')
        plt.savefig('logistic_confusion_matrix_image.png')
        plt.show()
    if model_name == "DecisionTree":
        loaded_model = joblib.load("model_weights_decision_tree.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)

        plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title='Confusion Matrix')
        plt.savefig('Decision Tree_confusion_matrix_image.png')
        plt.show()
    if model_name == "RandomForest":
        loaded_model = joblib.load("model_weights_random_forest.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)

        plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title='Confusion Matrix')
        plt.savefig('Random Forest_confusion_matrix_image.png')
        plt.show()
    if model_name == "SVM":
        loaded_model = joblib.load("model_weights_svm.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)

        plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title='Confusion Matrix')
        plt.savefig('SVM_confusion_matrix_image.png')
        plt.show()
    if model_name == "KNeighbors":
        loaded_model = joblib.load("model_weights_knn.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)

        plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title='Confusion Matrix')
        plt.savefig('KNN_confusion_matrix_image.png')
        plt.show()
@app.get("/sensitivity")
def calculate_sensitivity(model_name : str):
    global df, X_train, X_test, y_train, y_test
    sensitivity = None

    if model_name == "LogisticRegression":
        loaded_model = joblib.load("model_weights_logistic.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_sensitivities = []
        for class_of_interest in range(cm.shape[0]):
            true_positives = cm[class_of_interest, class_of_interest]
            false_negatives = sum(cm[class_of_interest, :]) - true_positives
            sensitivity = true_positives / (true_positives + false_negatives)
            class_sensitivities.append(sensitivity)
        overall_sensitivity = sum(class_sensitivities) / len(class_sensitivities)
    if model_name == "DecisionTree":
        loaded_model = joblib.load("model_weights_decision_tree.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_sensitivities = []
        for class_of_interest in range(cm.shape[0]):
            true_positives = cm[class_of_interest, class_of_interest]
            false_negatives = sum(cm[class_of_interest, :]) - true_positives
            sensitivity = true_positives / (true_positives + false_negatives)
            class_sensitivities.append(sensitivity)
        overall_sensitivity = sum(class_sensitivities) / len(class_sensitivities)
    if model_name == "SVM":
        loaded_model = joblib.load("model_weights_svm.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_sensitivities = []
        for class_of_interest in range(cm.shape[0]):
            true_positives = cm[class_of_interest, class_of_interest]
            false_negatives = sum(cm[class_of_interest, :]) - true_positives
            sensitivity = true_positives / (true_positives + false_negatives)
            class_sensitivities.append(sensitivity)
        overall_sensitivity = sum(class_sensitivities) / len(class_sensitivities)
    if model_name == "RandomForest":
        loaded_model = joblib.load("model_weights_random_forest.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_sensitivities = []
        for class_of_interest in range(cm.shape[0]):
            true_positives = cm[class_of_interest, class_of_interest]
            false_negatives = sum(cm[class_of_interest, :]) - true_positives
            sensitivity = true_positives / (true_positives + false_negatives)
            class_sensitivities.append(sensitivity)
        overall_sensitivity = sum(class_sensitivities) / len(class_sensitivities)
    if model_name == "KNeighbors":
        loaded_model = joblib.load("model_weights_knn.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_sensitivities = []
        for class_of_interest in range(cm.shape[0]):
            true_positives = cm[class_of_interest, class_of_interest]
            false_negatives = sum(cm[class_of_interest, :]) - true_positives
            sensitivity = true_positives / (true_positives + false_negatives)
            class_sensitivities.append(sensitivity)
        overall_sensitivity = sum(class_sensitivities) / len(class_sensitivities)
    return(f"Sensitivity : {overall_sensitivity:.4f}")

@app.get("/specificity")  
def calculate_specificity(model_name:  str):
    global df, X_train, X_test, y_train, y_test
    specificity = None

    if model_name == "LogisticRegression":
        loaded_model = joblib.load("model_weights_logistic.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_specificities = []
        for class_of_interest in range(cm.shape[0]):
            true_negatives = sum(cm[i, i] for i in range(cm.shape[0]) if i != class_of_interest)
            false_positives = sum(cm[i, class_of_interest] for i in range(cm.shape[0]) if i != class_of_interest)
            specificity = true_negatives / (true_negatives + false_positives)
            class_specificities.append(specificity)
        overall_specificity = sum(class_specificities) / len(class_specificities)
    if model_name == "Decision tree":
        loaded_model = joblib.load("model_weights_decision_tree.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_specificities = []
        for class_of_interest in range(cm.shape[0]):
            true_negatives = sum(cm[i, i] for i in range(cm.shape[0]) if i != class_of_interest)
            false_positives = sum(cm[i, class_of_interest] for i in range(cm.shape[0]) if i != class_of_interest)
            specificity = true_negatives / (true_negatives + false_positives)
            class_specificities.append(specificity)
        overall_specificity = sum(class_specificities) / len(class_specificities)
    if model_name == "SVM":
        loaded_model = joblib.load("model_weights_svm.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_specificities = []
        for class_of_interest in range(cm.shape[0]):
            true_negatives = sum(cm[i, i] for i in range(cm.shape[0]) if i != class_of_interest)
            false_positives = sum(cm[i, class_of_interest] for i in range(cm.shape[0]) if i != class_of_interest)
            specificity = true_negatives / (true_negatives + false_positives)
            class_specificities.append(specificity)
        overall_specificity = sum(class_specificities) / len(class_specificities)
    if model_name == "RandomForest":
        loaded_model = joblib.load("model_weights_random_forest.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_specificities = []
        for class_of_interest in range(cm.shape[0]):
            true_negatives = sum(cm[i, i] for i in range(cm.shape[0]) if i != class_of_interest)
            false_positives = sum(cm[i, class_of_interest] for i in range(cm.shape[0]) if i != class_of_interest)
            specificity = true_negatives / (true_negatives + false_positives)
            class_specificities.append(specificity)
        overall_specificity = sum(class_specificities) / len(class_specificities)
    if model_name == "KNeighbors":
        loaded_model = joblib.load("model_weights_knn.joblib")
        test_predictions = loaded_model.predict(X_test)

        cm = confusion_matrix(y_test, test_predictions)
        class_specificities = []
        for class_of_interest in range(cm.shape[0]):
            true_negatives = sum(cm[i, i] for i in range(cm.shape[0]) if i != class_of_interest)
            false_positives = sum(cm[i, class_of_interest] for i in range(cm.shape[0]) if i != class_of_interest)
            specificity = true_negatives / (true_negatives + false_positives)
            class_specificities.append(specificity)
        overall_specificity = sum(class_specificities) / len(class_specificities)
    return(f"Overall Specificity for Decision Tree: {overall_specificity:.4f}")

@app.get("/calculate-ROC")
def claculate_ROC(model_name : str):
    global df, X_train, X_test, y_train, y_test

    if model_name == "LogisticRegression":
        loaded_model = joblib.load("model_weights_logistic.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_predictions_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_predictions_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 8))

        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]), linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Logistic Regression')
        plt.legend(loc="lower right")
        plt.show()
    if model_name == "DecisionTree":
        loaded_model = joblib.load("model_weights_decision_tree.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_predictions_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_predictions_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 8))

        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]), linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Decision Tree')
        plt.legend(loc="lower right")
        plt.show()
    
    if model_name == "SVM":
        loaded_model = joblib.load("model_weights_svm.joblib")
        decision_values = loaded_model.decision_function(X_test)

        # Binarize the labels for multi-class ROC curve
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], decision_values[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), decision_values.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plotting the ROC curve
        plt.figure(figsize=(8, 8))

        # Plotting ROC curve for each class
        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        # Plotting micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]), linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for SVM')
        plt.legend(loc="lower right")
        plt.show()
    if model_name == "RandomForest":
        loaded_model = joblib.load("model_weights_random_forest.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_predictions_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_predictions_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 8))

        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]), linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Random Forest')
        plt.legend(loc="lower right")
        plt.show()
    if model_name == "KNeighbors":
        loaded_model = joblib.load("model_weights_knn.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_predictions_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_predictions_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 8))

        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]), linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Kneighbors')
        plt.legend(loc="lower right")
        plt.show()

@app.get("/calculate-AUC")
def calclulate_AUC(model_name : str):
    global df,X_train,y_train,X_test,y_test

    if model_name == "LogisticRegression":
        loaded_model = joblib.load("model_weights_logistic.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        # Binarize the labels for multi-class AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute AUC for each class
        class_auc = []
        for i in range(len(np.unique(y_test))):
            auc_i = roc_auc_score(y_test_bin[:, i], test_predictions_prob[:, i])
            class_auc.append(auc_i)

        # Macro-average AUC
        macro_auc = np.mean(class_auc)

        # Micro-average AUC
        micro_auc = roc_auc_score(y_test_bin.ravel(), test_predictions_prob.ravel())

        # Returning AUC values
    if model_name == "DecisionTree":
        loaded_model = joblib.load("model_weights_decision_tree.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        # Binarize the labels for multi-class AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute AUC for each class
        class_auc = []
        for i in range(len(np.unique(y_test))):
            auc_i = roc_auc_score(y_test_bin[:, i], test_predictions_prob[:, i])
            class_auc.append(auc_i)

        # Macro-average AUC
        macro_auc = np.mean(class_auc)

        # Micro-average AUC
        micro_auc = roc_auc_score(y_test_bin.ravel(), test_predictions_prob.ravel())

        # Returning AUC values
    if model_name == "SVM":
        loaded_model = joblib.load("model_weights_svm.joblib")
        test_predictions_prob = loaded_model.decision_function(X_test)

        # Binarize the labels for multi-class AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute AUC for each class
        class_auc = []
        for i in range(len(np.unique(y_test))):
            auc_i = roc_auc_score(y_test_bin[:, i], test_predictions_prob[:, i])
            class_auc.append(auc_i)

        # Macro-average AUC
        macro_auc = np.mean(class_auc)

        # Micro-average AUC
        micro_auc = roc_auc_score(y_test_bin.ravel(), test_predictions_prob.ravel())
    if model_name == "RandomForest":
        loaded_model = joblib.load("model_weights_random_forest.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        # Binarize the labels for multi-class AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute AUC for each class
        class_auc = []
        for i in range(len(np.unique(y_test))):
            auc_i = roc_auc_score(y_test_bin[:, i], test_predictions_prob[:, i])
            class_auc.append(auc_i)

        # Macro-average AUC
        macro_auc = np.mean(class_auc)

        # Micro-average AUC
        micro_auc = roc_auc_score(y_test_bin.ravel(), test_predictions_prob.ravel())
    if model_name == "KNeighbors":
        loaded_model = joblib.load("model_weights_knn.joblib")
        test_predictions_prob = loaded_model.predict_proba(X_test)

        # Binarize the labels for multi-class AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute AUC for each class
        class_auc = []
        for i in range(len(np.unique(y_test))):
            auc_i = roc_auc_score(y_test_bin[:, i], test_predictions_prob[:, i])
            class_auc.append(auc_i)

        # Macro-average AUC
        macro_auc = np.mean(class_auc)

        # Micro-average AUC
        micro_auc = roc_auc_score(y_test_bin.ravel(), test_predictions_prob.ravel())
    return JSONResponse(content={"class_auc": class_auc, "macro_auc": macro_auc, "micro_auc": micro_auc})
 
@app.get("/perform-decision-tree")
def perform_decision_tree():
    try:
        global df, X_train, X_test, y_train, y_test

        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train, y_train)

        weight_file = "model_weights_decision_tree.joblib"
        joblib.dump(dt_classifier, weight_file)

        loaded_model = joblib.load(weight_file)
        test_predictions = loaded_model.predict(X_test)
        accuracy = np.sum(y_test.values == test_predictions) / len(y_test.values)

        print(f"Decision Tree Accuracy: {accuracy}")

        results_df = pd.DataFrame({'Actual Output': y_test, 'DT_Predicted Output': test_predictions})
        results_df.to_csv('decision_tree_predictions.csv', index=False)

        message = f"Decision Tree applied successfully. Model weights saved in 'model_weights_decision_tree.joblib'. Accuracy: {accuracy}"
        return JSONResponse(content={"message": message, "accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": f"Something went wrong: {e}"})

@app.get("/perform-kneighbors")
def perform_Kneighbours():
    try:
        global df, X_train, X_test, y_train, y_test

        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_train)

        weight_file = "model_weights_knn.joblib"
        joblib.dump(knn_classifier, weight_file)

        loaded_model = joblib.load(weight_file)
        knn_test_predictions = loaded_model.predict(X_test)

        accuracy = np.sum(y_test.values == knn_test_predictions) / len(y_test.values)

        print(f"K-Neighbours Accuracy: {accuracy}")

        results_df = pd.DataFrame({'Actual Output': y_test, 'KNN_Predicted Output': knn_test_predictions})
        results_df.to_csv('knn_predictions.csv', index=False)

        message = f"K-Nearest Neighbors applied successfully. Model weights saved in 'model_weights_knn.joblib'. Accuracy: {accuracy}"
        return JSONResponse(content={"message": message, "accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": f"Something went wrong: {e}"})

@app.get("/perform-svm")
def apply_svm():

    try:
        global df, X_train, X_test, y_train, y_test

        svm_classifier = SVC(random_state=42)
        svm_classifier.fit(X_train, y_train)

        weight_file = "model_weights_svm.joblib"
        joblib.dump(svm_classifier, weight_file)

        loaded_model = joblib.load(weight_file)
        svm_test_predictions = loaded_model.predict(X_test)
        accuracy = np.sum(y_test.values == svm_test_predictions) / len(y_test.values)

        print(f"SVM Accuracy: {accuracy}")

        results_df = pd.DataFrame({'Actual Output': y_test, 'SVM_Predicted Output': svm_test_predictions})
        results_df.to_csv('svm_predictions.csv', index=False)

        message = f"SVM applied successfully. Model weights saved in 'model_weights_svm.joblib'. Accuracy: {accuracy}"
        return JSONResponse(content={"message": message, "accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": f"Something went wrong: {e}"})

@app.get("/perform-random-forest")
def perform_random_forest():
    try:
        global df, X_train, X_test, y_train, y_test

        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_train, y_train)

        weight_file = "model_weights_random_forest.joblib"
        joblib.dump(rf_classifier, weight_file)

        loaded_model = joblib.load(weight_file)
        test_predictions = loaded_model.predict(X_test)
        accuracy = np.sum(y_test.values == test_predictions) / len(y_test.values)

        print(f"Random Forest Accuracy: {accuracy}")

        results_df = pd.DataFrame({'Actual Output': y_test, 'RF_Predicted Output': test_predictions})
        results_df.to_csv('random_forest_predictions.csv', index=False)

        message = f"Random Forest applied successfully. Model weights saved in 'model_weights_random_forest.joblib'. Accuracy: {accuracy}"
        return JSONResponse(content={"message": message, "accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": f"Something went wrong: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")