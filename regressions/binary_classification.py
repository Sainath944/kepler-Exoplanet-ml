import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from xgboost import XGBClassifier # type: ignore



class astro:
    
    def random_forest(X_train, X_test, y_train, y_test, X, y):
        #random forest regression
        #starting the regression things
        model = RandomForestClassifier(n_estimators=100, random_state=42) #taking the regressor
        model.fit(X_train, y_train)#fitting the variables

        # Predictions
        y_pred = model.predict(X_test) #predicting the values

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)#it will provide us the details about the true positives, false positives, true negatives, falsse negatives with which we can get the precision f1 score and all otherr like used to check the correctness of the model we can get all the parameters which are used to check the correctness of the model

        # Classification Report
        print("Classification Report:\n", classification_report(y_test, y_pred))#it will give us all the pareameters which are used to checkk the correctness of the model

        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("ROC-AUC Score:", roc_auc)
        #------------------------------------------------------------------------------------------
        # Random Forest Classifier same payina unnadhe malla rasina but koddiga thedaga rasina anthe
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        print("Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
        print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
        print("Random Forest - ROC-AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

        # Feature Importance give us the features which are mostly impacting the output we can specify which can show up for instance we used to show top 10 features which impact the most output
        rf_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("Random Forest - Top 10 Feature Importances:\n", rf_importances.head(10))

        # Feature Importance
        feature_importance = model.feature_importances_ # this method is used to knew that which are featues are like the output depends on which feature more 
        feature_names = X.columns # we can able to see the top ten features here
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(10))
        plt.title("Top 10 Feature Importances")
        plt.show()


    def logistic_regression(X_train, X_test, y_train, y_test, X, y):
        # Logistic Regression
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)

        print("Logistic Regression - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
        print("Logistic Regression - Classification Report:\n", classification_report(y_test, y_pred_log))
        print("Logistic Regression - ROC-AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))


    def ada_boost(X_train, X_test, y_train, y_test, X, y):
        # Initialize AdaBoost with the SAMME algorithm
        ada_model = AdaBoostClassifier(n_estimators=50, algorithm='SAMME', random_state=42) # here we are using a specific thing called algorithm = damme because in the adaboost in the future versions its gonna be removed so samme will be in that place so to avoid the bugs and to maintain for the future proofe we are using the algorith = samme thing
        ada_model.fit(X_train, y_train)
        y_pred_ada = ada_model.predict(X_test)

        print("AdaBoost - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))
        print("AdaBoost - Classification Report:\n", classification_report(y_test, y_pred_ada))
        print("AdaBoost - ROC-AUC Score:", roc_auc_score(y_test, ada_model.predict_proba(X_test)[:, 1]))

        # Feature Importance
        ada_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': ada_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("AdaBoost - Top 10 Feature Importances:\n", ada_importances.head(10))

    def xg_boost(X_train, X_test, y_train, y_test, X, y):
        #creating separate train test sets for the xgboost
        xg_y_train = y_train.replace(-1, 1)#for the xgboost we should be only having the 0 or 1 it wont acccepts the -1 or any other numbers so i converted the -1 disposition into 1 here as we know in the disposition -1 represents the true positive and 0 represents the flase positive
        xg_y_test = y_test.replace(-1, 1)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, xg_y_train)
        y_pred_xgb = xgb_model.predict(X_test)

        print("XGBoost - Confusion Matrix:\n", confusion_matrix(xg_y_test, y_pred_xgb))
        print("XGBoost - Classification Report:\n", classification_report(xg_y_test, y_pred_xgb))
        print("XGBoost - ROC-AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

        # Feature Importance
        xgb_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("XGBoost - Top 10 Feature Importances:\n", xgb_importances.head(10))

    def k_neighbour(X_train, X_test, y_train, y_test, X, y):
        # K-Nearest Neighbors Classifier
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)

        print("K-Nearest Neighbors - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
        print("K-Nearest Neighbors - Classification Report:\n", classification_report(y_test, y_pred_knn))
        print("K-Nearest Neighbors - ROC-AUC Score:", roc_auc_score(y_test, knn_model.predict_proba(X_test)[:, 1]))





def main():
    #read the processed data
    data = pd.read_csv('your/path/to/processed_exoplanet_data.csv')

    # Separate target variable
    X = data.drop(columns=['disposition'])
    y = data['disposition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data.info()
    print(data.describe())

    random_forest(X_train, X_test, y_train, y_test, X, y)
    logistic_regression(X_train, X_test, y_train, y_test, X, y)
    ada_boost(X_train, X_test, y_train, y_test, X, y)
    xg_boost(X_train, X_test, y_train, y_test, X, y)
    k_neighbour(X_train, X_test, y_train, y_test, X, y)


if __name__ == "__main__":
    main()




