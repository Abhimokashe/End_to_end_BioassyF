def train_test_split(list_Xn_comp,list_y_resampled):
    """train test split of data
    args:list_Xn_comp,list_y_resampled
    Return:list_X_train,list_X_test,list_y_train,list_y_test"""

    list_X_train = []
    list_X_test = []
    list_y_train = []
    list_y_test = []

    for i in range(len(list_Xn_comp)):
        for j in range(len(list_y_resampled)):
            if i == j:
                from sklearn.model_selection import train_test_split
                X_train,X_test,y_train,y_test = train_test_split(list_Xn_comp[i],list_y_resampled[j],random_state=0,test_size=0.2)
                list_X_train.append(X_train)
                list_X_test.append(X_test)
                list_y_train.append(y_train)
                list_y_test.append(y_test)
    return list_X_train,list_X_test,list_y_train,list_y_test

def fitting_using_decision_tree_algorithm(list_X_train,list_y_train):
    """Fitting of model
    args: list_X_train,list_y_train
    Return: list_model"""
    list_model = []

    for i in range(len(list_X_train)):
        for j in range(len(list_y_train)):
            if i == j:
                from sklearn.tree import DecisionTreeClassifier
                dtc = DecisionTreeClassifier()
                model = dtc.fit(list_X_train[i],list_y_train[j])
                list_model.append(model)
    return list_model

def prediction_on_train_test_data(list_model,list_X_train,list_X_test):
    """Prediction on train test data
    args: list_model,list_X_train,list_X_test
    Return: list_y_pred_train,list_y_pred_test"""
    list_y_pred_train = []
    list_y_pred_test = []

    for i in range(len(list_model)):
        for j in range(len(list_X_train)):
            for k in range(len(list_X_test)):
                if i==j==k:
                    y_pred_train = list_model[i].predict(list_X_train[j])
                    list_y_pred_train.append(y_pred_train)
                    y_pred_test = list_model[i].predict(list_X_test[k])
                    list_y_pred_test.append(y_pred_test)
    return list_y_pred_train,list_y_pred_test

def model_evaluation_metrics(list_y_train,list_y_test,list_y_pred_train,list_y_pred_test,df_list_names):
    for i in range(len(list_y_train)):
        for j in range(len(list_y_pred_train)):
            for k in range(len(df_list_names)):
                for m in range(len(list_y_pred_test)):
                    if i == j == k == m:
                         from sklearn.metrics import accuracy_score,classification_report,f1_score,confusion_matrix
                         print(df_list_names[k])
                         print("accuracy_score_train={}".format(accuracy_score(list_y_train[i],list_y_pred_train[j])))
                         print()
                         print("Classification_report_train")
                         print(classification_report(list_y_train[i],list_y_pred_train[j]))
                         print()
                         print("Confusion_matrix_train")
                         print(confusion_matrix(list_y_train[i],list_y_pred_train[j]))
                         print('---------------------------------')
                         print()
                         print(df_list_names[k])
                         print("Accuracy_score_of_test={}".format(accuracy_score(list_y_test[i],list_y_pred_test[j])))
                         print()
                         print("Classification_report_test")
                         print(classification_report(list_y_test[i],list_y_pred_test[j]))
                         print()
                         print("Confusion_matrix_test")
                         print(confusion_matrix(list_y_test[i],list_y_pred_test[j]))
                         print('-----------------------------------------')
    return