from kedro.pipeline import Pipeline, node
from .nodes import train_test_split,fitting_using_decision_tree_algorithm,prediction_on_train_test_data,model_evaluation_metrics

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func=train_test_split,
                 inputs=["list_Xn_comp","list_y_resampled"],
                 outputs=["list_X_train","list_X_test","list_y_train","list_y_test"],
                 name="train_test_split_node",
            ),
            node(
                 func=fitting_using_decision_tree_algorithm,
                 inputs=["list_X_train","list_y_train"],
                 outputs="list_model",
                 name="fitting_using_decision_tree_algorithm_node",
            ),
            node(
                 func=prediction_on_train_test_data,
                 inputs=["list_model","list_X_train","list_X_test"],
                 outputs=["list_y_pred_train","list_y_pred_test"],
                 name="prediction_on_train_test_data_node",
            ),
            node(
                 func=model_evaluation_metrics,
                 inputs=["list_y_train","list_y_test","list_y_pred_train","list_y_pred_test","df_list_names"],
                 outputs= None,
                 name="model_evaluation_metrics_node",
            )
        ]
    )