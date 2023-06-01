from kedro.pipeline import Pipeline, node
from .nodes import preparing_list_of_train_test_data,concatenation_of_file,checking_missing_value,data_splitting,treating_missing_value,checking_imbalance_data,treating_imbalance_data,scaling_data,application_pca,plot_scree_plot


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func= preparing_list_of_train_test_data,
                 inputs= ["df_1284mr","df_1284r","df_1608mr","df_1608r","df_362r","df_373_439r","df_373r","df_439mr","df_439r","df_456r","df_604_644r","df_604r","df_644mr","df_644r","df_687_721r","df_687r","df_688r","df_721mr","df_721r","df_746_1284r","df_746r","df_1284mr_t","df_1284r_t","df_1608mr_t","df_1608r_t","df_362r_t","df_373_439r_t","df_373r_t","df_439mr_t","df_439r_t","df_456r_t","df_604_644r_t","df_604r_t","df_644mr_t","df_644r_t","df_687_721r_t","df_687r_t","df_688r_t","df_721mr_t","df_721r_t","df_746_1284r_t","df_746r_t"],
                 outputs = ["df_list_train","list_df_test","df_list_names"],
                 name= "preparing_list_of_train_test_data_node",
            ),
            node(
                 func= concatenation_of_file,
                 inputs= ["df_list_train","list_df_test"],
                 outputs = "list_df_concat",
                 name = "concatenation_of_file_node",
            ),
            node(
                 func= checking_missing_value,
                 inputs= ["list_df_concat","df_list_names"],
                 outputs= None,
                 name= "checking_missing_value_node",
            ),
            node(
                 func= data_splitting,
                 inputs= "list_df_concat",
                 outputs= ["list_X","list_y"],
                 name= "data_splitting",
            ),
            node(
                func= treating_missing_value,
                inputs="list_X",
                outputs= "list_X1",
                name= "treating_missing_value_node",
            ),
            node(
                 func= checking_imbalance_data,
                 inputs= ["list_X1","list_y","df_list_names"],
                 outputs= None,
                 name= "checking_imbalance_data_node",
            ),
            node(
                 func= treating_imbalance_data,
                 inputs= ["list_X1","list_y"],
                 outputs= ["list_X_resampled","list_y_resampled"],
                 name= "treating_imbalance_data_node",
            ),
            node(
                 func= scaling_data,
                 inputs= ["list_X_resampled","list_y_resampled"],
                 outputs = "list_Xscal",
                 name= "scaling_data_node",
            ),
            node(
                 func= application_pca,
                 inputs= "list_Xscal",
                 outputs= ["list_Xn_comp","list_pca","list_PC_components"],
                 name= "application_pca_node",
            ),
            node(
                 func= plot_scree_plot,
                 inputs= ["list_Xn_comp","list_pca","list_PC_components"],
                 outputs= None,
                 name= "plot_scree_plot_node"
            )
        ]
    )