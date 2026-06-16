MODEL_CONFIGS = {

    "borrower": {
        "drop_columns": [
            "seller_name",
            "servicer_name"
        ]
    },

    "institutional": {
        "drop_columns": []
    },

    "enhanced_bss": {
        "dataset": "bss"
    }
}

train_logistic()

train_random_forest()

train_xgboost()