####################################
#####     Pipeline - Kernstueck     #######
####################################



from data import load_dataset, preprocess
from models import train_model
from balancing import apply_balancing
from utils import evaluate
from utils import expand_grid
from sklearn.model_selection import StratifiedKFold


def run_all(config):
    results = []

    for dataset in config["data"]["datasets"]:
        print(f"\nDataset: {dataset['name']}")

        X_train, X_test, y_train, y_test = load_dataset(
            dataset["path"],
            config["data"]["target"],
            config["data"]["test_size"],
            config["data"]["random_state"]
        )

        X_train, X_test, y_train, y_test = preprocess(
            X_train, X_test, y_train, y_test,
            **config["preprocess"]
        )

        for method in config["balancing"]["methods"]:
            for exp in config["experiments"]:
                for params in expand_grid(exp["grid"]):

                    print(f"{dataset['name']} | {exp['name']} | {method} | {params}")

                    # ⚖️ Balancing NUR auf Training!
                    X_bal, y_bal = apply_balancing(X_train, y_train, method)

                    model = train_model(
                        X_bal, y_bal,
                        exp["algorithm"],
                        params
                    )

                    scores = evaluate(model, X_test, y_test)

                    results.append({
                        "dataset": dataset["name"],
                        "model": exp["name"],
                        "balancing": method,
                        **params,
                        **scores
                    })

    return results


