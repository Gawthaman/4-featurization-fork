import os
import warnings
from utils import set_seeds, featurize_data, measure_hardness
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="session")
def get_namespace():
    script_fname = "featurize_assignment_ans.py"
    script_content = open(script_fname).read()

    namespace = {}
    exec(script_content, namespace)
    return namespace


def test_task_a(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    assert "X_train" in get_namespace, "X_train is not defined"
    assert "y_train" in get_namespace, "y_train is not defined"
    assert "X_candidate" in get_namespace, "X_candidate is not defined"

    # test whether X_train has the correct shape
    user_X_train = get_namespace["X_train"]
    assert user_X_train.shape[1] == 13, "X_train has the wrong number of columns"


def test_task_b(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    # test whether the generation strategy model is set to BOTORCH_MODULAR
    assert (
        str(running_ax_client.generation_strategy.model.model) == "BoTorchModel"
    ), "The model is not set to BOTORCH_MODULAR"

    # test whether the following hardness values are in the ax_client data, 0.74, 18.2, 18.5, 1.02, 1.0, 10.44, 1.21, 18.0, 11.28, 9.9, 0.85, 5.96
    user_train_hardness = running_ax_client.get_trials_data_frame()["hardness"].values
    train_hardness = [
        0.74,
        18.2,
        18.5,
        1.02,
        1.0,
        10.44,
        1.21,
        18.0,
        11.28,
        9.9,
        0.85,
        5.96,
    ]

    # assert that the hardness values are in the ax_client data
    assert all(
        [hardness in user_train_hardness for hardness in train_hardness]
    ), "The hardness values are not in the ax_client data"


def test_task_c(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    # test whether the number of trials is 12+25
    assert (
        running_ax_client.experiment.num_trials == 37
    ), "Incorrect number of trials, exhaust your budget of 25 experiments"

    # assert tha there aren't any duplicate trials in the df
    user_trials_df = running_ax_client.get_trials_data_frame()
    assert (
        user_trials_df.duplicated().sum() == 0
    ), "There are duplicate trials in the trials dataframe, ensure that you aren't selecting the same material twice"


def test_task_d(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    # Optimal Composition: Cr0.05W0.95B4
    # Optimal Hardness: 46.45652707

    optimal_composition = "Cr0.05W0.95B4"
    user_optimal_composition = get_namespace["optimal_composition"]

    optimal_hardness = 46.45652707
    user_optimal_hardness = get_namespace["max_hardness"]

    assert (
        optimal_composition in user_optimal_composition
    ), "Optimal composition is incorrect"

    assert user_optimal_hardness == optimal_hardness, "Incorrect optimal hardness"


def test_task_e(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    most_important = "mode_crystal_radius"
    user_most_important = get_namespace["most_important"]

    correlation = -1
    user_correlation = get_namespace["correlation"]

    assert user_most_important == most_important, "Most important feature is incorrect"
    assert user_correlation == correlation, "Incorrect correlation sign"


def test_task_f(get_namespace):

    running_ax_client = get_namespace["ax_client"]

    user_n_hard_materials = get_namespace["n_hard_materials"]
    n_hard_materials = 6

    assert (
        user_n_hard_materials == n_hard_materials
    ), "Incorrect number of hard materials"
