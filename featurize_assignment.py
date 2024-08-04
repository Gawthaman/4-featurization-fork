# ======================================================================================
# ASSIGNMENT 4: Using Featurization to Optimize for Hardness

# Your goal is to use Honegumi and your knowledge of the Ax API to develop an
# optimization script to help find a composition that maximizes the yield strength
# of a developed alloy. Your experimental budget is limited to 25 experiments. A
# synthetic objective function has been provided that will serve as a proxy for real
# experimental measurements.
# ======================================================================================

from utils import set_seeds, featurize_data, measure_hardness

set_seeds()  # setting the random seed for reproducibility

# --------------------------------------------------------------------------------------
# TASK A: Featurize the data using the featurize_data function.
# --------------------------------------------------------------------------------------

import pandas as pd

train = pd.read_csv("data/train.csv")
candidate = pd.read_csv("data/candidate.csv")

# TODO: Your Code Goes Here

# --------------------------------------------------------------------------------------
# TASK B: Use Honegumi to set up the optimization problem and attach the training data.
# --------------------------------------------------------------------------------------
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties

from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model= # TODO: Your Code Goes Here,
            num_trials=-1,
            max_parallelism=3,
        ),
    ]
)

# TODO: Your Code Goes Here
# --------------------------------------------------------------------------------------
# TASK C: Build the optimization loop and and evaluate 25 candidates.
# --------------------------------------------------------------------------------------
from ax.core.observation import ObservationFeatures

# TODO: Your Code Goes Here

# --------------------------------------------------------------------------------------
# TASK D: Report the optimal composition and associated hardness.
# --------------------------------------------------------------------------------------

# TODO: Your Code Goes Here

# --------------------------------------------------------------------------------------
# TASK E: Report the most important feature and its correlation for predicting hardness.
# --------------------------------------------------------------------------------------

# TODO: Your Code Goes Here

# --------------------------------------------------------------------------------------
# TASK F: How many materials have a hardness greater than 43?
# --------------------------------------------------------------------------------------

# TODO: Your Code Goes Here