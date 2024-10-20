import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import dotenv
import os
from mostlyai import MostlyAI
from datallm import DataLLM

dotenv.load_dotenv()

print(os.getenv("MOSTLY_AI_API_KEY"))

# Initialize DataLLM
datallm = DataLLM(
    api_key=os.getenv("MOSTLY_AI_API_KEY"), base_url="https://data.mostly.ai"
)

# Generate mock HR data
hr_data = datallm.mock(
    n=100,  # number of samples
    data_description="HR system data for a fictitious company",
    columns={
        "employee_id": {
            "prompt": "Unique identifier for each employee",
            "dtype": "integer",
        },
        "name": {"prompt": "Full name of the employee", "dtype": "string"},
        "department": {
            "prompt": "Department where the employee works",
            "dtype": "category",
            "categories": ["HR", "Sales", "Marketing", "IT", "Finance"],
        },
        "position": {"prompt": "Job position of the employee", "dtype": "string"},
    },
    progress_bar=False,
)

# Enrich HR data with manager_id
hr_data["manager_id"] = datallm.enrich(
    data=hr_data,
    prompt="Manager ID for each employee, C-level employees have no manager",
    data_description="HR system data with manager relationships",
    dtype="integer",
    progress_bar=False,
)

# Set manager_id to None for C-level positions
c_level_positions = ["CEO", "CFO", "COO", "CTO", "CMO"]
hr_data.loc[
    hr_data["position"].str.contains("|".join(c_level_positions), case=False),
    "manager_id",
] = None


# Ensure manager_id points to existing employee_ids
employee_ids = hr_data["employee_id"].tolist()
hr_data["manager_id"] = hr_data.apply(
    lambda row: (
        np.random.choice(employee_ids) if pd.notnull(row["manager_id"]) else None
    ),
    axis=1,
)

# Define salary ranges for different positions
salary_ranges = {
    "HR": (40000, 60000),
    "Sales": (50000, 70000),
    "Marketing": (45000, 65000),
    "IT": (60000, 90000),
    "Finance": (55000, 80000),
    "C-level": (150000, 300000),
}


# Function to assign salary based on position
def assign_salary(position):
    for key, (low, high) in salary_ranges.items():
        if key in position:
            return np.random.randint(low, high)
    return np.random.randint(40000, 60000)  # Default range


# Add salary column
hr_data["salary"] = hr_data["position"].apply(assign_salary)
hr_data.to_csv("hr_data_mostly.csv", index=False)


# Generate mock performance review data
performance_reviews = pd.DataFrame(
    {
        "employee_id": hr_data["employee_id"],
        "manager_id": hr_data["manager_id"],
        "rating": [random.randint(1, 5) for _ in range(len(hr_data))],
        "review_date": [
            datetime.now() - timedelta(days=random.randint(0, 365))
            for _ in range(len(hr_data))
        ],
    }
)

# Remove reviews for C-level employees (no manager)
performance_reviews = performance_reviews.dropna(subset=["manager_id"])
performance_reviews.to_csv("performance_reviews_mostly.csv", index=False)
