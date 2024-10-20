import random
import faker
from numpy._core.numeric import astype
import pandas as pd
import numpy as np
from pandas.core import apply

fake = faker.Faker()

C_LEVEL_EXECUTIVES = ["CEO", "CTO", "CFO", "CIO", "COO"]


job_titles_by_department = {
    "Finance": [
        "Financial Analyst",
        "Accountant",
        "Controller",
        "Treasury Manager",
        "Finance Assistant",
    ],
    "IT": [
        "Software Engineer",
        "System Administrator",
        "Network Engineer",
        "IT Support Specialist",
        "DevOps Engineer",
    ],
    "Marketing": [
        "Marketing Coordinator",
        "Content Strategist",
        "SEO Specialist",
        "Social Media Manager",
        "Brand Manager",
    ],
    "HR": [
        "HR Specialist",
        "Recruiter",
        "Compensation Analyst",
        "Training Coordinator",
        "HR Assistant",
    ],
    "Sales": [
        "Sales Executive",
        "Account Manager",
        "Sales Associate",
        "Business Development Manager",
        "Sales Representative",
    ],
}


# First, create some base info about all employees.
# We will differentiate them later
def create_employee_basis(n=500):

    employees = []
    for _ in range(n):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = f"{first_name.lower()}.{last_name.lower()}@fake.company"
        gender = fake.random_element(elements=("Male", "Female"))
        hire_date = fake.date_this_decade(before_today=True, after_today=False)

        employees.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "gender": gender,
                "hire_date": hire_date,
            }
        )
    employees_df = pd.DataFrame(employees)
    employees_df["employee_id"] = range(1, len(employees) + 1)
    if employees_df.employee_id.duplicated().any():
        raise ValueError("Duplicate employee ids")
    return employees_df


# print(create_employee_basis())

# Next, we will assign departments to the employees,
# We will assume that 10% of the employees are managers
#

employee_basis = create_employee_basis()


def create_department_data():

    departments = pd.DataFrame(
        {
            "department_id": [1, 2, 3, 4, 5],
            "department_name": ["Sales", "Marketing", "Finance", "IT", "HR"],
        }
    )
    return departments


def extract_manager_ids_from_employees(employee_df, manager_count):
    manager_ids = employee_df.sample(n=manager_count)["employee_id"].tolist()

    return manager_ids


def create_manager_positions(departments_df):
    # Initialize empty lists to store positions and departments
    positions = []
    departments = []

    # Add C-level executive positions
    for position in C_LEVEL_EXECUTIVES:
        positions.append(position)
        departments.append("Board of Directors")

    # Add team lead and head of positions for each department
    for department in departments_df["department_name"]:
        positions.append(f"{department} Team Lead")
        positions.append(f"Head of {department}")
        departments.append(department)
        departments.append(department)

    # Create a DataFrame from the lists
    manager_positions = pd.DataFrame({"position": positions, "department": departments})

    return manager_positions


# TODO: Performance rating table
if __name__ == "__main__":
    employee_basis = create_employee_basis()

    departments = create_department_data()
    # every department has a "team lead" and a "head of" position
    # additionally, we will have some C-level executives.
    manager_number = (len(departments) * 2) + len(C_LEVEL_EXECUTIVES)
    print(f"Manager number: {manager_number}")
    manager_ids = extract_manager_ids_from_employees(
        employee_df=employee_basis, manager_count=manager_number
    )
    if len(manager_ids) != manager_number:
        raise ValueError("Not enough managers")
    manager_df = employee_basis[employee_basis["employee_id"].isin(manager_ids)]
    manager_positions = create_manager_positions(departments_df=departments)
    print(manager_positions)
    if len(manager_df) != len(manager_positions):
        raise ValueError("Not all managers have positions")
    manager_df["position"] = manager_positions["position"].values
    manager_df["department"] = manager_positions["department"].values

    print(manager_df)
    is_c_level = manager_df["position"].isin(C_LEVEL_EXECUTIVES)

    c_level_ids = manager_df[is_c_level]["employee_id"].tolist()  # Convert to a list

    # Handle the assignment to 'reports_to'
    manager_df["reports_to"] = np.where(
        is_c_level,  # Condition: If the manager is C-level
        np.nan,  # Use np.nan for C-level (reports to no one)
        [
            random.choice(c_level_ids) if c_level_ids else np.nan
            for _ in range(len(manager_df))
        ],  # Safely choose a C-level if available
    )
    manager_df["reports_to"] = manager_df["reports_to"].astype("Int64")

    print(
        f"Manager df: {manager_df[['employee_id', 'position', 'reports_to', 'department']]}"
    )
    print(manager_df.value_counts("position"))
    print(manager_df.value_counts("reports_to", dropna=False))

    subordinate_df = pd.DataFrame(
        employee_basis[~employee_basis["employee_id"].isin(manager_ids)]
    )
    subordinate_df["reports_to"] = np.random.choice(
        manager_df[~is_c_level]["employee_id"], len(subordinate_df), replace=True
    )
    manager_departments = manager_df[["employee_id", "department"]].copy()
    subordinate_df = (
        subordinate_df.merge(
            manager_departments,
            left_on="reports_to",
            right_on="employee_id",
            how="left",
            copy=True,
        )
        .drop("employee_id_y", axis=1)
        .rename(columns={"employee_id_x": "employee_id"})
    )
    subordinate_df["position"] = subordinate_df["department"].apply(
        lambda dept: np.random.choice(job_titles_by_department[dept])
    )
    print(subordinate_df[["employee_id", "reports_to", "department", "position"]])
    print(subordinate_df.columns)
    print(manager_df.columns)
    employee_df_final = pd.concat([manager_df, subordinate_df])
    employee_df_final.to_csv("HR/employee_data.csv", index=False)
    # TODO: Add Salary
