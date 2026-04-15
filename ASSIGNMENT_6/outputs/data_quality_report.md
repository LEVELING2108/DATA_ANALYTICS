| Dataset   | Column                | Original Dtype   | Action          | Final Dtype   |
|:----------|:----------------------|:-----------------|:----------------|:--------------|
| Insurance | age                   | int64            | Drop Duplicates | int32         |
| Insurance | sex                   | object           | Drop Duplicates | object        |
| Insurance | bmi                   | float64          | Drop Duplicates | float64       |
| Insurance | children              | int64            | Drop Duplicates | int32         |
| Insurance | smoker                | object           | Drop Duplicates | object        |
| Insurance | region                | object           | Drop Duplicates | object        |
| Insurance | charges               | float64          | Drop Duplicates | float64       |
| HR        | satisfaction_level    | float64          | Check Nulls     | float64       |
| HR        | last_evaluation       | float64          | Check Nulls     | float64       |
| HR        | number_project        | int64            | Check Nulls     | int64         |
| HR        | average_montly_hours  | int64            | Check Nulls     | int64         |
| HR        | time_spend_company    | int64            | Check Nulls     | int64         |
| HR        | Work_accident         | int64            | Check Nulls     | int64         |
| HR        | left                  | int64            | Check Nulls     | int64         |
| HR        | promotion_last_5years | int64            | Check Nulls     | int64         |
| HR        | Department            | object           | Check Nulls     | object        |
| HR        | salary                | object           | Check Nulls     | object        |