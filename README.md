# team08
xunyi and darion
--- 
Questions to ask:

- Do we maintain class imbalance when doing the train-test split, or only within train-validation split? Or do we use random split for train-test.

- 

To Do

- Fix preprocessing for checkboxes on frontend
- Implement plotly graphs
- 

---
# Takeaways

* Same min-max scaler object can be used for multiple columns
* One hot encoding requires different object for different columns

To Do List
- Plot the AUC graph to determine best threshold
- Clean up code and comments

---
Front End readme

- Write hook that constantly polls the backend for the preprocessing, allows them to get scaled results instantly
- 

Things to PCA on
["contract_type", "tenure_months", "total_long_distance_fee", "total_charges_quarter", "has_premium_tech_support", "num_dependents" ]

docker build -t ml-churn-backend -f Dockerfile.backend .





