# Git_Sp25_ADEC743002
Our team investigated how demographic and lifestyle factors impact individual medical insurance costs, using a publicly available dataset from Kaggle. The objective was to evaluate which predictive model—linear regression, decision tree, or random forest—offered the best tradeoff between accuracy and interpretability.

Our analysis showed that while random forest excelled in fitting the training data, it tended to overfit and underperformed on unseen data. In contrast, linear regression delivered the most balanced performance on the test set, making it the most generalizable model.

## Researching, Reviewing, and Investigating the Data (by Gina):
Researching the Data - Each team member spent time thinking of problems in mutual areas of interest and researching data to address those problems. We brought forth different datasets and discussed the advantages of how it could be used to analyze the problem, but also the challenges. We had mutual interest in healthcare and wanted to better analyze how different factors affect health insurance expenditures.

In particular, we were interested in which modeling approach—linear regression, decision tree, or random forest—provides the best balance of accuracy and interpretability when predicting medical insurance charges.

We agreed upon a dataset from Kaggle that included medical insurance charges as the target variable, and independent variables including:

	•	age: age of primary beneficiary
	•	sex: insurance contractor gender, female, male
	•	bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
	•	objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
	•	children: Number of children covered by health insurance / Number of dependents
	•	smoker: Smoking
	•	region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
	•	charges: Individual medical costs billed by health insurance

Reviewing the Data - This dataset was particularly interesting because it included demographic and lifestyle factors that we suspected have an impact on medical costs. For example, we hypothesized that someone’s age can impact their costs, because older people tend to have more health issues than younger people so they need more care or at least would have increased premiums (thus increased costs). We also hypothesized that smokers and individuals with higher BMIs would increase insurance costs. This dataset was a strong choice in incorporating key personal risk factors while also giving insight to demographic information.

While these variables are important in analyzing health insurance costs for our models, we recognized some challenges with the dataset. These include:

- Potential complex relationships between other factors and insurance costs not included in the dataset, such as employment status, income level, and whether the individual has a pre-existing condition. Without this additional explanatory data, we have some bias in our model.
- Some data points in the dataset could be outliers that need special attention or handling.
- Certain factors might interact in ways that significantly affect costs compared to their individual effects.

The dataset was likely cleaned to an extent already because there were no missing variables, an occurrence unlikely to happen with real, patient healthcare data. The dataset also lacks some context that can inform this study better, such as temporal data around these variables, including whether or not these medical charges are yearly charges or on some other time table. Despite these challenges, we thought this dataset was a strong example to apply our modeling approaches to, using linear regression as a baseline, and comparing performance between decision trees and random forest. Each of these variables are interpretable which helped our goal of choosing a model that best balances accuracy and interpretability when predicting medical insurance charges.

Importing and Investigating the Data (performed by Gina): For the steps to import and investigate the data, Gina led this effort. She imported the data, split the data into training and testing datasets and then saved both to feather format. 

The investigation portion included:

- Obtaining summary statistics for the data (understand what some sample values are, what are the lower and upper limits).
- Creating histograms to understand if any data is skewed.
- Creating violin plots to understand the distribution regarding how dense certain values are
- Creating boxplots to identify outliers.
- Understanding the frequency of categorical values.
- Seeing how each predictor varies with the target variable, charges, to identify any non-linearity and potential interaction effects
- Confirming there are no missing values.
- Conducting a collinearity and multicollinearity check to see if any variables have a strong relationship with one another.


Some insights from this analysis were that for age, there were more frequent values around age 20, so the data included a lot of young adults. The mean BMI was 30, meaning that on average people in the US represented in this dataset are obese. There were mostly 0 children on insurance for adults, which made logical sense considering most of the adults were young. Viewing charges by smoker status, it was not surprising to see the majority of nonsmokers have low health insurance costs, while smokers have higher costs. Males have somewhat higher insurance costs than females. Certain variables were skewed, such as Children and Charges which are right-skewed. There are some outliers, particularly BMI and the response, Charges. 

Collinearity is generally not an issue, however, multicollinearity is present for Age and BMI, detected through VIF. This is somewhat expected as older people in the U.S. tend to be more overweight. One interesting observation is that Charges tend to be higher for those with more children covered by insurance. This is counterintuitive, but could be that while children generally have routine care, children are generally healthier so their costs are overall lower. Some non-linearity was detected, particulary between BMI and Charges. The plots show almost no relationship until a threshold of Class 1 Obsesity (as defined by the CDC here: https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html), where it then jumps to high insurance costs associated with BMI.

In analyzing potential interaction effects, Age appears to have a more additive relationship with charges, when analyzing by category, indicating no interactive effects. However, BMI does show interactive effects when compared against Charges, by other predictor categories. The most extreme interactive effects appear to be with BMI and Smoker and BMI and Sex, indicating that these predictors together are more predictive of Charges then separately. 

## 📊 Key Visualizations (by Gina)

To support our exploratory analysis, we created several visualizations that helped us understand the relationships, distributions, and outliers within the dataset:

### 🔹 Distribution Plots
- **Histograms** for age, BMI, children, and charges  
- **Violin plots** to visualize charges by category (e.g., smoker status, region)  
- **Boxplots** to detect outliers in numeric features

![Violin Plot - Charges by Sex](Final/violin_plot_sex.png)

### 🔹 Correlation & Collinearity
- **Heatmap** of feature correlations  
- **VIF (Variance Inflation Factor)** analysis to detect multicollinearity  
  → Notable finding: age and BMI showed moderate VIF values, confirming correlation

![Correlation Heat Map](Final/correlation_map.png)

### 🔹 Non-linearity & Interactions
- **LOWESS curves** showing non-linear jumps in charges with BMI  
- **Interaction plots**:
  - `BMI × Smoker`: strong upward shift in charges
  - `BMI × Sex`: smaller but visible effect
  - `Age × Other predictors`: more additive than interactive
 
![BMI x Smoker Interaction](Final/interaction_chart_smoker.png)

> These plots informed our modeling decisions and feature engineering steps, specifically our inclusion of interaction terms like `bmi × smoker` and `bmi × sex`.


## Script Structure and Team Contributions
Our pipeline consisted of four main Python scripts, each aligned with a distinct project phase. Responsibilities were divided among team members based on strengths in data wrangling, modeling, and production deployment.

## Working Directory Note

To ensure the scripts run smoothly on any machine, we added logic to each core script to automatically set the working directory to the script’s location.

This avoids path errors caused by IDEs like Spyder (which reset the working directory to a user’s home folder) and ensures that file reads/writes (e.g., model, data, Excel input) work without manual changes.

```python
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
```


## 01_data_analysis.py – Exploratory Data Analysis (Gina)
- Imported and summarized the dataset.
- Created visualizations (histograms, violin plots, boxplots).
- Checked for missing values and outliers.
- Assessed variable distributions, collinearity (VIF), and non-linearity.
- Explored interaction effects between predictors (e.g., BMI × Smoker).

## 02_data_cleaning.py – Data Cleaning & Transformation (Kache)
- Defined a reusable prep_data() function for cleaning and transformation.
- Handled median imputation for age, bmi, and children.
- Encoded categorical variables (binary and one-hot).
- Engineered interaction terms (bmi × smoker, bmi × sex).

## 03_train_and_save_model.py – Model Training (Reuben)
- Loaded training data and applied cleaning.
- Trained an XGBoost regressor using GridSearchCV for hyperparameter tuning.
- Evaluated model performance with RMSE and MAE.
- Saved final model as final_model.pkl.

## 04_predict_from_excel.py – Excel Input Prediction (Kache)
- Accepted user-uploaded Excel (CSV) row.
- Applied prep_data() transformation and imputed missing fields.
- Aligned columns with the trained model.
- Predicted charges using the trained XGBoost model.
- Exported the result with predicted value back to CSV.


## Pipeflow Diagram

insurance.csv  
  ↓  
**01_data_analysis.py**  
  → Exploratory data analysis (EDA)  
  → Train/test split  
  → Save as `df_train.feather` and `df_test.feather`  
  ↓  
**02_data_cleaning.py**  
  → Apply `prep_data()` function  
  → Encode features, handle missing values, add interaction terms  
  → Save cleaned files  
  ↓  
**03_train_and_save_model.py**  
  → Train XGBoost model with `GridSearchCV`  
  → Evaluate RMSE and MAE  
  → Saved final model as `final_model.json` (XGBoost native format for cross-platform compatibility)
  
  ↓  
**04_predict_from_excel.py**  
  → Load Excel row  
  → Apply cleaning and column alignment  
  → Predict charges using XGBoost  
  → Save result as `excel_test_row_with_prediction.csv`



## Model Performance Summary

Trained a supervised regression model to predict medical insurance charges using customer demographic and health-related attributes. The key steps and results are outlined below:

Model Choice: XGBoost Regressor
We selected XGBoost, a powerful gradient boosting algorithm, due to its:
	- Ability to handle nonlinear relationships and interactions
	- Built-in regularization to reduce overfitting
	- High performance on structured/tabular datasets

Training Process:
1.	Data Source: Cleaned and preprocessed training data was loaded.
2.	Target Variable: charges (continuous numerical value representing insurance cost)
3.	Features Included:
	- Age, sex, BMI, number of children, smoker status, region (one-hot encoded)
	- Two interaction terms: bmi × smoker, bmi × sex
4.	Model Optimization:
	- Performed hyperparameter tuning using GridSearchCV with 5-fold cross-validation
5.	Parameters tuned:
	- n_estimators: number of boosting rounds
	- max_depth: max tree depth
	- learning_rate: step size shrinkage

Best Parameters Found:
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100
}

Explanation of Selected XGBoost Parameters:
In this project, we used XGBoost, a powerful gradient boosting algorithm, to predict medical charges. We tuned the following key hyperparameters using GridSearchCV and selected the best based on performance:
1. learning_rate (also called eta)
   	Selected Value: 0.1
   
	Role: Controls how much each tree contributes to the overall model.

	Why this works: A moderate value like 0.1 balances learning speed and accuracy. Smaller values (e.g., 0.01) require more trees but may generalize better, while larger values (e.g., 0.3) can lead to overfitting. This value was optimal based on cross-validation.

3. max_depth
	Selected Value: 3

	Role: Limits how deep each decision tree can go.

	Why this works: A smaller depth reduces model complexity and helps prevent overfitting. A depth of 3 captures important feature interactions without memorizing the training data.
5. n_estimators
	Selected Value: 100

	Role: Number of trees (boosting rounds) the model builds.

	Why this works: With a moderate learning rate, 100 estimators are enough to converge to a good solution without excessive computation time.

Why We Didn’t Use Other Advanced Parameters:
Although XGBoost provides options for tree methods (grow_gpu_hist, grow_quantile_histmaker, etc.) and interaction constraints, we focused on core parameters relevant to our dataset and use case. The advanced options are more applicable to:

	- Distributed or GPU training
 
	- Extremely large datasets
 
	- Fine-tuning control over feature interactions
 
Since our project focused on model interpretability and accuracy on tabular health data, these were unnecessary. These values reflect a well-regularized model with controlled complexity and learning rate.

Evaluation Metrics (on Training Set):

	RMSE (Root Mean Squared Error): 3920.00
 
	- Measures average magnitude of error and Penalizes larger errors more heavily
 
	MAE (Mean Absolute Error): 2145.73
 
	- Measures average absolute difference between predicted and actual values
 
The model predicts charges with an average error of around $2,100, which is acceptable given the dataset's variability.  These metrics indicate the model is able to predict insurance charges within a few thousand dollars on average, which is reasonable given the wide range of real-world charges in the dataset.

Final Output:
•	Trained model saved as final_model.pkl using pickle
•	Ready for use in real-time prediction with new user input (e.g., Excel file)


---

## Excel Prediction Tool

We created a script (`04_predict_from_excel.py`) that allows users to input a single row of new data via a CSV file (formatted like Excel) and receive a predicted insurance charge based on the trained model.


### How It Works:
1. Load a new row of input data (e.g., `excel_test_row.csv`)
2. Apply the `prep_data()` function to clean and transform the input
3. Align columns with the trained model
4. Predict medical insurance charges using the saved XGBoost model
5. Export the result to `excel_test_row_with_prediction.csv`

### Example Output:
The script adds a new column, `predicted_charges`, to the original input:

| age | sex    | bmi  | children | smoker | region    | predicted_charges |
|-----|--------|------|----------|--------|-----------|--------------------|
| 54  | female | 33.1 | 2        | yes    | southeast | 19,267.39          |

> *Missing values for `age`, `bmi`, or `children` are automatically imputed using training medians.*

## Final Model Reproducibility Confirmed (as of May 8, 2025)

The final model (saved as `final_model.json`) was successfully retrained using `03_train_and_save_model.py`, with **RMSE = 3835.30 and MAE = 2107.33**.

Using this model, `04_predict_from_excel.py` generated a prediction of **$45,411.15** for the input row stored in `excel_test_row.csv`. The output file `excel_test_row_with_prediction.csv` reflects this value.

This confirms full alignment between:
- The latest `prep_data()` logic in `02_data_cleaning.py` (includes interaction terms)
- The trained model in `final_model.json`
- The prediction script in `04_predict_from_excel.py`

All components are reproducible, modular, and reflect final project updates.

 
## Imputation Notes
To support real-world usage, median imputation is applied to the following numeric fields if missing:

- `age`: 39.5  
- `bmi`: 30.21  
- `children`: 1.0  

These values were derived from the training dataset. Imputation ensures robustness in the Excel prediction tool, but results may vary if inputs deviate significantly from these medians.


### Output Location:
The final file is saved as:  
`excel_test_row_with_prediction.csv`

---

## Case Study: Prediction Interpretation

**Input Summary:**
- Age: 54  
- Sex: Female  
- BMI: 33.1  
- Children: 2  
- Smoker: Yes  
- Region: Southeast  

**Predicted Charges:** $45,411.15

## Interpretation:

- This prediction reflects a high-risk medical profile resulting in significantly elevated costs. The individual is a smoker and has a BMI above 30, classifying them as obese, two of the strongest individual predictors of increased medical insurance charges. Importantly, our model includes interaction terms (`bmi × smoker`, `bmi × sex`), which capture the **compounding effect** of these variables.
  
- While being female and having two children may modestly reduce costs on average, these protective factors are outweighed by the behavioral risk factors. At age 54, the individual is older than the dataset average, but the model suggests that lifestyle-related predictors — especially the combination of smoking and obesity — have a more dominant effect on predicted charges.

- This case confirms that the model correctly identifies and weights **both main effects and interactions**, aligning with healthcare cost literature and underwriting practices. The high predicted charge of $45,411.15 is consistent with the elevated health risk this profile represents, reinforcing the importance of including interaction terms when modeling medical insurance costs.


## Limitations & Future Work

While our project provided valuable insights into how demographic and lifestyle factors affect medical insurance costs, several limitations remain:

### Limitations
- **Missing key variables**: The dataset did not include critical factors such as income level, employment status, pre-existing health conditions, or plan type. These omitted variables may lead to bias or reduce model accuracy.
- **No temporal context**: It’s unclear whether `charges` represent monthly, annual, or per-claim costs. Without time granularity, the interpretation of the output is constrained.
- **Synthetic nature of data**: The absence of missing values and some uniform distributions suggests the dataset may have been pre-cleaned or simulated, limiting its realism.
- **Overfitting in tree-based models**: Both Random Forest and Decision Tree models overfitted the training data, underperforming on unseen data without further tuning or regularization.
- **Model Serialization Note**:
While initial versions of the trained model were saved using `pickle`, we encountered platform compatibility issues when attempting to reload the model outside of the original environment. To address this, the final version of the trained model was exported using XGBoost’s native `.json` format (`final_model.json`), which ensures full cross-platform reproducibility.

Future users should be aware that `.pkl`-based models may not deserialize properly across systems unless dependencies are tightly controlled. We recommend using the `.json` format for broader portability.

### Future Work
- **Model expansion**: Future versions could explore additional algorithms like SVM, XGBoost (with deeper tuning), or neural networks to compare non-linear modeling approaches.
- **Explainability tools**: Incorporating SHAP or LIME could enhance model transparency, especially for use in insurance policy decision-making.
- **Real-world dataset**: Applying the workflow to actual claims data (with regulatory permission) would strengthen generalizability and ethical alignment.
- **Pipeline deployment**: With a few modifications, the Excel prediction pipeline could be converted into a lightweight Flask app or API endpoint for real-time use.

Despite these limitations, our project provided a strong foundation in modeling, data preparation, and interpretation, ready to be extended into more production-grade workflows.


## Conclusion & Acknowledgments

This project offered a hands-on opportunity to apply machine learning techniques to a real-world problem: predicting medical insurance charges based on demographic and lifestyle factors. We explored the full modeling pipeline—from exploratory data analysis and feature engineering to model training, evaluation, and deployment.

Key takeaways include:
- The importance of model generalization over complexity
- How interaction effects (e.g., BMI × Smoker) shape prediction outcomes
- The value of clean, modular code for collaborative workflows

We would like to thank Professor Veliche for their guidance and feedback throughout the course. This project helped reinforce concepts from ADEC 7430 while giving us a practical framework for future data science work.

Special thanks to:
- **Gina Occhipinti** for leading exploratory analysis and data visualization
- **Kache Lee** for data preparation, transformation, and Excel prediction tooling
- **Reuben Vincent** for model training, evaluation, and tuning with XGBoost

This repository demonstrates not just predictive modeling, but also the collaborative effort required to turn data into actionable insights.
