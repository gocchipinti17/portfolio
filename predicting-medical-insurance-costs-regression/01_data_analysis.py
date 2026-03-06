""" Author: Gina Occhipinti """

#%% ###################### Import Packages ###################### 
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


#%% ###################### Set Up WD ########################### 
# Set working directory to the script's folder
os.chdir(Path(__file__).resolve().parent)

#%% ###################### Import Data ############################
# Import data file
df = pd.read_csv('insurance.csv')

# Check dataset structure
print(df.info())
print(df.head())
      
""" Appears to be no missing values"""

#%%  ###################### Summary Stats #########################
# Print summary stats
df.describe()

#%%  ###################### Check Distributions ################
# Separate categ data
categ_X_cols = df[["region", "sex", "smoker", "children"]]

# Create for loop to enumerate over each col with a count plot
for i, column in enumerate(categ_X_cols.columns):
    plt.figure(i)
    sns.countplot(categ_X_cols, x = column)
    plt.title(f"Distribution of {column}")
    plt.show()
    
# Separate numeric data
numerical_cols = df[["age", "bmi", "children", "charges"]]

# Create for loop to enumerate over each col with a histogram plot
for i, column in enumerate(numerical_cols.columns):
    plt.figure(i)
    sns.histplot(numerical_cols, x = column)
    plt.title(f"Distribution of {column}")
    plt.show()
    
"""
- Age has a slight right skew (high count of people age 20) otherwise is uniformly distributed
- Sex and Region uniformly distributed
- BMI normally distributed
- Children has a right skew
- Significantly more non-smokers than smokers
- Charges has a right skew
"""

## Create violin plots
for i, column in enumerate(categ_X_cols.columns):
    plt.figure(i)
    sns.violinplot(x = column, y = "charges", data = df)
    plt.title(f"Violin Distribution of {column}")
    plt.savefig(f"violin_plot_{column}.png", bbox_inches="tight", dpi=300)
    plt.show()

"""
Jumps in high insurance costs tend to be associated with smokers. Males appear to be somewhat more costly.
"""

#%%  ###################### Check Outliers #####################
# Extract numerical cols for boxplots
numerical_cols = df[["age", "bmi", "children", "charges"]]

# Create for loop to enumerate over each col with a boxplot
for i, column in enumerate(numerical_cols.columns):
    plt.figure(i)
    plt.boxplot(numerical_cols[column])
    plt.title(f"Boxplot of {column}")
    plt.show()
    
"""
- BMI has some outliers - population generally obsese though
- Charges has some outliers - some have very expensive medical bills
"""

#%%  ###################### Check Missing Values ################
# Check instances of missing values in the data 
missing_values = df.isnull()
print(missing_values)

#%%  ###################### Check Collinearity ################
# Create function to one-hot encode categorical varibles for visualization
def prep_data(df):
    df_analysis = df.copy()

    # One-hot encode categorical variables
    df_analysis = pd.get_dummies(df_analysis, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)

    return df_analysis

df_analysis = prep_data(df)

## Check correlation using corr
# Update pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(df_analysis.corr())

# Correlation matrix visualization
corr = df_analysis.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, cmap='BrBG')
plt.savefig("correlation_map.png", bbox_inches="tight", dpi=300)
plt.show()

"""
Collinearity not an issue - appears to be few correlated variables
Strong correlation between charges and smoking
Weak correlations between:
    BMI and Charges
    Age and Charges
    BMI and Southeast
"""

#%%  ###################### Check Multicollinearity ################

## Check VIF
# Define X
X = df_analysis[['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

# Print VIF
print(vif_data)

"""
Have mutlicollinearity for Age and BMI.
Somewhat expected - older people in the US might be more overweight.
"""

#%%  ###################### Check Predictor/Response Relationships - Categ. Variables ################
## View charges by categorical variables
# Create for loop to enumerate over each categ col with a scatterplot
for i, column in enumerate(categ_X_cols.columns):
    plt.figure(i)
    sns.catplot(data=df, x=categ_X_cols[column], y="charges")
    plt.title(f"Charges by {column}")
    plt.show()
    
"""
Charges appear equal between Regions and Sex.
Clear disparity between Charges for Smokers vs. Non-Smokers
Higher Charges for those who have less children on insurance compared to more (interesting)
"""

#%%  ###################### Check Non-Linearity ################
####### View charges by numerical variables #####
#### Age #####
## Bin age to visualize more clearly

# Create copy of the data
df_analysis2 = df.copy()

# Create bins
bins_age = [0, 20, 35, 45, 55, 65]

# Create bin column and cut age
df_analysis2['binned_age'] = pd.cut(df_analysis2['age'], bins_age)

# Verify outcome
print(df_analysis2[['binned_age', 'age']])

# Check data type
df_analysis2['binned_age'].info()

# Plot binned column against charges
sns.catplot(data=df_analysis2, x='binned_age', y="charges")
plt.xticks(rotation=30)
plt.title("Charges vs. Binned Age")
plt.show()

# Use a LOWESS smoother
sns.regplot(data=df_analysis2, 
            x="age", 
            y="charges", 
            lowess=True, 
            scatter_kws={'alpha':0.3}, 
            line_kws={'color':'red'})
plt.title("Regression Plot Charges vs. Age")
plt.show()

#### BMI #####
## Bin BMI to visualize more clearly
# Create bins of BMI representing adult BMI categories from CDC
bins_bmi = [0, 18.5, 24.9, 29.9, 30, 34.9, 39.9, 100]

# Create bin column and cut bmi
df_analysis2['binned_bmi'] = pd.cut(df_analysis2['bmi'], bins_bmi, 
                                    labels=['Underweight', 'Healthy', 'Overweight', 'Obsese', 'Class 1 Obese', 'Class 2 Obese', 'Severe Obese'])

# Verify outcome
print(df_analysis2[['binned_bmi', 'bmi']])

# Check data type
df_analysis2['binned_bmi'].info()

# Plot binned column against charges
sns.catplot(data=df_analysis2, x='binned_bmi', y="charges")
plt.xticks(rotation=30)
plt.title("Charges vs. Binned BMI")
plt.show()

# Use a LOWESS smoother
sns.regplot(x="bmi", 
            y="charges", 
            data=df_analysis2, 
            lowess=True, 
            scatter_kws={'alpha':0.3}, 
            line_kws={'color':'red'})
plt.title("Regression Plot Charges vs. BMI")
plt.show()

"""
Age appears to have a numeric relationship with Charges. We see a straight line positive relationship
where increased Age causes Charges to increase. 

BMI appears to show no relationship between Charges and BMI until we get to the BMI range of Obese (class 1)
where it jumps suggesting some non-linearity)
CDC Adult BMI Categories: https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html 
"""

#%%  ###################### Identify Interactions ################
##### BMI vs. Charges by category #####
df_categs = df[["sex", "region", "smoker", "children"]]
for i, col in enumerate(df_categs.columns):
    plt.figure(i)
    sns.lmplot(x="bmi", y="charges", hue=col, data=df, scatter_kws={'alpha': 0.3})
    plt.title(f"BMI v. Charges by {col}")
    plt.savefig(f"interaction_chart_{col}.png",bbox_inches="tight", dpi=300)
    plt.show()

##### Age vs. Charges by category #####
for i, col in enumerate(df_categs.columns):
    plt.figure(i)
    sns.lmplot(x="age", y="charges", hue=col, data=df, scatter_kws={'alpha': 0.3})
    plt.title(f"Age v. Charges by {col}")
    plt.show()

"""
Age tends to be more additive - it does not appear to have any interactive effects with other variables.
BMI on the other hand does appear to interact with other predictors, particularly Smoker (status) and Sex.
"""

#%% ###################### Split Data ##############################
# Define X and y from df
X = df.iloc[:, 0:6]
y = df.iloc[:, 6]

# Create 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=42)
#%% ###################### Save Data to Feather ##############################

# Save train and test data in feather format
# Create train dataframe
df_train = X_train.join(y_train)
df_train

df_test = X_test.join(y_test)
df_test

# Save to feather
df_train.to_feather("df_train.feather")
df_test.to_feather("df_test.feather")