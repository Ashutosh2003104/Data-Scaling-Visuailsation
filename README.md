Data-Scaling-Visuailsation

This repository contains a Python script that demonstrates the effect of feature scaling on a dataset using StandardScaler from scikit-learn. The script visualizes the data before and after scaling using scatter plots. Below is a detailed explanation of the code and its purpose.

Table of Contents
    Introduction
    Code Explanation
    Dependencies
    Data Preparation
    Scaling the Data
    Visualization
    How to Run the Code
    Results
    Contributing
    

Introduction

Feature scaling is a crucial preprocessing step in machine learning. It ensures that all features contribute equally to the model's learning process by transforming the data to have a mean of 0 and a standard deviation of 1. This repository demonstrates the effect of scaling using StandardScaler and visualizes the transformation using scatter plots.

Code Explanation

Dependencies

The script uses the following Python libraries:

pandas: For data manipulation.
matplotlib: For data visualization.
scikit-learn: For data scaling.

Make sure you have these libraries installed before running the script. You can install them using pip:

pip install pandas matplotlib scikit-learn


Data Preparation

The dataset used in this example contains two features: Age and EstimatedSalary. The data is split into training and testing sets using train_test_split from scikit-learn.

from sklearn.model_selection import train_test_split

Sample Code:

                        # Example dataset
                        df = pd.DataFrame({
                            'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                            'EstimatedSalary': [50000, 60000, 70000, 80000, 90000, 100000,                         110000, 120000, 13000,140000],
                            'Purchased': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                        })
                        
                        # Split the data into training and testing sets
                        X_train, X_test, y_train, y_test =                                                         train_test_split(df.drop('Purchased', axis=1), df['Purchased'],                                        test_size=0.3, random_state=0)

Scaling the Data: 

The StandardScaler is used to standardize the features in the training set. The same scaling parameters are then applied to the test set to avoid data leakage.

Sample code:

                from sklearn.preprocessing import StandardScaler
                
                # Initialize the scaler
                scaler = StandardScaler()
                
                # Fit the scaler on the training data and transform it
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Transform the test data using the same scaler
                X_test_scaled = scaler.transform(X_test)
                
                # Convert scaled arrays back to DataFrames for visualization
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

Visualization

The script uses matplotlib to create side-by-side scatter plots comparing the data before and after scaling.


Sample Code:

                import matplotlib.pyplot as plt
                
                # Create subplots
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                
                # Plot the data before scaling
                ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
                ax1.set_title("Before Scaling")
                ax1.set_xlabel("Age")
                ax1.set_ylabel("Estimated Salary")
                
                # Plot the data after scaling
                ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'],                                 color='red')
                ax2.set_title("After Scaling")
                ax2.set_xlabel("Age (Scaled)")
                ax2.set_ylabel("Estimated Salary (Scaled)")
                
                # Display the plots
                plt.show()


How to Run the Code

Clone this repository:
        git clone https://github.com/your-username/data-scaling-visualization.git
        
Navigate to the repository:
        cd data-scaling-visualization

Install the required dependencies:
        pip install -r requirements.txt
        
Run the script:
        python scaling_visualization.py



Results:


The script generates two scatter plots:

Before Scaling: Shows the original distribution of the Age and EstimatedSalary features.

After Scaling: Shows the distribution of the features after standardization.

The plots demonstrate how scaling transforms the data, making it more suitable for machine learning algorithms that are sensitive to feature scales.


Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

Contact
For questions or feedback, please contact Ashutosh Roy at riddlesbyashutosh@gmail.com



