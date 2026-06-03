# Project Title: Google Play Store Apps Analysis and Rating Prediction

## 1. Project Description
The main goal of the project is to analyze data from the Google Play Store, discover useful patterns and trends, and apply a simple predictive modeling technique in order to enhance the developers understanding and knowledge of how and which factors influence applications ratings and bring some awareness regarding the user behavior so that development companies can improve their apps or create better ones.

## 2. Dataset
* **Source:** Kaggle (Google Play Store Apps) https://www.kaggle.com/datasets/lava18/google-play-store-apps
* **Description:** The dataset contains information about Android applications available on the Google Play Store. Main attributes include: Application Name, Category, Rating, Number of Reviews, Number of Installs, Price, Content Rating, Application Size and Android Version. The original dataset had 10.841 records and now the cleaned dataset has 8.196 records. Data cleaning removed duplicate applications, invalid ratings, and handled missing values.

## 3. Technology Stack
The following technologies were used throughout the project:
* **Language:** Python 3.x
* **Data Processing:** Pandas, PySpark
* **Storage:** CSV
* **Visualization:** Matplotlib, Power BI
* **Version Control:** GitHub
* **Additional Libraries: NumPy

## 4. Pipeline Architecture
The project follows the pipeline below:
1. **Data Ingestion:** Load the Google Play Store dataset from a CSV file.
2. **Preprocessing:** Removing duplicate records, invalid ratings, handling missing values, converting numerical fields to proper data types and creating the Size_MB feature
3. **Analysis:** Exploratory Data Analysis with Pandas and Matplotlib, creating visualization to identify trends and patterns, using PySpark for Big Data processing and aggregation and Predictive modeling technique to classify applications as High Rated or Low Rated
4. **Output:** Charts and visualizations saved in the project folder, PySpark analytical results, Predictive modeling results, Power BI dashboard for final presentation.

## 5. Installation & Usage
How can someone else run your code?
1. Clone the repository: `git clone https://github.com/alexmnt37/big-data-google-playstore-analysis.git`
2. Install dependencies: `pip install pandas numpy matplotlib pyspark`
3. Run the main notebook/script: `python src/01_data_cleaning.py; python src/02_eda.py; python src/03_pyspark_processing.py; python src/04_rating_prediction.py` Generated charts will be saved automatically inside: 'reports/figures/'

## 6. Key Insights
* **Insight 1:** Most applications on Google Play are free rather than paid
* **Insight 2:** Paid applications tend to have slightly higher average ratings compared to free applications.
* **Insight 3:** Applications with a large number of reviews and installs are more likely to receive higher ratings.
* **Insight 4:** A simple rule-based predictive model was implemented. The model learned thresholds from the training data and used them to predict whether an application would be highly rated.

## 7. Author
* **Name:** Muntean Alexandru
* **Course:** Big Data Fundamentals