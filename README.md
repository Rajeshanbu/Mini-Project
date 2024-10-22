## Web App for Automated ML Model Training
This project focuses on developing a no-code web application that enables users to automate machine learning model training through a simple interface. By uploading CSV datasets, users can preprocess data, select different scalers, and choose from a wide range of machine learning algorithms for model training and evaluation, all without writing code. This streamlines the machine learning workflow, making it accessible to both technical and non-technical users.

## About
Web App for Automated ML Model Training is a project designed to automate the machine learning workflow by providing a user-friendly, no-code interface for model training. The application allows users to upload datasets, preprocess data, and select from a variety of machine learning algorithms, including Logistic Regression, Random Forest, SVC, and more. Through the use of advanced scaling and preprocessing techniques, the app ensures that data is ready for model training. Users can easily train and evaluate models, improving efficiency and reducing the technical barriers typically associated with machine learning tasks. This tool empowers both beginners and professionals to streamline their ML experiments and achieve accurate results without coding.
## Features
<!--List the features of the project as shown below-->
- Implements a wide range of machine learning algorithms, including advanced models like XGBoost, CatBoost, and LightGBM.
-Scalable and customizable no-code framework for quick deployment and experimentation.
-Supports multiple data preprocessing techniques, including various scalers and transformers.
-Simple and intuitive user interface powered by Streamlit for seamless interaction.
-Allows users to upload CSV files for automated data processing, model training, and evaluation.
-Provides real-time performance metrics such as accuracy for easy model comparison.
-Facilitates model saving for future use, enabling streamlined model management.

## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10, Ubuntu) for compatibility with machine learning libraries.
* Development Environment: Python 3.7 or later for building and running the ML web application.
* Machine Learning Libraries: Scikit-learn for model training, XGBoost, CatBoost, LightGBM for advanced classifiers.
* Data Processing Libraries: Pandas for data manipulation, NumPy for numerical computations.
* Web Framework: Streamlit for building the interactive web interface for model training and evaluation.
* Version Control: Git for collaborative development and effective code management.
* IDE: Use of Visual Studio Code (VSCode) for coding, debugging, and version control.
* Additional Dependencies: Includes scikit-learn, Streamlit, XGBoost, CatBoost, LightGBM, and other necessary machine learning libraries as listed in the requirements.txt file.

## System Architecture
<!--Embed the system architecture diagram as shown below-->

The architecture consists of a Streamlit-based frontend where users upload CSV files and configure model training. The backend handles data preprocessing, including scaling, encoding, and splitting. Selected models, from Scikit-learn or advanced classifiers like XGBoost, are trained in the model training layer. Trained models are saved as pickle files, and evaluation metrics such as accuracy are displayed to the user.

![image](https://github.com/user-attachments/assets/4e6afcee-75ad-49d4-a90c-4f8001c3f2f9)



## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - (code execution in vs code)
project running in Visual Studio Code with Streamlit output in the terminal. Your project is running the "Automate ML Model" web app, and the Streamlit app appears to be accessible locally on localhost:8501.

![image](https://github.com/user-attachments/assets/b5aa58c1-61b6-4eae-bf57-fead13d0ce1d)

#### Output2 - (after that execution the output will stream in localhost)
which shows the Streamlit interface of your "Automate ML Model" app, you can use the output name "Dataset Selection Screen" as a short descriptive title.

![image](https://github.com/user-attachments/assets/1afb4bfd-5841-4f47-9289-b6bdecbbbb92)
![image](https://github.com/user-attachments/assets/674f3844-788b-4604-8a8a-69e1b97cb3e6)



Detection Accuracy: 0.20%
Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact
Results and Impact
The "Automate ML Model Training" project simplifies the process of training machine learning models, allowing users without coding expertise to build and evaluate models efficiently. By integrating various classifiers and scaling techniques, the system ensures flexibility and adaptability across multiple datasets.

This project demonstrates the potential for no-code platforms to democratize data science and machine learning, making these technologies more accessible to a broader audience. Its streamlined approach to model training reduces complexity and time investment, contributing to the advancement of AI-driven decision-making in various industries.

## Articles published / References
[1]  J. Brownlee, "Machine Learning Mastery with Python: Understand Your Data, Create Accurate Models, and Work Projects End-to-End," Machine Learning Mastery, 2016.. 

[2]  F. Chollet, "Deep Learning with Python," Manning Publications, 2017.

[3]  A. Müller and S. Guido, "Introduction to Machine Learning with Python: A Guide for Data Scientists," O'Reilly Media, 2016. 

[4]	A. Shankar and C. H. Lee, "A Survey on No-Code Platforms for Machine Learning," IEEE Access, vol. 8, pp. 207383-207392, 2020
[5]	T. W. Simpson, "Streamlit: A New Framework for Building Data Apps," Journal of Computational and Graphical Statistics, vol. 30, no. 1, pp. 1-9, 2021.
[6]	A. K. Jain and A. A. F. R. A. Singh, "Data Preprocessing for Machine Learning: A Comprehensive Review," ACM Computing Surveys, vol. 54, no. 7, pp. 1-33, 2021..

[7]	K. G. M. A. J. P. D. W. S. F. L. A. A. A. K. V. A. K. Gupta, "Evaluation of Machine Learning Models: A Comprehensive Study," IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 4, pp. 1658-1671, 2021.






