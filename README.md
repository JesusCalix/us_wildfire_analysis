# canonical_technical_assessment
Technical assessment for canonical interview process

How to run the project?

1. Clone or download the repository
2. Download and extract the dataset from the following link: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires?resource=download&select=FPA_FOD_20170508.sqlite
3. The extracted file should be an sqlfile. Store it in data/ folder
4. Create a virtual environment with python (optional) -> python -m venv venv
5. Install neccessary libraries: pip install -r requirements.txt
6. Run the script: python run.py
7. Access the webapplication on: http://127.0.0.1:5000/ 

************************************************************************************************************

<h3>Q1: Have wildfires become more or less frequent over time?</h3>
<h3>Q2: What counties are the most and least fire-prone?</h3>
<h3>Q3: Given the size, location and date, can you predict the cause of a wildfire?</h3>
