# canonical_technical_assessment
Technical assessment for canonical interview process

How to run the project?

1. Clone or download the repository
2. Download and extract the dataset from the following link: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires?resource=download&select=FPA_FOD_20170508.sqlite
3. The extracted file should be an sqlfile. Create a folder named "data" in the working folder. Store the sqlite file in data/ folder
4. Create a virtual environment with python (optional)
5. Install neccessary libraries: pip install -r requirements.txt
6. Run the script: python run.py
7. Access the webapplication on: http://127.0.0.1:5000/ 

************************************************************************************************************

<h3>Q1: Have wildfires become more or less frequent over time?</h3>

We can notice that there is not trend so we cannot determine if the fires are increasing or decreasing as the years pass on.

<h3>Q2: What counties are the most and least fire-prone?</h3>

The top 5 counties where fires happen the most are: Washington, Lincoln, Jackson, Marion and Cherokee.

<h3>Q3: Given the size, location and date, can you predict the cause of a wildfire?</h3>

The cause of the wildfires can be so many, for this model I decided to reduce the number of possible causes to improve model accuracy. I reduced them to: Nature, Accident, Human and Other. <br>

The prediction of the model is very low, and we can notice that it only performs well for nature label. Including more variables and doing a correlation analysis would definetely improve the model prediction accuracy.
