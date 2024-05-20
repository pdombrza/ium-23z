# Machine learning task
Project done in collaboration with [Błażej Ejzak](https://github.com/Blaczumba).

## Task
The task was to create a model to select best cache level for all music tracks for the next 24 hours.

## Repository structure
Data directiories contain available data, results_xgb directiory contains predictions and errors acquired when testing xgboost models. Test data directiory contains small samples of available datasets for testing purposes.

## Running the app
You have to be in the app directory to run the app
```bash
 cd app
 pip install -r requirements.txt
 uvicorn main:app --reload
```

## Used libraries
* FastAPI for the app itself
* xgboost and Pytorch for the models
* pandas and numpy for data analysis
* matplotlib and seaborn for data visualization
