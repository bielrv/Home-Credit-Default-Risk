# Home Credit Default Risk

## Setup

1. Clone current repo in desired parent folder
```bash
git clone https://github.com/bielrv/Home-Credit-Default-Risk.git
```

2. Create virtual environment `venv` using virtualenv
```bash
virtualenv -p python3 venv
```

3. Activate virtual environment
```
source venv/bin/activate
```

4. Install repo dependencies
```bash
pip install -r requirements.txt
```
5. Download competition dataset
```bash
kaggle competitions download -c home-credit-default-risk
```
*(kaggle library already in requirements)*

6. (Optional) - If you are using a kernel you might need to run:  
```bash
python3 -m pip install ipykernel    
python3 -m ipykernel install --user
```

## End-to-End Machine Learning Steps

How to go though a ML project E2E

1. Look at the big picture
2. Get the data
3. Discover and visualize the data to gain insights
4. Prepare the data for Machine Learning algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor, and maintain your solution

### 1. Look at the big picture
1. Frame the problem
2. Select a Performance Measure
3. Check the assumptions

### 2. Get the Data
1. Create the Workspace
2. Download the Data
3. Take a Quick Look at the Data Structure
4. Create a Test Set

### 3. Discover and visualize the data to gain insights
0. Try to get insight from a field expert for these steps
1. Create a copy of the data for exploration
2. Create Jupyter Notebook to keep record of the data exploration
3. Study each attribute and its characteristics:
    - Name
    - Type
    - % Missing values
    - % Zero values
    - Noisiness and type of noise
    - Is the data useful for the task?
    - Type of distribution
4. For supervised learning tasks, identify the target attribute
5. Visualize the data
6. Study correlations between attributes
7. Study how you would solve the problem manually
8. Identify the promising transformations you may want to apply
9. Identify extra useful data
10. Document what you have learned

- Visualizing data
- Looking for Correlations
- Experimenting with Attribute Combinations



### 4. Prepare the data for Machine Learning algorithms
1. Data Cleaning
2. Handling Text and Categorical Attributes
3. Custom Transformers
4. Feature Scaling
5. Transformation Pipelines

### 5. Select a model and train it
1. Train and Evaluating on the Training Set
2. Better Evaluation Using Cross-Validation

### 6. Fine-tune your model
1. Grid Search
2. Randomized Search
3. Ensemble Methods
4. Analyze the Best Models and Their Errors
5. Evaluate Your System on the Test Set

### 7. Present your solution

### 8. Launch, monitor, and maintain your solution


## Notes

Use `pip freeze > requirements.txt` to refresh requirements
