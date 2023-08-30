# Data Science Interview Assignment

## Introduction

If you read this file, you have passed our initial screening. Well done! :clap: :clap: :clap:

:rocket: The next step to join the Data Science team of [xtream](https://xtreamers.io) is this assignment. 
You will find several datasets: please choose **only one**.
For each dataset, we propose several challenges. You **do not need to
complete all of them**, but rather only the ones you feel comfortable about or the ones that interest you. 

:sparkles: Choose what really makes you shine!

:watch: The deadline for submission is **10 days** after you are provided with the link to this repository, so that you can move at your own pace.

:heavy_exclamation_mark: **Important**: you might feel the tasks are too broad, or the requirements are not
fully elicited. **This is done on purpose**: we wish to let you take your own way in 
extracting value from the data and in developing your own solutions.

### Deliverables

Please fork this repository and work on it as if you were taking on a real-world project. 
On the deadline, we will check out your work.

:heavy_exclamation_mark: **Important**: At the end of this README, you will find a blank "How to run" section. 
Please write there instructions on how to run your code.

### Evaluation

Your work will be assessed according to several criteria, for instance:

* Work Method
* Understanding of the business problem
* Understanding of the data
* Correctness, completeness, and clarity of the results
* Quality of the codebase
* Documentation

:heavy_exclamation_mark: **Important**: this is not a Kaggle competition, we do not care about model performance.
No need to get the best possible model: focus on showing your method and why you would be able to get there,
given enough time and support.

---

### Diamonds

**Problem type**: regression

**Dataset description**: [Diamonds readme](./datasets/diamonds/README.md)

Don Francesco runs a jewelry. He is a very rich fellow, but his past is shady: be sure not to make him angry.
Over the years, he collected data from 5000 diamonds.
The dataset provides physical features of the stones, as well as their value, as estimated by a respected expert.

#### Challenge 1

**Francesco wants to know which factors influence the value of a diamond**: he is not an expert, he wants simple and clear messages.
However, he trusts no one, and he hired another data scientist to get a second opinion on your work.
Create a Jupyter notebook to explain what Francesco should look at and why.
Your code should be understandable by a data scientist, but your text should be clear for a layman.

[Exploratory data analysis](eda.ipynb)

#### Challenge 2

Then, Francesco tells you that the expert providing him with the stone valuations disappeared.
**He wants you to develop a model to predict the value of a new diamond given its characteristics**.
He insists on a point: his customer are not easy-going, so he wants to know why a stone is given a certain value.
Create a Jupyter notebook to meet John's request.

[Catboost regression model](model.ipynb)


#### Challenge 3

Francesco likes your model! Now he wants to use it. To improve the model, Francesco is open to hire a new expert and 
let him value more stones.
**Create an automatic pipeline capable of training a new instance of your model from the raw dataset**. 

[Python script for model training](train.py)

#### Challenge 4

Finally, Francesco wants to embed your model in a web application, to allow for easy use by his employees.
**Develop a REST API to expose the model predictions**.

[A simple FastAPI script](api.py)

---

## How to run

### Prerequisites

1. Make sure you have Python 3.11 installed.
2. It's recommended to use a virtual environment for project dependencies.

### Installation

1. Clone this repository.
   ```
   git clone <REPO_URL>
   cd <REPO_DIRECTORY>
   ```

2. Install the required packages.
   ```
   pip install -r requirements.txt
   ```

### Usage

#### 1. Training the Model

Use the `train.py` script to train the model on your dataset.

```
python train.py --data_path /path/to/diamonds.csv
```

Additional parameters such as model depth, learning rate, etc., can be set using the command-line arguments. Run `python train.py -h` to see all available options.


#### 2. Running the API

Start the FastAPI server with:

```
uvicorn api:app --reload
```

Once the API is running, you can:

- Send a POST request to `http://127.0.0.1:8000/predict/` with diamond data in JSON format to get a price prediction.

   Example with `curl`:
   ```
   curl -X POST http://127.0.0.1:8000/predict/ \
   -H "Content-Type: application/json" \
   -d '{
       "carat": 0.3,
       "cut": "Good",
       "color": "E",
       "clarity": "SI2",
       "depth": 61.5,
       "table": 55,
       "x": 4.29,
       "y": 4.31,
       "z": 2.63
   }'
   ```

- Send a POST request to `http://127.0.0.1:8000/predict_csv/` with a CSV file containing diamond data to get predictions for each row.

   Example with `curl`:
   ```
   curl -X POST http://127.0.0.1:8000/predict_csv/ \
   -H "Content-Type: multipart/form-data" \
   -F "file=@/path/to/test_data.csv"
   ```

- Train a new model by sending a POST request to `http://127.0.0.1:8000/train/` with a CSV file containing diamond data.

### Notes

- Ensure that the diamond data provided has all the necessary columns (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`).