# Introduction

Assuming we have small or large text and we want to know which category the text belongs, how can we predict what the subject of the text is?
This question is related to the problem of classification, a classification problem is when the output variable is a category, In our scenario the output variable it’s like “politics, “food”, “science” etc.

In this project we present a model that deals with the classification problem. To solve the text classification problem we set a goal to our research: to see if we can build a model that will classify the subject of the text according to the frequency of words in each category.

In our research we focus on 7 categories:

* Technology
* Politics
* Sports
* Entertainment
* Weather
* Science
* Business
* Food

> For full project click [here](https://efi1397.github.io/HIT.DataScience.FinalProject/)

## Get Started

We created a web application for our model, in this secion we will describe how to get started with the web application.

1.  Open the project (or just the web app folder) in your IDE :  
```
git clone https://github.com/efi1397/HIT.DataScience.FinalProject.git
```

2.  Create venv to the project  
```
python3 -m venv /path/to/new/virtual/environment
```

3. Switch the source to `venv/bin/activate`


4. Execute `pip3` install command  

```
pip3 install -r requirements.txt
```

5. Execute the program  

```
python3 main.py
```

6. Open your browser at `localhost:8080`  

![image](https://user-images.githubusercontent.com/76474133/176117769-4a91579a-e55d-481e-8057-2108a164baef.png)
