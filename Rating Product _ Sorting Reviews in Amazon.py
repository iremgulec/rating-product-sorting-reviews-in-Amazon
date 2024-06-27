###################################################
# PROJECT: Rating Product & Sorting Reviews on Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is accurately calculating the post-purchase ratings of products.
# Solving this problem means greater customer satisfaction for the e-commerce site, better product visibility for sellers, and a seamless shopping experience for buyers.
# Another problem is the correct ranking of reviews given to products.
# Misleading reviews being highlighted can directly affect product sales, causing both financial loss and loss of customers.
# By solving these two fundamental problems, e-commerce sites and sellers can increase their sales while customers can complete their purchasing journey smoothly.

###################################################
# Dataset Story
###################################################

# This dataset, containing Amazon product data, includes various metadata along with product categories.
# It contains user ratings and reviews for the most reviewed product in the Electronics category.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: User Name
# helpful: Helpfulness rating of the review
# reviewText: Review
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time (unix)
# reviewTime: Review time (raw)
# day_diff: Number of days since the review
# helpful_yes: Number of times the review was found helpful
# total_vote: Number of votes the review received


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# TASK 1: Calculate the Average Rating Based on Recent Reviews and Compare it with the Existing Average Rating
###################################################

# In the provided dataset, users have given ratings and written reviews for a product.
# The goal of this task is to evaluate the given ratings by weighting them according to the date.
# The initial average rating should be compared with the weighted rating based on the date.

###################################################
# Step 1: Read the Dataset and Calculate the Average Rating of the Product.
###################################################

df = pd.read_csv("amazon_review.csv")
df["overall"].mean()

df.head()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################


###################################################
# Step 2: Calculate the Weighted Average Rating Based on the Date.
###################################################

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() # 4.696
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() # 4.64
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() # 4.57
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() # 4.45


# Determining Time-Based Average Weights
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100


time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22) # 4.59559316512811

df["overall"].mean() # 4.58


###################################################
# TASK 2: Determine the 20 Reviews to be Displayed on the Product Detail Page
###################################################


###################################################
# Step 1: Create the helpful_no Variable
###################################################

# Note:
# total_vote is the total number of up-down votes given to a review.
# up means helpful.
# The helpful_no variable does not exist in the dataset and needs to be created from the existing variables.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

df.head()

###################################################
# Step 2: Calculate and Add the score_pos_neg_diff, score_average_rating, and wilson_lower_bound Scores to the Data
###################################################


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate the Wilson Lower Bound Score

- The lower bound of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
- The calculated score is used for product ranking.
- Note:
If the scores are between 1-5, they are marked as 1-3 negative and 4-5 positive to make them suitable for Bernoulli.
This can also bring some problems. Therefore, it is necessary to perform Bayesian average rating.


    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

#####################
# score_pos_neg_diff
#####################


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_pos_neg_diff", ascending=False).head(20)



# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_average_rating", ascending=False).head(20)



# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)




