### StatQuest: Decision Trees

[Link](https://www.youtube.com/watch?v=7VeUPuFGJHk)

A DT asks a question and classifies based on the answer

<img src = "./data/img/diag1.png" height="700" width = "700" align="center">

Note: A classification can be categories or numeric

In the 2nd case we are using mouse wtto predict mouse size

More complex DT:

<img src = "./data/img/diag2.png" height="700" width = "700" align="center">

It combines numeric data:

<img src = "./data/img/diag3.png" height="700" width = "700" align="center">

With Yes/No data:

<img src = "./data/img/diag4.png" height="700" width = "700" align="center">

Notice that cut-off for Resting heart rate need not be same on both sides

<img src = "./data/img/diag5.png" height="700" width = "700" align="center">

Also order of questions need not be same on both sides

The final classifications may be repeated

U start at top and go down till u get to a pt where u cant go further

Thats how a sample is classified

#### Raw table of data to DT:

We want to create a tree that uses **chest pain, good blood circulation, blocked artery status** to predict
**heart disease(y/n)**

We want to decide which of **chest pain, good blood circulation, blocked artery status** should be root node

<img src = "./data/img/diag6.png" height="700" width = "700" align="center">

We start off by exploring how well Chest pain classifies heart disease and build a tree as shown below:

<img src = "./data/img/diag7.png" height="700" width = "700" align="center">


We build similar trees for Good Blood Circulation and blocked Artery

<img src = "./data/img/diag8.png" height="700" width = "700" align="center">

As shown above we dont kno the BA status for this patient. We skip it but there are other alternatives

As there are missing values for a feature the total number of patients in each tree is diff

<img src = "./data/img/diag9.png" height="700" width = "700" align="center">


** because none of the leaf nodes are 100% YES Heart disease or 100% NO they all are considered as "impure"**

To determine which separation is best we need a way to measure and compare impurity

#### Gini method to measure impurity

Gini impurity (GI) is calculated for each leaf node as shown below:

<img src = "./data/img/diag10.png" height="700" width = "700" align="center">

Similarly we calculate GI for right leaf node

The leaf nodes do not reppresent same number of patients

Thus total GI for using Chest pain as root node is the weighted avg of GI of the 2 nodes:

<img src = "./data/img/diag11.png" height="700" width = "700" align="center">


Similarly we calculate GI for all 3 possible root nodes

<img src = "./data/img/diag12.png" height="700" width = "700" align="center">


Good blood circulation has lowest impurity and it separates the people with or without heart disease the best

So first node (root) = GBC

After the split we get 2 leaf nodes

Left: (37 y, 127 n)

Right: (100 y, 33 n)

Now we need to figure out how to separate (and if we should separate further) these patients in the Left and Right

**Lets start with left:**

These are the patients with GBC == true

Just like before we separate these patients based on CP and calculate GI as before

We do same for Blocked Artery

GI for BA = 2.9

This is less than GI for CP and also less than GI for GBC

<img src = "./data/img/diag13.png" height="700" width = "700" align="center">

Thus we use BA in the left part

Resulting tree:

<img src = "./data/img/diag14.png" height="700" width = "700" align="center">

Now we will use CP to try and separate the L->L node(24/25)

These are the patients with GBC = true and BA = true

CP does a good job in separating the patients:

<img src = "./data/img/diag15.png" height="700" width = "700" align="center">

Now we look at node in Root->L->R (13/102)

Lets try and use CP to divide these 115 patients

Note : ** Vast majority (89%) of patients in this node dont have heart disease**

<img src = "./data/img/diag16.png" height="700" width = "700" align="center">

After separating we get a higher GI than before separating

So we make this node a leaf node

<img src = "./data/img/diag17.png" height="700" width = "700" align="center">

We have built the entire LHS of the tree

<img src = "./data/img/diag18.png" height="700" width = "700" align="center">


For RHS we follow same steps:

1. Calculate all GI scores

2. If node otself has lowest score, then there is no point in separating and the node becomes a leaf node

3. If separating the data results in an improvement, pick the separation with the lowest impurity value

Complete tree:

<img src = "./data/img/diag18.1.png" height="700" width = "700" align="center">


#### Numeric data in DT:

Imagine if our features were numeric not just Y/N:

<img src = "./data/img/diag19.png" height="700" width = "700" align="center">

1. Sort patients by wt (lowest to highest)

<img src = "./data/img/diag20.png" height="700" width = "700" align="center">

2. Calculate avg wts for all adjacent patients

3. Calculate GI for each avg wt

<img src = "./data/img/diag21.png" height="700" width = "700" align="center">

In the above diag GI is calculated for wt < 167.5

4. The lowest GI occurs when wt < 205 (GI=0.27)

<img src = "./data/img/diag22.png" height="700" width = "700" align="center">


So this is the cutoff that we will use when we compare wt to CP or BA



#### DT with ranked data and multiple choice data

Ranked data is similar to numeric data, except that now we calculate impurity scores for **all possiblle ranks**

So if rank is from 1 to 4 (4 being best), we calculate impurity scores as:

- rank <= 1

- rank <= 2

- rank <= 3

We dont need <=4 as it includes everyone

When there are multiple choices like color choices - B, R or G we calculate GI for each one as well as each possible combination

- B

- G

- R

- B or G

- B or R

- G or R

We dont need to calculate for B or R or G as it includes everyone





### StatQuest: Random Forests Part 1 - Building, Using and Evaluating

[Link](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&t=143s)

DTs are easy to build, use and interpret

But in practice, theyare not that awesome

> Trees have one aspect that prevents them from being the ideal tool for predictive learning, namely **inaccuracy**

** They work great with the data used to create them but are not flexible when it comes to classifying
new samples**

RF combines simplicity of DTs with flexibility resulting in a vast improvement in accuracy


#### Step 1 : Create a "bootstrapped" dataset

<img src = "./data/img/diag23.png" height="700" width = "700" align="center">

Say these 4 samples are entire dataset

To create a bootstrapped dataset that is same size as original we randomly select samples from original dataset

**We are allowed to pick the same sample more than once**

Say first sample in original dataset = S1

We create bootstrap dataset as: **S2, S1, S4, S4**

<img src = "./data/img/diag24.png" height="700" width = "700" align="center">

#### Step 2: Create a DT using Bootstrapped dataset but only use a random subset of vars (columns) at each step

In this example we will consider 2 vars at each step

Thus instead of considering all 4 vars (CP, GBC, BA, Wt) to figure out how to split the root node we randomly select 2 : GBC, BA

Say GBC did the best job at separating the samples

We used GBC, we grey it out, so that we can focus on rem vars

<img src = "./data/img/diag25.png" height="700" width = "700" align="center">

Now we have to figure out how to select vars for circled node:

<img src = "./data/img/diag26.png" height="700" width = "700" align="center">

Just like for the root we randomly select 2 vars from (CP, BA, wt)

We select CP and wt

Thus we build the tree by: 

1. using the bootstrapped dataset

2. considering a random subset of vars at each step

This is done for a single tree

Now we make a new bootstrapped dataset and build tree considering a random subset of vars at each step

Ideally we do this 100s of times

considering a random subset of vars at each step

<img src = "./data/img/diag27.png" height="700" width = "700" align="center">


Because of the randomness associated with creating the bootsrapped dataset and also due to choosing random columns for each step, RF results in a ** wide variety of DTs**

This variety makes RF more effevtive that DTs



#### Now that we have created the RF, how do we use it?

First we get data of a new patient

We want to predict if Heart disease or not

We take data and run down 1st tree

Output: Yes

We keep track of this result

<img src = "./data/img/diag28.png" height="700" width = "700" align="center">

Similarly we run data thru 2nd... last tree

We keep track of the results and see which option received most votes

Here Yes: 5 No : 1

So conclusion : YES

**Bagging** : Bootstrapping the data plus using the aggregrate to make a decision is called Bagging


#### Test accuracy of a RF

When we created the bootstrapped dataset we allowed duplicate entries in the bootstrapped dataset

<img src = "./data/img/diag29.png" height="700" width = "700" align="center">

As a result above entry was not included in the bootstrapped dataset

> Typically about 1/3 the original data does not end up in the bootstrapped dataset

These entries are called the **Out-of-Bag Dataset**

We know the results of OoB data

Say there is only 1 entry in OoB data = No

we use them to test

We run the data through our first DT

Result : No

Similarly we run throuugh all trees and keep track of the results

Then we chose the most common result: Here it is correct and = No

<img src = "./data/img/diag30.png" height="700" width = "700" align="center">

We repeat the process for all OoB samples for all trees

Some may be incorrectly labeled

**Accuracy**: Proportion of OoB Samples that were correctly claasified by the RF

The proportion of OoB smaples that were incorrectly classified is the **OoB Error**


We now know how to:

- Build a RF

- Use a RF

- Estimate accuracy of RF

<img src = "./data/img/diag31.png" height="700" width = "700" align="center">

We used 2 vars to make a decision at each step

Now we can compare OoB Error for RF built using 2vars per step to a RF built using 3 vars per step

We can test many diff settings and chose the most accurate RF

Process:

1. Build a RF

2. Est accuracy of RF

3. Change no of vars used per step

4. Repeat for a number of times and chose the RF that is most accurate

Typically we start by using the square of number (sq root?) and then try a few settings above and below that value




### StatQuest: Random Forests Part 2: Missing data and clustering

[Link](https://www.youtube.com/watch?v=nyxTdL_4Q-Q)

Lets see how RF deals with missing data

Missing data can be of 2 types:

<img src = "./data/img/diag32.png" height="700" width = "700" align="center">

- Missing data can be in original dataset
- It may be in a new sample we want to categorize

Lets start with **Missing data in the original dataset**:

We want to create a RF from the data

But we dont know if the 4th patient has BA or what is their wt

We make an initial guess that mey be bad and gradually refine the guess until it (hopefully) gets good

Initial guess for BA = most common value = No

Since wt is numeric our initial guess is the median val = 180

<img src = "./data/img/diag33.png" height="700" width = "700" align="center">

This is the dataset with the initial guesses

Now we want to refine our guesses

We do this by ** detemining which samples are similar to the one with the missing data**

#### Determining Similarity:

1. Build a RF

2. Run all of the data down all of the trees

Lets start by running all of the data down the 1st tree:

<img src = "./data/img/diag34.png" height="700" width = "700" align="center">

Say sample 3 and 4 ended up in the same leaf node

That means they are **simialar (that is how similarity is defined in RF)**

We keep track of similar samples using a **Proximity Matrix**

The PM has a row foreach sample and a col for each sample

<img src = "./data/img/diag35.png" height="700" width = "700" align="center">

As samples 3 and 4 are similar we put 1 there

Similarly we run all of the data down the 2nd tree

<img src = "./data/img/diag36.png" height="700" width = "700" align="center">

Now samples 2, 3 and 4 all ended up in the same leaf nodes

PM now:

<img src = "./data/img/diag37.png" height="700" width = "700" align="center">

We add 1 as the pairs come in smae leaf node again

We run all the data down the 3rd tree

Updated PM:

<img src = "./data/img/diag38.png" height="700" width = "700" align="center">

Ultimately, we run the data down all the trees and the PM fills in

<img src = "./data/img/diag39.png" height="700" width = "700" align="center">

Then we divide each proximity value by total number of trees (say we had 10 trees)

Updated PM:

<img src = "./data/img/diag40.png" height="700" width = "700" align="center">

Now we can use the proximity values for sample 4 to make better guesses about the missing data


For BA we calculate the weighted freq of Y and N using prox values as wts

<img src = "./data/img/diag41.png" height="700" width = "700" align="center">

Calculations:

> Freq of Yes = 1/3

> Freq of No = 2/3

> The wighted freq of Yes = Freq of Yes * The weight for Yes

> The weight for Yes = (Proximity of Yes)/(All proximities)

<img src = "./data/img/diag42.png" height="700" width = "700" align="center">

> The proximity for Yes = Proximity value for sample 2 (the only one with Yes)

> We divide that by sum of proximities for sample 4

<img src = "./data/img/diag43.png" height="700" width = "700" align="center">

> The weight for Yes = 0.1/(0.1 + 0.1 + 0.8) = 0.1

> The wighted freq of Yes = 1/3 * 0.1 = 0.03

Similarly,

> The wighted freq of No = Freq of No * The weight for No

> The weight for No = (0.1 + 0.8)/(0.1 + 0.1 + 0.8) = 0.9

> The wighted freq of No = 2/3 * 0.9 = 0.6

Since No has a way higher wt freq we chose No

**So our new, improved revised guess based on proximities for BA is No**

** Filling in missing values for wt:**

For wt we use proximities to calculate a weighted avg



> Weighted avg = (wt of sample 1 * wt avg wt of sample 1) + (wt of sample 2 * wt avg wt of sample 2) + (wt of sample 3 * wt avg wt of sample 3)


> wt avg wt of sample 1 = (proximity of sample 1) / (sum of proximities) = 0.1 / (0.1 + 0.1 + 0.8) = 0.1

<img src = "./data/img/diag44.png" height="700" width = "700" align="center">


Similarly we calculate wt avg wt of sample 2 and wt avg wt of sample 3

> Weighted avg = (125 * 0.1) + (180 * 0.1) + (210 * 0.8) = 198.5

wts used to calculate the weighted avg is based on proximities

So we fill missing val as 198.5


<img src = "./data/img/diag45.png" height="700" width = "700" align="center">

Now that we have revised our guesses a little bit, we do the whole thing over again..

- we build a RF
- run data thru the trees
- recalculate proximities
- recalculate missing vals
- we do this 6 or 7 times until the missing values converge i.e. no longer change each time we recalculate

___

#### Super Cool stuff with the PM:

<img src = "./data/img/diag46.png" height="700" width = "700" align="center">


We have already seen this PM 

This is the PM b4 we divided each value by 10

If samples 3 and 4 ended up in the same leaf node for all 10 trees:

<img src = "./data/img/diag47.png" height="700" width = "700" align="center">

We divide each number by 10

For Samples 3 and 4 the entry will be 1

1 in PM => samples are as close as they can be

Also

1 - prox value = distance

<img src = "./data/img/diag48.png" height="700" width = "700" align="center">

Thus it is possible to derive a ** Distance Matrix ** from the PM

Getting distance matrix (which is similar to corr matrix) means we can plot **Heat Maps**

<img src = "./data/img/diag49.png" height="700" width = "700" align="center">

We can also draw **MDS Plots**

<img src = "./data/img/diag50.png" height="700" width = "700" align="center">

---

#### Missing data in new sample that we want to categorize

Imagine that we have already built a RF and we wanted to classify a new patient

But the patient has missing data for BA

<img src = "./data/img/diag51.png" height="700" width = "700" align="center">


We dont know if patient has BA

So we need to make a guess about BA so that we can run the patient down all the trees in the forest

1. Create 2 copies of the data (Yes and No for Heart Disease 

<img src = "./data/img/diag52.png" height="700" width = "700" align="center">

2. Then we use the iterative method discussed about to make a good guess about the misssing values

<img src = "./data/img/diag53.png" height="700" width = "700" align="center">

3. These are the guesses that we came up with:

<img src = "./data/img/diag54.png" height="700" width = "700" align="center">

4. Then we run the 2 samples down the trees in the forest

5. Then we see which of the 2 is correctly labeled by the RF most number of times

6. The sample which is correctly labeled more times wins

---

### StatQuest: Random Forests in R

[Link](https://www.youtube.com/watch?v=6EXPYzbfLCE)


```R
# Import libraries:

library(ggplot2)

# cowplot improves ggplot2's default settings
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
library(cowplot)

library(randomForest)
```


```R
# install.packages('cowplot', repos='http://cran.us.r-project.org')

```

We  are going to use heart disease dataset from UCI ML repo


```R
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

data <- read.csv(url, header = FALSE)

head(data)
```


<table>
<thead><tr><th scope=col>V1</th><th scope=col>V2</th><th scope=col>V3</th><th scope=col>V4</th><th scope=col>V5</th><th scope=col>V6</th><th scope=col>V7</th><th scope=col>V8</th><th scope=col>V9</th><th scope=col>V10</th><th scope=col>V11</th><th scope=col>V12</th><th scope=col>V13</th><th scope=col>V14</th></tr></thead>
<tbody>
	<tr><td>63 </td><td>1  </td><td>1  </td><td>145</td><td>233</td><td>1  </td><td>2  </td><td>150</td><td>0  </td><td>2.3</td><td>3  </td><td>0.0</td><td>6.0</td><td>0  </td></tr>
	<tr><td>67 </td><td>1  </td><td>4  </td><td>160</td><td>286</td><td>0  </td><td>2  </td><td>108</td><td>1  </td><td>1.5</td><td>2  </td><td>3.0</td><td>3.0</td><td>2  </td></tr>
	<tr><td>67 </td><td>1  </td><td>4  </td><td>120</td><td>229</td><td>0  </td><td>2  </td><td>129</td><td>1  </td><td>2.6</td><td>2  </td><td>2.0</td><td>7.0</td><td>1  </td></tr>
	<tr><td>37 </td><td>1  </td><td>3  </td><td>130</td><td>250</td><td>0  </td><td>0  </td><td>187</td><td>0  </td><td>3.5</td><td>3  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>41 </td><td>0  </td><td>2  </td><td>130</td><td>204</td><td>0  </td><td>2  </td><td>172</td><td>0  </td><td>1.4</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>56 </td><td>1  </td><td>2  </td><td>120</td><td>236</td><td>0  </td><td>0  </td><td>178</td><td>0  </td><td>0.8</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
</tbody>
</table>



Lets label the cols:

[Data Manual](http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)

Only 14 used
      -- 1. #3  (age)       
      -- 2. #4  (sex)       
      -- 3. #9  (cp)        
      -- 4. #10 (trestbps)  
      -- 5. #12 (chol)      
      -- 6. #16 (fbs)       
      -- 7. #19 (restecg)   
      -- 8. #32 (thalach)   
      -- 9. #38 (exang)     
      -- 10. #40 (oldpeak)   
      -- 11. #41 (slope)     
      -- 12. #44 (ca)        
      -- 13. #51 (thal)      
      -- 14. #58 (num)       (the predicted attribute)
      
3 age: age in years

4 sex: sex (1 = male; 0 = female)

9 cp: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic
        
10 trestbps: resting blood pressure (in mm Hg on admission to the 
        hospital)
        
12 chol: serum cholestoral in mg/dl

16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)

19 restecg: resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST 
                    elevation or depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy
                    by Estes' criteria
                    
32 thalach: maximum heart rate achieved

38 exang: exercise induced angina (1 = yes; 0 = no)

40 oldpeak = ST depression induced by exercise relative to rest

41 slope: the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping
        
44 ca: number of major vessels (0-3) colored by flourosopy

51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

58 num: diagnosis of heart disease (angiographic disease status)
        -- Value 0: < 50% diameter narrowing
        -- Value 1: > 50% diameter narrowing
        (in any major vessel: attributes 59 through 68 are vessels)
        


```R
colnames(data) <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", 
                    "slope", "ca", "thal", "hd")

head(data)
```


<table>
<thead><tr><th scope=col>age</th><th scope=col>sex</th><th scope=col>cp</th><th scope=col>trestbps</th><th scope=col>chol</th><th scope=col>fbs</th><th scope=col>restecg</th><th scope=col>thalach</th><th scope=col>exang</th><th scope=col>oldpeak</th><th scope=col>slope</th><th scope=col>ca</th><th scope=col>thal</th><th scope=col>hd</th></tr></thead>
<tbody>
	<tr><td>63 </td><td>1  </td><td>1  </td><td>145</td><td>233</td><td>1  </td><td>2  </td><td>150</td><td>0  </td><td>2.3</td><td>3  </td><td>0.0</td><td>6.0</td><td>0  </td></tr>
	<tr><td>67 </td><td>1  </td><td>4  </td><td>160</td><td>286</td><td>0  </td><td>2  </td><td>108</td><td>1  </td><td>1.5</td><td>2  </td><td>3.0</td><td>3.0</td><td>2  </td></tr>
	<tr><td>67 </td><td>1  </td><td>4  </td><td>120</td><td>229</td><td>0  </td><td>2  </td><td>129</td><td>1  </td><td>2.6</td><td>2  </td><td>2.0</td><td>7.0</td><td>1  </td></tr>
	<tr><td>37 </td><td>1  </td><td>3  </td><td>130</td><td>250</td><td>0  </td><td>0  </td><td>187</td><td>0  </td><td>3.5</td><td>3  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>41 </td><td>0  </td><td>2  </td><td>130</td><td>204</td><td>0  </td><td>2  </td><td>172</td><td>0  </td><td>1.4</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>56 </td><td>1  </td><td>2  </td><td>120</td><td>236</td><td>0  </td><td>0  </td><td>178</td><td>0  </td><td>0.8</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
</tbody>
</table>



str() function gives us the structure of the data


```R
str(data)
```

    'data.frame':	303 obs. of  14 variables:
     $ age     : num  63 67 67 37 41 56 62 57 63 53 ...
     $ sex     : num  1 1 1 1 0 1 0 0 1 1 ...
     $ cp      : num  1 4 4 3 2 2 4 4 4 4 ...
     $ trestbps: num  145 160 120 130 130 120 140 120 130 140 ...
     $ chol    : num  233 286 229 250 204 236 268 354 254 203 ...
     $ fbs     : num  1 0 0 0 0 0 0 0 0 1 ...
     $ restecg : num  2 2 2 0 2 0 2 0 2 2 ...
     $ thalach : num  150 108 129 187 172 178 160 163 147 155 ...
     $ exang   : num  0 1 1 0 0 0 0 1 0 1 ...
     $ oldpeak : num  2.3 1.5 2.6 3.5 1.4 0.8 3.6 0.6 1.4 3.1 ...
     $ slope   : num  3 2 2 3 1 1 3 1 2 3 ...
     $ ca      : Factor w/ 5 levels "?","0.0","1.0",..: 2 5 4 2 2 2 4 2 3 2 ...
     $ thal    : Factor w/ 4 levels "?","3.0","6.0",..: 3 2 4 2 2 2 2 2 4 4 ...
     $ hd      : int  0 2 1 0 0 0 3 0 2 1 ...
    

Some of the cols are messed up

- sex is supposed to be a factor where 0: female and 1: male

- cp is supposed to be a factor where levels 1-3 represents diff types of pain and 4 represents no chest pain

- fbs is supposed to be a factor

- restecg is supposed to be a factor

- exang is supposed to be a factor

- slope is supposed to be a factor

- ca and thal are correctly called factors but one of the levels is "?" when we need it to be NA

#### Change "?" to NA:


```R
data[data == '?'] <- NA
```

To make data easier on the eye, convert 0s in sex to F and 1s to M

Then convert the col into a factor


```R
data[data$sex == 0,]$sex <- "F"

data[data$sex == 1,]$sex <- "M"

data$sex <- as.factor(data$sex)

head(data)
```


<table>
<thead><tr><th scope=col>age</th><th scope=col>sex</th><th scope=col>cp</th><th scope=col>trestbps</th><th scope=col>chol</th><th scope=col>fbs</th><th scope=col>restecg</th><th scope=col>thalach</th><th scope=col>exang</th><th scope=col>oldpeak</th><th scope=col>slope</th><th scope=col>ca</th><th scope=col>thal</th><th scope=col>hd</th></tr></thead>
<tbody>
	<tr><td>63 </td><td>M  </td><td>1  </td><td>145</td><td>233</td><td>1  </td><td>2  </td><td>150</td><td>0  </td><td>2.3</td><td>3  </td><td>0.0</td><td>6.0</td><td>0  </td></tr>
	<tr><td>67 </td><td>M  </td><td>4  </td><td>160</td><td>286</td><td>0  </td><td>2  </td><td>108</td><td>1  </td><td>1.5</td><td>2  </td><td>3.0</td><td>3.0</td><td>2  </td></tr>
	<tr><td>67 </td><td>M  </td><td>4  </td><td>120</td><td>229</td><td>0  </td><td>2  </td><td>129</td><td>1  </td><td>2.6</td><td>2  </td><td>2.0</td><td>7.0</td><td>1  </td></tr>
	<tr><td>37 </td><td>M  </td><td>3  </td><td>130</td><td>250</td><td>0  </td><td>0  </td><td>187</td><td>0  </td><td>3.5</td><td>3  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>41 </td><td>F  </td><td>2  </td><td>130</td><td>204</td><td>0  </td><td>2  </td><td>172</td><td>0  </td><td>1.4</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
	<tr><td>56 </td><td>M  </td><td>2  </td><td>120</td><td>236</td><td>0  </td><td>0  </td><td>178</td><td>0  </td><td>0.8</td><td>1  </td><td>0.0</td><td>3.0</td><td>0  </td></tr>
</tbody>
</table>



We convert the other cols into factors:


```R
data$cp = as.factor(data$cp)
data$fbs = as.factor(data$fbs)
data$restecg = as.factor(data$restecg)
data$exang = as.factor(data$exang)
data$slope = as.factor(data$slope)

```

Since ca and thal cols had ? in them R took it to be a col of strings

We convert these cols to int then convert them as factors


```R
data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal)
data$thal <- as.factor(data$thal)
```

Last thing is to make hd (heart disease as 0: Healthy, 1: Unhealthy)


```R
data$hd <- ifelse(test = data$hd == 0, yes = "Healthy", no = "Unhealthy")

data$hd <- as.factor(data$hd)
```


```R
head(data)
```


<table>
<thead><tr><th scope=col>age</th><th scope=col>sex</th><th scope=col>cp</th><th scope=col>trestbps</th><th scope=col>chol</th><th scope=col>fbs</th><th scope=col>restecg</th><th scope=col>thalach</th><th scope=col>exang</th><th scope=col>oldpeak</th><th scope=col>slope</th><th scope=col>ca</th><th scope=col>thal</th><th scope=col>hd</th></tr></thead>
<tbody>
	<tr><td>63       </td><td>M        </td><td>1        </td><td>145      </td><td>233      </td><td>1        </td><td>2        </td><td>150      </td><td>0        </td><td>2.3      </td><td>3        </td><td>2        </td><td>3        </td><td>Healthy  </td></tr>
	<tr><td>67       </td><td>M        </td><td>4        </td><td>160      </td><td>286      </td><td>0        </td><td>2        </td><td>108      </td><td>1        </td><td>1.5      </td><td>2        </td><td>5        </td><td>2        </td><td>Unhealthy</td></tr>
	<tr><td>67       </td><td>M        </td><td>4        </td><td>120      </td><td>229      </td><td>0        </td><td>2        </td><td>129      </td><td>1        </td><td>2.6      </td><td>2        </td><td>4        </td><td>4        </td><td>Unhealthy</td></tr>
	<tr><td>37       </td><td>M        </td><td>3        </td><td>130      </td><td>250      </td><td>0        </td><td>0        </td><td>187      </td><td>0        </td><td>3.5      </td><td>3        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
	<tr><td>41       </td><td>F        </td><td>2        </td><td>130      </td><td>204      </td><td>0        </td><td>2        </td><td>172      </td><td>0        </td><td>1.4      </td><td>1        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
	<tr><td>56       </td><td>M        </td><td>2        </td><td>120      </td><td>236      </td><td>0        </td><td>0        </td><td>178      </td><td>0        </td><td>0.8      </td><td>1        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
</tbody>
</table>




```R
str(data)
```

    'data.frame':	303 obs. of  14 variables:
     $ age     : num  63 67 67 37 41 56 62 57 63 53 ...
     $ sex     : Factor w/ 2 levels "F","M": 2 2 2 2 1 2 1 1 2 2 ...
     $ cp      : Factor w/ 4 levels "1","2","3","4": 1 4 4 3 2 2 4 4 4 4 ...
     $ trestbps: num  145 160 120 130 130 120 140 120 130 140 ...
     $ chol    : num  233 286 229 250 204 236 268 354 254 203 ...
     $ fbs     : Factor w/ 2 levels "0","1": 2 1 1 1 1 1 1 1 1 2 ...
     $ restecg : Factor w/ 3 levels "0","1","2": 3 3 3 1 3 1 3 1 3 3 ...
     $ thalach : num  150 108 129 187 172 178 160 163 147 155 ...
     $ exang   : Factor w/ 2 levels "0","1": 1 2 2 1 1 1 1 2 1 2 ...
     $ oldpeak : num  2.3 1.5 2.6 3.5 1.4 0.8 3.6 0.6 1.4 3.1 ...
     $ slope   : Factor w/ 3 levels "1","2","3": 3 2 2 3 1 1 3 1 2 3 ...
     $ ca      : Factor w/ 4 levels "2","3","4","5": 1 4 3 1 1 1 3 1 2 1 ...
     $ thal    : Factor w/ 3 levels "2","3","4": 2 1 3 1 1 1 1 1 3 3 ...
     $ hd      : Factor w/ 2 levels "Healthy","Unhealthy": 1 2 2 1 1 1 2 1 2 2 ...
    

Since we are going to be randomly sampling things, lets set the seed for the random no generator so that we can reproduce our results


```R
set.seed(42)
```

Now we impute values for the NAs in the dataset with **rfImpute()**

The 1st arg is **hd ~ . **

This means that we want the hd col to be predicted by the data in the other cols

data specifies the dataset

iter = 6: Here we specify how many RFs should rfImpute() build to estimate the mssing values

In theory, 4-6 iters are enough

Lastly, we save the results  i.e the dataset with imputed values instead of NAs as **data.imputed**


```R
data.imputed = rfImpute(hd ~ ., data = data, iter = 6)

head(data)
```

    ntree      OOB      1      2
      300:  17.49% 14.02% 21.58%
    ntree      OOB      1      2
      300:  17.16% 13.41% 21.58%
    ntree      OOB      1      2
      300:  17.49% 14.02% 21.58%
    ntree      OOB      1      2
      300:  17.16% 13.41% 21.58%
    ntree      OOB      1      2
      300:  17.16% 13.41% 21.58%
    ntree      OOB      1      2
      300:  17.16% 13.41% 21.58%
    


<table>
<thead><tr><th scope=col>age</th><th scope=col>sex</th><th scope=col>cp</th><th scope=col>trestbps</th><th scope=col>chol</th><th scope=col>fbs</th><th scope=col>restecg</th><th scope=col>thalach</th><th scope=col>exang</th><th scope=col>oldpeak</th><th scope=col>slope</th><th scope=col>ca</th><th scope=col>thal</th><th scope=col>hd</th></tr></thead>
<tbody>
	<tr><td>63       </td><td>M        </td><td>1        </td><td>145      </td><td>233      </td><td>1        </td><td>2        </td><td>150      </td><td>0        </td><td>2.3      </td><td>3        </td><td>2        </td><td>3        </td><td>Healthy  </td></tr>
	<tr><td>67       </td><td>M        </td><td>4        </td><td>160      </td><td>286      </td><td>0        </td><td>2        </td><td>108      </td><td>1        </td><td>1.5      </td><td>2        </td><td>5        </td><td>2        </td><td>Unhealthy</td></tr>
	<tr><td>67       </td><td>M        </td><td>4        </td><td>120      </td><td>229      </td><td>0        </td><td>2        </td><td>129      </td><td>1        </td><td>2.6      </td><td>2        </td><td>4        </td><td>4        </td><td>Unhealthy</td></tr>
	<tr><td>37       </td><td>M        </td><td>3        </td><td>130      </td><td>250      </td><td>0        </td><td>0        </td><td>187      </td><td>0        </td><td>3.5      </td><td>3        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
	<tr><td>41       </td><td>F        </td><td>2        </td><td>130      </td><td>204      </td><td>0        </td><td>2        </td><td>172      </td><td>0        </td><td>1.4      </td><td>1        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
	<tr><td>56       </td><td>M        </td><td>2        </td><td>120      </td><td>236      </td><td>0        </td><td>0        </td><td>178      </td><td>0        </td><td>0.8      </td><td>1        </td><td>2        </td><td>2        </td><td>Healthy  </td></tr>
</tbody>
</table>



After each iteration rfImpute() prints the Out-of-Bag(OOB) error rate

This should get smaller if the estimates are improving

Now that we have imputed the values, we build a RF


```R
model <- randomForest(hd ~ ., data = data.imputed, proximity = TRUE)
```

The 1st arg is hd ~ .

This means that we want the hd col to be predicted by the data in the other cols

We also want randomForest() to return the PM

We will use this to cluster the samples

Lastly, we save the randomForest and asspciated data like PM as **model**

Get summary of RF and how well it performed:


```R
model
```


    
    Call:
     randomForest(formula = hd ~ ., data = data.imputed, proximity = TRUE) 
                   Type of random forest: classification
                         Number of trees: 500
    No. of variables tried at each split: 3
    
            OOB estimate of  error rate: 16.5%
    Confusion matrix:
              Healthy Unhealthy class.error
    Healthy       141        23   0.1402439
    Unhealthy      27       112   0.1942446


Type of random forest: classification

If we had used the RF to predict wt or ht it would say "regression"

If we had omitted the thing RF was supposed to predict entirely, it would say "unsupervised" 

Number of trees: 500: how many trees are in RF

No. of variables tried at each split: 3

- how many cols of data were considered at each internal node

Classification trees have a default setting of sq root of no of vars

Regression trees have a default setting of no of vars div by 3

OOB estimate of  error rate: 16.5% : This means that 83.5% of the OoB samples were correctly classified by the RF

Helthy Unhealthy class.error
Helthy       141        23   0.1402439
Unhealthy     27       112   0.1942446

This is the Confusion Matrix


```R
head(model$err.rate)
```


<table>
<thead><tr><th scope=col>OOB</th><th scope=col>Healthy</th><th scope=col>Unhealthy</th></tr></thead>
<tbody>
	<tr><td>0.2672414</td><td>0.1969697</td><td>0.3600000</td></tr>
	<tr><td>0.2702703</td><td>0.2452830</td><td>0.3037975</td></tr>
	<tr><td>0.2616034</td><td>0.2692308</td><td>0.2523364</td></tr>
	<tr><td>0.2643678</td><td>0.2797203</td><td>0.2457627</td></tr>
	<tr><td>0.2795699</td><td>0.2894737</td><td>0.2677165</td></tr>
	<tr><td>0.2762238</td><td>0.2709677</td><td>0.2824427</td></tr>
</tbody>
</table>




```R
nrow(model$err.rate)
```


500


Each row in model$err.rate reflects the error rates at diff stages of creating the RF

The 1st row contains error rates after making 1st tree

2nd row contains error rates after making 1st 2 trees

... and so on

last row contains error rates after making all 500 trees



We want to construct a df which has the type of error in the rows rather than the cols


```R
print(rep(c(2,4), each = 4))

print(rep(c(2,4), times = 4))
```

    [1] 2 2 2 2 4 4 4 4
    [1] 2 4 2 4 2 4 2 4
    

Creating col: Type


```R
Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(model$err.rate))

Type
```


<ol class=list-inline>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'OOB'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Healthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
	<li>'Unhealthy'</li>
</ol>



Creating col: Trees


```R
Trees = rep(1:nrow(model$err.rate), times = 3)

Trees
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
	<li>7</li>
	<li>8</li>
	<li>9</li>
	<li>10</li>
	<li>11</li>
	<li>12</li>
	<li>13</li>
	<li>14</li>
	<li>15</li>
	<li>16</li>
	<li>17</li>
	<li>18</li>
	<li>19</li>
	<li>20</li>
	<li>21</li>
	<li>22</li>
	<li>23</li>
	<li>24</li>
	<li>25</li>
	<li>26</li>
	<li>27</li>
	<li>28</li>
	<li>29</li>
	<li>30</li>
	<li>31</li>
	<li>32</li>
	<li>33</li>
	<li>34</li>
	<li>35</li>
	<li>36</li>
	<li>37</li>
	<li>38</li>
	<li>39</li>
	<li>40</li>
	<li>41</li>
	<li>42</li>
	<li>43</li>
	<li>44</li>
	<li>45</li>
	<li>46</li>
	<li>47</li>
	<li>48</li>
	<li>49</li>
	<li>50</li>
	<li>51</li>
	<li>52</li>
	<li>53</li>
	<li>54</li>
	<li>55</li>
	<li>56</li>
	<li>57</li>
	<li>58</li>
	<li>59</li>
	<li>60</li>
	<li>61</li>
	<li>62</li>
	<li>63</li>
	<li>64</li>
	<li>65</li>
	<li>66</li>
	<li>67</li>
	<li>68</li>
	<li>69</li>
	<li>70</li>
	<li>71</li>
	<li>72</li>
	<li>73</li>
	<li>74</li>
	<li>75</li>
	<li>76</li>
	<li>77</li>
	<li>78</li>
	<li>79</li>
	<li>80</li>
	<li>81</li>
	<li>82</li>
	<li>83</li>
	<li>84</li>
	<li>85</li>
	<li>86</li>
	<li>87</li>
	<li>88</li>
	<li>89</li>
	<li>90</li>
	<li>91</li>
	<li>92</li>
	<li>93</li>
	<li>94</li>
	<li>95</li>
	<li>96</li>
	<li>97</li>
	<li>98</li>
	<li>99</li>
	<li>100</li>
	<li>101</li>
	<li>102</li>
	<li>103</li>
	<li>104</li>
	<li>105</li>
	<li>106</li>
	<li>107</li>
	<li>108</li>
	<li>109</li>
	<li>110</li>
	<li>111</li>
	<li>112</li>
	<li>113</li>
	<li>114</li>
	<li>115</li>
	<li>116</li>
	<li>117</li>
	<li>118</li>
	<li>119</li>
	<li>120</li>
	<li>121</li>
	<li>122</li>
	<li>123</li>
	<li>124</li>
	<li>125</li>
	<li>126</li>
	<li>127</li>
	<li>128</li>
	<li>129</li>
	<li>130</li>
	<li>131</li>
	<li>132</li>
	<li>133</li>
	<li>134</li>
	<li>135</li>
	<li>136</li>
	<li>137</li>
	<li>138</li>
	<li>139</li>
	<li>140</li>
	<li>141</li>
	<li>142</li>
	<li>143</li>
	<li>144</li>
	<li>145</li>
	<li>146</li>
	<li>147</li>
	<li>148</li>
	<li>149</li>
	<li>150</li>
	<li>151</li>
	<li>152</li>
	<li>153</li>
	<li>154</li>
	<li>155</li>
	<li>156</li>
	<li>157</li>
	<li>158</li>
	<li>159</li>
	<li>160</li>
	<li>161</li>
	<li>162</li>
	<li>163</li>
	<li>164</li>
	<li>165</li>
	<li>166</li>
	<li>167</li>
	<li>168</li>
	<li>169</li>
	<li>170</li>
	<li>171</li>
	<li>172</li>
	<li>173</li>
	<li>174</li>
	<li>175</li>
	<li>176</li>
	<li>177</li>
	<li>178</li>
	<li>179</li>
	<li>180</li>
	<li>181</li>
	<li>182</li>
	<li>183</li>
	<li>184</li>
	<li>185</li>
	<li>186</li>
	<li>187</li>
	<li>188</li>
	<li>189</li>
	<li>190</li>
	<li>191</li>
	<li>192</li>
	<li>193</li>
	<li>194</li>
	<li>195</li>
	<li>196</li>
	<li>197</li>
	<li>198</li>
	<li>199</li>
	<li>200</li>
	<li>201</li>
	<li>202</li>
	<li>203</li>
	<li>204</li>
	<li>205</li>
	<li>206</li>
	<li>207</li>
	<li>208</li>
	<li>209</li>
	<li>210</li>
	<li>211</li>
	<li>212</li>
	<li>213</li>
	<li>214</li>
	<li>215</li>
	<li>216</li>
	<li>217</li>
	<li>218</li>
	<li>219</li>
	<li>220</li>
	<li>221</li>
	<li>222</li>
	<li>223</li>
	<li>224</li>
	<li>225</li>
	<li>226</li>
	<li>227</li>
	<li>228</li>
	<li>229</li>
	<li>230</li>
	<li>231</li>
	<li>232</li>
	<li>233</li>
	<li>234</li>
	<li>235</li>
	<li>236</li>
	<li>237</li>
	<li>238</li>
	<li>239</li>
	<li>240</li>
	<li>241</li>
	<li>242</li>
	<li>243</li>
	<li>244</li>
	<li>245</li>
	<li>246</li>
	<li>247</li>
	<li>248</li>
	<li>249</li>
	<li>250</li>
	<li>251</li>
	<li>252</li>
	<li>253</li>
	<li>254</li>
	<li>255</li>
	<li>256</li>
	<li>257</li>
	<li>258</li>
	<li>259</li>
	<li>260</li>
	<li>261</li>
	<li>262</li>
	<li>263</li>
	<li>264</li>
	<li>265</li>
	<li>266</li>
	<li>267</li>
	<li>268</li>
	<li>269</li>
	<li>270</li>
	<li>271</li>
	<li>272</li>
	<li>273</li>
	<li>274</li>
	<li>275</li>
	<li>276</li>
	<li>277</li>
	<li>278</li>
	<li>279</li>
	<li>280</li>
	<li>281</li>
	<li>282</li>
	<li>283</li>
	<li>284</li>
	<li>285</li>
	<li>286</li>
	<li>287</li>
	<li>288</li>
	<li>289</li>
	<li>290</li>
	<li>291</li>
	<li>292</li>
	<li>293</li>
	<li>294</li>
	<li>295</li>
	<li>296</li>
	<li>297</li>
	<li>298</li>
	<li>299</li>
	<li>300</li>
	<li>301</li>
	<li>302</li>
	<li>303</li>
	<li>304</li>
	<li>305</li>
	<li>306</li>
	<li>307</li>
	<li>308</li>
	<li>309</li>
	<li>310</li>
	<li>311</li>
	<li>312</li>
	<li>313</li>
	<li>314</li>
	<li>315</li>
	<li>316</li>
	<li>317</li>
	<li>318</li>
	<li>319</li>
	<li>320</li>
	<li>321</li>
	<li>322</li>
	<li>323</li>
	<li>324</li>
	<li>325</li>
	<li>326</li>
	<li>327</li>
	<li>328</li>
	<li>329</li>
	<li>330</li>
	<li>331</li>
	<li>332</li>
	<li>333</li>
	<li>334</li>
	<li>335</li>
	<li>336</li>
	<li>337</li>
	<li>338</li>
	<li>339</li>
	<li>340</li>
	<li>341</li>
	<li>342</li>
	<li>343</li>
	<li>344</li>
	<li>345</li>
	<li>346</li>
	<li>347</li>
	<li>348</li>
	<li>349</li>
	<li>350</li>
	<li>351</li>
	<li>352</li>
	<li>353</li>
	<li>354</li>
	<li>355</li>
	<li>356</li>
	<li>357</li>
	<li>358</li>
	<li>359</li>
	<li>360</li>
	<li>361</li>
	<li>362</li>
	<li>363</li>
	<li>364</li>
	<li>365</li>
	<li>366</li>
	<li>367</li>
	<li>368</li>
	<li>369</li>
	<li>370</li>
	<li>371</li>
	<li>372</li>
	<li>373</li>
	<li>374</li>
	<li>375</li>
	<li>376</li>
	<li>377</li>
	<li>378</li>
	<li>379</li>
	<li>380</li>
	<li>381</li>
	<li>382</li>
	<li>383</li>
	<li>384</li>
	<li>385</li>
	<li>386</li>
	<li>387</li>
	<li>388</li>
	<li>389</li>
	<li>390</li>
	<li>391</li>
	<li>392</li>
	<li>393</li>
	<li>394</li>
	<li>395</li>
	<li>396</li>
	<li>397</li>
	<li>398</li>
	<li>399</li>
	<li>400</li>
	<li>401</li>
	<li>402</li>
	<li>403</li>
	<li>404</li>
	<li>405</li>
	<li>406</li>
	<li>407</li>
	<li>408</li>
	<li>409</li>
	<li>410</li>
	<li>411</li>
	<li>412</li>
	<li>413</li>
	<li>414</li>
	<li>415</li>
	<li>416</li>
	<li>417</li>
	<li>418</li>
	<li>419</li>
	<li>420</li>
	<li>421</li>
	<li>422</li>
	<li>423</li>
	<li>424</li>
	<li>425</li>
	<li>426</li>
	<li>427</li>
	<li>428</li>
	<li>429</li>
	<li>430</li>
	<li>431</li>
	<li>432</li>
	<li>433</li>
	<li>434</li>
	<li>435</li>
	<li>436</li>
	<li>437</li>
	<li>438</li>
	<li>439</li>
	<li>440</li>
	<li>441</li>
	<li>442</li>
	<li>443</li>
	<li>444</li>
	<li>445</li>
	<li>446</li>
	<li>447</li>
	<li>448</li>
	<li>449</li>
	<li>450</li>
	<li>451</li>
	<li>452</li>
	<li>453</li>
	<li>454</li>
	<li>455</li>
	<li>456</li>
	<li>457</li>
	<li>458</li>
	<li>459</li>
	<li>460</li>
	<li>461</li>
	<li>462</li>
	<li>463</li>
	<li>464</li>
	<li>465</li>
	<li>466</li>
	<li>467</li>
	<li>468</li>
	<li>469</li>
	<li>470</li>
	<li>471</li>
	<li>472</li>
	<li>473</li>
	<li>474</li>
	<li>475</li>
	<li>476</li>
	<li>477</li>
	<li>478</li>
	<li>479</li>
	<li>480</li>
	<li>481</li>
	<li>482</li>
	<li>483</li>
	<li>484</li>
	<li>485</li>
	<li>486</li>
	<li>487</li>
	<li>488</li>
	<li>489</li>
	<li>490</li>
	<li>491</li>
	<li>492</li>
	<li>493</li>
	<li>494</li>
	<li>495</li>
	<li>496</li>
	<li>497</li>
	<li>498</li>
	<li>499</li>
	<li>500</li>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
	<li>7</li>
	<li>8</li>
	<li>9</li>
	<li>10</li>
	<li>11</li>
	<li>12</li>
	<li>13</li>
	<li>14</li>
	<li>15</li>
	<li>16</li>
	<li>17</li>
	<li>18</li>
	<li>19</li>
	<li>20</li>
	<li>21</li>
	<li>22</li>
	<li>23</li>
	<li>24</li>
	<li>25</li>
	<li>26</li>
	<li>27</li>
	<li>28</li>
	<li>29</li>
	<li>30</li>
	<li>31</li>
	<li>32</li>
	<li>33</li>
	<li>34</li>
	<li>35</li>
	<li>36</li>
	<li>37</li>
	<li>38</li>
	<li>39</li>
	<li>40</li>
	<li>41</li>
	<li>42</li>
	<li>43</li>
	<li>44</li>
	<li>45</li>
	<li>46</li>
	<li>47</li>
	<li>48</li>
	<li>49</li>
	<li>50</li>
	<li>51</li>
	<li>52</li>
	<li>53</li>
	<li>54</li>
	<li>55</li>
	<li>56</li>
	<li>57</li>
	<li>58</li>
	<li>59</li>
	<li>60</li>
	<li>61</li>
	<li>62</li>
	<li>63</li>
	<li>64</li>
	<li>65</li>
	<li>66</li>
	<li>67</li>
	<li>68</li>
	<li>69</li>
	<li>70</li>
	<li>71</li>
	<li>72</li>
	<li>73</li>
	<li>74</li>
	<li>75</li>
	<li>76</li>
	<li>77</li>
	<li>78</li>
	<li>79</li>
	<li>80</li>
	<li>81</li>
	<li>82</li>
	<li>83</li>
	<li>84</li>
	<li>85</li>
	<li>86</li>
	<li>87</li>
	<li>88</li>
	<li>89</li>
	<li>90</li>
	<li>91</li>
	<li>92</li>
	<li>93</li>
	<li>94</li>
	<li>95</li>
	<li>96</li>
	<li>97</li>
	<li>98</li>
	<li>99</li>
	<li>100</li>
	<li>101</li>
	<li>102</li>
	<li>103</li>
	<li>104</li>
	<li>105</li>
	<li>106</li>
	<li>107</li>
	<li>108</li>
	<li>109</li>
	<li>110</li>
	<li>111</li>
	<li>112</li>
	<li>113</li>
	<li>114</li>
	<li>115</li>
	<li>116</li>
	<li>117</li>
	<li>118</li>
	<li>119</li>
	<li>120</li>
	<li>121</li>
	<li>122</li>
	<li>123</li>
	<li>124</li>
	<li>125</li>
	<li>126</li>
	<li>127</li>
	<li>128</li>
	<li>129</li>
	<li>130</li>
	<li>131</li>
	<li>132</li>
	<li>133</li>
	<li>134</li>
	<li>135</li>
	<li>136</li>
	<li>137</li>
	<li>138</li>
	<li>139</li>
	<li>140</li>
	<li>141</li>
	<li>142</li>
	<li>143</li>
	<li>144</li>
	<li>145</li>
	<li>146</li>
	<li>147</li>
	<li>148</li>
	<li>149</li>
	<li>150</li>
	<li>151</li>
	<li>152</li>
	<li>153</li>
	<li>154</li>
	<li>155</li>
	<li>156</li>
	<li>157</li>
	<li>158</li>
	<li>159</li>
	<li>160</li>
	<li>161</li>
	<li>162</li>
	<li>163</li>
	<li>164</li>
	<li>165</li>
	<li>166</li>
	<li>167</li>
	<li>168</li>
	<li>169</li>
	<li>170</li>
	<li>171</li>
	<li>172</li>
	<li>173</li>
	<li>174</li>
	<li>175</li>
	<li>176</li>
	<li>177</li>
	<li>178</li>
	<li>179</li>
	<li>180</li>
	<li>181</li>
	<li>182</li>
	<li>183</li>
	<li>184</li>
	<li>185</li>
	<li>186</li>
	<li>187</li>
	<li>188</li>
	<li>189</li>
	<li>190</li>
	<li>191</li>
	<li>192</li>
	<li>193</li>
	<li>194</li>
	<li>195</li>
	<li>196</li>
	<li>197</li>
	<li>198</li>
	<li>199</li>
	<li>200</li>
	<li>201</li>
	<li>202</li>
	<li>203</li>
	<li>204</li>
	<li>205</li>
	<li>206</li>
	<li>207</li>
	<li>208</li>
	<li>209</li>
	<li>210</li>
	<li>211</li>
	<li>212</li>
	<li>213</li>
	<li>214</li>
	<li>215</li>
	<li>216</li>
	<li>217</li>
	<li>218</li>
	<li>219</li>
	<li>220</li>
	<li>221</li>
	<li>222</li>
	<li>223</li>
	<li>224</li>
	<li>225</li>
	<li>226</li>
	<li>227</li>
	<li>228</li>
	<li>229</li>
	<li>230</li>
	<li>231</li>
	<li>232</li>
	<li>233</li>
	<li>234</li>
	<li>235</li>
	<li>236</li>
	<li>237</li>
	<li>238</li>
	<li>239</li>
	<li>240</li>
	<li>241</li>
	<li>242</li>
	<li>243</li>
	<li>244</li>
	<li>245</li>
	<li>246</li>
	<li>247</li>
	<li>248</li>
	<li>249</li>
	<li>250</li>
	<li>251</li>
	<li>252</li>
	<li>253</li>
	<li>254</li>
	<li>255</li>
	<li>256</li>
	<li>257</li>
	<li>258</li>
	<li>259</li>
	<li>260</li>
	<li>261</li>
	<li>262</li>
	<li>263</li>
	<li>264</li>
	<li>265</li>
	<li>266</li>
	<li>267</li>
	<li>268</li>
	<li>269</li>
	<li>270</li>
	<li>271</li>
	<li>272</li>
	<li>273</li>
	<li>274</li>
	<li>275</li>
	<li>276</li>
	<li>277</li>
	<li>278</li>
	<li>279</li>
	<li>280</li>
	<li>281</li>
	<li>282</li>
	<li>283</li>
	<li>284</li>
	<li>285</li>
	<li>286</li>
	<li>287</li>
	<li>288</li>
	<li>289</li>
	<li>290</li>
	<li>291</li>
	<li>292</li>
	<li>293</li>
	<li>294</li>
	<li>295</li>
	<li>296</li>
	<li>297</li>
	<li>298</li>
	<li>299</li>
	<li>300</li>
	<li>301</li>
	<li>302</li>
	<li>303</li>
	<li>304</li>
	<li>305</li>
	<li>306</li>
	<li>307</li>
	<li>308</li>
	<li>309</li>
	<li>310</li>
	<li>311</li>
	<li>312</li>
	<li>313</li>
	<li>314</li>
	<li>315</li>
	<li>316</li>
	<li>317</li>
	<li>318</li>
	<li>319</li>
	<li>320</li>
	<li>321</li>
	<li>322</li>
	<li>323</li>
	<li>324</li>
	<li>325</li>
	<li>326</li>
	<li>327</li>
	<li>328</li>
	<li>329</li>
	<li>330</li>
	<li>331</li>
	<li>332</li>
	<li>333</li>
	<li>334</li>
	<li>335</li>
	<li>336</li>
	<li>337</li>
	<li>338</li>
	<li>339</li>
	<li>340</li>
	<li>341</li>
	<li>342</li>
	<li>343</li>
	<li>344</li>
	<li>345</li>
	<li>346</li>
	<li>347</li>
	<li>348</li>
	<li>349</li>
	<li>350</li>
	<li>351</li>
	<li>352</li>
	<li>353</li>
	<li>354</li>
	<li>355</li>
	<li>356</li>
	<li>357</li>
	<li>358</li>
	<li>359</li>
	<li>360</li>
	<li>361</li>
	<li>362</li>
	<li>363</li>
	<li>364</li>
	<li>365</li>
	<li>366</li>
	<li>367</li>
	<li>368</li>
	<li>369</li>
	<li>370</li>
	<li>371</li>
	<li>372</li>
	<li>373</li>
	<li>374</li>
	<li>375</li>
	<li>376</li>
	<li>377</li>
	<li>378</li>
	<li>379</li>
	<li>380</li>
	<li>381</li>
	<li>382</li>
	<li>383</li>
	<li>384</li>
	<li>385</li>
	<li>386</li>
	<li>387</li>
	<li>388</li>
	<li>389</li>
	<li>390</li>
	<li>391</li>
	<li>392</li>
	<li>393</li>
	<li>394</li>
	<li>395</li>
	<li>396</li>
	<li>397</li>
	<li>398</li>
	<li>399</li>
	<li>400</li>
	<li>401</li>
	<li>402</li>
	<li>403</li>
	<li>404</li>
	<li>405</li>
	<li>406</li>
	<li>407</li>
	<li>408</li>
	<li>409</li>
	<li>410</li>
	<li>411</li>
	<li>412</li>
	<li>413</li>
	<li>414</li>
	<li>415</li>
	<li>416</li>
	<li>417</li>
	<li>418</li>
	<li>419</li>
	<li>420</li>
	<li>421</li>
	<li>422</li>
	<li>423</li>
	<li>424</li>
	<li>425</li>
	<li>426</li>
	<li>427</li>
	<li>428</li>
	<li>429</li>
	<li>430</li>
	<li>431</li>
	<li>432</li>
	<li>433</li>
	<li>434</li>
	<li>435</li>
	<li>436</li>
	<li>437</li>
	<li>438</li>
	<li>439</li>
	<li>440</li>
	<li>441</li>
	<li>442</li>
	<li>443</li>
	<li>444</li>
	<li>445</li>
	<li>446</li>
	<li>447</li>
	<li>448</li>
	<li>449</li>
	<li>450</li>
	<li>451</li>
	<li>452</li>
	<li>453</li>
	<li>454</li>
	<li>455</li>
	<li>456</li>
	<li>457</li>
	<li>458</li>
	<li>459</li>
	<li>460</li>
	<li>461</li>
	<li>462</li>
	<li>463</li>
	<li>464</li>
	<li>465</li>
	<li>466</li>
	<li>467</li>
	<li>468</li>
	<li>469</li>
	<li>470</li>
	<li>471</li>
	<li>472</li>
	<li>473</li>
	<li>474</li>
	<li>475</li>
	<li>476</li>
	<li>477</li>
	<li>478</li>
	<li>479</li>
	<li>480</li>
	<li>481</li>
	<li>482</li>
	<li>483</li>
	<li>484</li>
	<li>485</li>
	<li>486</li>
	<li>487</li>
	<li>488</li>
	<li>489</li>
	<li>490</li>
	<li>491</li>
	<li>492</li>
	<li>493</li>
	<li>494</li>
	<li>495</li>
	<li>496</li>
	<li>497</li>
	<li>498</li>
	<li>499</li>
	<li>500</li>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
	<li>7</li>
	<li>8</li>
	<li>9</li>
	<li>10</li>
	<li>11</li>
	<li>12</li>
	<li>13</li>
	<li>14</li>
	<li>15</li>
	<li>16</li>
	<li>17</li>
	<li>18</li>
	<li>19</li>
	<li>20</li>
	<li>21</li>
	<li>22</li>
	<li>23</li>
	<li>24</li>
	<li>25</li>
	<li>26</li>
	<li>27</li>
	<li>28</li>
	<li>29</li>
	<li>30</li>
	<li>31</li>
	<li>32</li>
	<li>33</li>
	<li>34</li>
	<li>35</li>
	<li>36</li>
	<li>37</li>
	<li>38</li>
	<li>39</li>
	<li>40</li>
	<li>41</li>
	<li>42</li>
	<li>43</li>
	<li>44</li>
	<li>45</li>
	<li>46</li>
	<li>47</li>
	<li>48</li>
	<li>49</li>
	<li>50</li>
	<li>51</li>
	<li>52</li>
	<li>53</li>
	<li>54</li>
	<li>55</li>
	<li>56</li>
	<li>57</li>
	<li>58</li>
	<li>59</li>
	<li>60</li>
	<li>61</li>
	<li>62</li>
	<li>63</li>
	<li>64</li>
	<li>65</li>
	<li>66</li>
	<li>67</li>
	<li>68</li>
	<li>69</li>
	<li>70</li>
	<li>71</li>
	<li>72</li>
	<li>73</li>
	<li>74</li>
	<li>75</li>
	<li>76</li>
	<li>77</li>
	<li>78</li>
	<li>79</li>
	<li>80</li>
	<li>81</li>
	<li>82</li>
	<li>83</li>
	<li>84</li>
	<li>85</li>
	<li>86</li>
	<li>87</li>
	<li>88</li>
	<li>89</li>
	<li>90</li>
	<li>91</li>
	<li>92</li>
	<li>93</li>
	<li>94</li>
	<li>95</li>
	<li>96</li>
	<li>97</li>
	<li>98</li>
	<li>99</li>
	<li>100</li>
	<li>101</li>
	<li>102</li>
	<li>103</li>
	<li>104</li>
	<li>105</li>
	<li>106</li>
	<li>107</li>
	<li>108</li>
	<li>109</li>
	<li>110</li>
	<li>111</li>
	<li>112</li>
	<li>113</li>
	<li>114</li>
	<li>115</li>
	<li>116</li>
	<li>117</li>
	<li>118</li>
	<li>119</li>
	<li>120</li>
	<li>121</li>
	<li>122</li>
	<li>123</li>
	<li>124</li>
	<li>125</li>
	<li>126</li>
	<li>127</li>
	<li>128</li>
	<li>129</li>
	<li>130</li>
	<li>131</li>
	<li>132</li>
	<li>133</li>
	<li>134</li>
	<li>135</li>
	<li>136</li>
	<li>137</li>
	<li>138</li>
	<li>139</li>
	<li>140</li>
	<li>141</li>
	<li>142</li>
	<li>143</li>
	<li>144</li>
	<li>145</li>
	<li>146</li>
	<li>147</li>
	<li>148</li>
	<li>149</li>
	<li>150</li>
	<li>151</li>
	<li>152</li>
	<li>153</li>
	<li>154</li>
	<li>155</li>
	<li>156</li>
	<li>157</li>
	<li>158</li>
	<li>159</li>
	<li>160</li>
	<li>161</li>
	<li>162</li>
	<li>163</li>
	<li>164</li>
	<li>165</li>
	<li>166</li>
	<li>167</li>
	<li>168</li>
	<li>169</li>
	<li>170</li>
	<li>171</li>
	<li>172</li>
	<li>173</li>
	<li>174</li>
	<li>175</li>
	<li>176</li>
	<li>177</li>
	<li>178</li>
	<li>179</li>
	<li>180</li>
	<li>181</li>
	<li>182</li>
	<li>183</li>
	<li>184</li>
	<li>185</li>
	<li>186</li>
	<li>187</li>
	<li>188</li>
	<li>189</li>
	<li>190</li>
	<li>191</li>
	<li>192</li>
	<li>193</li>
	<li>194</li>
	<li>195</li>
	<li>196</li>
	<li>197</li>
	<li>198</li>
	<li>199</li>
	<li>200</li>
	<li>201</li>
	<li>202</li>
	<li>203</li>
	<li>204</li>
	<li>205</li>
	<li>206</li>
	<li>207</li>
	<li>208</li>
	<li>209</li>
	<li>210</li>
	<li>211</li>
	<li>212</li>
	<li>213</li>
	<li>214</li>
	<li>215</li>
	<li>216</li>
	<li>217</li>
	<li>218</li>
	<li>219</li>
	<li>220</li>
	<li>221</li>
	<li>222</li>
	<li>223</li>
	<li>224</li>
	<li>225</li>
	<li>226</li>
	<li>227</li>
	<li>228</li>
	<li>229</li>
	<li>230</li>
	<li>231</li>
	<li>232</li>
	<li>233</li>
	<li>234</li>
	<li>235</li>
	<li>236</li>
	<li>237</li>
	<li>238</li>
	<li>239</li>
	<li>240</li>
	<li>241</li>
	<li>242</li>
	<li>243</li>
	<li>244</li>
	<li>245</li>
	<li>246</li>
	<li>247</li>
	<li>248</li>
	<li>249</li>
	<li>250</li>
	<li>251</li>
	<li>252</li>
	<li>253</li>
	<li>254</li>
	<li>255</li>
	<li>256</li>
	<li>257</li>
	<li>258</li>
	<li>259</li>
	<li>260</li>
	<li>261</li>
	<li>262</li>
	<li>263</li>
	<li>264</li>
	<li>265</li>
	<li>266</li>
	<li>267</li>
	<li>268</li>
	<li>269</li>
	<li>270</li>
	<li>271</li>
	<li>272</li>
	<li>273</li>
	<li>274</li>
	<li>275</li>
	<li>276</li>
	<li>277</li>
	<li>278</li>
	<li>279</li>
	<li>280</li>
	<li>281</li>
	<li>282</li>
	<li>283</li>
	<li>284</li>
	<li>285</li>
	<li>286</li>
	<li>287</li>
	<li>288</li>
	<li>289</li>
	<li>290</li>
	<li>291</li>
	<li>292</li>
	<li>293</li>
	<li>294</li>
	<li>295</li>
	<li>296</li>
	<li>297</li>
	<li>298</li>
	<li>299</li>
	<li>300</li>
	<li>301</li>
	<li>302</li>
	<li>303</li>
	<li>304</li>
	<li>305</li>
	<li>306</li>
	<li>307</li>
	<li>308</li>
	<li>309</li>
	<li>310</li>
	<li>311</li>
	<li>312</li>
	<li>313</li>
	<li>314</li>
	<li>315</li>
	<li>316</li>
	<li>317</li>
	<li>318</li>
	<li>319</li>
	<li>320</li>
	<li>321</li>
	<li>322</li>
	<li>323</li>
	<li>324</li>
	<li>325</li>
	<li>326</li>
	<li>327</li>
	<li>328</li>
	<li>329</li>
	<li>330</li>
	<li>331</li>
	<li>332</li>
	<li>333</li>
	<li>334</li>
	<li>335</li>
	<li>336</li>
	<li>337</li>
	<li>338</li>
	<li>339</li>
	<li>340</li>
	<li>341</li>
	<li>342</li>
	<li>343</li>
	<li>344</li>
	<li>345</li>
	<li>346</li>
	<li>347</li>
	<li>348</li>
	<li>349</li>
	<li>350</li>
	<li>351</li>
	<li>352</li>
	<li>353</li>
	<li>354</li>
	<li>355</li>
	<li>356</li>
	<li>357</li>
	<li>358</li>
	<li>359</li>
	<li>360</li>
	<li>361</li>
	<li>362</li>
	<li>363</li>
	<li>364</li>
	<li>365</li>
	<li>366</li>
	<li>367</li>
	<li>368</li>
	<li>369</li>
	<li>370</li>
	<li>371</li>
	<li>372</li>
	<li>373</li>
	<li>374</li>
	<li>375</li>
	<li>376</li>
	<li>377</li>
	<li>378</li>
	<li>379</li>
	<li>380</li>
	<li>381</li>
	<li>382</li>
	<li>383</li>
	<li>384</li>
	<li>385</li>
	<li>386</li>
	<li>387</li>
	<li>388</li>
	<li>389</li>
	<li>390</li>
	<li>391</li>
	<li>392</li>
	<li>393</li>
	<li>394</li>
	<li>395</li>
	<li>396</li>
	<li>397</li>
	<li>398</li>
	<li>399</li>
	<li>400</li>
	<li>401</li>
	<li>402</li>
	<li>403</li>
	<li>404</li>
	<li>405</li>
	<li>406</li>
	<li>407</li>
	<li>408</li>
	<li>409</li>
	<li>410</li>
	<li>411</li>
	<li>412</li>
	<li>413</li>
	<li>414</li>
	<li>415</li>
	<li>416</li>
	<li>417</li>
	<li>418</li>
	<li>419</li>
	<li>420</li>
	<li>421</li>
	<li>422</li>
	<li>423</li>
	<li>424</li>
	<li>425</li>
	<li>426</li>
	<li>427</li>
	<li>428</li>
	<li>429</li>
	<li>430</li>
	<li>431</li>
	<li>432</li>
	<li>433</li>
	<li>434</li>
	<li>435</li>
	<li>436</li>
	<li>437</li>
	<li>438</li>
	<li>439</li>
	<li>440</li>
	<li>441</li>
	<li>442</li>
	<li>443</li>
	<li>444</li>
	<li>445</li>
	<li>446</li>
	<li>447</li>
	<li>448</li>
	<li>449</li>
	<li>450</li>
	<li>451</li>
	<li>452</li>
	<li>453</li>
	<li>454</li>
	<li>455</li>
	<li>456</li>
	<li>457</li>
	<li>458</li>
	<li>459</li>
	<li>460</li>
	<li>461</li>
	<li>462</li>
	<li>463</li>
	<li>464</li>
	<li>465</li>
	<li>466</li>
	<li>467</li>
	<li>468</li>
	<li>469</li>
	<li>470</li>
	<li>471</li>
	<li>472</li>
	<li>473</li>
	<li>474</li>
	<li>475</li>
	<li>476</li>
	<li>477</li>
	<li>478</li>
	<li>479</li>
	<li>480</li>
	<li>481</li>
	<li>482</li>
	<li>483</li>
	<li>484</li>
	<li>485</li>
	<li>486</li>
	<li>487</li>
	<li>488</li>
	<li>489</li>
	<li>490</li>
	<li>491</li>
	<li>492</li>
	<li>493</li>
	<li>494</li>
	<li>495</li>
	<li>496</li>
	<li>497</li>
	<li>498</li>
	<li>499</li>
	<li>500</li>
</ol>




```R
Error = c(model$err.rate[, "OOB"],
         model$err.rate[, "Healthy"],
         model$err.rate[, "Unhealthy"])

length(Error)
```


1500


Making the df:


```R
oob.error.data <- data.frame(Trees = Trees, Type = Type, Error = Error)

head(oob.error.data)

nrow(oob.error.data)
```


<table>
<thead><tr><th scope=col>Trees</th><th scope=col>Type</th><th scope=col>Error</th></tr></thead>
<tbody>
	<tr><td>1        </td><td>OOB      </td><td>0.2672414</td></tr>
	<tr><td>2        </td><td>OOB      </td><td>0.2702703</td></tr>
	<tr><td>3        </td><td>OOB      </td><td>0.2616034</td></tr>
	<tr><td>4        </td><td>OOB      </td><td>0.2643678</td></tr>
	<tr><td>5        </td><td>OOB      </td><td>0.2795699</td></tr>
	<tr><td>6        </td><td>OOB      </td><td>0.2762238</td></tr>
</tbody>
</table>




1500


Now we plot this error


```R
ggplot(data = oob.error.data, aes(x=Trees, y=Error)) + geom_line(aes(color = Type))
```




![png](DT%20and%20RF_files/DT%20and%20RF_54_1.png)


The blue line shows the error rates while classifying **Unhealthy** patients

The green line shows the overall **OOB Error Rate**. So its in the middle (avg) of the 2

The red line shows the error rates while classifying **healthy** patients

We see in general the error rate dec when the RF has more trees

**If we added more trees would the error rate go down further?**

Lets make a RF with 1000 trees


```R
model2 <- randomForest(hd ~ ., data = data.imputed, ntree = 1000, proximity = TRUE)

model2
```


    
    Call:
     randomForest(formula = hd ~ ., data = data.imputed, ntree = 1000,      proximity = TRUE) 
                   Type of random forest: classification
                         Number of trees: 1000
    No. of variables tried at each split: 3
    
            OOB estimate of  error rate: 16.5%
    Confusion matrix:
              Healthy Unhealthy class.error
    Healthy       142        22   0.1341463
    Unhealthy      28       111   0.2014388


OOB error rate is same as before

And confusion matrix tells us we did no better than before


```R
Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(model2$err.rate))
Trees = rep(1:nrow(model2$err.rate), times = 3)
Error = c(model2$err.rate[, "OOB"],
         model2$err.rate[, "Healthy"],
         model2$err.rate[, "Unhealthy"])

oob.error.data <- data.frame(Trees = Trees, Type = Type, Error = Error)

ggplot(data = oob.error.data, aes(x=Trees, y=Error)) + geom_line(aes(color = Type))
```




![png](DT%20and%20RF_files/DT%20and%20RF_58_1.png)


The error rates stabilize right after 500 trees

So adding more trees would not help

But we would not have known this had we not added more trees

#### Optimal no of vars at each internal node

This is done using the param: mtry


```R
# Create an empty vector:

oob.values = vector(length = 10)

for (i in 1: 10){
    
    # build an RF using "i" to determine no of vars to try at  each step
    
    temp.model = randomForest(hd ~ ., data = data.imputed, mtry = i, ntree = 1000)
    
    # print(temp.model)
    
    # reqd oob value is the 1st col (OOB) of last (after building 1000) tree
    
    oob.value <- temp.model$err.rate[nrow(temp.model$err.rate),1]
    
    # Stre OOB error rate for current model
    
    oob.values[i] <- oob.value
}

oob.values
```


<ol class=list-inline>
	<li>0.171617161716172</li>
	<li>0.171617161716172</li>
	<li>0.161716171617162</li>
	<li>0.184818481848185</li>
	<li>0.174917491749175</li>
	<li>0.194719471947195</li>
	<li>0.181518151815182</li>
	<li>0.201320132013201</li>
	<li>0.188118811881188</li>
	<li>0.194719471947195</li>
</ol>



The 3rd value i.e no of vars = 3 is the optimal value

Coincidentally this is the default value
