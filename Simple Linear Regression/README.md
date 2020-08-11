# Introduction

On a warm summers day nothing compares to a cold pint in good company, but on a cold, damp and rainy day the last thought on anyones mind is of heading to the pub. The relationship between the consumption of beer and weather is an obvious one but just how much does the weather affect our cravings for this most splendid of beverages?

The [dataset](https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo) we'll be working with comes from Kaggle and contains values for beer consumption along with various weather statistics for 365 days in the area of São Paulo, Brazil. Let's **hop** into looking at how we can use linear regression to model this relationship.

<div align='center'><img src='beer.jpeg'></div>

# Packages and settings


```python
# For plotting data
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Patch

# For importing tabulated data
import pandas as pd

# For representing and operating on vectors and matracies
import numpy as np

# Set the graph image to display as a support vector graphic
%config InlineBackend.figure_format = 'svg'

# Prevent pandas dataframe preview from taking up too much vertical space
pd.options.display.max_rows = 7
```

# Simple linear regression

With simple linear regression we aim to predict a dependent variable, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'>, given a single independent variable, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'>, by uncovering a linear relationship between the two. To demonstrate this process we'll investigate the relationship between the **volume of beer consumed**, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'>, over the course of a given day and the **average temperature**, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'>, for that very same day.

Our data comes in the form of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24m%24'> pairs of observations of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'>.  Both <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> are represented by seperate column vectors whose elements can be paired together using single subscript notation to produce observations of the form (<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x_%7Bi%7D%24'>, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y_%7Bi%7D%24'>).

<div align='center'><img src='https://render.githubusercontent.com/render/math?math=%24%0A%5Cbegin%7Balign%2A%7D%0A%26%0A%5Cvec%7Bx%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20x_%7Bi%7D%0A%5Cend%7Bbmatrix%7D%20%3D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20x_%7B1%7D%20%5C%5C%0A%20%20%20%20x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%20x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%26%0A%5Cvec%7By%7D%20%3D%20%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20y_%7Bi%7D%0A%5Cend%7Bbmatrix%7D%20%3D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20y_%7B1%7D%20%5C%5C%0A%20%20%20%20y_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%20y_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%5Cend%7Balign%2A%7D%0A%24'></div>

To solidify this representation further let's import and display our data using the python package, pandas.


```python
# Read in our dataset
df = pd.read_csv('beer.csv')
# Select a subset of the columns of our data
df = df[['avg_temp(c)', 'beer_consumption(l)']]
# Display our data
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_temp(c)</th>
      <th>beer_consumption(l)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27.30</td>
      <td>25.461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27.02</td>
      <td>28.972</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24.82</td>
      <td>30.814</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>21.68</td>
      <td>22.309</td>
    </tr>
    <tr>
      <th>363</th>
      <td>21.38</td>
      <td>20.467</td>
    </tr>
    <tr>
      <th>364</th>
      <td>24.76</td>
      <td>22.446</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 2 columns</p>
</div>



# Visualisation


```python
# Assign avgerage temperature to x and shape it into a column vector
x = df[['avg_temp(c)']].values.reshape(-1, 1)
# Assign beer consumption to y and shape it into a column vector
y = df[['beer_consumption(l)']].values.reshape(-1, 1)

# Configure the graph display size
fig, ax = plt.subplots(figsize=(11, 6))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add grid lines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
# Add title and axes labels
ax.set_title('Beer consumption as a function of temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Beer Consumption (Litres)')

# Plot observations of x and y on our graph
ax.scatter(x, y, alpha=0.70, color='xkcd:golden yellow', s=30)

# Plot lines and text for a
ax.plot([16, 21], [20, 20], color='xkcd:bright red', linestyle='-', linewidth=2)
ax.plot([21, 26], [26, 26], color='xkcd:bright red', linestyle='-', linewidth=2)
t = plt.text(18.5, 20, '$a$', horizontalalignment='center', verticalalignment='center', fontsize=14)
t.set_bbox(dict(facecolor='white', edgecolor='none'))
t = plt.text(23.5, 26, '$a$', horizontalalignment='center', verticalalignment='center', fontsize=14)
t.set_bbox(dict(facecolor='white', edgecolor='none'))

# Plot lines and text for b
ax.plot([21, 21], [20, 26], color='xkcd:charcoal', linestyle='-', linewidth=2)
ax.plot([26, 26], [26, 32], color='xkcd:charcoal', linestyle='-', linewidth=2)
t = plt.text(21, 23, '$b$', horizontalalignment='center', verticalalignment='center', fontsize=14)
t.set_bbox(dict(facecolor='white', edgecolor='none'))
t = plt.text(26, 29, '$b$', horizontalalignment='center', verticalalignment='center', fontsize=14)
t.set_bbox(dict(facecolor='white', edgecolor='none'))

plt.show()
```


![svg](output_8_0.svg)


In the above graph we observe an interesting trend of our data, namely, a **positive**, **linear** correlation between beer consumption and temperature.
- The correlation is **positive** because, on average, **beer consumption increases as temperature increases**.
- The correlation is **linear** because, on average, **a fixed increase in temperature, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24a%24'>, produces a fixed increase in beer consumption, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24b%24'>**.

This linear relationship of our data is exactly what we aim to model with simple linear regression.

# A hypothesis for our data

We model the linear relationship between <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Cvec%7Bx%7D%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Cvec%7By%7D%24'> using a mapping function, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%5Ccolon%20%5Cvec%7Bx%7D%20%5Cto%20%5Cvec%7By%7D%24'>, which is known as the hypothesis function and comes in the form <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%3B%20%5Ctheta_i%29%20%3D%20%5Ctheta_0%20%2B%20%5Ctheta_1%5Cvec%7Bx%7D%24'>. 

- <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%3B%20%5Ctheta_i%29%24'> is our **prediction** of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> given the variable <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'> and parameters <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_i%24'>.
- <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_0%24'> corresponds to the **y-intercept** of a line that fits the data.
- <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_1%24'> corresponds to **the rate of change of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> with respect to <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'>**, or the gradient of a line that fits the data.

<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%3B%20%5Ctheta_i%29%24'> refers to a function <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%24'> that takes a variable, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'>, and parameters <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_i%24'> as intput. The difference between a variable and a parameter, is a semantic one in that each of the parameters <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_i%24'> takes on a fixed, although currently unknown value, while the variable, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'>, will take on many different values from our dataset.

As both <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> are represented by column vectors we can compute <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%24'> via scalar multiplication and addition as follows.

<div align='center'><img src='https://render.githubusercontent.com/render/math?math=%24%0A%5Cbegin%7Balign%2A%7D%0Ah%28x%29%20%3D%20%5Ctheta_0%20%2B%20%5Ctheta_1%5Cvec%7Bx%7D%0A%3D%20%5Ctheta_0%20%2B%20%5Ctheta_1%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20x_%7B1%7D%20%5C%5C%0A%20%20%20%20x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%20x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%3D%20%5Ctheta_0%20%2B%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Ctheta_0x_%7B1%7D%20%5C%5C%0A%20%20%20%20%5Ctheta_0x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%20%5Ctheta_0x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%3D%20%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Ctheta_0%20%2B%20%5Ctheta_0x_%7B1%7D%20%5C%5C%0A%20%20%20%20%5Ctheta_0%20%2B%20%5Ctheta_0x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%20%5Ctheta_0%20%2B%20%5Ctheta_0x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%5Cend%7Balign%2A%7D%0A%24'></div>

Using some sensible values for <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_i%24'> let us now compute our prediction of beer consumption for each temperature in our dataset, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x_i%29%24'>, and plot this against the actual value for beer consumption in our dataset, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y_i%24'>, for comparison.


```python
# Assign avgerage temperature to x
x = df[['avg_temp(c)']].values

# Assign beer consumption to y
y = df[['beer_consumption(l)']].values

# Assign some guesses of theta_0, the y-intercept and theta_1, the gradient
theta = np.array([
    [0], 
    [1.19]
])

# Compute h(x) via scalar multiplication
h = theta[0] + thet[1] * x

# Configure the graph display size
fig, ax = plt.subplots(figsize=(11, 6))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid lines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Add title and axes labels
ax.set_title('Beer Consumption as a function of temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Beer Consumption (Litres)')

# Plot pairs of x and y on our graph
ax.scatter(x, y, alpha=0.70, color='xkcd:golden yellow', s=30)

# Plot h(x)
ax.plot(x, h, 'xkcd:charcoal', label='$h(x)$', linewidth=2)

# Add a legend to the graph
ax.legend()

# Display the graph
plt.show()
```


![svg](output_12_0.svg)



# Computing the hypothesis function via matrix multiplication

We just learned how we can compute <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%24'> via scalar multiplication and addition, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20%3D%20%5Ctheta_0%20%2B%20%5Ctheta_1%5Cvec%7Bx%7D%24'>, but there exists another method of computing <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%24'> that is both more compact and more efficient, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20%3D%20X%5Ctheta%24'>.

To derive <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20%3D%20X%5Ctheta%24'> from <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20%3D%20%5Ctheta_0%20%2B%20%5Cvec%7Bx%7D%5Ctheta_1%24'> we:
- Add a new column, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x_0%24'>, to <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x%24'> to produce the matrix <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24X%24'>.

<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%0A%5Cbegin%7Balign%2A%7D%0A%26%26%0A%5Cvec%7Bx_%7B0%7D%7D%20%3D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%201%20%5C%5C%0A%20%20%20%201%20%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%20%0A%20%20%20%201%0A%5Cend%7Bbmatrix%7D%0A%26%26%0AX%20%3D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7B1%7D%20%5C%5C%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%20%20%20%20%20%26%20%5Cvdots%20%5C%5C%20%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%26%26%0A%5Cvec%7B%5Ctheta%7D%20%3D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Ctheta_0%20%5C%5C%0A%20%20%20%20%5Ctheta_1%0A%5Cend%7Bbmatrix%7D%0A%5Cend%7Balign%2A%7D%0A%24'>

- Rewrite <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%24'> as a linear combination.

<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%0A%5Cbegin%7Balign%2A%7D%0A%26%26%0Ah%28x%29%20%5C%3B%3D%5C%3B%20%5Ctheta_0%20%2B%20%5Ctheta_1x_1%20%5C%3B%0A%3D%5C%3B%20%5Ctheta_0x_0%20%2B%20%5Ctheta_1x_1%20%5C%3B%0A%5Cend%7Balign%2A%7D%0A%24'>
<p markdown="1"> - Show that the linear combination <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_0x_0%20%2B%20%5Ctheta_1x_1%24'> is equal to <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24X%5Ctheta%24'>.</p>

<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%0A%5Cbegin%7Balign%2A%7D%0A%26%26%0A%5Ctheta_0x_0%20%2B%20%5Ctheta_1x_1%20%20%5C%3B%3D%5C%3B%0A%5Ctheta_%7B0%7D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%201%5C%5C%0A%20%20%20%201%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%0A%20%20%20%201%5C%5C%0A%5Cend%7Bbmatrix%7D%0A%2B%0A%5Ctheta_%7B1%7D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20x_%7B1%7D%5C%5C%0A%20%20%20%20x_%7B2%7D%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%0A%20%20%20%20x_%7Bm%7D%5C%5C%0A%5Cend%7Bbmatrix%7D%0A%5C%3B%3D%5C%3B%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Ctheta_%7B0%7D%20%2B%20%5Ctheta_%7B1%7Dx_%7B1%7D%5C%5C%0A%20%20%20%20%5Ctheta_%7B0%7D%20%2B%20%5Ctheta_%7B1%7Dx_%7B2%7D%5C%5C%0A%20%20%20%20%5Cvdots%20%5C%5C%0A%20%20%20%20%5Ctheta_%7B0%7D%20%2B%20%5Ctheta_%7B1%7Dx_%7Bm%7D%5C%5C%0A%5Cend%7Bbmatrix%7D%0A%5C%3B%3D%5C%3B%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7B1%7D%20%5C%5C%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7B2%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%20%20%20%20%20%26%20%5Cvdots%20%5C%5C%20%0A%20%20%20%201%20%20%20%20%20%20%26%20x_%7Bm%7D%0A%5Cend%7Bbmatrix%7D%0A%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Ctheta_0%20%5C%5C%0A%20%20%20%20%5Ctheta_1%0A%5Cend%7Bbmatrix%7D%0A%5C%3B%3D%5C%3B%0AX%5Ctheta%0A%5Cend%7Balign%2A%7D%0A%24'>

Now let us validate that the result of these two methods are indeed the same by equating them in code.


```python
# Assign avgerage temperature to x
x = df[['avg_temp(c)']].values

# Assign beer consumption to y
y = df[['beer_consumption(l)']].values

# Assign some guesses of theta_0, the y-intercept and theta_1, the gradient
theta = np.array([
    [3],
    [1.1]
])

# Compute h via scalar multiplication
h1 = theta[0] + theta[1] * x

# Add x0 to x
x = np.hstack([np.ones(x.shape), x])
# Comput h via matrix multiplication
h2 = x @ theta

# Verify that each row of h1 is equal to each row of h2
print(np.all(h1 == h2))
```

    True


# Measuring the fit of the hypothesis to our data

As you can see from the preceding graph, our values of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_%7B0%7D%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_%7B1%7D%24'> produce a mapping that fits our data reasonably well, but we can also see that a different set of values would probably produce a better fit. Moving away from visual inspection, how can we say with certainty which hypothesis best fits our data? The method of least squares regression provides us an answer in the form of a single metric that we can use to compare each of our hypotheses.


```python
# Assign avgerage temperature to x
x = df[['avg_temp(c)']].values
ind = np.argsort(x.squeeze())
x = x[ind]

# Assign beer consumption to y
y = df[['beer_consumption(l)']].values
y = y[ind]

# Assign some guesses of theta_0, the y-intercept and theta_1, the gradient
theta = np.array([
    [3],
    [1.1]
])

# Compute h(x)
h = theta[0] + theta[1] * x

# Configure the graph display size
fig, ax = plt.subplots(figsize=(11, 6))

# Plot h(x)
ax.plot(x, h, 'xkcd:charcoal', label='$h(x)$', linewidth=2)
# Plot the data points of (x, y) on our graph
ax.scatter(x, y, alpha=0.70, color='xkcd:golden yellow', s=30)

sample = (np.absolute(h - y).squeeze() < 5) & (np.absolute(h - y).squeeze() > 2)
x, y, h = x[sample], y[sample], h[sample]
spacing = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x)) / 14)
sample = np.argsort(np.absolute(x - spacing), axis=0)[0, :]
# sample = np.array([0, 1, 3, 5, 8, 16, 26, 30, 40, 48, 63, 85, 102, 118, 126, 132, 136, 142, 143, 145])
x, y, h = x[sample], y[sample], h[sample]

# Plot the residuals
ax.add_collection(LineCollection(np.hstack([x, y, x, h]).reshape((-1, 2, 2)), color='xkcd:bright red', linestyles='-', label='$residual$', linewidth=2))
# Create rectangles to represent the square of our residuals
residual2 = [Rectangle((a, b), abs(c - b) / 2.5 * 0.894, c - b) for a, b, c in zip(x, y, h)]
# Plot the rectangles
ax.add_collection(PatchCollection(residual2, color="xkcd:black", edgecolor='none', alpha=0.25, label='${residual}^2$'))

# Add title and axes labels
ax.set_title('Beer consumption as a function of temperature')
ax.set_xlabel('Temperature / $x_1$ (°C)')
ax.set_ylabel('Beer Consumption / $y$ (litres)')
    
# Remove top and right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
    
# Add grid lines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Add custom legend to the second subplot that includes a label for the square of our residuals
ax.legend(handles=ax.get_legend_handles_labels()[0] + [Patch(color='xkcd:black', label='${residual}^2$', alpha=0.5)])

# Display the graph
plt.show()
```


![svg](output_18_0.svg)


Residuals (also known as the errors of prediction) are defined as the differences between our predictions of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> and the observed values of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'>, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20-%20y%24'>, and are depicted by the red lines in the graph above. We can use the mean average of the residuals as an indicator of how well our hypothesis fits our data; a smaller value implies a better fitting hypothesis.

<div align='center'><img src='https://render.githubusercontent.com/render/math?math=%24%5Cmin_%7Bh%28x%29%7D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7B1%7D%20h%28x%29%20-%20y%24'></div>

You may have noticed that when <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%24'> is less than <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24y%24'> the corresponding residual takes on a negative value. As our residuals represent the error of prediction, a negative error implies a prediction better than the observed value, which is impossible, and could cancel out other errors during summation. For this reason we ensure that all values of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24h%28x%29%20-%20y%24'> are positive by taking their square before summation. The squared residuals can also be visualised in the above graph. <div align='center'><img src='https://render.githubusercontent.com/render/math?math=%24%5Cmin_%7Bh%28x%29%7D%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7B1%7D%20%28h%28x%29%20-%20y%29%5E2%24'></div>

Thus we have derived the least squares regression cost function, where cost refers to the fact that larger values are inherently bad. The preferential use of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%28h%28x%29%20-%20y%29%5E2%24'> over <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%7Ch%28x%29%20-%20y%7C%24'> to fix the sign of the residuals may at first look like an overcomplication but does infact reward us in several ways, the most important of which, is that the derivative of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%28h%28x%29%20-%20y%29%5E2%24'> is continuous while the derivative of  <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%7Ch%28x%29%20-%20y%7C%24'> is not. The significance of this fact may not manifest immediately but will later become clear when we discuss the topic of gradient descent as a method of finding optimal model parameters in a subsequent notebook.

# Finding the best fitting hypothesis for our data

There are several methods for finding optimal values of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_0%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_1%24'>, but two common methods are gradient descent and analytical analysis via normal equations.

The key difference between the two methods lies in their algrithmic complexity. Gradient descent has a complexity of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24O%28n%5E2%29%24'> while analytical analysis via normal equation has a complexity of <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24O%28n%5E3%29%24'>, where <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24n%24'> is the number of explanatory variables, <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24x_1%2C%20...%2C%20x_n%24'>. This makes gradient descent better suited to problems with many explanatory variables, typically tens of thoundsands.

As we are working with data that contains just one explanatory variable we will look at using the linear algebra library numpy to solve for <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_0%24'> and <img valign='middle' src='https://render.githubusercontent.com/render/math?math=%24%5Ctheta_1%24'>.


```python
# Assign avgerage temperature to x and reshape it into a column vector
x = df[['avg_temp(c)']].values.reshape(-1, 1)

# Assign beer consumption to y and reshape it into a column vector
y = df[['beer_consumption(l)']].values.reshape(-1, 1)

# Add x0
x = np.hstack([np.ones((len(x), 1)), x])

# Find optimal values of theta0 and theta1 then reshape into a column vector
theta = np.linalg.lstsq(x, y, rcond=None)[0].reshape(-1, 1)

# Compute h
h = x @ theta

# Configure the graph display size
fig, ax = plt.subplots(figsize=(11, 6))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid lines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Add title and axes labels
ax.set_title('Beer consumption as a function of temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Beer Consumption (Litres)')

# Plot pairs of x and y on our graph
ax.scatter(x[:, 1], y, alpha=0.70, color='xkcd:golden yellow', s=30)

# Plot h(x)
ax.plot(x[:, 1], h, 'xkcd:charcoal', label='$h(x)$', linewidth=2)

# Add a legend to the graph
ax.legend()

# Display the graph
plt.show()
```


![svg](output_22_0.svg)


# Making a prediction

Let's imagine for a second that we're a pub owner in São Paulo and we are trying to decide if we have enough barrels of beer for an approaching day that has a temperature forecast of **25.32°C**. Using our linear regression model we can make a sensible prediction of beer consumption based on data we've collected over the previous year.


```python
# Assign avgerage temperature to x and reshape it into a column vector
x = df[['avg_temp(c)']].values.reshape(-1, 1)

# Assign beer consumption to y and reshape it into a column vector
y = df[['beer_consumption(l)']].values.reshape(-1, 1)

# Add x0
x = np.hstack([np.ones((len(x), 1)), x])

# Find optimal values of theta0 and theta1 then reshape into a column vector
theta = np.linalg.lstsq(x, y, rcond=None)[0].reshape(-1, 1)

h = np.array([[1, 25.32]]) @ theta

print("Beer consumption prediction for %s°C: %s " % (25.32, h[0][0]))
```

    Beer consumption prediction for 25.32°C: 28.655333120934497 

