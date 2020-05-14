{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 align=\"center\"> Logistic Regression </h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 2: Load the Data and Libraries\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "plt.style.use(\"ggplot\")\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pylab import rcParams\n",
    "rcParams['figure.figsize'] = 12, 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>DMV_Test_1</th>\n",
       "      <th>DMV_Test_2</th>\n",
       "      <th>Results</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>34.623660</td>\n",
       "      <td>78.024693</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>30.286711</td>\n",
       "      <td>43.894998</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>35.847409</td>\n",
       "      <td>72.902198</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>60.182599</td>\n",
       "      <td>86.308552</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>79.032736</td>\n",
       "      <td>75.344376</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   DMV_Test_1  DMV_Test_2  Results\n",
       "0   34.623660   78.024693        0\n",
       "1   30.286711   43.894998        0\n",
       "2   35.847409   72.902198        0\n",
       "3   60.182599   86.308552        1\n",
       "4   79.032736   75.344376        1"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"DMV_Written_Tests.csv\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 100 entries, 0 to 99\n",
      "Data columns (total 3 columns):\n",
      "DMV_Test_1    100 non-null float64\n",
      "DMV_Test_2    100 non-null float64\n",
      "Results       100 non-null int64\n",
      "dtypes: float64(2), int64(1)\n",
      "memory usage: 2.4 KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "scores = data[['DMV_Test_1', 'DMV_Test_2']].values\n",
    "results = data['Results'].values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 3: Visualize the Data\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "passed = (results == 1).reshape(100, 1)\n",
    "failed = (results == 0).reshape(100, 1)\n",
    "\n",
    "ax = sns.scatterplot(x = scores[passed[:, 0], 0],\n",
    "                     y = scores[passed[:, 0], 1],\n",
    "                     marker = \"^\",\n",
    "                     color = \"green\",\n",
    "                     s = 60)\n",
    "sns.scatterplot(x = scores[failed[:, 0], 0],\n",
    "                y = scores[failed[:, 0], 1],\n",
    "                marker = \"X\",\n",
    "                color = \"red\",\n",
    "                s = 60)\n",
    "\n",
    "ax.set(xlabel=\"DMV Written Test 1 Scores\", ylabel=\"DMV Written Test 2 Scores\")\n",
    "ax.legend([\"Passed\", \"Failed\"])\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 4: Define the Logistic Sigmoid Function $\\sigma(z)$\n",
    "---\n",
    "\n",
    "$$ \\sigma(z) = \\frac{1}{1+e^{-z}}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def logistic_function(x):    \n",
    "    return 1/ (1 + np.exp(-x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_function(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 5: Compute the Cost Function $J(\\theta)$ and Gradient\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The objective of logistic regression is to minimize the cost function\n",
    "\n",
    "$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} [ y^{(i)}log(h_{\\theta}(x^{(i)})) + (1 - y^{(i)})log(1 - (h_{\\theta}(x^{(i)}))]$$\n",
    "\n",
    "where the gradient of the cost function is given by\n",
    "\n",
    "$$ \\frac{\\partial J(\\theta)}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (h_{\\theta}(x^{(i)}) - y^{(i)})x_j^{(i)}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_cost(theta, x, y):\n",
    "    m = len(y)\n",
    "    y_pred = logistic_function(np.dot(x , theta))\n",
    "    error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))\n",
    "    cost = -1 / m * sum(error)\n",
    "    gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))\n",
    "    return cost[0] , gradient"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 6: Cost and Gradient at Initialization\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cost at initialization 0.693147180559946\n",
      "Gradient at initialization: [[-0.1       ]\n",
      " [-0.28122914]\n",
      " [-0.25098615]]\n"
     ]
    }
   ],
   "source": [
    "mean_scores = np.mean(scores, axis=0)\n",
    "std_scores = np.std(scores, axis=0)\n",
    "scores = (scores - mean_scores) / std_scores #standardization\n",
    "\n",
    "rows = scores.shape[0]\n",
    "cols = scores.shape[1]\n",
    "\n",
    "X = np.append(np.ones((rows, 1)), scores, axis=1) #include intercept\n",
    "y = results.reshape(rows, 1)\n",
    "\n",
    "theta_init = np.zeros((cols + 1, 1))\n",
    "cost, gradient = compute_cost(theta_init, X, y)\n",
    "\n",
    "print(\"Cost at initialization\", cost)\n",
    "print(\"Gradient at initialization:\", gradient)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 7: Gradient Descent\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Minimize the cost function $J(\\theta)$ by updating the below equation and repeat until convergence\n",
    "$\\theta_j := \\theta_j - \\alpha \\frac{\\partial J(\\theta)}{\\partial \\theta_j}$ (simultaneously update $\\theta_j$ for all $j$)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient_descent(x, y, theta, alpha, iterations):\n",
    "    costs = []\n",
    "    for i in range(iterations):\n",
    "        cost, gradient = compute_cost(theta, x, y)\n",
    "        theta -= (alpha * gradient)\n",
    "        costs.append(cost)\n",
    "    return theta, costs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "theta, costs = gradient_descent(X, y, theta_init, 1, 200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Theta after running gradient descent: [[1.50850586]\n",
      " [3.5468762 ]\n",
      " [3.29383709]]\n",
      "Resulting cost: 0.2048938203512014\n"
     ]
    }
   ],
   "source": [
    "print(\"Theta after running gradient descent:\", theta)\n",
    "print(\"Resulting cost:\", costs[-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 8: Plotting the Convergence of $J(\\theta)$\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot $J(\\theta)$ against the number of iterations of gradient descent:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(costs)\n",
    "plt.xlabel(\"Iterations\")\n",
    "plt.ylabel(\"$J(\\Theta)$\")\n",
    "plt.title(\"Values of Cost Function over iterations of Gradient Descent\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 9: Plotting the decision boundary\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$h_\\theta(x) = \\sigma(z)$, where $\\sigma$ is the logistic sigmoid function and $z = \\theta^Tx$\n",
    "\n",
    "When $h_\\theta(x) \\geq 0.5$ the model predicts class \"1\":\n",
    "\n",
    "$\\implies \\sigma(\\theta^Tx) \\geq 0.5$\n",
    "\n",
    "$\\implies \\theta^Tx \\geq 0$ predict class \"1\" \n",
    "\n",
    "Hence, $\\theta_1 + \\theta_2x_2 + \\theta_3x_3 = 0$ is the equation for the decision boundary, giving us \n",
    "\n",
    "$ x_3 = \\frac{-(\\theta_1+\\theta_2x_2)}{\\theta_3}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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VLIpjYaIiIiItMsJJ9Ty2mslXHVVJStXpjJkSB5LlqQRCEQ7stigxFpERERE2i01NcTUqZWsXVvCwIH15Odncsopft57LzHaoUWdEmsRERER6bDvf7+BJ5/cyqJFZXz7rY/Ro/3k52dQUdF7W6MrsRYRERGRTvF4YMyYaoqKirnwwp08+mgagwfn8fTTqb2yNboSaxERERHpkvT0ELfcsp01a0o44IAAv/lNFmeemcPHHydEO7QepcRaRERERCJi4MAGVq4sZf78cj78MJERI3IpKOhHVVXvKA9RYi0iIiIiEeP1wrnnOq3RzzijmkWL+jF0aC4vvpgS9+UhSqxFREREJOJycoL89rflPPtsKf36hbjoomwmTMjmyy/jtzW6EmsRERER6TY/+1kdL75YwqxZFbz5ZhLHHZfHwoV9qa2NdmSRp8RaRERERLpVYiJMmrSToqJihg+v4fbb0zn++Dxefz2+WqMrsRYRERGRHtG/f5A//KGMxx7bSjAI48b5mTw5k+++i4+UND4+hYiIiIi4xnHH1bJuXTHXXlvJmjWpDB6cx8MP96GhIdqRdY0SaxERERHpcSkpcN11laxdW8yRR9Zx440ZjBqVy9/+5t7W6EqsRURERCRqDjkkwLJl23jggW1s3erl1FNzueGGDLZtc9/a10qsRURERCSqPB44+eQaioqK+fWvd/DEE2kMGZLHk0+mEgxGO7r2U2ItIiIiIjGhb98QN964nRdfLOGQQwJce20WY8fm8MEH7miNrsRaRERERGLKD3/YwLPPlvLb35bxyScJjByZyy23pMd850Yl1iIiIiISc7xeOOusajZsKGbcuCp27vTgifGya3fMq4uIiIhIr5SdHeL22ytcUWutGWsRERERiXleF2StLghRRERERCT2KbEWEREREYkAJdYiIiIiIhGgxFpEREREJAKUWIuIiIiIRIASaxERERGRCFBiLSIiIiISAUqsRUREREQiQIm1iIiIiEgEKLEWEREREYkAJdYiIiIiIhGgxFpEREREJAKUWIuIiIiIRIASaxERERGRCFBiLSIiIiISAUqsRUREREQiQIm1iIiIiEgEeEKhULRj6CzXBi4iIiIiruNpawc3z1h7mrsZY/7W0mO6dfym8dR4xvJN46nxjOWbxlPjGcs3jWenbm1yc2ItIiIiIhIzlFiLiIiIiERAPCbW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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x = X[passed[:, 0], 1],\n",
    "                y = X[passed[:, 0], 2],\n",
    "                marker = \"^\",\n",
    "                color = \"green\",\n",
    "                s = 60)\n",
    "ax = sns.scatterplot(x = X[failed[:, 0], 1],\n",
    "                    y = X[failed[:, 0], 2],\n",
    "                    marker = \"X\",\n",
    "                    color = \"red\",\n",
    "                    s = 60)\n",
    "\n",
    "ax.legend([\"Passed\", \"Failed\"])\n",
    "ax.set(xlabel=\"DMV Written Test 1 Scores\", ylabel=\"DMV Written Test 2 Scores\")\n",
    "\n",
    "x_boundary = np.array([np.min(X[:, 1]), np.max(X[:, 1])])\n",
    "y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]\n",
    "\n",
    "sns.lineplot(x = x_boundary, y = y_boundary, color=\"blue\")\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 10: Predictions using the optimized $\\theta$ values\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$h_\\theta(x) = x\\theta$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(theta, x):\n",
    "    results = x.dot(theta)\n",
    "    return results > 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Accuracy: 89 %\n"
     ]
    }
   ],
   "source": [
    "p = predict(theta, X)\n",
    "print(\"Training Accuracy:\", sum(p==y)[0],\"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A person who scores 50 and 79 on their DMV written tests have a 0.71 probability of passing.\n"
     ]
    }
   ],
   "source": [
    "test = np.array([50,79])\n",
    "test = (test - mean_scores)/std_scores\n",
    "test = np.append(np.ones(1), test)\n",
    "probability = logistic_function(test.dot(theta))\n",
    "print(\"A person who scores 50 and 79 on their DMV written tests have a\",\n",
    "      np.round(probability[0], 2),\"probability of passing.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
