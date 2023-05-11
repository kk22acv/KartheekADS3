#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import cluster_tools
import errors


# In[12]:


def read_data(fname = 'Population, total.csv', fname2 = 'Electric power consumption (kWh per capita).csv'):
    df = pd.read_csv(fname)
    df = df.melt(id_vars = 'date', var_name = 'country', value_name = 'population')
    df1 = pd.read_csv(fname2)
    df1 = df1.melt(id_vars = 'date', var_name = 'country', value_name = 'power_cons')
    df = df.loc[df1.power_cons.notna(), :].reset_index(drop = True)
    df1 = df1.loc[df1.power_cons.notna(), :].reset_index(drop = True)
    df_main = pd.DataFrame({"population":df.population, 'power_cons':df1.power_cons})
    df_scaled, df_min, df_max = cluster_tools.scaler(df_main) # scale
    return df, df1, df_main, df_scaled, df_min, df_max

df, df1, df_main, df_scaled, df_min, df_max = read_data()


# In[33]:


# Compute correlation
cluster_tools.map_corr(df_scaled)
plt.title("Correlation Heatmap")
plt.show()


# In[6]:


cost = [0]*9
for k in range(1,10):
    kmeans = KMeans(n_clusters = k, n_init = 'auto')
    kmeans.fit(df_scaled)
    cost[k-1] = kmeans.inertia_
plt.plot(range(1, 10), cost, 'bx-')
plt.title('Elbow Plot')
plt.xlabel('k')
plt.ylabel('inertia')
plt.grid()
plt.show()


# In[14]:


# Function to plot clusters
def plot_clusters(df, df_scaled, k = 4):
    # Input dataframe should be scaled
    kmeans = KMeans(n_clusters = k, n_init = 'auto')
    kmeans.fit(df_scaled)
    c_centers_main = cluster_tools.backscale(kmeans.cluster_centers_, df_min, df_max) # back scaled
    df['label'] = kmeans.labels_
    color_codes = ['red', 'blue', 'yellow', 'green']
    for i in range(k):
        x = df.loc[df['label'] == i, 'population']
        y = df.loc[df['label'] == i, 'power_cons']
        plt.scatter(x, y, color = color_codes[i])
        plt.scatter(c_centers_main[i, 0], c_centers_main[i, 1], marker = '3', s = 100, color = 'black')
    plt.xlabel('Population')
    plt.ylabel('Power Consumption')
    plt.title('Clusters of Power Consumption by Population')
    plt.grid(alpha = 0.5)
    plt.show()
plot_clusters(df_main, df_scaled, k = 4)


# In[34]:


# Compare power consumption within clusters
def plot_cluster(df, df_main, k, plot = True):    
    filter_ = (df_main.label == k)
    c, d = list(df.country[filter_].unique()), df.date[filter_] # country, date
    dfc = df_main.loc[filter_, ['population', 'power_cons']]
    dfc['date'] = d
    dfc = dfc.groupby('date').agg({'population':'mean', 'power_cons':'mean'})
    print(f"Countries in Cluster {k+1}\n-----------------------------\n{' '.join(c)}\n")
    if plot:
        dfc.plot('population', 'power_cons', kind = 'scatter')
        plt.ylabel('Power Consumption')
        plt.title(f"Cluster {k+1}")
        plt.grid()
        plt.show()
    else:
        return dfc

for i in range(2,4):
    plot_cluster(df, df_main, k = i)


# In[35]:


# Curve fitting data for clusters
def quad_c3(x, a, b, c):
    return a*x**2 + b*x + c

def linear_c4(x, a, b):
    return a*x + b

def fit_and_plot(dfc, cluster = 3):
    # Only for cluster 3 and 4
    xraw = dfc['population'].to_numpy()
    idx = np.argsort(xraw)
    xraw = xraw[idx]
    yraw = dfc['power_cons'].to_numpy()[idx]
    if cluster == 3:
        params, pcov = curve_fit(quad_c3, xraw, yraw)
        yfit = quad_c3(xraw, *params) # predicted
        sigmas = np.sqrt(np.diag(pcov)) # Sigmas
        func = quad_c3
    elif cluster == 4:
        params, pcov = curve_fit(linear_c4, xraw, yraw)
        yfit = linear_c4(xraw, *params) # predicted
        sigmas = np.sqrt(np.diag(pcov)) # Sigmas
        func = linear_c4
    else:
        return "Invalid Cluster Number"
    
    dfc.plot('population', 'power_cons', kind = 'scatter', label = 'Actual Power Consumption')
    plt.plot(xraw, yfit, label = "Estimated Power Consumption", color = 'green')
    if all(sigmas > 1): # Check if all sigmas are greater than 1
        ylow, yup = errors.err_ranges(xraw, func, params, sigmas)
        plt.fill_between(xraw, ylow, yup, alpha = 0.3, label = 'Confidence Range')
    plt.title("Actual and Estimates")
    plt.ylabel('Power Consumption')
    plt.grid()
    plt.legend()
    plt.show()


# In[36]:


# Third Cluster
cluster = 3
dfc3 = plot_cluster(df, df_main, k = cluster-1, plot = False)
fit_and_plot(dfc3, cluster = 3)


# In[37]:


# Fourth Cluster
cluster = 4
dfc3 = plot_cluster(df, df_main, k = cluster-1, plot = False)
fit_and_plot(dfc3, cluster = 3)


# In[ ]:




