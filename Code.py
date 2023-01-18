##### Import Libraries
import pandas as pd
import numpy as np
import statistics as stat

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

##### Upload Data
JD_user_data = pd.read_csv("JD_user_data.csv")
JD_sku_data = pd.read_csv("JD_sku_data.csv")
JD_order_data = pd.read_csv("JD_order_data.csv")
JD_inventory_data = pd.read_csv("JD_inventory_data.csv")
JD_network_data = pd.read_csv("JD_network_data.csv")
JD_delivery_data = pd.read_csv("JD_delivery_data.csv")
#JD_click_data = pd.read_csv("JD_click_data.csv")

##### Merge SKU and Order Data
df_merged = pd.merge(JD_order_data, JD_sku_data, how = "left", on=["sku_ID"])
df_merged_head = df_merged.head(1000)
df_merged_tail = df_merged.tail(1000)
print("Number of SKUs total:",len(JD_sku_data.sku_ID.unique()))
print("Number of SKUs sold:",len(df_merged.sku_ID.unique())) 
print("Proportion of SKUs sold:",len(df_merged.sku_ID.unique())/len(JD_sku_data.sku_ID.unique()))
print("Number of Orders:",len(df_merged))
print("Number of Customers:",len(JD_user_data.user_ID.unique()))
print("Start Date:",df_merged.sort_values(by="order_date").order_date.to_list()[0])
print("End Date:",df_merged.sort_values(by="order_date").order_date.to_list()[-1])

##### Preprocessing: Make order day column
order_day = []
for date in df_merged.order_date:
    order_day.append(int(date[-2:]))
df_merged.insert(4, "order_day", order_day)

##### Preprocessing: Make order day name column and business day column 
order_day_name = []
order_day_business_day = []
for date in df_merged.order_date:
    day = int(date[-2:])
    
    if day in [1,8,15,22,29]:
        order_day_name.append('Thursday')
        order_day_business_day.append(1)
    
    elif day in [2,9,16,23,30]:
        order_day_name.append('Friday')
        order_day_business_day.append(1)
    
    elif day in [3,10,17,24,31]:
        order_day_name.append('Saturday')
        order_day_business_day.append(0)
    
    elif day in [4,11,18,25]:
        order_day_name.append('Sunday')
        order_day_business_day.append(0)
    
    elif day in [5,12,19,26]:
        order_day_name.append('Monday')
        order_day_business_day.append(1)
    
    elif day in [6,13,20,27]:
        order_day_name.append('Tuesday')
        order_day_business_day.append(1)
    
    elif day in [7,14,21,28]:
        order_day_name.append('Wednesday')
        order_day_business_day.append(1)

df_merged['order_day_name'] = order_day_name
df_merged['order_day_business_day'] = order_day_business_day

##### Preprocessing: Make total discount and 
df_merged["total_discount"] = 1-(df_merged.final_unit_price/df_merged.original_unit_price)
df_merged["sales"] = df_merged.final_unit_price*df_merged.quantity
df_merged["total_discount"] = df_merged["total_discount"].replace(np.inf,1)
df_merged["total_discount"] = df_merged["total_discount"].replace(np.nan,0)

##### Split into Training (first three weeks of month) and Test Data (last week of month)
df_train  = df_merged[df_merged.order_day < 24]
df_test  = df_merged[df_merged.order_day >= 24]

##### Aggregate Clustering Data 
list_sku = []
for sku in df_train.sku_ID.unique():
    df_sku_i = df_train[df_train.sku_ID==sku]
    dict_sku_i = {"sku_ID": sku,
                  "quantity": int(df_sku_i.quantity.sum()),
                  "average_price": int(df_sku_i[df_sku_i.gift_item!=1].original_unit_price.mean()) if len(df_sku_i[df_sku_i.gift_item!=1]) != 0 else 0, #average price of non-gift items
                  "type": df_sku_i.type_x.to_list()[0],
                  "brand_ID": df_sku_i.brand_ID.to_list()[0],
                  "attribute1": df_sku_i.attribute1.to_list()[0],
                  "attribute2": df_sku_i.attribute2.to_list()[0]
                  }
    list_sku.append(dict_sku_i)
df_cluster = pd.DataFrame(list_sku)
  
### Optimal Clustering: Quantity, Average Price, Type 
X = df_cluster[["quantity", "average_price", "type"]]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
list_silhouette_STD = []
num_skus = []
num_orders = []
training_data_obs = []
for i in range(2,11):
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0)
    model = kmeans.fit(X_std)
    labels = model.predict(X_std)
    avg_silhouette_score = silhouette_score(X_std,labels)
    
    df_cluster['cluster_label'] = labels
    num_skus.append(df_cluster.cluster_label.value_counts())
    
   
    df_train_new_CLUST = pd.merge(df_train, df_cluster[["sku_ID","cluster_label"]], how = "left", on=["sku_ID"])
    num_orders.append(df_train_new_CLUST.cluster_label.value_counts())

    row_aggs = []
    for j in range(0,i):
        row_aggs.append(len(df_train_new_CLUST[df_train_new_CLUST.cluster_label == j].drop_duplicates(["sku_ID","order_day"])))
    training_data_obs.append(row_aggs)
    
    dict_silhouette = {"number_of_clusters": i,
                       "silhouette_sscore": avg_silhouette_score
                       }
    list_silhouette_STD.append(dict_silhouette)
df_silhouette_STD = pd.DataFrame(list_silhouette_STD)

### Clustering n=2: Quantity, Average Price, Type (Standardized)
n=2
X = df_cluster[["quantity", "average_price", "type"]]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=n, random_state=0)
model = kmeans.fit(X_std)
labels = model.predict(X_std)
df_cluster['cluster_label'] = labels
print("Number of SKUs in each cluster:\n",df_cluster.cluster_label.value_counts())
df_train_new = pd.merge(df_train, df_cluster[["sku_ID","cluster_label"]], how = "left", on=["sku_ID"])
print("Number of Orders in each cluster:\n",df_train_new.cluster_label.value_counts())

### Clustering n=2: Scatter Plots
sns.scatterplot(
    data=df_cluster, 
    x='quantity', 
    y='average_price', 
    hue='cluster_label',
    palette=['red', 'orange']
    )
plt.title("Quantity vs Average Price (n="+str(n)+")")
plt.xlabel('Quantity')
plt.ylabel('Average Price')
plt.show()

sns.scatterplot(
    data=df_cluster, 
    x='type', 
    y='quantity', 
    hue='cluster_label',
    palette=['red', 'orange']
    )
plt.title("Type vs Quantity (n="+str(n)+")")
plt.xlabel('Type')
plt.ylabel('Quantity')
plt.show()

sns.scatterplot(
    data=df_cluster, 
    x='type', 
    y='average_price', 
    hue='cluster_label',
    palette=['red', 'orange']
    )
plt.title("Type vs Average Price (n="+str(n)+")")
plt.xlabel('Type')
plt.ylabel('Average Price')
plt.show()

### Add cluster labels to data 
df_train_new_new = df_train_new
df_cluster_0 = df_train_new_new[df_train_new_new.cluster_label==0]
df_cluster_1 = df_train_new_new[df_train_new_new.cluster_label==1]

df_test_cluster_0 = df_test[df_test.sku_ID.isin(list(df_cluster_0.sku_ID))]
df_test_cluster_0['cluster_label'] = [0]*len(df_test_cluster_0)
df_cluster_0_new = df_cluster_0.append(df_test_cluster_0)

df_test_cluster_1 = df_test[df_test.sku_ID.isin(list(df_cluster_1.sku_ID))]
df_test_cluster_1['cluster_label'] = [1]*len(df_test_cluster_1)
df_cluster_1_new = df_cluster_1.append(df_test_cluster_1)






##### Feature Engineering 
def create_lags(df_cluster_3):

    ### Handle attribute null vales: make them averages of SKUs in the saem cluster
    attribute1_mean = df_cluster_3[(df_cluster_3.attribute1.notna())&(df_cluster_3.attribute1 != '-')].attribute1.astype(float).mean()
    attribute2_mean = df_cluster_3[(df_cluster_3.attribute2.notna())&(df_cluster_3.attribute2 != '-')].attribute2.astype(float).mean()
    df_cluster_3.attribute1 = df_cluster_3.attribute1.replace(np.nan, attribute1_mean)
    df_cluster_3.attribute1 = df_cluster_3.attribute1.replace('-', attribute1_mean)
    df_cluster_3.attribute2 = df_cluster_3.attribute2.replace(np.nan, attribute2_mean)
    df_cluster_3.attribute2 = df_cluster_3.attribute2.replace('-', attribute2_mean)

    ### Daily averages
    list_day_i = []
    for sku in df_cluster_3.sku_ID.unique():
        df_sku_i = df_cluster_3[(df_cluster_3.sku_ID == sku)&(df_cluster_3.gift_item != 1)] ### No gift items 
            
        for day in df_sku_i.order_day.unique():
            df_day_i = df_sku_i[df_sku_i.order_day == day]


            dict_day_i = {"sku_ID": sku,
                          "order_day": day,
                          "day_of_week": list(df_day_i.order_day_name)[0],
                          "weekday": list(df_day_i.order_day_business_day)[0],
                          "avg_final_price": df_day_i.final_unit_price.mean(),
                          "avg_total_discount": df_day_i.total_discount.mean(),
                          "avg_direct_discount": df_day_i.direct_discount_per_unit.mean(),
                          "avg_quantity_discount": df_day_i.quantity_discount_per_unit.mean(),
                          "avg_bundle_discount": df_day_i.bundle_discount_per_unit.mean(),
                          "avg_coupon_discount": df_day_i.coupon_discount_per_unit.mean(),
                          "type": float(list(df_day_i.type_x)[0]),
                          "attribute1": float(list(df_day_i.attribute1)[0]),
                          "attribute2": float(list(df_day_i.attribute2)[0]),
                          "avg_final_price_other": df_cluster_3[(df_cluster_3.order_day == day)&(df_cluster_3.sku_ID != sku)].final_unit_price.mean(),
                          "sum_quantity": df_day_i.quantity.sum(),
                          "sum_sales": df_day_i.sales.sum(),
                          "gift_items_in_orders": df_cluster_3[df_cluster_3.order_ID.isin(df_day_i.order_ID.unique())].gift_item.sum() ##### CHNAGE: GIFT ITEMS IN ORDER
                          }
            list_day_i.append(dict_day_i)
    features_cluster_3 = pd.DataFrame(list_day_i)

     ### Lagged daily averages
    df_lagged_list = []
    for sku in features_cluster_3.sku_ID.unique():
        df_sku_i = features_cluster_3[features_cluster_3.sku_ID == sku]

        df_sku_i['avg_total_discount_lag'] = [np.nan]+df_sku_i.avg_total_discount.to_list()[:-1]
        df_sku_i['avg_direct_discount_lag'] = [np.nan]+df_sku_i.avg_direct_discount.to_list()[:-1]
        df_sku_i['avg_quantity_discount_lag'] = [np.nan]+df_sku_i.avg_quantity_discount.to_list()[:-1]
        df_sku_i['avg_bundle_discount_lag'] = [np.nan]+df_sku_i.avg_bundle_discount.to_list()[:-1]
        df_sku_i['avg_coupon_discount_lag'] = [np.nan]+df_sku_i.avg_coupon_discount.to_list()[:-1]
        df_sku_i['avg_final_price_other_lag'] = [np.nan]+df_sku_i.avg_final_price_other.to_list()[:-1]
        df_sku_i['sum_quantity_lag'] = [np.nan]+df_sku_i.sum_quantity.to_list()[:-1]
        df_sku_i['sum_sales_lag'] = [np.nan]+df_sku_i.sum_sales.to_list()[:-1]
        df_sku_i['sum_sales_lag'] = [np.nan]+df_sku_i.sum_sales.to_list()[:-1]
        df_sku_i['gift_items_in_orders_lag'] = [np.nan]+df_sku_i.gift_items_in_orders.to_list()[:-1] ##### CHNAGE: GIFT ITEMS IN ORDER


        df_lagged_list.append(df_sku_i)

    df_lagged = pd.concat(df_lagged_list).sort_values(by="order_day")
    
    return df_lagged

df_cluster_0_train_test_og = create_lags(df_cluster_0_new)
df_cluster_0_train_test = df_cluster_0_train_test_og.dropna()
df_cluster_0_train_test['type'] = df_cluster_0_train_test['type'].replace(2,0)
df_cluster_0_train = df_cluster_0_train_test[df_cluster_0_train_test.order_day < 24]
df_cluster_0_test = df_cluster_0_train_test[df_cluster_0_train_test.order_day >= 24]

df_cluster_1_train_test_og = create_lags(df_cluster_1_new)
df_cluster_1_train_test = df_cluster_1_train_test_og.dropna()
df_cluster_1_train_test['type'] = df_cluster_1_train_test['type'].replace(2,0)
df_cluster_1_train = df_cluster_1_train_test[df_cluster_1_train_test.order_day < 24]
df_cluster_1_test = df_cluster_1_train_test[df_cluster_1_train_test.order_day >= 24]

### Feature Importance 
df_xxxx = pd.get_dummies(df_cluster_0_train.append(df_cluster_1_train), columns = ['day_of_week'])

xxxx_train = df_xxxx[['avg_final_price','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','day_of_week_Monday','day_of_week_Tuesday','day_of_week_Wednesday','day_of_week_Thursday','day_of_week_Friday','sum_quantity_lag','avg_coupon_discount_lag','avg_bundle_discount_lag','avg_quantity_discount_lag','avg_direct_discount_lag','weekday','order_day','gift_items_in_orders_lag']]
yyyy_train = df_xxxx[['sum_quantity']]

randomforest_xxxx = RandomForestRegressor(random_state=0)
model_xxxx = randomforest_xxxx.fit(xxxx_train,yyyy_train)

feat_imp_xxxx = pd.DataFrame()
feat_imp_xxxx['Feature'] = xxxx_train.columns
feat_imp_xxxx['Importance'] = model_xxxx.feature_importances_
feat_imp_xxxx = feat_imp_xxxx.sort_values(by="Importance", ascending=False)

### Correlation
corr_xxxx = df_xxxx[['sum_quantity','avg_final_price','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','day_of_week_Monday','day_of_week_Tuesday','day_of_week_Wednesday','day_of_week_Thursday','day_of_week_Friday','sum_quantity_lag','avg_coupon_discount_lag','avg_bundle_discount_lag','avg_quantity_discount_lag','avg_direct_discount_lag','weekday','order_day','gift_items_in_orders_lag']].corr()

### Cluster 0 Demand Model
X_train_cluster_0 = df_cluster_0_train[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday','gift_items_in_orders_lag']]
y_train_cluster_0 = df_cluster_0_train[['sum_quantity']]

X_test_cluster_0 = df_cluster_0_test[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday','gift_items_in_orders_lag']]
y_test_cluster_0 = df_cluster_0_test[['sum_quantity']]

randomforest_cluster_0 = RandomForestRegressor(random_state=0)
model_cluster_0 = randomforest_cluster_0.fit(X_train_cluster_0, y_train_cluster_0)
y_test_pred_cluster_0 = model_cluster_0.predict(X_test_cluster_0).astype(int)

df_cluster_0_test['y_act'] = df_cluster_0_test.sum_quantity
df_cluster_0_test['y_pred'] = y_test_pred_cluster_0

mse_cluster_0 = mean_squared_error(df_cluster_0_test.y_act, df_cluster_0_test.y_pred)
print(mse_cluster_0)

mad_cluster_0 = mean_absolute_error(df_cluster_0_test.y_act, df_cluster_0_test.y_pred)
print(mad_cluster_0)

len(df_cluster_0_test[df_cluster_0_test['y_act'] < df_cluster_0_test['y_pred']])/len(df_cluster_0_test)
len(df_cluster_0_test[df_cluster_0_test['y_act'] > df_cluster_0_test['y_pred']])/len(df_cluster_0_test)
len(df_cluster_0_test[df_cluster_0_test['y_act'] == df_cluster_0_test['y_pred']])/len(df_cluster_0_test)

### Cluster 1 Demand Model
X_train_cluster_1 = df_cluster_1_train[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday','gift_items_in_orders_lag']]
y_train_cluster_1 = df_cluster_1_train[['sum_quantity']]

X_test_cluster_1 = df_cluster_1_test[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday','gift_items_in_orders_lag']]
y_test_cluster_1 = df_cluster_1_test[['sum_quantity']]

randomforest_cluster_1 = RandomForestRegressor(random_state=0)
model_cluster_1 = randomforest_cluster_1.fit(X_train_cluster_1, y_train_cluster_1)
y_test_pred_cluster_1 = model_cluster_1.predict(X_test_cluster_1).astype(int)

df_cluster_1_test['y_act'] = df_cluster_1_test.sum_quantity
df_cluster_1_test['y_pred'] = y_test_pred_cluster_1

mse_cluster_1 = mean_squared_error(df_cluster_1_test.y_act, df_cluster_1_test.y_pred)
print(mse_cluster_1)

mad_cluster_1 = mean_absolute_error(df_cluster_1_test.y_act, df_cluster_1_test.y_pred)
print(mad_cluster_1)

len(df_cluster_1_test[df_cluster_1_test['y_act'] < df_cluster_1_test['y_pred']])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test['y_act'] > df_cluster_1_test['y_pred']])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test['y_act'] == df_cluster_1_test['y_pred']])/len(df_cluster_1_test)


### Cluster 0 Price Optimization
prices_list = []
for price in df_cluster_0_test.avg_final_price:
    price_25 = price*0.25
    price_50 = price*0.5
    price_75 = price*0.75
    price_100 = price
    price_125 = price*1.25
    price_150 = price*1.50
    price_175 = price*1.75
    price_200 = price*2
    
    prices = [price_25, price_50, price_75, price_100, price_125, price_150, price_175, price_200]
    prices_list.append(prices)
df_cluster_0_test['price_options'] = prices_list

new_prices = []
new_revenues_grand = []
df_cluster_0_test_used = df_cluster_0_test[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday', 'gift_items_in_orders_lag','price_options']]
for row in range(len(df_cluster_0_test_used)):
    price_options = df_cluster_0_test_used.iloc[row,-1]
    new_revenues = []
    for price in price_options:
        new_vars = [price]+list(df_cluster_0_test_used.iloc[row,1:-1])
        new_demand = randomforest_cluster_0.predict([new_vars])
        new_revenue = price*int(new_demand)
        new_revenues.append(new_revenue)
    new_price = price_options[new_revenues.index(max(new_revenues))]
    new_prices.append(new_price)
    new_revenues_grand.append(max(new_revenues))
            
df_cluster_0_test["avg_price_old"] = df_cluster_0_test.avg_final_price  
df_cluster_0_test["avg_quant_old"] = df_cluster_0_test.y_act
df_cluster_0_test["avg_sales_old"] = df_cluster_0_test.avg_final_price*df_cluster_0_test.sum_quantity  
df_cluster_0_test["avg_price_new"] = new_prices

avg_quant_new = []
for i,j in zip(new_revenues_grand, new_prices):
    if j!=0:
        avg_quant_new.append(i/j)
    else:
        avg_quant_new.append(np.nan)

df_cluster_0_test["avg_quant_new"] = avg_quant_new
df_cluster_0_test["avg_sales_new"] = new_revenues_grand      
     
### Cluster 0 Result Metrics
df_cluster_0_test.avg_sales_old.sum()
df_cluster_0_test.avg_sales_new.sum()
len(df_cluster_0_test[df_cluster_0_test.avg_price_old < df_cluster_0_test.avg_price_new])/len(df_cluster_0_test)
len(df_cluster_0_test[df_cluster_0_test.avg_price_old > df_cluster_0_test.avg_price_new])/len(df_cluster_0_test)
len(df_cluster_0_test[df_cluster_0_test.avg_price_old == df_cluster_0_test.avg_price_new])/len(df_cluster_0_test)

### Cluster 1 Price Optimization
prices_list = []
for price in df_cluster_1_test.avg_final_price:
    price_25 = price*0.25
    price_50 = price*0.5
    price_75 = price*0.75
    price_100 = price
    price_125 = price*1.25
    price_150 = price*1.50
    price_175 = price*1.75
    price_200 = price*2
    
    prices = [price_25, price_50, price_75, price_100, price_125, price_150, price_175, price_200]
    prices_list.append(prices)
df_cluster_1_test['price_options'] = prices_list

new_prices = []
new_revenues_grand = []
df_cluster_1_test_used = df_cluster_1_test[['avg_final_price','sum_quantity_lag','avg_total_discount_lag','avg_final_price_other_lag','attribute1','attribute2','type','order_day', 'weekday','gift_items_in_orders_lag','price_options']]
for row in range(len(df_cluster_1_test_used)):
    price_options = df_cluster_1_test_used.iloc[row,-1]
    new_revenues = []
    for price in price_options:
        new_vars = [price]+list(df_cluster_1_test_used.iloc[row,1:-1])
        new_demand = randomforest_cluster_1.predict([new_vars])
        new_revenue = price*int(new_demand)
        new_revenues.append(new_revenue)
    new_price = price_options[new_revenues.index(max(new_revenues))]
    new_prices.append(new_price)
    new_revenues_grand.append(max(new_revenues))
            
df_cluster_1_test["avg_price_old"] = df_cluster_1_test.avg_final_price  
df_cluster_1_test["avg_quant_old"] = df_cluster_1_test.y_act
df_cluster_1_test["avg_sales_old"] = df_cluster_1_test.avg_final_price*df_cluster_1_test.sum_quantity  
df_cluster_1_test["avg_price_new"] = new_prices
df_cluster_1_test["avg_quant_new"] = [i/j for i, j in zip(new_revenues_grand, new_prices)]
df_cluster_1_test["avg_sales_new"] = new_revenues_grand      

### Cluster 1 Result Metrics
df_cluster_1_test.avg_sales_old.sum()
df_cluster_1_test.avg_sales_new.sum()
len(df_cluster_1_test[df_cluster_1_test.avg_price_old < df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test.avg_price_old > df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test.avg_price_old == df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)


### Grand Total Result Metrics
df_cluster_1_test = df_cluster_0_test.append(df_cluster_1_test)

len(df_cluster_1_test[df_cluster_1_test['y_act'] < df_cluster_1_test['y_pred']])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test['y_act'] > df_cluster_1_test['y_pred']])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test['y_act'] == df_cluster_1_test['y_pred']])/len(df_cluster_1_test)

df_cluster_1_test.avg_sales_old.sum()
df_cluster_1_test.avg_sales_new.sum()
len(df_cluster_1_test[df_cluster_1_test.avg_price_old < df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test.avg_price_old > df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)
len(df_cluster_1_test[df_cluster_1_test.avg_price_old == df_cluster_1_test.avg_price_new])/len(df_cluster_1_test)


