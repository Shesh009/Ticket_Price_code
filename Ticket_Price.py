import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st


#Price Function
def price(a,b,c,d):
    x=0
    if d=='Balcony':
        if (a>=7 and b>=7 and c>=7):
            x=500
        elif (a>=7 and b>=7 and (c>=4 and c<7)):
            x=450
        elif (a>=7 and b>=7 and (c>=0 and c<4)):
            x=400
        elif ((a>=4 and a<7) and (b>=4 and b<7) and c>=7):
            x=450
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=4 and c<7)):
            x=400
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=0 and c<4)):
            x=350
        elif ((a>=0 and a<4) and (b>=0 and b<4) and c>=7):
            x=400
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=4 and c<7)):
            x=350
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=0 and c<4)):
            x=300
        else:
            x=50
    elif d=='First Class':
        if (a>=7 and b>=7 and c>=7):
            x=400
        elif (a>=7 and b>=7 and (c>=4 and c<7)):
            x=350
        elif (a>=7 and b>=7 and (c>=0 and c<4)):
            x=300
        elif ((a>=4 and a<7) and (b>=4 and b<7) and c>=7):
            x=350
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=4 and c<7)):
            x=300
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=0 and c<4)):
            x=250
        elif ((a>=0 and a<4) and (b>=0 and b<4) and c>=7):
            x=300
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=4 and c<7)):
            x=250
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=0 and c<4)):
            x=200
        else:
            x=50
    elif d=="Second Class":
        if (a>=7 and b>=7 and c>=7):
            x=300
        elif (a>=7 and b>=7 and (c>=4 and c<7)):
            x=250
        elif (a>=7 and b>=7 and (c>=0 and c<4)):
            x=200
        elif ((a>=4 and a<7) and (b>=4 and b<7) and c>=7):
            x=250
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=4 and c<7)):
            x=200
        elif ((a>=4 and a<7) and (b>=4 and b<7) and (c>=0 and c<4)):
            x=150
        elif ((a>=0 and a<4) and (b>=0 and b<4) and c>=7):
            x=200
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=4 and c<7)):
            x=150
        elif ((a>=0 and a<4) and (b>=0 and b<4) and (c>=0 and c<4)):
            x=100
        else:
            x=50

    return x
#Predict Function
def predict_price(seat_t,pop,cast,talk):
    pred_array=[]
    if seat_t=="Balcony":
        pred_array=pred_array+[1,0,0]
    elif seat_t=="First Class":
        pred_array=pred_array+[0,1,0]
    elif seat_t=="Second Class":
        pred_array=pred_array+[0,0,1]

    pred_array=pred_array+[pop,cast,talk]
    pred_array=np.array([pred_array])
    pred=reg1.predict(pred_array)
    return int(pred)

popularity = []
for i in range(0,100):
    popularity.append(random.randint(1,10))

data=pd.DataFrame(popularity,columns=['Popularity'])

cast_crew_rating=[]
for i in range(0,100):
    cast_crew_rating.append(popularity[i])
data["Cast_crew_rating"]=cast_crew_rating

theater_rating=[]
for i in range(0,100):
    theater_rating.append(random.randint(1,10))
data['Theater_rating']=theater_rating

seats=['Balcony',"First Class","Second Class"]
seat_type=[]
for i in range(0,100):
    seat_type.append(random.choice(seats))
data['Seat_type']=seat_type

time_sh=['Morning','Afternoon','Evening','Night']
show_time=[]
for i in range(0,100):
    show_time.append(random.choice(time_sh))
data["show_time"]=show_time

days_bef=[]
for i in range(0,100):
    days_bef.append(random.randint(0,5))
data['days_before']=days_bef

occup=['0%','25%','50%','75%']
occupancy=[]
for i in range(0,100):
    occupancy.append(random.choice(occup))
data['occupancy']=occupancy

price1=[]
for i in range(0,100):
    price1.append(price(popularity[i],cast_crew_rating[i],theater_rating[i],seat_type[i]))
data["Price"]=price1

data1=data[["Popularity","Cast_crew_rating","Theater_rating","Seat_type","Price"]]

seats1={"Balcony":1,"First Class":2,"Second Class":3}
data2=data1.replace({"Seat_type":seats1})

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
data3=ct.fit_transform(data2)

cols=['Balcony','First Class','Second Class','Popularity','Cast_crew_rating','Theater_rating','Price']
data4=pd.DataFrame(data3,columns=cols)

x=data4.iloc[:,:-1].values
y=data4.iloc[:,-1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

reg1=LinearRegression()
reg1.fit(x_train,y_train)

y_pred1=reg1.predict(x_test)

st.header("TICKET PRICE PREDICTION")
st.text_input("MOVIE NAME")
st.text_input("THEATER NAME")
col0,col1,col2=st.columns(3)
pop1=col0.number_input("POPULARITY",min_value=0.0,max_value=10.0,step=0.5)
cast1=col1.number_input("CAST AND CREW RATING",min_value=0.0,max_value=10.0,step=0.5)
talk1=col2.number_input("THEATER RATING",min_value=0.0,max_value=10.0,step=0.5)
col3,col4,col5=st.columns(3)
seat1=col3.selectbox("SEAT TYPE",seats1,index=0)
col4.selectbox("SHOW TIME",time_sh,index=2)
num=col5.number_input("NUMBER OF TICKETS",min_value=1,step=1)
col6,col7,col8,col9,col10=st.columns(5)
col11,col12,col13=st.columns(3)
if col8.button("TOTAL PRICE"):
    y=predict_price(seat_t=seat1,pop=pop1,cast=cast1,talk=talk1)
    col12.write(f"Each ticket : {y}")
    col12.success("Total price for {0} tickets is {1}.RS".format(num,y*num))