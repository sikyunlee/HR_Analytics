#########################################################
# COVID-19 Data Analytics - UCLA Anderson 2020 Datathon #
#########################################################

library(dplyr)
library(magrittr)
library(tibble)
library(tidyverse)
library(ggplot2)
library(highcharter)
library(lubridate)
library(prophet)
library(RCurl)
library(ModelMetrics)
library(caTools)
library(rpart)
library(rpart.plot)
library(ROCR)

#' 
#' 
#' 
#' 
#' 


data <- read.csv('us-counties.csv')
data <- as_tibble(data)


#' Aggregate Calculation
#' @description Calculates descriptive statistics of COVID19 data from NY Times
#'   Git data
#' @param data Aggregate data provided by NY Times Git
#' @return Summary of CovID19 First Case, Last Case, Total Cases, Total Deaths
#'
#' @export
agg_calculations <- function(data) {
  `%>%` <- magrittr:: `%>%`
  dplyr::summarise(
    First_Case = min(date, na.rm = T), 
    Last_Case = max(date, na.rm = T), 
    Total_Cases = sum(cases, na.rm = T), 
    Total_Deaths = sum(deaths, na.rm = T)) %>%
    glimpse
}

#Overall Descriptive Statistics
data_calculations <- data %>%
  summarise(date, cases, deaths) %>%
  mutate(date = as.Date(date)) %>%
  #group_by(date) %>%
  mutate(lag_cases = lag(cases))%>%
  mutate(daily_cases = cases - lag_cases) %>%
  mutate(lag_deaths = lag(deaths)) %>%
  mutate(daily_deaths = deaths - lag_deaths) %>%
  select(-lag_cases, -lag_deaths) %>%
  dplyr::summarise(
    First_Case = min(date, na.rm = T),
    Last_Case = max(date, na.rm = T),
    Total_Cases = sum(cases, na.rm = T),
    Total_Deaths = sum(deaths, na.rm = T)) %>%
  glimpse

#Daily cases grouped by state for each date
daily_cases <- data %>%
  group_by(date) %>%
  mutate(daily_case = dplyr::n()) %>%
  ungroup() %>%
  group_by(state) %>%
  glimpse

#Daily Cases in Washington
washington_daily_cases <- data %>%
  select(-county, -fips) %>%
  filter(state == "Washington") %>%
  group_by(date) %>%
  #group_by(state) %>%
  summarise(cum_case = sum(cases, na.rm = T),
            cum_death = sum(deaths, na.rm = T)) %>%
  mutate(cum_case_lag = lag(cum_case)) %>%
  mutate(daily_case = cum_case - cum_case_lag) %>%
  mutate(cum_death_lag = lag(cum_death)) %>%
  mutate(daily_death = cum_death - cum_death_lag) %>%
  select(-cum_case_lag, -cum_death_lag) %>%
  mutate(state = "washington") %>%
  glimpse

#Daily cases in California
california_daily_cases <- data %>%
  select(-county, -fips) %>%
  filter(state == "California") %>%
  group_by(date) %>%
  #group_by(state) %>%
  summarise(cum_case = sum(cases, na.rm = T),
            cum_death = sum(deaths, na.rm = T)) %>%
  mutate(cum_case_lag = lag(cum_case)) %>%
  mutate(daily_case = cum_case - cum_case_lag) %>%
  mutate(cum_death_lag = lag(cum_death)) %>%
  mutate(daily_death = cum_death - cum_death_lag) %>%
  select(-cum_case_lag, -cum_death_lag) %>%
  mutate(state = "california") %>%
  glimpse

#Daily cases in Texas
texas_daily_cases <- data %>%
  select(-county, -fips) %>%
  filter(state == "Texas") %>%
  group_by(date) %>%
  #group_by(state) %>%
  summarise(cum_case = sum(cases, na.rm = T),
            cum_death = sum(deaths, na.rm = T)) %>%
  mutate(cum_case_lag = lag(cum_case)) %>%
  mutate(daily_case = cum_case - cum_case_lag) %>%
  mutate(cum_death_lag = lag(cum_death)) %>%
  mutate(daily_death = cum_death - cum_death_lag) %>%
  select(-cum_case_lag, -cum_death_lag) %>%
  mutate(state = "Texas") %>%
  glimpse

#Daily cases in New York
new_york_daily_cases <- data %>%
  select(-county, -fips) %>%
  filter(state == "New York") %>%
  group_by(date) %>%
  #group_by(state) %>%
  summarise(cum_case = sum(cases, na.rm = T),
            cum_death = sum(deaths, na.rm = T)) %>%
  mutate(cum_case_lag = lag(cum_case)) %>%
  mutate(daily_case = cum_case - cum_case_lag) %>%
  mutate(cum_death_lag = lag(cum_death)) %>%
  mutate(daily_death = cum_death - cum_death_lag) %>%
  select(-cum_case_lag, -cum_death_lag) %>%
  mutate(state = "New York") %>%
  glimpse

#Daily cases in Illinois
illinois_daily_cases <- data %>%
  select(-county, -fips) %>%
  filter(state == "Illinois") %>%
  group_by(date) %>%
  #group_by(state) %>%
  summarise(cum_case = sum(cases, na.rm = T),
            cum_death = sum(deaths, na.rm = T)) %>%
  mutate(cum_case_lag = lag(cum_case)) %>%
  mutate(daily_case = cum_case - cum_case_lag) %>%
  mutate(cum_death_lag = lag(cum_death)) %>%
  mutate(daily_death = cum_death - cum_death_lag) %>%
  select(-cum_case_lag, -cum_death_lag) %>%
  mutate(state = "Illinois") %>%
  glimpse

#Plot Daily increase per date for each state
five_states <- bind_rows(california_daily_cases, washington_daily_cases) 
five_states <- bind_rows(five_states, texas_daily_cases) 
five_states <- bind_rows(five_states, new_york_daily_cases) 
five_states <- bind_rows(five_states, illinois_daily_cases) 
unique(five_states$state) #check to see if bind was done ok

#Output: plot daily cases for all five states
ggplot(data = five_states,
       aes(x = date, y = daily_case, color = state)) + 
  geom_line() +
  theme_minimal() + 
  geom_point()

#Output: plot daily deaths for all five states
ggplot(data = five_states,
       aes(x = date, y = daily_death, color = state)) + 
  geom_line() + 
  theme_minimal() +
  geom_point()


### Step 3: Predicting COVID19 cases with RNN LSTM
library(keras)
library(caret)

#Let's use Washington state daily cases data to predict WA COVID19 cases
#We will use past 14 days data to predict with batch_size 16 and total epochs 1

wash_cases <- washington_daily_cases$daily_case
#set some parameters for the RNN LSTM model
max_len <- 14 # number of days that will be considered
batch_size <- 16 # number of sequence days that will be considered for one-time for training
total_epochs <- 4 # number of times the entire dataset will be explored while training the model

set.seed(100) # setting seed for reproducability

#list of start indexes from the COVID19 data for overlapping chunks
start_indexes <- seq(1, length(washington_daily_cases$date) - (max_len + 1), by = 3)

#create an empty matrix to store the data in
covid_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)

#fill out matrix with the overlapping slices of the dataset
#this will create an extra column with blanks to fill-in after predicting that day
for (i in 1:length(start_indexes)){
  covid_matrix[i,] <- wash_cases[start_indexes[i]:(start_indexes[i] + max_len)]
}

#check if all columns are numeric and do not have NAs
covid_matrix <- covid_matrix * 1 #OK

if(anyNA(covid_matrix)){
  covid_matrix <- na.omit(covid_matrix)
}

#Split the dataset into 14 past days and 1 day we want to predict
X <- covid_matrix[, -ncol(covid_matrix)]
y <- covid_matrix[, ncol(covid_matrix)]

#Split and X and y data into training and testing set with caret library
training_index <- createDataPartition(y, p = 0.9,
                                      list = FALSE,
                                      times = 1)

X_train <- array(X[training_index,], dim = c(length(training_index), max_len, 1))
y_train <- y[training_index]

X_test <- array(X[-training_index,], dim = c(length(y) - length(training_index),
                                             max_len, 1))
y_test <- y[-training_index]


#Initialize the model from Keras
#Start with a sequential model by adding layers sequentially
model <- keras_model_sequential()

#Add Input layer, hidden layers, and output layer
#For simplicity, this model won't go through complex searches for optimal layers

#Input layer
dim(X_train)
#first dimension is the number of training data
#second dim is the length of input sequence (=max_len)
#third dim is the number of features we have in data (in this case, 1)

model %>%
  layer_dense(input_shape = dim(X_train)[2:3], units = max_len)

#Create hidden layers
model %>%
  layer_simple_rnn(units = 8)

#Specify sigmoid activation function for Output layer
model %>%
  layer_dense(units = 1, activation = 'sigmoid') 

#Check out the structure of the model
summary(model)

#Add Loss function, Optimizer option, and set metrics
model %>% compile(loss = 'binary_crossentropy',
                  optimizer = 'RMSprop',
                  metrics = c('accuracy'))


#Training the model
trained_model <- model %>% fit(
  x = X_train, #sequence used for predicting
  y = y_train, #sequence that we're predicting
  batch_size = batch_size, # how many samples to pass to our odel at a time
  epochs = total_epochs, #how many times we'll look @ the whole dataset
  validation_split = 0.1 #how much data to hold out for testing in the future
)

#Check the model
trained_model #accuracy is terrible! only 13% 

#Plot trained model
plot(trained_model)

#Try predicting future cases --> Predicts Classes
#cases <- model %>% predict(X_test, batch_size = batch_size)
#table(y_test, cases)

#For another solution, let's try Prophet package created by FB data Science team

##########
#Trying Prophet Functions
##########
us_data <- data %>%
  mutate(date = as.Date(date)) %>%
  select(date, cases, deaths) %>%
  group_by(date) %>%
  summarise(cum_cases = sum(cases, na.rm = T),
            cum_deaths = sum(deaths, na.rm = T)) %>%
  mutate(lag_cases = lag(cum_cases)) %>%
  mutate(daily_cases = cum_cases - lag_cases) %>%
  mutate(lag_deaths = lag(cum_deaths)) %>%
  mutate(daily_deaths = cum_deaths - lag_deaths) %>%
  select(-lag_cases, -lag_deaths) %>%
  arrange(date) %>%
  glimpse

#Predicting daily cases into the future
#install.packages('prophet')
#library('prophet')
df <- us_data %>%
  mutate(ds = date) %>%
  mutate(y = daily_cases) %>%
  select(ds, y) %>%
  glimpse

#create model for training
m <- prophet(df)

#set number of days to predict; in this case 31 days
future <- make_future_dataframe(m, periods = 31)
tail(future) #to check up to when this will predict; in this case, 8/21

#predict using the predict function
forecast <- predict(m, future)
#show the tail of the predicted number --> yhat
#along with predicted intervals in yhat_lower, yhat_upper
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

plot(m, forecast)

#Output: plot predicted daliy cases
#ggplot(data = forecast,
#       aes(x = ds, y = yhat)) + 
#  geom_line() +
#  theme_minimal() + 
#  geom_point()

#Break down the forecast by trend, weekly seasonality
prophet_plot_components(m, forecast)

#Interactive plot of the forecast
dyplot.prophet(m, forecast)

predicted <- forecast %>%
  select(ds, yhat) %>%
  glimpse

#Let's test our predicted values to the actual ones in NYtimes data
#Get the URL and load into R
url <- 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
new_data <- read.csv(url)

#Munge the data in to the data format above
new_df <- new_data %>%
  mutate(data = as.Date(date)) %>%
  select(date, cases, deaths) %>%
  select(date, cases, deaths) %>%
  group_by(date) %>%
  summarise(cum_cases = sum(cases, na.rm=T),
            cum_deaths = sum(deaths, na.rm = T)) %>%
  mutate(lag_cases = lag(cum_cases)) %>%
  mutate(daily_cases = cum_cases- lag_cases) %>%
  mutate(lag_deaths = lag(cum_deaths)) %>%
  mutate(daily_deaths = cum_deaths - lag_deaths) %>%
  select(-lag_cases, -lag_deaths) %>%
  arrange(date) %>%
  glimpse

new_df <- new_df %>%
  mutate(ds = date) %>%
  mutate(y = daily_cases) %>%
  select(ds, y) %>%
  glimpse

#Compare the predicted daily cases, deaths vs actual
#First, round the predicted yhat (cases) into whole numbers
predicted <- predicted %>%
  mutate(yhat = round(yhat, 0)) %>%
  glimpse

#Calculate MSE, RMSE. MAE
mse <- mean((new_df$y - predicted$yhat)^2, na.rm = T)
mse

rmse <- sqrt(mean((new_df$y - predicted$yhat)^2, na.rm = T))
rmse

mae <- mean(abs(new_df$y - predicted$yhat), na.rm = T)
mae

#############

#############
#CART on COVID19 data
#############
set.seed(100)

#Split based on time 60:40
train = us_data[2: round(60*nrow(us_data)/100, 0), ]
test = us_data[round(60*nrow(us_data)/100 + 1): nrow(us_data), ]

#Train model
covid.rpart <- rpart(daily_cases ~., data = train)
covid.rpart
summary(covid.rpart)

#Check model tree
prp(covid.rpart, extra = 1)

#Predict
covid.predict = predict(covid.rpart, newdata = test)

#Calculate MSE, RMSE. MAE
cmse <- mean((test$daily_cases - covid.predict)^2, na.rm = T)
cmse

crmse <- sqrt(mean((test$daily_cases - covid.predict)^2, na.rm = T))
crmse

cmae <- mean(abs(test$daily_cases - covid.predict), na.rm = T)
cmae

###############################

############
#Comparison between two models
############

comparison <- data.frame('Metrics' = c('MSE', 'RMSE', 'MAE'), 
                         'Prophet' = c(mse, rmse, mae), 
                         'CART' = c(cmse, crmse, cmae),
                         'Diff' = c(mse-cmse, rmse-crmse, mae-cmae))

# Conclusion: Prophet performs better in all metrics
################################


##################################
#Additional: EDA of Data
##################################
#Get top 5 states with highest cum cases of COVID19
state_cases <- data %>%
  select(-county, -fips) %>%
  group_by(state) %>%
  summarise(total_case = sum(cases, na.rm = T),
            total_death = sum(deaths, na.rm = T)) %>%
  arrange(desc(total_case), desc(total_death)) %>%
  glimpse

#Get top 5 states with highest cum deaths of COVID19
state_death <- data %>%
  select(-county, -fips) %>%
  group_by(state) %>%
  summarise(total_case = sum(cases, na.rm = T),
            total_death = sum(deaths, na.rm = T)) %>%
  arrange(desc(total_death), desc(total_case)) %>%
  glimpse

#Plot Top 5 States with highest cum cases of COVID19
ggplot(data = head(state_cases, 5), 
       aes(x = state, y = total_case, fill = state)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = total_case), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  geom_point()

#Plot Top 5 States with highest cum deaths of COVID19
ggplot(data = head(state_death, 5), 
       aes(x = state, y = total_death, fill = state)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = total_death), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  geom_point()

#Plot cumulative increase of COVID19 for U.S. Total
plot_daily_cases <- daily_cases
ggplot(data = plot_daily_cases, 
        aes(x = date, y = daily_case)) + 
  geom_line() +
  geom_point()

