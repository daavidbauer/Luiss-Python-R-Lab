################# IMPORT AND CLEAN DATASET ##################

games <- read.csv("C:/Users/david/OneDrive - LUISS Libera Università Internazionale degli Studi Sociali Guido Carli/David Master/Python and R for Data Science/Python Documentation/games.csv")
View(games)

#get a summary of all columns in the data frame
summary(games) 

#get the total number of rows containing nulls
sum(!complete.cases(games))

#since the number of rows is very low (less than 1% of the entire row amount) 
#we simply drop those rows
games <- na.omit(games)

#check for amount duplicate rows (there are none)
sum(duplicated(games))

#Drop the duplicate titles
games <- games[!duplicated(games[["Title"]]), ]

#Drop extra index column and summary and reviews since performing text 
#analysis is outside the scope of this project
games_clean <- games[,-(which(names(games) %in% c('Reviews','Summary', 'X')))]
head(games_clean)

#a function to turn e.g, "8.8k" into an integer of 8800
k_to_int <- function(lst) {
  values <- c()
  
for (value in lst){  
  
    if (grepl('[Kk]', value)) {
      value <- gsub('[Kk]', '', value)
      
      if (grepl('\\.', value)) {
        values <- c(values, (as.integer(gsub('\\.', '', value)) * 100))
      } else {
        values <- c(values, as.integer(value) * 1000)
        }
      
    } else {
      values <- c(values, as.numeric(value))
    }
  } 
  return (values)
}

#apply the function to all related columns 
games_clean[, c("Times.Listed", "Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist")] <- lapply(games_clean[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist")], k_to_int)

# remove all games where the release date is to be determined
games_clean <- games_clean[-(which(games_clean["Release.Date"] == 'releases on TBD')),]

#adding columns for the season the game was published in 
get_season <- function(column) {
  seasons <- c()
  
  for (date in column) {
    
    month <- as.Date(date, format = "%b %d, %Y")
    month <- as.integer(format(month, "%m"))
    
    if (!is.na(month) && 2 < month && month < 6) {
      seasons <- c(seasons, 'Spring')
    } else if (!is.na(month) && 5 < month && month < 9) {
      seasons <- c(seasons, 'Summer')
    } else if (!is.na(month) && 8 < month && month < 12) {
      seasons <- c(seasons,'Fall')
    } else {
      seasons <- c(seasons,'Winter')
    }
  }
  return(seasons)
}

#add the season to the data frame as a column 
games_clean$Season <- get_season(games_clean$Release.Date)

#only leave the month and the year from the Release.date variable as 
#new columns in the data frame
months <- c()
years <- c()

for (date in games_clean$Release.Date){
  
  if (!is.na(date)){
    
    months <- c(months, substr(date, 1, 3))
    
    year <- substr(date, nchar(date) - 4, nchar(date))
    year_as_date <- as.Date(paste0(year, "-01-01"))
    year <- as.integer(format(year_as_date, "%Y"))
    
    years <- c(years, year)
  } 
} 

games_clean$Release.Month <- months
games_clean$Release.Year <- years

#drop the original release date column 
games_clean <- games_clean[,-which(names(games_clean) == "Release.Date")]

#Get only the first genre of each title so it can be used as a dummy later
library(dplyr)
genre_lst = c()

for (genre in games_clean$Genres){
  genre <- gsub("\\[", '',genre) %>%
  gsub("\\]", '', .) %>%
  gsub(",","", .) %>%
  gsub("'","", .) %>%
  strsplit(" ")
genre_lst <- c(genre_lst, genre[[1]][1])  
}

games_clean["Genres"] <- genre_lst

games_clean <- na.omit(games_clean)

#split the teams string into individual items in a list for each game and get main (first)
team_lst = c()

for (team in games_clean$Team){
  team <- gsub("\\[", '',team) %>%
    gsub("\\]", '', .) %>%
    gsub("'","", .) %>%
    strsplit(",")
  team_lst <- c(team_lst, team[[1]][1])  
}
games_clean["Team"] <- team_lst

View(games_clean)

###################### EXPLORATORY STATISTICS ######################

#plot the pearson correlation of the numeric variables in the dataset
install.packages("corrplot")
library(corrplot)

numeric_vars <- games_clean[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist", "Rating","Release.Year")]

cor_matrix <- cor(numeric_vars)
cor_matrix
corrplot(cor_matrix, method = "color", col = 
           colorRampPalette(c("turquoise", "white", "indianred2"))(50), 
         tl.col = "black", 
         tl.srt = 35, 
         addCoef.col = "black", 
         number.cex = 0.7)

#constructing pair plots for seeing the relationship of the variables 
pairs(numeric_vars)

## Plot Games per month
library(dplyr)
library(ggplot2)

games_clean %>%
  group_by(Release.Month) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Release.Month, y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games per Month",
       x = "Months",
       y = "Number of Games")

## Games per season
games_clean %>%
  group_by(Season) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Season, y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games per Season",
       x = "Season",
       y = "Number of Games")+
  theme_minimal() 

## Games per year
games_clean %>%
  group_by(Release.Year) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Release.Year, y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games per Year",
       x = "Years",
       y = "Number of Games")+
  theme_minimal() 

## Genres Total
games_clean %>%
  group_by(Genres) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = reorder(Genres, Number_of_Games), y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games Genre",
       x = "Genres",
       y = "Number of Games") +
  theme_minimal() +
  coord_flip()

# Mean Rating
games_clean %>%
  group_by(Release.Year) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = Release.Year, y = Mean_Rating)) +
  geom_line(color = "darkviolet", size = 0.75) +
  labs(title = "Mean Rating per Release Year",
       x = "Mean Rating",
       y = "Release Year")+
  theme_minimal() 

# Genre Rating
games_clean %>%
  group_by(Genres) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = reorder(Genres, Mean_Rating), y = Mean_Rating, fill =Mean_Rating)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Mean Genre Rating",
       x = "Genres",
       y = "Mean Rating") +
  theme_minimal() +
  coord_flip()

#Team Total
games_clean %>%
  group_by(Team) %>%
  summarise(Number_of_Games = n()) %>%
  top_n(10, Number_of_Games) %>%
  ggplot(aes(x = reorder(Team, Number_of_Games), y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Biggest Video Game Producers",
       x = "Teams",
       y = "Number of Video Games")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

#get all rows with video games where Nintendo was a producing team 
nintendo_games<- games_clean[(which(grepl("Nintendo", games_clean$Team))),]

#Nintendo game releases per month
nintendo_games %>%
  group_by(Release.Month) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Release.Month, y = Number_of_Games, fill = Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "salmon")+
  labs(title = 'Games Released per Month by Nintendo',
       x = "Release Month",
       y = "Number of Games")+
  theme_minimal() 
  
#Nintendo game releases per season
nintendo_games %>%
  group_by(Season) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Season, y = Number_of_Games, fill = Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "salmon")+
  labs(title = 'Games Released per Season by Nintendo',
       x = "Season",
       y = "Number of Games")+
  theme_minimal() 

#Nintendo game releases per season
nintendo_games %>%
  group_by(Genres) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Genres, y = Number_of_Games, fill = Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "salmon")+
  labs(title = 'Games Released per Season by Nintendo',
       x = "Season",
       y = "Number of Games")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

#Nintendo Genres Totals
nintendo_games %>%
  group_by(Genres) %>%
  summarise(Number_of_Games = n()) %>%
  mutate(Percentage = (Number_of_Games / sum(Number_of_Games)) * 100) %>%
  ggplot(aes(x = "", y = Percentage, fill = Genres)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Nintendo Genres Totals", fill = "Genres")

# Nintendo Per Year
nintendo_games %>%
  group_by(Release.Year) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Release.Year, y = Number_of_Games, fill = Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "tomato")+
  labs(title = 'Games Released per Year by Nintendo',
       x = "Release Year",
       y = "Number of Games")+
  theme_minimal()

# Nintendo Mean Rating per Year
nintendo_games %>%
  group_by(Release.Year) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = Release.Year, y = Mean_Rating)) +
  geom_line(color = "tomato", size = 0.75) +
  labs(title = "Mean Rating per Release Year",
       x = "Mean Rating",
       y = "Release Year")+
  theme_minimal() 

# Nintendo Mean Rating per Genre
nintendo_games %>%
  group_by(Genres) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = reorder(Genres, Mean_Rating), y = Mean_Rating, fill =Mean_Rating)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "tomato") +
  labs(title = "Mean Genre Rating",
       x = "Genres",
       y = "Mean Rating") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

###################### MODEL BUILDING AND TESTING #######################

##even thought the correlation in our data set is sub-optimal 
#we tried building a linear regression model to experience the OSEMN Pipeline 
#from beginning to the end

#create dummy variables for Genre, Release Month and Season
games_fin <- cbind(games_clean[, -which(
  names(games_clean) %in% c("Genres"))], 
                     model.matrix(~ Genres - 1, data = games_clean))

games_fin <- cbind(games_fin[, -which(
  names(games_fin) %in% c("Release.Month"))], 
  model.matrix(~ Release.Month - 1, data = games_fin))

games_fin <- cbind(games_fin[, -which(
  names(games_fin) %in% c("Season"))], 
  model.matrix(~ Season - 1, data = games_fin))

#drop Team and Title since they have too many unique values to be turned into a dummy 
games_fin<- games_fin[,-(which(names(games_fin) %in% c("Team", "Title", "Times.Listed")))]

# Perform min-max scaling
min_max <- function(x) {(x - min(x)) / (max(x) - min(x))}
games_fin[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist",
              "Release.Year")] <- lapply(games_fin[, c( "Number.of.Reviews", "Plays","Playing", 
                          "Backlogs", "Wishlist", "Release.Year")], min_max)

# split the final games data set into 70% training and 30% test data 
train <- games_fin[1:(1082*0.7), ]
test <- games_fin[(1082*0.7):1082, ]

#build the linear regression model 
lin_fit <- lm(Rating ~ ., data = train)

#get summary statistics
summary(lin_fit)

#use the model to predict using the test set
y_pred <- predict(lin_fit, newdata = test)

#get a root mean squared error evaluation of the model
print(RMSE <- sqrt(mean((test$Rating - y_pred)^2)))
