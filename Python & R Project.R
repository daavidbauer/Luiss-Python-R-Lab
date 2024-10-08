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

#Turn the genres in each row into list containing them as individual items
library(dplyr)
genre_lst = c()

for (genre in games_clean$Genres){
  genre <- gsub("\\[", '',genre) %>%
  gsub("\\]", '', .) %>%
  gsub(",","", .) %>%
  gsub("'","", .) %>%
  strsplit(" ")
genre_lst <- c(genre_lst, genre)  
}

games_clean$Genres <- genre_lst

games_clean <- na.omit(games_clean)

#split the teams string into individual items in a list for each game 
team_lst = c()

for (team in games_clean$Team){
  team <- gsub("\\[", '',team) %>%
    gsub("\\]", '', .) %>%
    gsub("'","", .) %>%
    strsplit(",")
  team_lst <- c(team_lst, team)  
}

games_clean$Team <- team_lst

#display the cleaned data set 
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

#construct pair plots for seeing the relationship of the variables 
pairs(numeric_vars)

#Kernel Density plots of the numeric variables for visualizing their frequency 
library(ggplot2)
library(tidyr)

numeric_vars <- numeric_vars %>%
  gather(key = "variable", value = "value")

ggplot(data = numeric_vars, aes(x = value)) +
  geom_density() +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Kernel Density Plot", x = "Value", y = "Density") +
  theme_minimal()

#plot ratings per season as violin plots
games_clean %>%
  ggplot(aes(x = Season, y = Rating, fill = Season)) +
  geom_violin() +
  geom_boxplot(width = 0.2, fill = "darkviolet", color = "orange") +
  labs(title = "Ratings per Season", x = "Season", y = "Rating") +
  theme_minimal()

#plot ratings per month as violin plots
games_clean %>%
  ggplot(aes(x = Release.Month, y = Rating)) +
  geom_violin() +
  geom_boxplot(width = 0.2, fill = "darkviolet", color = "orange") +
  labs(title = "Ratings per Month", x = "Month", y = "Rating") +
  theme_minimal()

#plot Games per month
games_clean %>%
  group_by(Release.Month) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Release.Month, y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games per Month",
       x = "Months",
       y = "Number of Games")

#plot Games per season
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

#plot games per year
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

#plot total number of games per genre
games_clean %>%
  group_by(Genres = sapply(Genres, function(x) x[1])) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = reorder(Genres, Number_of_Games), y = Number_of_Games, fill =Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Games Genre",
       x = "Genres",
       y = "Number of Games") +
  theme_minimal() +
  coord_flip()

#plot mean rating over the years
games_clean %>%
  group_by(Release.Year) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = Release.Year, y = Mean_Rating)) +
  geom_line(color = "darkviolet", size = 0.75) +
  labs(title = "Mean Rating per Release Year",
       x = "Mean Rating",
       y = "Release Year")+
  theme_minimal() 

#plot mean genre ratings
games_clean %>%
  group_by(Genres = sapply(Genres, function(x) x[1])) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = reorder(Genres, Mean_Rating), y = Mean_Rating, fill =Mean_Rating)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "darkviolet", high = "orange") +
  labs(title = "Mean Genre Rating",
       x = "Genres",
       y = "Mean Rating") +
  theme_minimal() +
  coord_flip()

#T plot the numbers of games per team
games_clean %>%
  group_by(Team = sapply(Team, function(x) x[1])) %>%
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

#get violin plots comparing nintendo and not nintendo games 
games_clean_nintendo <- games_clean %>%
  mutate(Nintendo = ifelse(grepl("Nintendo", games_clean$Team), "Nintendo", "Other"))

ggplot(games_clean_nintendo, aes(x = Nintendo, y = Rating, Fill = Nintendo)) +
  geom_violin() +
  geom_boxplot(width = 0.2, fill = "darkviolet", color = "orange") +
  labs(x = "Teams", y = "Rating") +
  theme_minimal()

#get all rows with video games where Nintendo was a producing team 
nintendo_games<- games_clean[(which(grepl("Nintendo", games_clean$Team))),]

#plot nintendo game releases per month
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
  
#plot nintendo game releases per season
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

#plot nintendo game releases per genre
nintendo_games %>%
  group_by(Genres = sapply(Genres, function(x) x[1])) %>%
  summarise(Number_of_Games = n()) %>%
  ggplot(aes(x = Genres, y = Number_of_Games, fill = Number_of_Games)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "salmon")+
  labs(title = 'Games Released per Season by Nintendo',
       x = "Season",
       y = "Number of Games")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

#plot nintendo genre total game numbers
nintendo_games %>%
  group_by(Genres = sapply(Genres, function(x) x[1])) %>%
  summarise(Number_of_Games = n()) %>%
  mutate(Percentage = (Number_of_Games / sum(Number_of_Games)) * 100) %>%
  ggplot(aes(x = "", y = Percentage, fill = Genres)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Nintendo Genres Totals", fill = "Genres")

#plot nintendo games per year
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

#plot nintendo games mean rating per year
nintendo_games %>%
  group_by(Release.Year) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = Release.Year, y = Mean_Rating)) +
  geom_line(color = "tomato", size = 0.75) +
  labs(title = "Mean Rating per Release Year",
       x = "Mean Rating",
       y = "Release Year")+
  theme_minimal() 

#plot nintendo mean rating per genre
nintendo_games %>%
  group_by(Genres = sapply(Genres, function(x) x[1])) %>%
  summarise(Mean_Rating = mean(Rating)) %>%
  ggplot(aes(x = reorder(Genres, Mean_Rating), y = Mean_Rating, fill =Mean_Rating)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "steelblue1", high = "tomato") +
  labs(title = "Mean Genre Rating",
       x = "Genres",
       y = "Mean Rating") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

#text length analysis 

games_words <- games
mean_rating <- mean(games_words$Rating)

##review length distribution 
#data frame games above & below mean

above_mean <- games_words[which(games_words$Rating > mean_rating),]
below_mean <- games_words[which(games_words$Rating < mean_rating),]

above_mean_review_lengths = sapply(strsplit(above_mean$Reviews, split = " "), length)
below_mean_review_lengths = sapply(strsplit(below_mean$Reviews, split = " "), length)

review_lengths <- data.frame(lenghts = c(above_mean_review_lengths, below_mean_review_lengths),
                              above_below_mean = rep(c("Above", "Below")))

#create a two-layered histogram
ggplot(review_lengths, aes(x = lenghts, fill = above_below_mean)) +
  geom_histogram(position = "identity", alpha = 0.7, bins = 30) +
  labs(title = "Review Length Distribution Above vs. Below Mean Rating",
       x = "Length",
       y = "Frequency") +
  scale_fill_manual(values = c("Above" = "burlywood1", "Below" = "coral3")) +
  theme_minimal()

#summary Length Distribution 
#data frame games above & below mean

above_mean_summary_lengths = sapply(strsplit(above_mean$Summary, split = " "), length)
below_mean_summary_lengths = sapply(strsplit(below_mean$Summary, split = " "), length)

summary_lengths <- data.frame(lenghts = c(above_mean_summary_lengths, below_mean_summary_lengths),
                              above_below_mean = rep(c("Above", "Below")))

#create a two-layered histogram
ggplot(summary_lengths, aes(x = lenghts, fill = above_below_mean)) +
  geom_histogram(position = "identity", alpha = 0.7, bins = 30) +
  labs(title = "Summary Length Distribution Above vs. Below Mean Rating",
       x = "Length",
       y = "Frequency") +
  scale_fill_manual(values = c("Above" = "burlywood1", "Below" = "coral3")) +
  theme_minimal()

###################### MODEL BUILDING AND TESTING #######################

##even thought the correlation in our data set is sub-optimal 
#we tried building a linear regression model to experience the OSEMN Pipeline 
#from the beginning to the end

#create dummy variables for Genre, Release Month and Season
games_fin <- cbind(games_clean[, -which(
  names(games_clean) %in% c("Release.Month"))], 
  model.matrix(~ Release.Month - 1, data = games_clean))

games_fin <- cbind(games_fin[, -which(
  names(games_fin) %in% c("Season"))], 
  model.matrix(~ Season - 1, data = games_fin))

games_fin$Genres <- sapply(games_fin$Genres, function(x) x[1])
games_fin <- na.omit(games_fin)

games_fin <- cbind(games_fin[, -which(
  names(games_fin) %in% c("Genres"))], 
  model.matrix(~ Genres - 1, data = games_fin))

#drop Team and Title since they have too many unique values to be turned into a dummy 
games_fin<- games_fin[,-(which(names(games_fin) %in% c("Team", "Title", "Times.Listed")))]

#perform standard scaling (more robust with linear regression)
games_fin_scaled <- as.data.frame(scale(games_fin[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist",
              "Release.Year")]))
games_fin[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist",
              "Release.Year")] <- games_fin_scaled[, c("Number.of.Reviews", "Plays","Playing", "Backlogs", "Wishlist",
                                                "Release.Year")]

#split the final games data set into 70% training and 30% test data 
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

################################################################################
