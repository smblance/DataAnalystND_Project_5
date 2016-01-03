setwd('~/DataAnalyst/Projects/DataAnalystND_Project_5/ud120-projects/final_project/')
data = read.csv('final_project_data.csv')

library(ggplot2)

data <- subset(data, name != 'TOTAL')
ggplot(data = data, aes(salary, bonus)) +
  geom_point()


feature_stats <- data.frame(row.names = names(data))
for (col in names(data)) {
  feature_stats[col, 'notNaN'] <- nrow(subset(data, data[[col]] != 'NaN'))
  feature_stats[col, 'unique_values'] <- length(unique(data[[col]]))
}

ggplot(data, aes(poi)) +
  geom_histogram()
