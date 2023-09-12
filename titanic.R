# Load required libraries
library(titanic)
library(dplyr)
library(tidyr)
library(ggplot2)
library(glmnet)

# Load Titanic dataset
data("titanic_train")
titanic_data <- titanic_train

# Step 2: Data Cleaning and Preprocessing
# Remove unnecessary columns and missing values
titanic_data <- titanic_data %>%
  select(Survived, Pclass, Age, Sex, SibSp, Parch, Embarked) %>%
  drop_na()

# Convert categorical variables to factors
titanic_data$Sex <- factor(titanic_data$Sex)
titanic_data$Pclass <- factor(titanic_data$Pclass)
titanic_data$Embarked <- factor(titanic_data$Embarked)

# Split the data into training and testing sets
set.seed(123)
sample_index <- sample(1:nrow(titanic_data), 0.7 * nrow(titanic_data))
train_data <- titanic_data[sample_index, ]
test_data <- titanic_data[-sample_index, ]

# Step 3: Create an initial logistic regression model
initial_model <- glm(Survived ~ ., data = train_data, family = "binomial")

# Step 4: Analyze the significance of each factor
summary(initial_model)

accuracy <- mean(binary_predictions == test_data$Survived)
cat("Accuracy on the test set using significant predictors:", accuracy, "\n")



# Create a new data frame with only significant predictors
significant_data <- titanic_data[, c("Survived", "Pclass","Age","Sex","SibSp")]

# Step 6: Split the data into training and testing sets
set.seed(123)
sample_index1 <- sample(1:nrow(significant_data), 0.7 * nrow(significant_data))
train_data1 <- significant_data[sample_index1, ]
test_data1 <- significant_data[-sample_index1, ]



# Step 8: Create a logistic regression model using only the significant predictors
new_model <- glm(Survived ~ ., data = train_data1, family = "binomial")

# Step 9: Analyze the new model
summary(new_model)

# Make predictions on the test set using the new model
predictions <- predict(new_model, newdata = test_data1, type = "response")

# Convert probabilities to binary predictions (0 or 1)
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)
summary(new_model)

# Calculate accuracy on the test set
accuracy1 <- mean(binary_predictions == test_data1$Survived)
cat("Accuracy on the test set using significant predictors:", accuracy1, "\n")