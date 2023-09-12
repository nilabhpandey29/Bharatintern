# Load necessary libraries for model evaluation
library(pROC)
library(caret)

# 1. Deviance comparison
dev_initial <- deviance(initial_model)
dev_new <- deviance(new_model)

cat("Deviance for the initial model:", dev_initial, "\n")
cat("Deviance for the new model:", dev_new, "\n")

# 2. AIC and BIC comparison
aic_initial <- AIC(initial_model)
bic_initial <- BIC(initial_model)

aic_new <- AIC(new_model)
bic_new <- BIC(new_model)

cat("AIC for the initial model:", aic_initial, "\n")
cat("BIC for the initial model:", bic_initial, "\n")

cat("AIC for the new model:", aic_new, "\n")
cat("BIC for the new model:", bic_new, "\n")

# 3. Pseudo R-squared comparison (Cox and Snell R-squared)
# Fit the null model for Cox and Snell R-squared
null_model <- glm(Survived ~ 1, data = train_data, family = "binomial")

# Calculate Cox and Snell R-squared for the initial model
cox_snell_r_squared_initial <- 1 - exp(-2 * (logLik(initial_model) - logLik(null_model)))

# Fit the null model for Cox and Snell R-squared for the new model
null_model_new <- glm(Survived ~ 1, data = train_data1, family = "binomial")

# Calculate Cox and Snell R-squared for the new model
cox_snell_r_squared_new <- 1 - exp(-2 * (logLik(new_model) - logLik(null_model_new)))

cat("Cox and Snell R-squared for the initial model:", cox_snell_r_squared_initial, "\n")
cat("Cox and Snell R-squared for the new model:", cox_snell_r_squared_new, "\n")


# 4. ROC curve and AUC comparison
# Generate predictions for the initial model
initial_predictions <- predict(initial_model, newdata = test_data, type = "response")

# Generate predictions for the new model
new_predictions <- predict(new_model, newdata = test_data1, type = "response")

# Calculate ROC curve and AUC for the initial model
roc_initial <- roc(test_data$Survived, initial_predictions)
auc_initial <- auc(roc_initial)

# Calculate ROC curve and AUC for the new model
roc_new <- roc(test_data1$Survived, new_predictions)
auc_new <- auc(roc_new)

cat("AUC for the initial model:", auc_initial, "\n")
cat("AUC for the new model:", auc_new, "\n")


cat("AUC for the initial model:", auc_initial, "\n")
cat("AUC for the new model:", auc_new, "\n")

# 5. Likelihood ratio test
lr_test <- lrtest(new_model, initial_model)
cat("Likelihood Ratio Test p-value:", lr_test$p.value, "\n")

# 6. Confusion matrices
# Generate binary predictions for the initial model
binary_predictions_initial <- ifelse(initial_predictions > threshold, 1, 0)

# Create a confusion matrix for the initial model
conf_matrix_initial <- confusionMatrix(data = factor(binary_predictions_initial, levels = c(0, 1)),
                                       reference = factor(test_data$Survived, levels = c(0, 1)))

# Display the confusion matrix for the initial model
print(conf_matrix_initial)

# Generate binary predictions for the new model
binary_predictions_new <- ifelse(new_predictions > threshold, 1, 0)

# Create a confusion matrix for the new model
conf_matrix_new <- confusionMatrix(data = factor(binary_predictions_new, levels = c(0, 1)),
                                   reference = factor(test_data1$Survived, levels = c(0, 1)))

# Display the confusion matrix for the new model
print(conf_matrix_new)


