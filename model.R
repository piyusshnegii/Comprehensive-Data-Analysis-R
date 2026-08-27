# ==========================================================
# Week 3 Task: Statistical Analysis and Predictive Modeling with R
# Dataset: Titanic Passenger Data (cleaned in Week 1)
# Goal: Predict Survived (binary classification) via Logistic Regression
# ==========================================================

set.seed(42)
options(width = 100)
dir.create("plots_wk3", showWarnings = FALSE)

suppressPackageStartupMessages({
  library(caret)
  library(pROC)
})

titanic <- read.csv("titanic_cleaned.csv", stringsAsFactors = FALSE)
titanic$Survived <- as.integer(titanic$Survived)

cat("===== DATASET DIMENSIONS =====\n")
print(dim(titanic))

## ==========================================================
## 1. HYPOTHESIS TESTING / EXPLORATORY STATISTICAL ANALYSIS
## ==========================================================

## --- 1.1 Normality test on Age and Fare (Shapiro-Wilk on a sample, since n=891 > 5000 limit is fine but let's report) ---
cat("\n===== SHAPIRO-WILK NORMALITY TEST: Age =====\n")
print(shapiro.test(titanic$Age))

cat("\n===== SHAPIRO-WILK NORMALITY TEST: Fare_Capped =====\n")
print(shapiro.test(titanic$Fare_Capped))

## --- 1.2 T-test: Is mean Age different between Survived vs Died? ---
cat("\n===== WELCH TWO-SAMPLE T-TEST: Age by Survival =====\n")
t_age <- t.test(Age ~ Survived, data = titanic)
print(t_age)

## --- 1.3 T-test: Is mean Fare different between Survived vs Died? ---
cat("\n===== WELCH TWO-SAMPLE T-TEST: Fare by Survival =====\n")
t_fare <- t.test(Fare_Capped ~ Survived, data = titanic)
print(t_fare)

## --- 1.4 Chi-square test: Sex vs Survival ---
cat("\n===== CHI-SQUARE TEST: Sex vs Survival =====\n")
chi_sex <- chisq.test(table(titanic$Sex, titanic$Survived))
print(chi_sex)

## --- 1.5 Chi-square test: Pclass vs Survival ---
cat("\n===== CHI-SQUARE TEST: Pclass vs Survival =====\n")
chi_class <- chisq.test(table(titanic$Pclass, titanic$Survived))
print(chi_class)

## --- 1.6 Chi-square test: Embarked vs Survival ---
cat("\n===== CHI-SQUARE TEST: Embarked vs Survival =====\n")
chi_embarked <- chisq.test(table(titanic$Embarked, titanic$Survived))
print(chi_embarked)

## --- 1.7 Correlation test: Fare vs Age ---
cat("\n===== PEARSON CORRELATION TEST: Age vs Fare =====\n")
print(cor.test(titanic$Age, titanic$Fare_Capped))

## ==========================================================
## 2. TRAIN / TEST SPLIT
## ==========================================================
model_data <- titanic[, c("Survived", "Pclass", "Sex_Encoded", "Age", "Fare_Capped",
                           "FamilySize", "Embarked_C", "Embarked_Q", "Cabin_Known")]
model_data$Survived <- factor(model_data$Survived, levels = c(0, 1), labels = c("Died", "Survived"))

train_idx <- createDataPartition(model_data$Survived, p = 0.8, list = FALSE)
train_data <- model_data[train_idx, ]
test_data  <- model_data[-train_idx, ]

cat("\n===== TRAIN / TEST SPLIT =====\n")
cat("Training rows:", nrow(train_data), " | Testing rows:", nrow(test_data), "\n")
cat("Training class balance:\n"); print(prop.table(table(train_data$Survived)))
cat("Testing class balance:\n"); print(prop.table(table(test_data$Survived)))

## ==========================================================
## 3. MODEL BUILDING - Logistic Regression with 10-fold Cross-Validation
## ==========================================================
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                      summaryFunction = twoClassSummary, savePredictions = "final")

log_model <- train(Survived ~ Pclass + Sex_Encoded + Age + Fare_Capped +
                      FamilySize + Embarked_C + Embarked_Q + Cabin_Known,
                    data = train_data, method = "glm", family = "binomial",
                    trControl = ctrl, metric = "ROC")

cat("\n===== CROSS-VALIDATION RESULTS (10-fold) =====\n")
print(log_model)

cat("\n===== LOGISTIC REGRESSION MODEL SUMMARY (final model) =====\n")
print(summary(log_model$finalModel))

## Odds ratios for interpretability
cat("\n===== ODDS RATIOS =====\n")
print(round(exp(coef(log_model$finalModel)), 3))

## ==========================================================
## 4. MODEL EVALUATION ON HELD-OUT TEST SET
## ==========================================================
test_probs <- predict(log_model, newdata = test_data, type = "prob")[, "Survived"]
test_pred  <- predict(log_model, newdata = test_data)

cat("\n===== CONFUSION MATRIX (Test Set, threshold = 0.5) =====\n")
cm <- confusionMatrix(test_pred, test_data$Survived, positive = "Survived")
print(cm)

## ROC / AUC
roc_obj <- roc(response = test_data$Survived, predictor = test_probs, levels = c("Died", "Survived"))
cat("\n===== TEST SET AUC =====\n")
print(auc(roc_obj))

png("plots_wk3/roc_curve.png", width = 800, height = 700, res = 130)
plot(roc_obj, col = "#337AB7", lwd = 2.5, main = "ROC Curve - Logistic Regression (Test Set)")
abline(a = 0, b = 1, lty = 2, col = "grey60")
text(0.3, 0.2, paste("AUC =", round(auc(roc_obj), 3)), cex = 1.1)
dev.off()

## Confusion matrix heatmap
cm_table <- as.data.frame(cm$table)
png("plots_wk3/confusion_matrix.png", width = 800, height = 700, res = 130)
par(mar = c(5, 6, 4, 2))
mat <- matrix(cm_table$Freq, nrow = 2, byrow = FALSE,
              dimnames = list(Predicted = c("Died", "Survived"), Actual = c("Died", "Survived")))
image(1:2, 1:2, t(mat)[, 2:1], axes = FALSE, col = colorRampPalette(c("white", "#337AB7"))(50),
      xlab = "Actual", ylab = "Predicted", main = "Confusion Matrix (Test Set)")
axis(1, at = 1:2, labels = c("Died", "Survived")); axis(2, at = 1:2, labels = c("Survived", "Died"))
for (i in 1:2) for (j in 1:2) text(i, j, mat[3 - j, i], cex = 1.8, font = 2)
dev.off()

## ==========================================================
## 5. DIAGNOSTIC ANALYSIS
## ==========================================================

## 5a. Residual diagnostics on final glm (fit on full training data)
final_glm <- log_model$finalModel

png("plots_wk3/residual_diagnostics.png", width = 1000, height = 800, res = 120)
par(mfrow = c(2, 2))
plot(final_glm)
dev.off()

## 5b. Variable importance
cat("\n===== VARIABLE IMPORTANCE (|z-value| from logistic model) =====\n")
var_imp <- varImp(log_model)
print(var_imp)

png("plots_wk3/variable_importance.png", width = 800, height = 600, res = 130)
plot(var_imp, main = "Variable Importance - Logistic Regression")
dev.off()

## 5c. Multicollinearity check (VIF)
cat("\n===== VARIANCE INFLATION FACTORS (multicollinearity check) =====\n")
print(car::vif(final_glm))

cat("\n===== SCRIPT COMPLETE =====\n")
