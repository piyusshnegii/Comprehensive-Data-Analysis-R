# ==========================================================
# Week 1 Task: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic Passenger Data (Kaggle / Data Science Dojo)
# ==========================================================

options(width = 100)
dir.create("plots", showWarnings = FALSE)

## ---------------------------------------------------------
## 1. LOAD DATA
## ---------------------------------------------------------
titanic <- read.csv("titanic.csv", stringsAsFactors = FALSE)

cat("\n===== STRUCTURE OF RAW DATA (str) =====\n")
str(titanic)

cat("\n===== DIMENSIONS =====\n")
print(dim(titanic))

cat("\n===== FIRST 5 ROWS =====\n")
print(head(titanic, 5))

cat("\n===== SUMMARY OF RAW DATA (summary) =====\n")
print(summary(titanic))

## ---------------------------------------------------------
## 2. MISSING VALUE ANALYSIS
## ---------------------------------------------------------
cat("\n===== MISSING VALUES PER COLUMN (before cleaning) =====\n")
missing_counts <- sapply(titanic, function(x) sum(is.na(x) | x == ""))
print(missing_counts)

missing_pct <- round(100 * missing_counts / nrow(titanic), 2)
cat("\n===== MISSING VALUE PERCENTAGE =====\n")
print(missing_pct)

## ---------------------------------------------------------
## 3. DATA CLEANING
## ---------------------------------------------------------

# 3a. Cabin has ~77% missing -> too sparse to impute meaningfully.
#     Convert to a binary indicator instead of dropping the column outright.
titanic$Cabin_Known <- ifelse(titanic$Cabin == "" | is.na(titanic$Cabin), 0, 1)
titanic$Cabin <- NULL

# 3b. Embarked has 2 missing values -> impute with the mode (most frequent port).
embarked_mode <- names(sort(table(titanic$Embarked[titanic$Embarked != ""]), decreasing = TRUE))[1]
titanic$Embarked[titanic$Embarked == "" | is.na(titanic$Embarked)] <- embarked_mode

# 3c. Age has ~20% missing -> impute with the median (robust to outliers),
#     stratified by Pclass and Sex to keep the imputation realistic.
titanic$Age_Imputed <- 0
for (p in unique(titanic$Pclass)) {
  for (s in unique(titanic$Sex)) {
    idx <- titanic$Pclass == p & titanic$Sex == s
    med_age <- median(titanic$Age[idx], na.rm = TRUE)
    na_idx <- idx & is.na(titanic$Age)
    titanic$Age[na_idx] <- med_age
    titanic$Age_Imputed[na_idx] <- 1
  }
}

# 3d. Fare: check for missing/zero fares (data entry issue), impute with median by class.
if (any(is.na(titanic$Fare))) {
  for (p in unique(titanic$Pclass)) {
    idx <- titanic$Pclass == p
    med_fare <- median(titanic$Fare[idx], na.rm = TRUE)
    na_idx <- idx & is.na(titanic$Fare)
    titanic$Fare[na_idx] <- med_fare
  }
}

cat("\n===== MISSING VALUES AFTER CLEANING =====\n")
print(sapply(titanic, function(x) sum(is.na(x))))

## ---------------------------------------------------------
## 4. OUTLIER DETECTION (IQR method) - Fare and Age
## ---------------------------------------------------------
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  sum(x < lower | x > upper)
}

cat("\n===== OUTLIER COUNTS (IQR method, 1.5x rule) =====\n")
cat("Fare outliers:", detect_outliers(titanic$Fare), "out of", nrow(titanic), "\n")
cat("Age outliers :", detect_outliers(titanic$Age), "out of", nrow(titanic), "\n")

# Cap extreme Fare outliers at the 99th percentile (winsorization) instead of deleting rows,
# to preserve sample size while limiting the influence of extreme values.
fare_cap <- quantile(titanic$Fare, 0.99, na.rm = TRUE)
titanic$Fare_Capped <- ifelse(titanic$Fare > fare_cap, fare_cap, titanic$Fare)

png("plots/boxplot_fare.png", width = 900, height = 600, res = 120)
par(mfrow = c(1, 2))
boxplot(titanic$Fare, main = "Fare - Before Capping", col = "tomato", ylab = "Fare")
boxplot(titanic$Fare_Capped, main = "Fare - After Capping (99th pct)", col = "seagreen", ylab = "Fare")
dev.off()

png("plots/boxplot_age.png", width = 700, height = 600, res = 120)
boxplot(titanic$Age, main = "Age Distribution (post-imputation)", col = "steelblue", ylab = "Age")
dev.off()

## ---------------------------------------------------------
## 5. FEATURE ENGINEERING / TRANSFORMATION
## ---------------------------------------------------------

# 5a. Encode categorical variables
# Sex -> binary
titanic$Sex_Encoded <- ifelse(titanic$Sex == "male", 1, 0)

# Embarked -> one-hot encoding
titanic$Embarked_C <- ifelse(titanic$Embarked == "C", 1, 0)
titanic$Embarked_Q <- ifelse(titanic$Embarked == "Q", 1, 0)
titanic$Embarked_S <- ifelse(titanic$Embarked == "S", 1, 0)

# 5b. Family size feature
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$IsAlone <- ifelse(titanic$FamilySize == 1, 1, 0)

# 5c. Normalization (min-max scaling) of numeric columns for modeling readiness
min_max_scale <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
titanic$Age_Scaled  <- min_max_scale(titanic$Age)
titanic$Fare_Scaled <- min_max_scale(titanic$Fare_Capped)

cat("\n===== STRUCTURE AFTER CLEANING & FEATURE ENGINEERING =====\n")
str(titanic)

write.csv(titanic, "titanic_cleaned.csv", row.names = FALSE)

## ---------------------------------------------------------
## 6. EXPLORATORY DATA ANALYSIS
## ---------------------------------------------------------
cat("\n===== SUMMARY STATISTICS (cleaned numeric columns) =====\n")
print(summary(titanic[, c("Age", "Fare", "FamilySize", "Survived")]))

cat("\n===== SURVIVAL RATE OVERALL =====\n")
print(round(prop.table(table(titanic$Survived)) * 100, 2))

cat("\n===== SURVIVAL RATE BY SEX =====\n")
print(round(prop.table(table(titanic$Sex, titanic$Survived), margin = 1) * 100, 2))

cat("\n===== SURVIVAL RATE BY PASSENGER CLASS =====\n")
print(round(prop.table(table(titanic$Pclass, titanic$Survived), margin = 1) * 100, 2))

cat("\n===== CORRELATION MATRIX (numeric features) =====\n")
num_cols <- titanic[, c("Survived", "Pclass", "Age", "Fare", "FamilySize", "Sex_Encoded", "Cabin_Known")]
corr_matrix <- round(cor(num_cols, use = "complete.obs"), 2)
print(corr_matrix)

# Visualization 1: Survival counts by sex
png("plots/survival_by_sex.png", width = 800, height = 600, res = 120)
counts <- table(titanic$Sex, titanic$Survived)
barplot(counts, beside = TRUE, col = c("lightpink", "lightblue"),
        legend.text = rownames(counts), args.legend = list(x = "topright"),
        names.arg = c("Died", "Survived"), main = "Survival Count by Sex",
        ylab = "Number of Passengers")
dev.off()

# Visualization 2: Survival rate by passenger class
png("plots/survival_by_class.png", width = 800, height = 600, res = 120)
class_surv <- prop.table(table(titanic$Pclass, titanic$Survived), margin = 1)[, 2] * 100
barplot(class_surv, col = "darkorange", names.arg = c("1st Class", "2nd Class", "3rd Class"),
        main = "Survival Rate (%) by Passenger Class", ylab = "Survival Rate (%)", ylim = c(0, 100))
dev.off()

# Visualization 3: Age distribution histogram
png("plots/age_histogram.png", width = 800, height = 600, res = 120)
hist(titanic$Age, breaks = 20, col = "cornflowerblue", main = "Age Distribution of Passengers",
     xlab = "Age (years)")
dev.off()

# Visualization 4: Fare distribution by class
png("plots/fare_by_class.png", width = 800, height = 600, res = 120)
boxplot(Fare_Capped ~ Pclass, data = titanic, col = c("gold", "lightgreen", "lightcoral"),
        main = "Fare Distribution by Passenger Class (capped)", xlab = "Passenger Class", ylab = "Fare")
dev.off()

# Visualization 5: Correlation heatmap (base R)
png("plots/correlation_heatmap.png", width = 900, height = 750, res = 120)
par(mar = c(6, 8, 4, 2))
image(1:ncol(corr_matrix), 1:nrow(corr_matrix), t(corr_matrix)[, nrow(corr_matrix):1],
      axes = FALSE, xlab = "", ylab = "", main = "Correlation Heatmap",
      col = colorRampPalette(c("firebrick", "white", "steelblue"))(50))
axis(1, at = 1:ncol(corr_matrix), labels = colnames(corr_matrix), las = 2, cex.axis = 0.8)
axis(2, at = 1:nrow(corr_matrix), labels = rev(rownames(corr_matrix)), las = 2, cex.axis = 0.8)
for (i in 1:nrow(corr_matrix)) {
  for (j in 1:ncol(corr_matrix)) {
    text(j, nrow(corr_matrix) - i + 1, corr_matrix[i, j], cex = 0.7)
  }
}
dev.off()

cat("\n===== SCRIPT COMPLETE. Cleaned data written to titanic_cleaned.csv =====\n")
cat("Plots saved in ./plots/\n")
