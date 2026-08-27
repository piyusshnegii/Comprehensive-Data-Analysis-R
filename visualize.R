# ==========================================================
# Week 2 Task: Data Visualization and Insight Communication with R
# Dataset: Titanic Passenger Data (cleaned in Week 1)
# Library: ggplot2
# ==========================================================

library(ggplot2)

dir.create("plots_wk2", showWarnings = FALSE)

titanic <- read.csv("titanic_cleaned.csv", stringsAsFactors = FALSE)
titanic$Survived_Label <- factor(titanic$Survived, levels = c(0, 1), labels = c("Died", "Survived"))
titanic$Pclass_Label   <- factor(titanic$Pclass, levels = c(1, 2, 3), labels = c("1st Class", "2nd Class", "3rd Class"))

theme_report <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

## ---------------------------------------------------------
## CHART 1: Grouped Bar Chart - Survival counts by Class and Sex
## ---------------------------------------------------------
p1 <- ggplot(titanic, aes(x = Pclass_Label, fill = Survived_Label)) +
  geom_bar(position = "dodge", color = "black", linewidth = 0.2) +
  facet_wrap(~Sex) +
  scale_fill_manual(values = c("Died" = "#D9534F", "Survived" = "#5CB85C")) +
  labs(title = "Survival Counts by Passenger Class and Sex",
       subtitle = "Grouped bar chart split by gender",
       x = "Passenger Class", y = "Number of Passengers", fill = "Outcome") +
  theme_report

ggsave("plots_wk2/01_bar_survival_class_sex.png", p1, width = 8, height = 5, dpi = 150)

## ---------------------------------------------------------
## CHART 2: Scatter Plot - Age vs Fare, colored by Survival
## ---------------------------------------------------------
p2 <- ggplot(titanic, aes(x = Age, y = Fare_Capped, color = Survived_Label)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("Died" = "#D9534F", "Survived" = "#5CB85C")) +
  labs(title = "Age vs. Fare Paid, Colored by Survival Outcome",
       subtitle = "Scatter plot (Fare capped at 99th percentile to reduce distortion)",
       x = "Age (years)", y = "Fare Paid (capped)", color = "Outcome") +
  theme_report

ggsave("plots_wk2/02_scatter_age_fare.png", p2, width = 8, height = 5, dpi = 150)

## ---------------------------------------------------------
## CHART 3: Histogram - Age distribution, faceted by survival
## ---------------------------------------------------------
p3 <- ggplot(titanic, aes(x = Age, fill = Survived_Label)) +
  geom_histogram(binwidth = 5, color = "white", alpha = 0.85, position = "identity") +
  facet_wrap(~Survived_Label, ncol = 1) +
  scale_fill_manual(values = c("Died" = "#D9534F", "Survived" = "#5CB85C")) +
  labs(title = "Age Distribution: Died vs. Survived",
       subtitle = "Histogram, 5-year bins",
       x = "Age (years)", y = "Number of Passengers") +
  theme_report + theme(legend.position = "none")

ggsave("plots_wk2/03_histogram_age_by_survival.png", p3, width = 8, height = 6, dpi = 150)

## ---------------------------------------------------------
## CHART 4: Line Chart - Survival rate trend across age groups
## ---------------------------------------------------------
titanic$AgeBin <- cut(titanic$Age, breaks = seq(0, 80, by = 5), include.lowest = TRUE)
age_surv <- aggregate(Survived ~ AgeBin, data = titanic, FUN = function(x) mean(x) * 100)
age_surv$AgeMid <- seq(2.5, by = 5, length.out = nrow(age_surv))

p4 <- ggplot(age_surv, aes(x = AgeMid, y = Survived)) +
  geom_line(color = "#337AB7", linewidth = 1.1) +
  geom_point(color = "#337AB7", size = 2.5) +
  geom_hline(yintercept = 38.38, linetype = "dashed", color = "grey50") +
  annotate("text", x = 65, y = 42, label = "Overall average: 38.4%", size = 3.3, color = "grey40") +
  labs(title = "Survival Rate Trend Across Age Groups",
       subtitle = "Line chart, 5-year age bins",
       x = "Age (years, bin midpoint)", y = "Survival Rate (%)") +
  theme_report

ggsave("plots_wk2/04_line_survival_by_age.png", p4, width = 8, height = 5, dpi = 150)

## ---------------------------------------------------------
## CHART 5: Boxplot - Fare distribution by Class and Survival
## ---------------------------------------------------------
p5 <- ggplot(titanic, aes(x = Pclass_Label, y = Fare_Capped, fill = Survived_Label)) +
  geom_boxplot(alpha = 0.85, outlier.size = 0.8) +
  scale_fill_manual(values = c("Died" = "#D9534F", "Survived" = "#5CB85C")) +
  labs(title = "Fare Paid by Class and Survival Outcome",
       subtitle = "Boxplot (Fare capped at 99th percentile)",
       x = "Passenger Class", y = "Fare Paid (capped)", fill = "Outcome") +
  theme_report

ggsave("plots_wk2/05_boxplot_fare_class_survival.png", p5, width = 8, height = 5, dpi = 150)

## ---------------------------------------------------------
## CHART 6: Stacked Bar - Embarkation port composition by class
## ---------------------------------------------------------
titanic$Embarked_Label <- factor(titanic$Embarked, levels = c("C", "Q", "S"),
                                  labels = c("Cherbourg", "Queenstown", "Southampton"))

p6 <- ggplot(titanic, aes(x = Pclass_Label, fill = Embarked_Label)) +
  geom_bar(position = "fill", color = "black", linewidth = 0.2) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("Cherbourg" = "#F0AD4E", "Queenstown" = "#5BC0DE", "Southampton" = "#337AB7")) +
  labs(title = "Port of Embarkation Composition by Passenger Class",
       subtitle = "100% stacked bar chart",
       x = "Passenger Class", y = "Proportion of Passengers", fill = "Port") +
  theme_report

ggsave("plots_wk2/06_stackedbar_embarked_class.png", p6, width = 8, height = 5, dpi = 150)

cat("All 6 visualizations saved to plots_wk2/\n")
cat("\n--- Age-bin survival table used for Chart 4 ---\n")
print(age_surv[, c("AgeBin", "Survived")])
