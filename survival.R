# Survival analysis of customer lifetime: Kaplan-Meier curves + Cox proportional hazards.
# Input: data/processed/survival.csv  (written by `make features`)
#   churned = 1 if no order in the last 90 days of the data window, else censored
# NOTE: this script has not been executed by the author of the starter repo; run and debug it.

library(survival)

df <- read.csv("data/processed/survival.csv", stringsAsFactors = TRUE)
df$duration_days <- pmax(df$duration_days, 1)          # Surv() dislikes zero-length spells
df$housing_type  <- relevel(df$housing_type, ref = "HDB 4rm")
df$age_band      <- relevel(df$age_band, ref = "25-34")
df$acquisition_channel <- relevel(df$acquisition_channel, ref = "Organic")

dir.create("reports/figures", showWarnings = FALSE, recursive = TRUE)

# ---- Kaplan-Meier by acquisition channel ----
km <- survfit(Surv(duration_days, churned) ~ acquisition_channel, data = df)
png("reports/figures/km_by_channel.png", width = 900, height = 600, res = 120)
plot(km, col = 1:5, lwd = 2, xlab = "Days since signup", ylab = "Share of customers still active",
     main = "Customer survival by acquisition channel")
legend("topright", legend = levels(df$acquisition_channel), col = 1:5, lwd = 2, cex = 0.8)
dev.off()
print(survdiff(Surv(duration_days, churned) ~ acquisition_channel, data = df))   # log-rank test

# ---- Cox proportional hazards ----
cox <- coxph(Surv(duration_days, churned) ~ housing_type + age_band + acquisition_channel + region,
             data = df)
print(summary(cox))                       # hazard ratios = exp(coef)
print(cox.zph(cox))                       # check the proportional-hazards assumption; discuss violations

# Median survival by segment (put these in your write-up / Tableau)
print(summary(km)$table[, c("records", "events", "median")])
