library(ggplot2)
library(lmerTest)
library(nlme)
setwd("/Users/zijiao/home/research/unit-vln/code/lmer_data/clip-envdrop/stop") #change this to wherever the stop_samples.csv is
#####################
#Stop Experiment    #
#####################

stop_data <- as.data.frame(read.csv("no_end1_ahead_partial0.csv")) #change file name here
stop_data$intervention_yes_no = as.numeric(stop_data$intervention_yes_no) #switch to numeric
stop_data <- groupedData(stop_prob ~ intervention_yes_no | scan_id/path_id/step_id, stop_data) #group it with nested scan/path/step data
stop_data$scan_id = factor(stop_data$scan_id, ordered=FALSE)
stop_data$path_id = factor(stop_data$path_id, ordered=FALSE)
stop_data$step_id = factor(stop_data$step_id, ordered=FALSE)


# https://bookdown.org/steve_midway/DAR/random-effects.html <-- good reference

# Assume IID paired draws -- stop_prob = w*intervention + b_scan/path/step
fm1 <- lmer(stop_prob  ~  intervention_yes_no + (1|scan_id:path_id:step_id), data=stop_data)
fm1
anova(fm1)

# Assume paired draws with scan-level effects -- step_prob = (w+w_scan)*intervention + b_scan/path/step
fm2 <- lmer(stop_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id) + (1|scan_id:path_id:step_id), data=stop_data)
fm2
anova(fm2)

# Assume paired draws with scan and trajectory level effects -- step_prob = (w+w_scan/traj)*intervention + b_scan/path/step
fm3 <- lmer(stop_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id:path_id) + (1|scan_id:path_id), data=stop_data)
fm3
anova(fm3)
