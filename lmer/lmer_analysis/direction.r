library(ggplot2)
library(lmerTest)
library(nlme)
setwd("/Users/zijiao/home/research/unit-vln/code/lmer_data/envdrop-imagenet/direction") #change this to wherever the stop_samples.csv is
#####################
#Stop Experiment    #
#####################

direction_data <- as.data.frame(read.csv("forward.csv")) #load dat
direction_data$intervention_yes_no = as.numeric(direction_data$intervention_yes_no) #switch to numeric
direction_data <- groupedData(accum_prob ~ intervention_yes_no | scan_id/path_id/step_id, direction_data) #group it with nested scan/path/step data
direction_data$scan_id = factor(direction_data$scan_id, ordered=FALSE)
direction_data$path_id = factor(direction_data$path_id, ordered=FALSE)
direction_data$step_id = factor(direction_data$step_id, ordered=FALSE)


# https://bookdown.org/steve_midway/DAR/random-effects.html <-- good reference

# Assume IID paired draws -- stop_prob = w*intervention + b_scan/path/step
fm1 <- lmer(accum_prob  ~  intervention_yes_no + (1|scan_id:path_id:step_id), data=direction_data)
fm1
anova(fm1)

# Assume paired draws with scan-level effects -- step_prob = (w+w_scan)*intervention + b_scan/path/step
fm2 <- lmer(accum_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id) + (1|scan_id:path_id:step_id), data=direction_data)
fm2
anova(fm2)

# Assume paired draws with scan and trajectory level effects -- step_prob = (w+w_scan/traj)*intervention + b_scan/path/step
fm3 <- lmer(accum_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id:path_id) + (1|scan_id:path_id), data=direction_data)
fm3
anova(fm3)