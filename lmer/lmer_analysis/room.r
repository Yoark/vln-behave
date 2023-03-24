library(ggplot2)
library(lmerTest)
library(nlme)
setwd("/Users/zijiao/home/research/unit-vln/code/lmer_data/envdrop-imagenet/room") #change this to wherever the stop_samples.csv is
#####################
#Stop Experiment    #
#####################

room_data <- as.data.frame(read.csv("1_hop_delta.csv")) #load dat
room_data$intervention_yes_no = as.numeric(room_data$intervention_yes_no) #switch to numeric
room_data <- groupedData(dist2goal ~ intervention_yes_no | scan_id/path_id/step_id, room_data) #group it with nested scan/path/step data
room_data$scan_id = factor(room_data$scan_id, ordered=FALSE)
room_data$path_id = factor(room_data$path_id, ordered=FALSE)
room_data$step_id = factor(room_data$step_id, ordered=FALSE)


# https://bookdown.org/steve_midway/DAR/random-effects.html <-- good reference

# Assume IID paired draws -- stop_prob = w*intervention + b_scan/path/step
fm1 <- lmer(dist2goal  ~  intervention_yes_no + (1|scan_id:path_id:step_id), data=room_data)
fm1
anova(fm1)

# Assume paired draws with scan-level effects -- step_prob = (w+w_scan)*intervention + b_scan/path/step
fm2 <- lmer(dist2goal  ~  intervention_yes_no + (intervention_yes_no|scan_id) + (1|scan_id:path_id:step_id), data=room_data)
fm2
anova(fm2)

# Assume paired draws with scan and trajectory level effects -- step_prob = (w+w_scan/traj)*intervention + b_scan/path/step
fm3 <- lmer(dist2goal  ~  intervention_yes_no + (intervention_yes_no|scan_id:path_id) + (1|scan_id:path_id), data=room_data)
fm3
anova(fm3)