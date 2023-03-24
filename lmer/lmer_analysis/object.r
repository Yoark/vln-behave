library(ggplot2)
library(lmerTest)
library(nlme)
setwd("/Users/zijiao/home/research/unit-vln/code/lmer_data/envdrop-imagenet/object") #change this to wherever the stop_samples.csv is
#####################
#Stop Experiment    #
#####################

object_data <- as.data.frame(read.csv("total_objects.csv")) #load dat
object_data$intervention_yes_no = as.numeric(object_data$intervention_yes_no) #switch to numeric
object_data <- groupedData(accum_prob ~ intervention_yes_no | scan_id/path_id/step_id, object_data) #group it with nested scan/path/step data
object_data$scan_id = factor(object_data$scan_id, ordered=FALSE)
object_data$path_id = factor(object_data$path_id, ordered=FALSE)
object_data$step_id = factor(object_data$step_id, ordered=FALSE)


# https://bookdown.org/steve_midway/DAR/random-effects.html <-- good reference

# Assume IID paired draws -- stop_prob = w*intervention + b_scan/path/step
fm1 <- lmer(accum_prob  ~  intervention_yes_no + (1|scan_id:path_id:step_id), data=object_data)
fm1
anova(fm1)

# Assume paired draws with scan-level effects -- step_prob = (w+w_scan)*intervention + b_scan/path/step
fm2 <- lmer(accum_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id) + (1|scan_id:path_id:step_id), data=object_data)
fm2
anova(fm2)

# Assume paired draws with scan and trajectory level effects -- step_prob = (w+w_scan/traj)*intervention + b_scan/path/step
fm3 <- lmer(accum_prob  ~  intervention_yes_no + (intervention_yes_no|scan_id:path_id) + (1|scan_id:path_id), data=object_data)
fm3
anova(fm3)