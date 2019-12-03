rm(list=ls())
library(tidyverse)
testout<-read.csv("/Users/lisa/Documents/SFB1294_B05/SceneWalk/Scenewalk_python/tests/testout.txt", header=F, sep=",")
# testout$asd<- 12
testout <- extract(testout, V1, c("pars"), "(\\[.*?\\])", remove=FALSE)
testout$V1 <- NULL
testout <- testout[complete.cases(testout), ]
testout <- data.frame(testout)
names(testout)<- c("pars")

testout$pars <- gsub(pattern = "(subtractive)", replacement = "sub", x = testout$pars)
testout$pars <- gsub(pattern = "(pre)", replacement = "presac", x = testout$pars)
testout$pars <- gsub(pattern = "(e-)", replacement = "emin", x = testout$pars)
# testoutdf <-data.frame(testoutdf)
colnamessa<- c("locdep_decay_switch", "exponents", "shifts", "att_map_init_type", "inhib_method", "durations", "omega_prevloc", "cb_sd_y", "cb_sd_x", "first_fix_OmegaAttention", "shift_size", "sigmaShift", "zeta", "inhibStrength", "lamb", "gamma", "sigmaInhib", "sigmaAttention", "omegaInhib", "omegaAttention")

cleaned <- separate(testout, col="pars", into=colnamessa, sep = "-")


basic_config <- cleaned[,1:5]
unique(basic_config)

data_long <- gather(cleaned, name, val, locdep_decay_switch:omegaAttention, factor_key=TRUE)


p<- ggplot(data_long, aes(x=val))+
  facet_wrap(~name)+
  geom_histogram(stat="count")
p

asd <- cleaned[cleaned$gamma!="15",]
