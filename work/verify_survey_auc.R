args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "fail")
d <- read.csv(args[1])
stopifnot(all(d$weight > 0), !anyNA(d[c("stratum", "psu", "domain")]))
design <- svydesign(ids = ~psu, strata = ~stratum, weights = ~weight,
                    data = d, nest = TRUE)
sub <- subset(design, domain == 1)
# Python supplies the derivative per observation divided by its survey weight;
# svytotal independently aggregates its variance over the full survey design.
result <- svytotal(~influence_per_weight, sub)
write.csv(data.frame(se = unname(SE(result)), df = degf(sub),
                     variance_method = "survey::svytotal of AUC influence"),
          args[2], row.names = FALSE)
