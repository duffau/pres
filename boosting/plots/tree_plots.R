rm(list=ls()); gc()
library(rpart)
library(stringr)
library(latex2exp)
library(extrafont)
set.seed(1)
setwd("~/repos/talks/boosting/plots")

r_count <- 0
t_count <- 0
xnames <- 1:4


threshold <- function() {
  t_count <<- t_count + 1
  paste("t_", t_count, sep="")
}

region_name <- function() {
  r_count <<- r_count + 1
  paste("R_", r_count, sep="")
}

xname <- function() {
  paste("X_", xnames[t_count %% length(xnames) + 1], sep="")
}


translate_labels <- function(labels) {
  new_labels <- c()
  m <- str_match(labels, "X(\\d+)([<|>]).*")
  for (i in 1:nrow(m)) {
    num <- m[i,2]
    rel <- m[i,3]
    if(is.na(labels[i])) {
      new_labels <- c(new_labels, "")
    } else if (length(grep("R", labels[i]))) {
      new_labels <- c(new_labels, TeX(paste("$",region_name(), "$", sep="")))
    } else if (is.na(num)) {
      new_labels <- c(new_labels, labels[i])
    } else {
      new_labels <- c(new_labels, TeX(paste("$",xname(), " ",rel, " ", threshold(), "$", sep="")))
    }
  }
  new_labels
}

n <- 100
region <- sample(c("R1", "R2"), n, replace = T)
X1 <- rnorm(n)
X2 <- rnorm(n)
X3 <- rnorm(n)
X4 <- rnorm(n)
data1 <- data.frame(
  region, X1, X2, X3, X4
)

label <- function(x, y, labels, ...) {
  argg <- c(as.list(environment()), list(...))
  print(argg)
  labels <- translate_labels(labels)
  print(labels)
  text(x, y, labels, ...)
} 


fit1 <- rpart(region ~ X1, data = data1,  control = list(maxdepth = 1))
fit2 <- rpart(region ~ X1 + X2, data = data1,  control = list(maxdepth = 2))
fit3 <- rpart(region ~ X1 + X2 + X3, data = data1,  control = list(maxdepth = 3))
fit4 <- rpart(region ~ X1 + X2 + X3 + X4, data = data1,  control = list(maxdepth = 4))
fits <- list(fit1, fit2, fit3, fit4)

cex <- 1.5
svg("tree-depth.svg")
loadfonts(device = "win")
par(mfrow = c(2,2), xpd = NA, family = "LM Roman 10")
for( i in 1:4) {
  r_count <<- 0
  t_count <<- 0
  plot(fits[[i]], main=paste("Depth =", i),cex.main=cex)
  text(
    fits[[i]], 
    use.n = F, 
    FUN=label,
    cex=cex
  )
} 
dev.off()