library(MASS)
data(mcycle)

X <- mcycle$times

#最小値
X_min <- min(X)

#最大値
X_max <- max(X)

#正規化
X_scale <- scale(X, center = X_min, scale = (X_max - X_min))

y <- mcycle$accel
z <- y - mean(y)
## GP regression 
GPmodel = GP_fit(X_scale, z) 
print(GPmodel, digits=4) 
plot(GPmodel)

