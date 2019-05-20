### Lasso
#install.packages("glmnet")
library(glmnet)
library(MASS)

head(Boston)
dim(Boston)

X <- Boston[,1:13]
y <- Boston[,14]

X <- scale(X)      # 説明変数を標準化
y <- y - mean(y)   # 目的変数を中心化

# Lasso推定
res <- glmnet(x=X, y=y)

# 解パス図描画
plot(res, xvar="lambda", label=TRUE, xlab="正則化パラメータの対数値", 
     ylab="回帰係数", col="black", lwd=2.5)

# CVの計算
res.cv <- cv.glmnet(x=X, y=y)
# CV値の推移をプロット
plot(res.cv, xlab="正則化パラメータの対数値", ylab="２乗誤差")

# CV値が最小となる正則化パラメータ値を出力
res.cv$lambda.min

# 正則化パラメータの値を固定
res1 <- glmnet(x=X, y=y, lambda=res.cv$lambda.min)  
res1$beta  # 係数の推定値

