##### グループLassoの適用
library(grpreg)
data(Lung)
X <- Lung$X    # 説明変数
y <- Lung$y[,1]
group <- Lung$group

# グループlassoの実行
res <- grpreg(X, y, group, penalty="grLasso")

# 解パス図は次のプログラムでも作成可能
plot(res, xlab="正則化パラメータの値", ylab="回帰係数")
res$beta[,10] # 回帰係数の推定値

# 解パス図の作成
par(mar=c(5, 5, 1, 1))
plot(0, 0, xlim=c(-0.01,40), ylim=range(res$beta[-1,]), 
     type="n", xlab="正則化パラメータの値", ylab="回帰係数")
for(i in c(1:14)){
  lines(res$lambda, res$beta[i+1, ], lty=as.numeric(group)[i],
        lwd=2)
  text(-0.01, res$beta[i+1,100], i, cex=1)
}
