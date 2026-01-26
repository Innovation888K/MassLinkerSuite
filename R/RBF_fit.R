library(pracma)

rbf_fit = function(x, y,kernel_num=20,peak_center=15,peak_width=c(5,60),max_iterations=3000) {
  # 建立设计矩阵
  centers=sample_by_intensity(kernel_num,abs(y),x)
  if(is.null(centers)){
    return(list(weights=rep(0,kernel_num),centers=rep(0,kernel_num),sigma=rep(0,kernel_num)))
  }
  sigmas=optimize_sigma(x,y,centers,peak_center,peak_width,max_iterations)
  G = matrix(nrow = length(x), ncol = length(centers))
  for (i in seq_along(centers)) {
    G[, i] = rbf_kernel(x, centers[i], sigmas[i])
  }
  
  # 求权重
  weights = pinv(G) %*% y
  
  return(list(weights = weights, centers = centers, sigma = sigmas))
}

rbf_kernel = function(x, c, sigma) {
  exp(-(x - c)^2 / (2 * sigma^2))
}

# RBF 预测函数
rbf_predict = function(rbf_model, x) {
  weights = rbf_model$weights
  centers = rbf_model$centers
  sigma = rbf_model$sigma
  
  G = matrix(nrow = length(x), ncol = length(centers))
  for (i in seq_along(centers)) {
    G[, i] = rbf_kernel(x, centers[i], sigma[i])
  }
  
  y_pred = G %*% weights
  return(y_pred)
}

sample_by_intensity=function(sample_time,intensity_list,candidates){
  set.seed(50)
  ret=c()
  prefix_sum=numeric(length = length(intensity_list)+1)
  prefix_sum[1]=0
  for (i in 1:length(intensity_list)) {
    prefix_sum[i+1]=prefix_sum[i]+intensity_list[i]
  }
  samples=runif(sample_time,min=0,max=sum(intensity_list))
  for (i in samples) {
    for (j in 1:length(prefix_sum)) {
      if(i<prefix_sum[j]){
        ret=c(ret,candidates[j-1])
        break
      }
    }
  }
  return(sort(ret))
}

compute_gradient <- function(x, y, centers, weights, sigmas) {
  N = length(x)
  P = length(centers)
  grad_sigmas = numeric(P)
  phi = numeric(P)
  partial_derivative = numeric(P)
  G_all = matrix(nrow = length(x), ncol = length(centers))
  for (j in seq_along(centers)) {
    G_all[, j] = rbf_kernel(x, centers[j], sigmas[j])
  }
  weights = pinv(G_all) %*% y
  # 计算模型的预测输出
  y_pred = rowSums(G_all %*% weights)
  for (i in 1:P) {
    partial_derivative[i]=mean(2*(y_pred-y)*weighted.mean(i)*y_pred*(x-centers[i])^2/sigmas[i]^3)
  }
  return(partial_derivative)
}

optimize_sigma =function(x,y,centers,peak_center,peak_width,max_iterations=3000){
  sigmas <- rep(peak_center, length(centers))
  learning_rate <- 0.0001
  for (iter in seq_len(max_iterations)) {
    # E 步：计算每个样本对每个 RBF 的响应
    G_all = matrix(nrow = length(x), ncol = length(centers))
    for (j in seq_along(centers)) {
      G_all[, j] = rbf_kernel(x, centers[j], sigmas[j])
    }
    weights = pinv(G_all) %*% y
    # 计算模型的预测输出
    y_pred = rowSums(G_all %*% weights)
    # M 步：使用梯度下降更新 sigmas
    grad_sigmas = compute_gradient(x, y, centers, weights, sigmas)
    sigmas = sigmas - learning_rate * grad_sigmas
    sigmas=ifelse(sigmas>peak_width[2],peak_width[2],ifelse(sigmas<peak_width[1],peak_width[1],sigmas))
    # 检查收敛性
  }
  return(sigmas)
}
















