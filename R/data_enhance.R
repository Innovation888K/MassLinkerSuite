library(mzR)
library(MSnbase)
library(ProtGenerics)
args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1] 
output=args[2] 
enhance_num=args[3]
dir.create(output, showWarnings = FALSE)
#work_dir=paste0("D:\\质谱不鉴定\\MTBLS3144\\split",work)
for (f in list.files(work_dir)) {
  file_name=paste0(work_dir,"\\",f)
  file1 <- mzR::openMSfile(file_name)
  file2 <- MSnbase::readMSData(
    files = file_name,
    msLevel. = c(1,2),
    mode = "onDisk",
    verbose = TRUE
  )
  header <- mzR::header(file1)
  set.seed(42)
  # 获取原始数据
  peaks <- ProtGenerics::peaks(object = file1)
  rt <- ProtGenerics::rtime(object = file2)
  
  # 在这里对rt和peaks进行修改
  # 例如：将rt的值增加10%
  rt_frame=data.frame(rt=rt,idx=1:length(rt))
  rt_diff=rnorm(enhance_num,0,10)
  rt_diff[rt_diff<0-rt[1]]=-rt[1]
  # 例如：将peaks的值增加20%
  
  
  peaks1=lapply(1:enhance_num,function(m) return(lapply(peaks,function(x){
    x=data.frame(x)
    mz_shift=rnorm(nrow(x),0,5)*ifelse(x$mz<400,400,x$mz)/1000000#5% shift>10ppm
    x$mz=x$mz+mz_shift
    int_shift=rnorm(nrow(x),0,0.06)#2.5% shift>10%
    x$intensity=x$intensity*(1-int_shift)
    return(as.matrix(x))
  })))
  
  for (i in 1:enhance_num) {
    header$retentionTime <- rt_diff[i]
    output_file <- paste0(output,"\\",i,'-',f)
    print(output_file)
    mzR::writeMSData(
      object = peaks1[[i]],  # 修改后的峰数据
      file = output_file,       # 输出文件名
      header = mzR::header(file1)
    )
  }
  mzR::close(file1)
}


