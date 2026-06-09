# LC-MS raw data augmentation script for MassLinker.
#
# This script generates augmented LC-MS raw files by applying small stochastic
# perturbations to m/z values, peak intensities, and retention times. The goal
# is to expand the training set while preserving the overall metabolomic signal
# structure for downstream MassLinker tokenization and supervised modeling.
#
# Command-line arguments:
#   args[1] : work_dir
#       Directory containing raw LC-MS files.
#   args[2] : output
#       Directory used to save augmented raw files.
#   args[3] : enhance_num
#       Number of augmented files generated for each input file.

library(mzR)

args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1]
output <- args[2]
enhance_num <- as.integer(args[3])

# Create the output directory if it does not already exist.
dir.create(output, showWarnings = FALSE, recursive = TRUE)

# Set a fixed seed for reproducible augmentation.
set.seed(42)

# Select supported LC-MS raw data formats.
files <- list.files(
  work_dir,
  pattern = "\\.(mzXML|mzML|cdf|CDF)$",
  full.names = TRUE
)

# Process each raw LC-MS file.
for (file_name in files) {
  message("Processing: ", file_name)

  # Open the raw MS file and extract scan headers, peak lists, and retention time.
  file1 <- mzR::openMSfile(file_name)
  header <- mzR::header(file1)
  peaks <- ProtGenerics::peaks(object = file1)
  rt <- header$retentionTime

  # Generate sample-level retention-time shifts.
  rt_diff <- rnorm(enhance_num, mean = 0, sd = 10)

  for (i in seq_len(enhance_num)) {
    # Apply scan-level m/z and intensity perturbations.
    augmented_peaks <- lapply(peaks, function(x) {
      x <- data.frame(x)

      if (nrow(x) == 0) {
        return(as.matrix(x))
      }

      # Add a small m/z shift scaled by peak m/z.
      mz_shift <- rnorm(
        nrow(x),
        mean = 0,
        sd = 5
      ) * ifelse(x$mz < 400, 400, x$mz) / 1000000

      x$mz <- x$mz + mz_shift

      # Apply bounded multiplicative intensity perturbation.
      int_shift <- rnorm(nrow(x), mean = 0, sd = 0.06)
      int_shift <- pmax(pmin(int_shift, 0.3), -0.3)
      x$intensity <- x$intensity * (1 - int_shift)

      # Keep intensities non-negative.
      x$intensity[x$intensity < 0] <- 0

      return(as.matrix(x))
    })

    # Apply sample-level retention-time perturbation.
    header_i <- header
    header_i$retentionTime <- rt + rt_diff[i]
    header_i$retentionTime[header_i$retentionTime < 0] <- 0

    # Prefix the original filename with the augmentation index.
    sample_name <- basename(file_name)
    output_file <- file.path(output, paste0(i, "-", sample_name))

    message("Writing: ", output_file)

    # Write the augmented MS data to disk.
    mzR::writeMSData(
      object = augmented_peaks,
      file = output_file,
      header = header_i
    )
  }

  # Close the input MS file handle.
  mzR::close(file1)
}
