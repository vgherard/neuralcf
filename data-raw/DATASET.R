## code to prepare `DATASET` dataset goes here
ml_100k <- readr::read_tsv(
	"data-raw/ml_100k/u.data",
	col_names = c("user", "item", "rating", "timestamp"),
	col_types = c("iidi")
)

usethis::use_data(ml_100k, overwrite = TRUE)
