.regularizer_l2 <- function(l2_penalty) {
	if (is.null(l2_penalty))
		return(NULL)
	return(regularizer_l2(l2_penalty))
}

vectorize_args <- function(..., len)
{
	lapply(list(...), function(x) {
		if (is.null(x))
			return(x)
		if (length(x) == 1)
			return(replicate(len, x))
		assertthat::assert_that(length(x) == len)
		return(x)
	})
}

vectorizer <- function(x, len) {

}
