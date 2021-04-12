is_positive_vector <- function(x) {
	assertthat::assert_that(
		is.numeric(x),
		msg = paste0(deparse(match.call()$x), " must be numeric.")
	)
	assertthat::assert_that(
		assertthat::noNA(x),
		all(x > 0),
		msg = paste0(
			deparse(match.call()$x), " entries must be positive."
			)
		)

	return(TRUE)
}

is_mlp_units <- function(x) {
	assertthat::assert_that(is_positive_vector(x))
	assertthat::assert_that(
		all(x == as.integer(x)),
		msg = paste(deparse(match.call()$x), "values must be integer.")
		)
	return(TRUE)
}

is_vectorizable <- function(x, len) {
	assertthat::assert_that(
		length(x) %in% c(1, len),
		msg = paste0(
			deparse(match.call()$x),
			" must have length either 1 or ", len, "."
			)
		)
	return(TRUE)
}

is_activation <- function(x, len = NULL) {
	if (is.null(x))
		return(TRUE)
	assertthat::assert_that(
		is.character(x) || is.list(x),
		msg = paste0(
			deparse(match.call()$x),
			" must be either a character vector or a list.")
	)
	if (!is.null(len))
		assertthat::assert_that(is_vectorizable(x, len))

	for (i in seq_along(x))
		tryCatch(tensorflow::tf$keras$activations$get(x[[i]]),
			 error = function(cnd) {
			 	msg <- paste0(deparse(match.call()$x), "[[", i,
			 		      "]] is not a valid activation ",
			 		      "function.")
				assertthat::assert_that(FALSE, msg = msg)
				}
			 )

	return(TRUE)
}

is_l2_penalty <- function(x, len = NULL) {
	if (is.null(x))
		return(TRUE)
	assertthat::assert_that(is_positive_vector(x))
	if (!is.null(len))
		assertthat::assert_that(is_vectorizable(x, len))

	return(TRUE)
}

is_dropout_rate <- is_l2_penalty
