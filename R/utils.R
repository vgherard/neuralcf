.regularizer_l2 <- function(l2_penalty) {
	if (is.null(l2_penalty))
		return(NULL)
	return(regularizer_l2(l2_penalty))
}
