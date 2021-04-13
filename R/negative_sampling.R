#' @export
negative_generator <- function(
	user_input,
	item_input,
	num_negatives,
	num_items,
	min_index = 1,
	max_index = length(user_input),
	batch_size = 128
	)
{
	# ...Assertions...

	user_input <- user_input[min_index:max_index]
	item_input <- item_input[min_index:max_index]
	len_augmented_data <- (max_index - min_index + 1) * (1 + num_negatives)

	num_users <- max(user_input)
	data <- NULL
	i <- len_augmented_data + 1

	function() {
		if (i + batch_size >= len_augmented_data) {
			i <<- 1
			data <<- add_negatives(
				user_input = user_input,
				item_input = item_input,
				num_items = num_items,
				num_users = num_users,
				num_negatives = num_negatives
			)
		}

		len <- min(batch_size, len_augmented_data - i + 1)
		index <- i:(i + len - 1) # N.B.: preserves ALTREP
		res <- list(
			list(data[["user"]][index], data[["item"]][index]),
			data[["label"]][index]
		)
		i <<- i + len

		return(res)
	}
}
