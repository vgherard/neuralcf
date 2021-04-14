#' @export
recommender_pop <- function(
	data,
	mode = "item",
	feedback = "explicit",
	user_col = "user",
	item_col = "item",
	rating_col = "rating"
	)
{
	# ...Argument checking...
	if (mode == "item") {
		ids <- data[[item_col]]
		num_ids <- max(ids)
		num_objects <- max(data[[user_col]])
	} else {
		ids <- data[[user_col]]
		num_ids <- max(ids)
		num_objects <- max(data[[item_col]])
	}

	if (feedback == "explicit") {
		ratings <- data[[rating_col]]
		lt <- compute_average_ratings(ids, ratings)
	} else {
		lt <- compute_average_interactions(ids, num_objects)
	}

	if (mode == "item")
		function(user, item)
			return_average_prediction(item, lt, num_ids)
	else
		function(user, item)
			return_average_prediction(user, lt, num_ids)

}
