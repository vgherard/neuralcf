#' @export
gmf_recommender <- function(
	num_users,
	num_items,
	emb_dim,
	out_activation = NULL,
	emb_l2_penalty = NULL,
	out_l2_penalty = NULL
	)
{
	gmf_input <- list(
		layer_input(1L, name = "user_input", dtype = "int32"),
		layer_input(1L, name = "item_input", dtype = "int32")
		)

	gmf_output <- layer_gmf(
		gmf_input,
		n = c(num_users, num_items),
		emb_dim = emb_dim,
		out_activation = out_activation,
		emb_l2_penalty = emb_l2_penalty,
		out_l2_penalty = out_l2_penalty,
		name = "gmf"
		)

	keras_model(gmf_input, gmf_output)
}

#' @export
mlp_recommender <- function(
	num_users,
	num_items,
	emb_dim,
	hid_units = integer(0),
	hid_activation = NULL,
	out_activation = NULL,
	emb_l2_penalty = NULL,
	hid_l2_penalty = NULL,
	out_l2_penalty = NULL,
	hid_dropout_rate = NULL
	)
{
	user_input <- layer_input(1L, name = "user_input", dtype = "int32")
	item_input <- layer_input(1L, name = "item_input", dtype = "int32")

	user_embedding <- layer_embedding(user_input,
					  input_dim = num_users + 1,
					  output_dim = emb_dim[[1]])
	item_embedding <- layer_embedding(item_input,
					  input_dim = num_items + 1,
					  output_dim = emb_dim[[2]])

	mlp_input <- layer_concatenate(list(user_embedding, item_embedding))

	mlp_output <- layer_mlp(
		mlp_input,
		units = hid_units,
		activation = hid_activation,
		l2_penalty = hid_l2_penalty,
		dropout = hid_dropout_rate,
		name = "mlp"
	)

	output <- layer_dense(
		mlp_output,
		units = 1L,
		activation = out_activation,
		kernel_regularizer = regularizer_l2(out_l2_penalty)
		)

	keras_model(list(user_input, item_input), output)
}
