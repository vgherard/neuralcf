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
		dot_activation = out_activation,
		emb_l2_penalty = emb_l2_penalty,
		dot_l2_penalty = out_l2_penalty,
		name = "gmf"
		)

	keras_model(gmf_input, gmf_output)
}

#' @export
mlp_recommender <- function(
	num_users,
	num_items,
	emb_dim,
	emb_l2_penalty = NULL,
	hid_units = integer(0),
	hid_activation = NULL,
	hid_l2_penalty = NULL,
	hid_dropout_rate = NULL,
	out_activation = NULL,
	out_l2_penalty = NULL
	)
{
	check_args_mlp_rec(
		num_users,
		num_items,
		emb_dim,
		emb_l2_penalty,
		out_activation,
		out_l2_penalty
		) # Arguments for the hidden layer are checked by layer_mlp()
	c(emb_dim, emb_l2_penalty) %<-%
		vectorize_args(emb_dim, emb_l2_penalty, len = 2)

	user_input <- layer_input(1L, name = "user_input", dtype = "int32")
	item_input <- layer_input(1L, name = "item_input", dtype = "int32")

	user_embedding <- layer_embedding(user_input,
					  input_dim = num_users + 1,
					  output_dim = emb_dim[[1]]
					  )
	item_embedding <- layer_embedding(item_input,
					  input_dim = num_items + 1,
					  output_dim = emb_dim[[2]]
					  )

	mlp_input <- layer_concatenate(list(user_embedding, item_embedding))
	mlp_input <- layer_flatten(mlp_input)


	mlp_output <- layer_mlp(
		mlp_input,
		units = hid_units,
		activation = hid_activation,
		l2_penalty = hid_l2_penalty,
		dropout_rate = hid_dropout_rate,
		name = "mlp"
	)

	output <- layer_dense(
		mlp_output,
		units = 1L,
		activation = out_activation,
		kernel_regularizer = .regularizer_l2(out_l2_penalty)
		)

	keras_model(list(user_input, item_input), output)
}

#' @export
neumf_recommender <- function(
	num_users,
	num_items,
	gmf_emb_dim,
	gmf_dot_activation = NULL,
	gmf_emb_l2_penalty = NULL,
	gmf_dot_l2_penalty = NULL,
	mlp_emb_dim,
	mlp_hid_units = integer(0),
	mlp_hid_activation = NULL,
	mlp_emb_l2_penalty = NULL,
	mlp_hid_l2_penalty = NULL,
	mlp_hid_dropout_rate = NULL,
	out_activation = NULL,
	out_l2_penalty = NULL
) {
	check_args_mlp_rec(
		num_users,
		num_items,
		mlp_emb_dim,
		mlp_emb_l2_penalty,
		out_activation,
		out_l2_penalty
	) # Arguments for the hidden layer are checked by layer_*() creators
	c(mlp_emb_dim, mlp_emb_l2_penalty) %<-%
		vectorize_args(mlp_emb_dim, mlp_emb_l2_penalty, len = 2)

	input <- list(
		layer_input(1L, name = "user_input", dtype = "int32"),
		layer_input(1L, name = "item_input", dtype = "int32")
	)

	gmf_output <- layer_gmf(
		input,
		n = c(num_users, num_items),
		emb_dim = gmf_emb_dim,
		dot_activation = gmf_dot_activation,
		emb_l2_penalty = gmf_emb_l2_penalty,
		dot_l2_penalty = gmf_dot_l2_penalty,
		name = "gmf"
	)

	n <- c(num_users, num_items)
	mlp_input <- lapply(1:2, function(i) {
		layer_embedding(input[[i]],
				input_dim = n[[i]] + 1,
				output_dim = mlp_emb_dim[[i]]
				)
	}) %>% layer_concatenate %>% layer_flatten

	mlp_output <- layer_mlp(
		mlp_input,
		units = mlp_hid_units,
		activation = mlp_hid_activation,
		l2_penalty = mlp_hid_l2_penalty,
		dropout_rate = mlp_hid_dropout_rate,
		name = "mlp"
	)

	gmf_mlp_output <- layer_concatenate(list(gmf_output, mlp_output))

	output <- layer_dense(
		gmf_mlp_output,
		units = 1L,
		activation = out_activation,
		kernel_regularizer = .regularizer_l2(out_l2_penalty)
	)

	keras_model(input, output)
}

#------------------------------ Helpers ---------------------------------------#

check_args_mlp_rec <- function(
	num_users,
	num_items,
	emb_dim,
	emb_l2_penalty = NULL,
	out_activation = NULL,
	out_l2_penalty = NULL
	)
{
	tryCatch(
		# try
		assertthat::assert_that(
			is_n_inputs(c(num_users, num_items)),
			is_emb_dim(emb_dim, len = 2),
			is_l2_penalty(emb_l2_penalty, len = 2),
			is_activation(out_activation, len = 1),
			is_l2_penalty(out_l2_penalty, len = 1)
		)
		,
		# catch
		error = function(cnd)
			rlang::abort(cnd$message, class = "domain_error")
	)
}
