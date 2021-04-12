# Basic R6 class for Multi-Layer Perceptron blocks
MLPLayer <- R6::R6Class("MLPLayer",

	inherit = keras::KerasLayer,

	private = list(

		#Layer defining variables
		n_layers = NULL,
		units = NULL,
		l2_penalty = NULL,
		dropout_rate = NULL,

		# Weights
		W = list(),
		b = list(),

		# Activations
		activation = NULL,

		# Utilities
		W_shape = function(i, input_shape) {
			if (i == 1)
				list(input_shape[[2]], private$units[[i]])
			else
				list(private$units[[i - 1]], private$units[[i]])
		},
		b_shape = function(i) {
			list(1L, private$units[[i]])
		},
		l2_reg = function(i) {
			.regularizer_l2(private$l2_penalty[[i]])
		},

		lin_forward = function(x, i) {
			return(k_dot(x, private$W[[i]]) + private$b[[i]])
		},
		forward = function(x, i) {
			x <- private$lin_forward(x, i)
			return(private$activation[[i]](x))
		}
	),

	active = ,

	public = list(
		dense = list(),
		dropout = list(),

		initialize = function(
			units, activation, l2_penalty, dropout_rate
			)
		{
			private$n_layers <- length(units)
			private$units <- as.integer(units)
			private$activation <- lapply(activation, function(x){
				tensorflow::tf$keras$activations$get(x)
			})
			private$l2_penalty <- l2_penalty
			private$dropout_rate <- dropout_rate
		},

		build = function(input_shape)
		{
			for (i in seq_len(private$n_layers)) {
				private$W[[i]] <- self$add_weight(
					name = paste("W", i, sep = "_"),
					shape = private$W_shape(i, input_shape),
					initializer = "glorot_uniform",
					regularizer = private$l2_reg(i),
					trainable = TRUE
					)
				private$b[[i]] <- self$add_weight(
					name = paste("b", i, sep = "_"),
					shape = private$b_shape(i),
					initializer = "zeros",
					trainable = TRUE
					)
				}
		},

		call = function(x, mask = NULL, training = FALSE)
		{
			if (training && !is.null(private$dropout)) {
				for (i in seq_len(private$n_layers)) {
					x <- private$forward(x, i)
					x <- k_dropout(x,
						       private$dropout_rate[[i]]
						       )
				}
			} else {
				for (i in seq_len(private$n_layers))
					x <- private$forward(x, i)
			}

			return(x)
		},

		compute_output_shape = function(input_shape) {
			out_units <- private$units[[private$n_layers]]
			reticulate::tuple(input_shape[[1]], out_units)
		}

		#,print = function(...){}
	)
)

# Wrapper for MLP layer constructor
#' @export
layer_mlp <- function(
	object,
	units,
	activation = NULL,
	l2_penalty = NULL,
	dropout_rate = NULL,
	name = NULL,
	trainable = TRUE
	)
{
	check_args_mlp(units, activation, l2_penalty, dropout_rate)
	c(units, activation, l2_penalty, dropout_rate) %<-%
		vectorize_args_mlp(units, activation, l2_penalty, dropout_rate)
	keras::create_layer(
		MLPLayer, object, list(units = units,
				       activation = activation,
				       l2_penalty = l2_penalty,
				       dropout_rate = dropout_rate,
				       name = name,
				       trainable = trainable
				       )
		)
}

check_args_mlp <- function(units, activation, l2_penalty, dropout_rate)
{
	tryCatch(
		# try
		assertthat::assert_that(
			is_mlp_units(units),
			is_activation(activation, len = length(units)),
			is_l2_penalty(l2_penalty, len = length(units)),
			is_dropout_rate(dropout_rate, len = length(units))
		)
		,
		# catch
		error = function(cnd)
			rlang::abort(cnd$message, class = "domain_error")
	)
}

vectorizer <- function(x, len) {
	if (is.null(x))
		return(x)
	if (length(x) == 1)
		return(replicate(len, x))
	assertthat::assert_that(length(x) == len)
	return(x)
}

vectorize_args_mlp <- function(units, activation, l2_penalty, dropout_rate)
{
	len <- length(units)
	list(units,
	     vectorizer(activation, len),
	     vectorizer(l2_penalty, len),
	     vectorizer(dropout_rate, len)
	     ) # return
}
