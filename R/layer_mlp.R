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
