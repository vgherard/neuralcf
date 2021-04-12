MLPLayer <- R6::R6Class("MLPLayer",

	inherit = keras::KerasLayer,

	private = ,
	active = ,

	public = list(
		dense = list(),
		dropout = list(),

		initialize = function(units, activation, l2_penalty, dropout)
		{
			units <- as.integer(units)

			for (i in seq_along(units)) {
				.regularizer <- regularizer_l2(l2_penalty[[i]])
				self$dense[[i]] <- layer_dense(
					object = ,
					units = units[[i]],
					activation = activation[[i]],
					kernel_regularizer = .regularizer,
					trainable = TRUE
					)

				rate <- dropout[[i]]
				if (!is.null(rate)) {
					self$dropout[[i]] <-
						layer_dropout(rate = rate)
				}

			}
		},

		call = function(x, mask = NULL) {
			for (i in seq_along(self$dense))
			{
				x <- self$dense[[i]](x)
				if (!is.null(self$dropout)) {
					x <- self$dropout[[i]](x)
				}
			}
			return(x)
		},

		compute_output_shape = function(input_shape) {
			n <- length(self$dense)
			reticulate::tuple(
				input_shape[[1]], self$dense[[n]]$units
				)
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
	dropout = NULL,
	name = NULL,
	trainable = TRUE
	)
{
	keras::create_layer(
		MLPLayer, object, list(units = units,
				       activation = activation,
				       l2_penalty = l2_penalty,
				       dropout = dropout,
				       name = name,
				       trainable = trainable
				       )
		)
}
