
<!-- README.md is generated from README.Rmd. Please edit that file -->

# neuralcf

<!-- badges: start -->
<!-- badges: end -->

This package provides support for implementing recommender systems using
the R Keras interface.

## Installation

You can install the development version from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("vgherard/neuralcf")
```

## Example

``` r
library(neuralcf)
```

This example shows how to train and evaluate a neural network with the
[NeuMF](https://dl.acm.org/doi/10.1145/3038912.3052569) architecture,
for explicit feedback recommendations, using the MovieLense-100k
dataset:

``` r
data <- ml_100k
head(data)
#>   user item rating timestamp
#> 1  196  242      3 881250949
#> 2  186  302      3 891717742
#> 3   22  377      1 878887116
#> 4  244   51      2 880606923
#> 5  166  346      1 886397596
#> 6  298  474      4 884182806
c(num_users, num_items) %<-% c(max(ml_100k[["user"]]), max(ml_100k[["item"]]))
```

We define our model using the sequential Keras API (all the steps
explicitly illustrated here can be performed automatically through the
utility `neuralcf::neumf_recommender()`). Our input is a pair of (user,
item) integer ids:

``` r
input <- list(
    layer_input(1L, name = "user_input", dtype = "int32"),
    layer_input(1L, name = "item_input", dtype = "int32")
    )
```

A Generalized Matrix Factorization (GMF) layer embeds ‘user’ and ‘item’
in the same latent space and performs element-wise multiplication:

``` r
gmf_output <- layer_gmf(
        input,
        n = c(num_users, num_items),
        emb_dim = 64,
        emb_l2_penalty = 1e-4,
        name = "gmf"
    )
```

In parallel, we perform two independent embeddings of ‘user’ and ‘item’,
and process the concatenated input through a Multi-Layer Perceptron
(MLP) with four hidden layers:

``` r
mlp_input <- lapply(1:2, function(i) {
        layer_embedding(input[[i]],
                input_dim = c(num_users, num_items)[[i]] + 1,
                output_dim = 32,
                embeddings_initializer = "glorot_uniform",
                embeddings_regularizer = regularizer_l2(1e-4) 
                )
    }) %>% layer_concatenate %>% layer_flatten
mlp_output <- layer_mlp(
        mlp_input,
        units = c(64, 32, 16),
        activation = "relu",
        l2_penalty = c(1e-4, 3e-5, 1e-5),
        dropout_rate = c(0.3, 0.2, 0.1),
        name = "mlp"
    )
```

Finally, we concatenate the GMF and MLP outputs and project to a single
output dimension (a positive number, rating of ‘user’ to ‘item’):

``` r
gmf_mlp_output <- layer_concatenate(list(gmf_output, mlp_output))
output <- layer_dense(
        gmf_mlp_output,
        units = 1L,
        activation = "relu",
        kernel_regularizer = regularizer_l2(1e-4)
    )
```

We finalize our model using:

``` r
model <- keras_model(input, output)
```

We train the model on 90% of the MovieLense data using Adam:

``` r
compile(model,
    loss = "mean_squared_error",
    optimizer = optimizer_adam(),
    metrics = c("mean_squared_error", "mean_absolute_error")
    )

train_index <- 1:(.9 * nrow(data))
train <- data[train_index, ]
h <- fit(model,
    x = list(train[["user"]], train[["item"]]),
    y = ml_100k[["rating"]],
    batch_size = 256L,
    epochs = 20L,
    validation_split = .1,
    callbacks = callback_early_stopping(monitor = "val_mean_squared_error"),
    )
```

Here’s a view on the loss and metric functions during training:

``` r
plot(h)
```

<img src="man/figures/README-unnamed-chunk-9-1.png" width="100%" />

We finally evaluate on the remaining of the data:

``` r
test <- data[-train_index, ]
evaluate(model, x = list(test[["user"]], test[["item"]]), y = test[["rating"]])
#>                loss  mean_squared_error mean_absolute_error 
#>           0.9149609           0.8911576           0.7551823
```
