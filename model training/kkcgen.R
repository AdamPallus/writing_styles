#SET GPU
library(tensorflow)
config <- list()
config$gpu_options$allow_growth = TRUE

session_conf <- do.call(tf$ConfigProto, config)
## 


library(keras)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)

# Parameters --------------------------------------------------------------

maxlen <- 20

# Data Preparation --------------------------------------------------------

model<- load_model_hdf5('KKC4.hd5')
text<- readRDS('KKC4text.RDS')
chars<- readRDS('KKC4chars.RDS')

# Cut the text in semi-redundant sequences of maxlen characters
dataset <- map(
  seq(1, length(text) - maxlen - 1, by = 5), 
  ~list(sentece = text[.x:(.x + maxlen - 1)], next_char = text[.x + maxlen])
)

dataset <- transpose(dataset)

# Vectorization
X <- array(0, dim = c(length(dataset$sentece), maxlen, length(chars)))
y <- array(0, dim = c(length(dataset$sentece), length(chars)))

for(i in 1:length(dataset$sentece)){
  
  X[i,,] <- sapply(chars, function(x){
    as.integer(x == dataset$sentece[[i]])
  })
  
  y[i,] <- as.integer(chars == dataset$next_char[[i]])
  
}

# Model Definition --------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_lstm(256, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  # layer_lstm(256, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(64, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(64, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(64, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  layer_lstm(64, input_shape = c(maxlen, length(chars))) %>%
  # layer_dropout(rate = 0.5) %>%
  layer_dense(length(chars)) %>%
  layer_activation("softmax")

# optimizer <- optimizer_rmsprop(lr = 0.01)
optimizer <- optimizer_rmsprop(lr = 0.01)

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)

# Training & Results ----------------------------------------------------

sample_mod <- function(preds, temperature = 1){
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp(preds))
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}

  
  hist<-model %>% fit(
    X, y,
    # batch_size = 128,
    batch_size =2000,
    epochs = 20,
    verbose=2
  )
  
  for(diversity in c(0.2, 0.5, 1, 1.2)){
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - maxlen), size = 1)
    sentence <- text[start_index:(start_index + maxlen - 1)]
    generated <- ""
    
    for(i in 1:400){
      
      x <- sapply(chars, function(x){
        as.integer(x == sentence)
      })
      x <- array_reshape(x, c(1, dim(x)))
      
      preds <- predict(model, x)
      next_index <- sample_mod(preds, diversity)
      
      next_char <- chars[next_index]
      
      generated <- str_c(generated, next_char, collapse = "")
      sentence <- c(sentence[-1], next_char)
      
    }
    
    cat(generated)
    cat("\n\n")
    
  }


#After training

{
  sink('KKC5-1.txt')
  summary(model)
  # for(diversity in c(0.2, 0.4, 0.6,0.8, 1, 1.2,1.5)){
  for(diversity in c(0.005,.1,.4,.6,.8,1)){
    
    # poem_length=as.integer(runif(n=1,min=200,max=2000))
    poem_length=2000
    # for(diversity in c(0.80)){  
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - maxlen), size = 1)
    sentence <- text[start_index:(start_index + maxlen - 1)]
    generated <- ""

    
    for(i in 1:poem_length){
      
      x <- sapply(chars, function(x){
        as.integer(x == sentence)
      })
      x <- array_reshape(x, c(1, dim(x)))
      
      preds <- predict(model, x)
      # next_index <- sample_mod(preds, runif(1)+0.5)
      next_index <- sample_mod(preds, diversity)
      next_char <- chars[next_index]
      
      generated <- str_c(generated, next_char, collapse = "")
      sentence <- c(sentence[-1], next_char)
      
    }
    cat(generated)
    # cat(originalsentence,generated)
    cat("\n\n")
    
  }
  sink()
}
save_model_hdf5(model,'KKC5x.hd5')
