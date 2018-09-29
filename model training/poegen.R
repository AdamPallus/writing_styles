library(keras)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)

# Parameters --------------------------------------------------------------

maxlen <- 50

# Data Preparation --------------------------------------------------------


#SET GPU
library(tensorflow)
config <- list()
config$gpu_options$allow_growth = TRUE

session_conf <- do.call(tf$ConfigProto, config)
## 

library(stringi)
z<-readLines('POEtrim.txt',encoding="UTF-8",skipNul=TRUE)
# z<-stri_trans_general(z, "latin-ascii")

# z<- z[-1]

z<-str_replace_all(z,"[1234567890*_]","")
# 
# z<- readChar('slowregard2.txt',file.info('slowregard2.txt')$size)

text <- z %>%
  # str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE,lowercase=FALSE)

# bad = c("\\","¿","€",">","™")

# text[text %in% bad] = ""

print(sprintf("corpus length: %d", length(text)))

chars <- text %>%
  unique() %>%
  sort()

print(sprintf("total chars: %d", length(chars)))  

# Cut the text in semi-redundant sequences of maxlen characters
dataset <- map(
  seq(1, length(text) - maxlen - 1, by = 3), 
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
  # layer_lstm(128, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_dropout(rate = 0.2) %>%
  # layer_lstm(256, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(128, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(64, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  # layer_lstm(64, input_shape = c(maxlen, length(chars)),return_sequences = TRUE) %>%
  layer_lstm(1024, input_shape = c(maxlen, length(chars))) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(length(chars)) %>%
  layer_activation("softmax")

# optimizer <- optimizer_rmsprop(lr = 0.01)
optimizer <- optimizer_rmsprop(lr = 0.005)

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

{
sink('TrainingPoe-128x2dropout.txt',append=TRUE)
for(iteration in 1:5){
  
  cat(sprintf("iteration: %02d ---------------\n\n", (iteration*20)))
  
  hist<-model %>% fit(
    X, y,
    # batch_size = 128,
    batch_size =2000,
    epochs = 10,
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
}
sink()
}
#After training

{
  sink('Poe4-1.txt')
  summary(model)
  # for(diversity in c(0.2, 0.4, 0.6,0.8, 1, 1.2,1.5)){
  for(diversity in c(0.01,0.1,0.2,.1,.4,.6,.8,1)){
    # for (repeats in 1:3){
    # poem_length=as.integer(runif(n=1,min=200,max=2000))
    poem_length=2000
    # for(diversity in c(0.80)){  
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - maxlen), size = 1)
    sentence <- text[start_index:(start_index + maxlen - 1)]
    generated <- ""
    
    #alternative way to seed with custom string:
    # sentence<-"Just then Auri opened her eyes"
    # sentence<-"hey liz, what's up??" #must be 20 characters
    # sentence<-"can't feel\nyour face"
    # sentence<- "o the feeling of shocking fuzz"
    # cat(sentence)
    # originalsentence<-sentence
    # sentence<-strsplit(sentence,split=NULL)[[1]]
   
    # diversity=1
    
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
    
  # }
  sink()
}

save_model_hdf5(model,'POEmodel4.hd5') 
saveRDS(text,'POE1text.RDS')
saveRDS(chars,'POE1chars.RDS')

#note: SlowRegard7 uses two layers of 128 
#note: SlowRegard8 uses 4 layers of 64
#note: SlowRegard9 uses 3 layers of 128 64 and 64
#note: SlowRegard10 uses 2 layers of 128 and 2 dropout layers of 0.5
#11 is 1 512 layer
#slow regard 12 is 3 128 layers

#eemodel3 used two 256 layers with dropout of 0.2

#eemodel4 used a 64 and a 512 with dropout of 0.2

#eemodel5 used a 128 and 128 with 0.5 dropout and reduced sentence length of 20 