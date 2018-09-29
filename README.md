## Learning distinct writing styles using LSTM networks


Try it! http://neurodeep.shinyapps.io/writerchoice

### This app uses recurrent neural networks trained to learn the writing style of a wide array of authors

I trained recurrent neural networks to predict the next character in large corpuses of available writing from authors with distinct styles. Poetry from EE Cummings, Poe and Shakespere; verse from Chaucer and Malory; Grimm Fairy Tales and modern work by Patrick Rothfuss. 

Each corpus was trained using a different network architecture adjusted to the size and complexity of the corpus. 

The creativity slider allows the network to generate novel text at the risk of occasionally diverging from standard English. 

This project uses Keras for R with Tensorflow and is deployed using Shiny

